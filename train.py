# -*- coding: utf-8 -*-
"""
COLD RoBERTa Baseline 训练代码
复现论文: COLD: A Benchmark for Chinese Offensive Language Detection (EMNLP 2022)

论文中COLDETECTOR的核心架构:
- Backbone: bert-base-chinese (论文原文) / hfl/chinese-roberta-wwm-ext (HuggingFace发布版本)
- 分类头: [CLS] -> Linear -> sigmoid, 二分类 (cross-entropy loss)
- 优化器: BertAdam (linear warmup 0.05 + decay)
- 学习率: 5e-5
- Batch size: 64
- Max epochs: 30 (early stopping)

注: 论文Section 4.1用bert-base-chinese, HuggingFace发布的roberta-base-cold用chinese-roberta-wwm-ext
    此处复现HuggingFace发布版本, 即基于RoBERTa的版本
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm
import json

from preprocess import prepare_binary_splits


class COLDDataset(Dataset):
    """COLD文本分类数据集"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def evaluate(model, dataloader, device):
    """在验证集/测试集上评估"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    # 分类别指标
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1]
    )

    results = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "non_offensive": {"precision": prec[0], "recall": rec[0], "f1": f1[0]},
        "offensive": {"precision": prec[1], "recall": rec[1], "f1": f1[1]},
    }
    return results, all_preds


def train(args):
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    data = prepare_binary_splits(args.data_dir)
    train_texts, train_labels = data["train"]
    dev_texts, dev_labels = data["dev"]
    test_texts, test_labels = data["test"]

    # 加载tokenizer和模型
    # 论文COLDETECTOR使用bert-base-chinese
    # HuggingFace发布版使用hfl/chinese-roberta-wwm-ext (即roberta-base-cold的base model)
    print(f"Loading model: {args.model_name}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    model.to(device)

    # 构建Dataset和DataLoader
    train_dataset = COLDDataset(train_texts, train_labels, tokenizer, args.max_length)
    dev_dataset = COLDDataset(dev_texts, dev_labels, tokenizer, args.max_length)
    test_dataset = COLDDataset(test_texts, test_labels, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 优化器: 论文使用BertAdam with linear warmup(0.05) + decay
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 训练循环 (with early stopping)
    best_dev_f1 = 0.0
    patience_counter = 0
    best_epoch = 0

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        # 验证
        dev_results, _ = evaluate(model, dev_loader, device)
        print(f"\nEpoch {epoch} | Train Loss: {avg_loss:.4f}")
        print(f"  Dev Acc: {dev_results['accuracy']:.4f} | Dev Macro-F1: {dev_results['macro_f1']:.4f}")
        print(f"  Offensive  - P: {dev_results['offensive']['precision']:.4f} "
              f"R: {dev_results['offensive']['recall']:.4f} F1: {dev_results['offensive']['f1']:.4f}")
        print(f"  Non-Offen. - P: {dev_results['non_offensive']['precision']:.4f} "
              f"R: {dev_results['non_offensive']['recall']:.4f} F1: {dev_results['non_offensive']['f1']:.4f}")

        # Early stopping based on macro-F1
        if dev_results["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_results["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"  -> Best model saved (Macro-F1: {best_dev_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
                break

    # 加载最佳模型并在测试集上评估
    print("\n" + "=" * 60)
    print("Loading best model for test evaluation...")
    model = BertForSequenceClassification.from_pretrained(args.output_dir)
    model.to(device)

    test_results, test_preds = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Macro-F1: {test_results['macro_f1']:.4f}")
    print(f"  Offensive  - P: {test_results['offensive']['precision']:.4f} "
          f"R: {test_results['offensive']['recall']:.4f} F1: {test_results['offensive']['f1']:.4f}")
    print(f"  Non-Offen. - P: {test_results['non_offensive']['precision']:.4f} "
          f"R: {test_results['non_offensive']['recall']:.4f} F1: {test_results['non_offensive']['f1']:.4f}")

    # 细粒度评估
    if data["test_fine_grained"] is not None:
        fine_grained_eval(test_preds, data["test_fine_grained"])

    # 保存结果
    with open(os.path.join(args.output_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)

    print(f"\n论文参考指标: Accuracy=0.81, Macro-F1=0.82 (bert-base-chinese)")
    print(f"HuggingFace roberta-base-cold参考: Accuracy=0.8275, Macro-F1=0.8239")


def fine_grained_eval(preds, fine_grained_labels):
    """
    论文Table 6: 按细粒度类别评估检测准确率

    细粒度标签:
    0: Other Non-Offensive (安全-其他非冒犯)
    1: Attack Individual   (冒犯-攻击个人)  -> 预测应为1(offensive)
    2: Attack Group         (冒犯-攻击群体)  -> 预测应为1(offensive)
    3: Anti-Bias            (安全-反偏见)    -> 预测应为0(safe)

    关键发现:
    - COLDETECTOR对Attack Individual/Group检测较好 (79.51%/85.49%)
    - 但对Anti-Bias的识别很差 (38.32%), 因为Anti-Bias常采用"先承认后否认"的表达方式
      例如"女性在职场经常被歧视，但我不认为这是对的"
      分类器容易只关注前半句而忽略后面的反偏见陈述
    """
    fg_map = {0: "Other Non-Offen", 1: "Attack Individual",
              2: "Attack Group", 3: "Anti-Bias"}

    # 细粒度标签对应的正确二分类标签
    fg_to_binary = {0: 0, 1: 1, 2: 1, 3: 0}

    print("\n--- Fine-grained Evaluation (per subcategory accuracy) ---")
    for fg_label in [1, 2, 3, 0]:
        indices = [i for i, fl in enumerate(fine_grained_labels) if fl == fg_label]
        if not indices:
            continue
        correct_binary = fg_to_binary[fg_label]
        correct = sum(1 for i in indices if preds[i] == correct_binary)
        acc = correct / len(indices)
        print(f"  {fg_map[fg_label]:20s}: {acc:.4f} ({correct}/{len(indices)})")


def parse_args():
    parser = argparse.ArgumentParser(description="Train COLD RoBERTa baseline")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext",
                        help="预训练模型名称. 论文用bert-base-chinese, HF发布版用hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--data_dir", type=str, default="COLDataset")
    parser.add_argument("--output_dir", type=str, default="output/roberta-cold")
    parser.add_argument("--max_length", type=int, default=128,
                        help="最大序列长度. 数据集平均长度~49字符, 128足够覆盖")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="论文设置: batch_size=64")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="论文设置: learning_rate=5e-5")
    parser.add_argument("--epochs", type=int, default=30,
                        help="论文设置: max_epoch=30, 配合early stopping")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="论文设置: linear warmup proportion=0.05")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
