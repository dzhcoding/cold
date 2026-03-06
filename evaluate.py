# -*- coding: utf-8 -*-
"""
COLD 评估与推理模块
支持:
1. 加载训练好的模型或HuggingFace官方 roberta-base-cold 进行推理
2. 在测试集上做完整评估 (二分类 + 细粒度)
3. 单条/批量文本的冒犯性检测
"""

import argparse
import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report

from preprocess import prepare_binary_splits


def load_model(model_path: str, device: str = None):
    """加载模型和tokenizer"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict(texts, model, tokenizer, device, batch_size=64):
    """
    批量预测

    Returns:
        preds: list of int (0=non-offensive, 1=offensive)
        probs: list of float (offensive的概率)
    """
    all_preds = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoding = tokenizer(
            batch_texts,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs[:, 1].cpu().tolist())  # offensive的概率

    return all_preds, all_probs


def evaluate_on_testset(model_path: str, data_dir: str = "COLDataset"):
    """完整测试集评估, 复现论文Table 5和Table 6"""
    model, tokenizer, device = load_model(model_path)
    data = prepare_binary_splits(data_dir)

    test_texts, test_labels = data["test"]
    preds, probs = predict(test_texts, model, tokenizer, device)

    # 二分类指标 (论文Table 5)
    acc = accuracy_score(test_labels, preds)
    macro_f1 = f1_score(test_labels, preds, average="macro")

    print("=" * 60)
    print("Binary Classification Results (Table 5 format)")
    print("=" * 60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Macro-F1:  {macro_f1:.4f}")
    print()
    print(classification_report(
        test_labels, preds,
        target_names=["Non-Offensive", "Offensive"],
        digits=4
    ))

    # 细粒度评估 (论文Table 6)
    if data["test_fine_grained"] is not None:
        fg_labels = data["test_fine_grained"]
        fg_map = {0: "Other Non-Offen.", 1: "Attack Individual",
                  2: "Attack Group", 3: "Anti-Bias"}
        fg_to_binary = {0: 0, 1: 1, 2: 1, 3: 0}

        print("=" * 60)
        print("Fine-grained Subcategory Accuracy (Table 6 format)")
        print("=" * 60)
        print(f"{'Category':<22} {'Accuracy':>10} {'Count':>8}")
        print("-" * 42)

        for fg_label in [1, 2, 3, 0]:
            indices = [i for i, fl in enumerate(fg_labels) if fl == fg_label]
            if not indices:
                continue
            correct_binary = fg_to_binary[fg_label]
            correct = sum(1 for i in indices if preds[i] == correct_binary)
            sub_acc = correct / len(indices)
            print(f"  {fg_map[fg_label]:<20} {sub_acc:>8.2%} {len(indices):>8}")

        print()
        print("论文参考值 (COLDETECTOR / bert-base-chinese):")
        print("  Attack Individual:  79.51%")
        print("  Attack Group:       85.49%")
        print("  Anti-Bias:          38.32%  <- 最难的类别")
        print("  Other Non-Offen.:   89.29%")
        print("  Overall Accuracy:   81.06%")


def interactive_demo(model_path: str):
    """交互式冒犯性检测demo"""
    model, tokenizer, device = load_model(model_path)

    label_map = {0: "Non-Offensive (安全)", 1: "Offensive (冒犯)"}

    print("=" * 60)
    print("COLD 冒犯性文本检测器 (输入 q 退出)")
    print("=" * 60)

    while True:
        text = input("\n请输入文本: ").strip()
        if text.lower() == "q":
            break
        if not text:
            continue

        preds, probs = predict([text], model, tokenizer, device)
        pred = preds[0]
        prob = probs[0]

        print(f"  预测: {label_map[pred]}")
        print(f"  冒犯概率: {prob:.4f}")


def batch_predict_demo(model_path: str):
    """批量预测示例, 复现HuggingFace页面的demo"""
    model, tokenizer, device = load_model(model_path)

    # HuggingFace页面的示例
    texts = [
        "你就是个傻逼！",         # -> Offensive (直接攻击个人)
        "黑人很多都好吃懒做，偷奸耍滑！",  # -> Offensive (攻击群体)
        "男女平等，黑人也很优秀。",      # -> Non-Offensive (反偏见)
    ]

    preds, probs = predict(texts, model, tokenizer, device)

    label_map = {0: "Non-Offensive", 1: "Offensive"}
    print("\nBatch Prediction Results:")
    print("-" * 60)
    for text, pred, prob in zip(texts, preds, probs):
        print(f"  [{label_map[pred]:>14}] (p={prob:.4f}) {text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="thu-coai/roberta-base-cold",
                        help="模型路径, 可以是本地训练的或HuggingFace的thu-coai/roberta-base-cold")
    parser.add_argument("--data_dir", type=str, default="COLDataset")
    parser.add_argument("--mode", type=str, default="eval",
                        choices=["eval", "demo", "batch"],
                        help="eval: 测试集评估, demo: 交互式, batch: 批量示例")
    args = parser.parse_args()

    if args.mode == "eval":
        evaluate_on_testset(args.model_path, args.data_dir)
    elif args.mode == "demo":
        interactive_demo(args.model_path)
    elif args.mode == "batch":
        batch_predict_demo(args.model_path)
