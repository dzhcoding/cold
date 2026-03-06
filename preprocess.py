# -*- coding: utf-8 -*-
"""
COLD数据集预处理模块
复现论文: COLD: A Benchmark for Chinese Offensive Language Detection (EMNLP 2022)

数据预处理流程:
1. 原始数据来源: 知乎(Zhihu)和微博(Weibo)的评论
2. 关键词查询采集 + 子话题爬取 (race/gender/region三个主题)
3. 后处理: 保留长度5-200 token的样本, 清洗emoji/URL/用户名/空白字符
4. Model-in-the-loop半自动标注 (训练集), 人工精标 (测试集)
"""

import re
import pandas as pd
from typing import Tuple


def clean_text(text: str) -> str:
    """
    复现论文的后处理清洗逻辑 (Appendix B.2):
    - 清洗emoji、URL、用户名、多余空白
    - 保留长度5-200 token的样本
    """
    if not isinstance(text, str):
        return ""
    # 移除URL
    text = re.sub(r'https?://\S+', '', text)
    # 移除@用户名
    text = re.sub(r'@\S+', '', text)
    # 移除emoji (Unicode emoji范围, 注意不能覆盖CJK字符区U+4E00-U+9FFF)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U0001F900-\U0001FA9F"  # supplemental symbols
        "\U0001FA00-\U0001FAFF"  # chess symbols & extended-A
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub('', text)
    # 移除微博表情标签 [xxx]
    text = re.sub(r'\[[\u4e00-\u9fff\w]+\]', '', text)
    # 合并多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def filter_by_length(text: str, min_len: int = 5, max_len: int = 200) -> bool:
    """论文要求: 保留长度在5-200 token之间的样本"""
    return min_len <= len(text) <= max_len


def load_dataset(data_dir: str = "COLDataset") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载COLD数据集的train/dev/test分割

    标签体系:
    - train/dev: 二分类 (0=safe, 1=offensive)
    - test: 细粒度四分类
        0: Other Non-Offensive (安全-其他非冒犯)
        1: Attack Individual   (冒犯-攻击个人)
        2: Attack Group         (冒犯-攻击群体)
        3: Anti-Bias            (安全-反偏见)

    Returns:
        (train_df, dev_df, test_df)
    """
    train_df = pd.read_csv(f"{data_dir}/train.csv", encoding="utf-8-sig")
    dev_df = pd.read_csv(f"{data_dir}/dev.csv", encoding="utf-8-sig")
    test_df = pd.read_csv(f"{data_dir}/test.csv", encoding="utf-8-sig")

    # 清洗文本
    for df in [train_df, dev_df, test_df]:
        df["TEXT"] = df["TEXT"].apply(clean_text)

    # 统一测试集的二分类标签 (细粒度 -> 二分类)
    # fine-grained-label 1,2 -> offensive(1); 0,3 -> safe(0)
    if "fine-grained-label" in test_df.columns:
        test_df["binary_label"] = test_df["fine-grained-label"].apply(
            lambda x: 1 if x in [1, 2] else 0
        )

    return train_df, dev_df, test_df


def get_label_stats(df: pd.DataFrame, label_col: str = "label") -> dict:
    """打印标签分布统计"""
    counts = df[label_col].value_counts().to_dict()
    total = len(df)
    stats = {k: {"count": v, "ratio": v / total} for k, v in counts.items()}
    return stats


def prepare_binary_splits(data_dir: str = "COLDataset"):
    """
    准备二分类任务的数据分割, 供训练使用

    论文数据规模:
    - Train: 25,726 (Offen: 15,934 / Non-Offen: 16,223 合计32,157 含dev)
    - Dev:   6,431
    - Test:  5,323 (Attack Individual: 288, Attack Group: 1819,
                     Anti-Bias: 668, Other Non-Offen: 2548)
    """
    train_df, dev_df, test_df = load_dataset(data_dir)

    train_texts = train_df["TEXT"].tolist()
    train_labels = train_df["label"].tolist()

    dev_texts = dev_df["TEXT"].tolist()
    dev_labels = dev_df["label"].tolist()

    test_texts = test_df["TEXT"].tolist()
    # 测试集使用映射后的二分类标签
    test_labels = test_df["binary_label"].tolist()

    print(f"Train: {len(train_texts)} samples")
    print(f"  Label distribution: {get_label_stats(train_df)}")
    print(f"Dev:   {len(dev_texts)} samples")
    print(f"  Label distribution: {get_label_stats(dev_df)}")
    print(f"Test:  {len(test_texts)} samples")
    print(f"  Label distribution: {get_label_stats(test_df, 'binary_label')}")

    if "fine-grained-label" in test_df.columns:
        fg_map = {0: "Other Non-Offen", 1: "Attack Individual",
                  2: "Attack Group", 3: "Anti-Bias"}
        print(f"  Fine-grained: {test_df['fine-grained-label'].map(fg_map).value_counts().to_dict()}")

    return {
        "train": (train_texts, train_labels),
        "dev": (dev_texts, dev_labels),
        "test": (test_texts, test_labels),
        "test_fine_grained": test_df["fine-grained-label"].tolist() if "fine-grained-label" in test_df.columns else None,
    }


if __name__ == "__main__":
    data = prepare_binary_splits()
