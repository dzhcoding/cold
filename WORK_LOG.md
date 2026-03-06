# COLD 项目复现工作日志

## 一、项目概述

复现论文 **"COLD: A Benchmark for Chinese Offensive Language Detection"** (EMNLP 2022, 清华大学CoAI组)

- 论文地址: https://arxiv.org/abs/2201.06025
- 数据集仓库: https://github.com/thu-coai/COLDataset
- 官方模型: https://huggingface.co/thu-coai/roberta-base-cold

---

## 二、环境搭建

### 2.1 仓库克隆

```bash
cd d:/code/COLD
git clone https://github.com/thu-coai/COLDataset.git .
```

仓库结构:
```
COLD/
├── COLDataset/
│   ├── train.csv    # 25,726条 (二分类)
│   ├── dev.csv      #  6,431条 (二分类)
│   └── test.csv     #  5,323条 (四分类细粒度标签)
├── LICENSE
└── README.md
```

### 2.2 依赖安装

运行环境: Windows 10 Pro, Python 3.14, CPU (无GPU)

```bash
pip install torch transformers tqdm pandas scikit-learn
```

安装的核心版本:
- PyTorch 2.10.0 (CPU)
- Transformers 5.3.0
- pandas 3.0.1, scikit-learn 1.8.0

---

## 三、论文研究解读

### 3.1 核心贡献

1. **COLDATASET**: 37,480条中文冒犯性语言数据集，覆盖种族/性别/地域三大主题
2. **COLDETECTOR**: 基于BERT/RoBERTa的基线检测器
3. **生成模型安全评估**: 用检测器评估CPM/CDialGPT/EVA的冒犯性生成问题

### 3.2 数据采集与预处理流程

**数据来源**: 知乎 + 微博

**采集方式**:
- 关键词查询: 预定义目标群体关键词 (如 黑人/女权/地域黑 等)，在平台搜索获取高密度数据
- 子话题爬取: 直接爬取热门子话题下的评论，覆盖更广泛表达

**后处理** (Appendix B.2):
- 移除 emoji、URL、用户名、多余空白字符
- 保留长度 5-200 token 的样本

**标注流程 (Model-in-the-loop)**:
- 训练集: 半自动标注，6轮迭代 (分类器预测 → 按分数分bin → 抽样10%检查 → 准确率≥90%直接采纳，否则整bin重标)
- 测试集: 完全人工标注，四分类细粒度标签

### 3.3 标签体系

**训练集/验证集 (二分类)**:
| label | 含义 |
|-------|------|
| 0 | Non-Offensive (安全) |
| 1 | Offensive (冒犯) |

**测试集 (四分类细粒度)**:
| fine-grained-label | 含义 | 二分类映射 | 数量 |
|-------------------|------|-----------|------|
| 0 | Other Non-Offensive (其他非冒犯) | 0 (safe) | 2,548 |
| 1 | Attack Individual (攻击个人) | 1 (offensive) | 288 |
| 2 | Attack Group (攻击群体) | 1 (offensive) | 1,819 |
| 3 | Anti-Bias (反偏见) | 0 (safe) | 668 |

### 3.4 冒犯性判定边界

**官方定义**: 任何形式的针对个人或群体的攻击性内容，包括隐含或直接的冒犯: 粗鲁、不尊重、侮辱、威胁、亵渎，以及任何使他人不适的有毒内容。

**关键难点**:
- **Anti-Bias 最难判定** (准确率仅~40%): 反偏见内容常用"先承认后否认"句式，如 "女性在职场被歧视，但我认为这不对"，模型容易只关注前半句而误判
- **关键词 ≠ 冒犯**: 敏感词在冒犯和非冒犯文本中都出现，仅靠关键词匹配 accuracy 只有54%
- **隐含冒犯**: 不含脏话但表达歧视态度的内容对模型挑战更大

### 3.5 COLDETECTOR 架构

```
Input Text → [CLS] + Tokens → BERT/RoBERTa Encoder → [CLS]隐藏状态 → Linear → sigmoid → 二分类
```

**训练超参数** (Appendix B.3):
| 参数 | 值 |
|------|-----|
| Backbone | bert-base-chinese (论文) / hfl/chinese-roberta-wwm-ext (HF发布版) |
| Optimizer | BertAdam (AdamW) with linear warmup + decay |
| Learning rate | 5e-5 |
| Batch size | 64 |
| Max epochs | 30 (配合 early stopping) |
| Warmup ratio | 0.05 |
| Loss | Binary cross-entropy |

### 3.6 论文对比的6种方法

| 方法 | 思路 | Accuracy |
|------|------|----------|
| Random | 随机猜 | 50% |
| KEYMAT | 14k敏感词关键词匹配 | 54% |
| PSELFDET | Prompt-based 自检测 | 59% |
| TJIGDET | 英文 Jigsaw 翻译成中文训练 | 60% |
| BAIDU TC | 百度文本审核API | 63% |
| **COLDETECTOR** | **BERT在COLDATASET上微调** | **81%** |

---

## 四、编写的复现代码

### 4.1 preprocess.py — 数据预处理模块

功能:
- `clean_text()`: 文本清洗 (移除URL/用户名/emoji/微博表情标签/多余空白)
- `load_dataset()`: 加载 train/dev/test 三个分割，自动映射测试集细粒度标签到二分类
- `prepare_binary_splits()`: 准备训练用的数据分割，打印标签分布统计

**修复记录**: emoji 正则中 `\U000024C2-\U0001F251` 范围过大覆盖了 CJK 字符区 (U+4E00-U+9FFF)，导致所有中文被清除。已修复为精确的 emoji 专用范围。

### 4.2 train.py — RoBERTa 基线训练代码

功能:
- `COLDDataset`: PyTorch Dataset，封装 tokenizer 编码
- `train()`: 完整训练循环 (含 early stopping、学习率warmup调度、梯度裁剪)
- `evaluate()`: 计算 accuracy / macro-F1 / 分类别 P/R/F1
- `fine_grained_eval()`: 按四个细粒度子类别分别计算检测准确率

使用方式:
```bash
# 用 RoBERTa 从头训练 (需GPU)
python train.py --model_name hfl/chinese-roberta-wwm-ext --batch_size 64 --lr 5e-5

# 用 BERT 从头训练 (复现论文原始设置)
python train.py --model_name bert-base-chinese --batch_size 64 --lr 5e-5
```

### 4.3 evaluate.py — 评估与推理模块

功能:
- `evaluate_on_testset()`: 在测试集上做完整评估 (二分类Table 5 + 细粒度Table 6)
- `interactive_demo()`: 交互式单条文本检测
- `batch_predict_demo()`: 批量预测示例

使用方式:
```bash
# 用官方模型评估
python evaluate.py --model_path thu-coai/roberta-base-cold --mode eval

# 交互式demo
python evaluate.py --model_path thu-coai/roberta-base-cold --mode demo

# 批量示例
python evaluate.py --model_path thu-coai/roberta-base-cold --mode batch
```

### 4.4 ANNOTATION_GUIDELINES.md — 标注指南文档

详细记录了论文的冒犯性判定定义、四分类标注体系、判定边界难点、三大主题关键词等。

---

## 五、评估结果

### 5.1 使用官方 roberta-base-cold 模型在测试集上的复现结果

**二分类指标** (对应论文 Table 5):

| 指标 | 我们的结果 | HuggingFace 参考 | 论文参考 (BERT) |
|------|-----------|-----------------|----------------|
| Accuracy | **82.66%** | 82.75% | 81% |
| Macro-F1 | **82.30%** | 82.39% | 82% |

分类别详细指标:

| 类别 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| Non-Offensive | 90.02% | 80.19% | 84.82% |
| Offensive | 74.08% | 86.43% | 79.78% |

**细粒度子类别准确率** (对应论文 Table 6):

| 类别 | 我们的结果 | 论文 (BERT) | 差异 |
|------|-----------|------------|------|
| Attack Individual | **83.68%** | 79.51% | +4.17% |
| Attack Group | **86.86%** | 85.49% | +1.37% |
| Anti-Bias | **41.62%** | 38.32% | +3.30% |
| Other Non-Offen. | **90.31%** | 89.29% | +1.02% |

### 5.2 结果分析

1. **复现成功**: 所有指标与论文/HuggingFace参考值高度吻合
2. **RoBERTa 略优于 BERT**: 各子类别准确率均有1-4%的提升，符合预期
3. **Anti-Bias 仍是最大挑战**: 即使用 RoBERTa 也仅 41.62%，验证了论文的核心发现——模型难以理解"先承认后否认"的反偏见表达
4. **Offensive Recall 高于 Precision**: 模型倾向于将更多内容判定为冒犯 (recall 86.43% > precision 74.08%)，说明存在一定程度的过度敏感

---

## 六、后续可做的工作

1. **路径B: 从头训练** — 在GPU环境下用 `train.py` 从头微调，验证训练流程能否复现论文指标
2. **Anti-Bias 误判分析** — 深入分析被误判为 Offensive 的 Anti-Bias 样本，理解模型失败模式
3. **backbone 对比** — 对比 bert-base-chinese vs hfl/chinese-roberta-wwm-ext 的效果差异
4. **按主题分析** — 分别评估 race/gender/region 三个主题的检测表现
5. **改进方向** — 针对 Anti-Bias 难题，探索长距离语义理解或对比学习等改进方案
