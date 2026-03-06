# 第 08 章 高效微调技术 (Parameter-Efficient Fine-Tuning, PEFT)

## 1. PEFT 背景（为什么需要高效微调）

### 1.1 传统微调的问题

随着大语言模型规模不断增长，全量微调（Full Fine-Tuning）面临严峻挑战：

| 问题 | 描述 | 影响 |
|------|------|------|
| **显存需求高** | 7B 模型全量微调需要 80GB+ 显存 | 大多数用户无法承担 |
| **计算成本高** | 需要更新所有参数 | 训练时间长，电费高 |
| **存储开销大** | 每个任务保存完整模型副本 | 存储成本指数增长 |
| **灾难性遗忘** | 微调后丢失预训练知识 | 泛化能力下降 |

**显存需求示例（以 7B 模型为例）：**

```
模型权重 (FP16):     14 GB
梯度 (FP16):         14 GB
优化器状态 (Adam):   42 GB (3x 模型参数)
激活值：             10-20 GB
---------------------------
总计：              80-90 GB
```

### 1.2 PEFT 的核心思想

**Parameter-Efficient Fine-Tuning (PEFT)** 的核心思想：

> **冻结预训练模型的大部分参数，仅微调少量额外参数**

**优势：**

1. **显存效率**：减少 70-90% 显存占用
2. **计算效率**：训练速度提升 2-5 倍
3. **存储效率**：仅保存少量适配器参数（几 MB 到几百 MB）
4. **多任务支持**：同一基座模型可加载多个适配器
5. **减少遗忘**：预训练知识得到更好保留

### 1.3 PEFT 方法分类

```
PEFT 方法
├── 低秩适配 (Low-Rank Adaptation)
│   ├── LoRA
│   ├── QLoRA
│   └── AdaLoRA
├── 提示微调 (Prompt Tuning)
│   ├── Prefix Tuning
│   ├── P-Tuning v1/v2
│   └── Soft Prompts
├── 适配器方法 (Adapter Methods)
│   ├── Houlsby Adapter
│   ├── Pfeiffer Adapter
│   └── Compacter
└── 其他方法
    ├── IA³ (Infused Adapter by Inhibiting and Amplifying)
    ├── LLaMA-Adapter
    └── DoRA (Weight-Decomposed Low-Rank Adaptation)
```

---

## 2. LoRA 原理与实现

### 2.1 LoRA 核心思想

**LoRA (Low-Rank Adaptation)** 由微软在 2021 年提出，核心洞察：

> **模型微调过程中的参数变化具有低秩特性**

**传统微调：**
```
W' = W + ΔW  (ΔW 与 W 同维度，全量更新)
```

**LoRA 微调：**
```
W' = W + ΔW = W + BA
其中：
- W: 冻结的预训练权重 (d × k)
- B: 可训练矩阵 (d × r)
- A: 可训练矩阵 (r × k)
- r: 秩 (rank)，通常 r << d, k
```

### 2.2 LoRA 架构图

```
输入 x
  │
  ├──────────────────────┐
  │                      ↓
  │                  冻结权重 W
  │                      ↓
  │                    Wx
  │                      │
  │    ┌────────────┐    │
  └───→│ 可训练矩阵 A │    │
       └────────────┘    │
              ↓          │
       ┌────────────┐    │
       │ 可训练矩阵 B │    │
       └────────────┘    │
              ↓          │
            BAx          │
              │          │
              ↓          ↓
              └───── (+) ─────→ 输出 h = Wx + BAx
```

### 2.3 LoRA 数学推导

**前向传播：**
```
h = Wx + ΔWx = Wx + BAx
```

**参数量对比：**
- 全量微调：d × k 个参数
- LoRA: (d + k) × r 个参数

**示例（7B 模型的 attention 层）：**
```
d = 4096, k = 4096, r = 8

全量微调：4096 × 4096 = 16,777,216 参数
LoRA: (4096 + 4096) × 8 = 65,536 参数

减少比例：99.6%
```

### 2.4 LoRA 实现细节

**1. 初始化策略：**
- A 矩阵：高斯随机初始化 N(0, σ²)
- B 矩阵：零初始化
- 确保初始时 ΔW = 0

**2. 缩放因子：**
```
h = Wx + (α/r) × BAx
```
其中 α 是超参数，通常设为 r 或 2r

**3. 应用位置：**
- 主要应用于 attention 的 Q、V 投影
- 也可应用于 MLP 层
- 实验表明 Q+V 组合效果最好

### 2.5 LoRA 超参数选择

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| rank (r) | 8-64 | 越大表达能力越强，但参数量增加 |
| alpha (α) | r 或 2r | 缩放因子 |
| dropout | 0.05-0.1 | 防止过拟合 |
| 目标模块 | q_proj, v_proj | 或 all linear 层 |

---

## 3. QLoRA 量化微调

### 3.1 QLoRA 核心思想

**QLoRA (Quantized LoRA)** 在 LoRA 基础上引入量化技术，进一步降低显存需求：

```
QLoRA = 4-bit 量化基座模型 + LoRA 微调
```

### 3.2 量化技术

**NF4 (Normal Float 4-bit)：**

- 针对正态分布优化的 4-bit 数据类型
- 比标准 4-bit 量化保留更多信息
- 特别适合量化神经网络权重

**双重量化 (Double Quantization)：**

```
第一次量化：权重 → 4-bit
第二次量化：量化常数 → 8-bit
```

额外减少约 0.4 bits/parameter

### 3.3 QLoRA 显存对比

| 方法 | 7B 模型显存 | 13B 模型显存 |
|------|------------|-------------|
| 全量微调 (FP16) | ~80 GB | ~160 GB |
| LoRA (FP16) | ~16 GB | ~32 GB |
| **QLoRA (4-bit)** | **~8 GB** | **~16 GB** |

### 3.4 QLoRA 实现要点

```python
# 使用 bitsandbytes 进行 4-bit 量化
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
```

---

## 4. Prefix Tuning / P-Tuning

### 4.1 Prefix Tuning

**核心思想：** 在输入序列前添加可训练的前缀向量

```
原始输入：[x1, x2, ..., xn]
Prefix Tuning: [p1, p2, ..., pm, x1, x2, ..., xn]
```

**特点：**
- 仅在前缀层添加可训练参数
- 不修改模型主体权重
- 适用于生成任务

### 4.2 P-Tuning v1

**改进点：** 使用提示编码器生成前缀

```
离散提示 → 提示编码器 → 连续前缀向量
```

**优势：**
- 连续提示比离散提示更易优化
- 跨任务泛化能力更强

### 4.3 P-Tuning v2

**核心改进：**

1. **多层前缀**：在每一层都添加前缀，不仅是输入层
2. **重参数化**：使用 MLP 编码前缀，提高稳定性
3. **多任务学习**：支持分类和生成任务

**架构：**
```
Layer 1: [Prefix_1] + Hidden_1
Layer 2: [Prefix_2] + Hidden_2
...
Layer N: [Prefix_N] + Hidden_N
```

### 4.4 三种方法对比

| 特性 | Prefix Tuning | P-Tuning v1 | P-Tuning v2 |
|------|--------------|-------------|-------------|
| 前缀位置 | 仅输入层 | 仅输入层 | 所有层 |
| 参数量 | 较少 | 较少 | 中等 |
| 适用任务 | 生成 | 生成 | 生成 + 分类 |
| 性能 | 中等 | 中等 | 较好 |

---

## 5. Adapter 方法

### 5.1 Houlsby Adapter

**架构：**
```
输入 → LayerNorm → Adapter → LayerNorm → 输出
              ↓           ↓
         下投影 (↓)   上投影 (↑)
              ↓           ↓
           激活函数 (ReLU/GELU)
```

**位置：** 在 Transformer 的 FFN 层后添加

### 5.2 Pfeiffer Adapter

**改进：**
- 仅在 FFN 后添加 adapter
- 不在 attention 后添加
- 减少参数量，保持性能

### 5.3 Compacter

**核心思想：** 使用低秩分解和 Kronecker 积进一步压缩 adapter

```
传统 Adapter: W_down (d×m), W_up (m×d)
Compacter: 使用 Kronecker 积分解，参数减少 100x
```

### 5.4 Adapter 参数量

| 方法 | 每层参数 | 7B 模型总参数 |
|------|---------|--------------|
| Houlsby | ~0.5M | ~16M |
| Pfeiffer | ~0.25M | ~8M |
| Compacter | ~0.01M | ~0.3M |

---

## 6. 对比与选择建议

### 6.1 方法对比总表

| 方法 | 显存效率 | 训练速度 | 性能 | 易用性 | 推荐场景 |
|------|---------|---------|------|--------|---------|
| **LoRA** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 通用首选 |
| **QLoRA** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 显存受限 |
| **P-Tuning v2** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 分类任务 |
| **Adapter** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 多任务学习 |
| **全量微调** | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 充足资源 |

### 6.2 选择建议

**场景 1：消费级显卡 (8-16GB)**
- 推荐：QLoRA (4-bit)
- 可微调：7B-13B 模型

**场景 2：单卡 A100/A800 (40-80GB)**
- 推荐：LoRA (FP16)
- 可微调：13B-70B 模型

**场景 3：多卡集群**
- 推荐：全量微调 或 LoRA
- 可微调：任意规模

**场景 4：多任务部署**
- 推荐：Adapter 或 LoRA
- 优势：共享基座，切换适配器

### 6.3 最佳实践

1. **从 LoRA 开始**：大多数场景的最佳平衡点
2. **显存不足用 QLoRA**：性能损失很小 (<5%)
3. **rank 选择**：从 r=8 开始，根据效果调整
4. **目标模块**：优先选择 attention 的 q_proj, v_proj
5. **学习率**：LoRA 通常使用 1e-4 到 2e-4

---

## 7. 代码实现示例

### 7.1 使用 HuggingFace PEFT 库

```python
from peft import LoraConfig, get_peft_model, TaskType

# 定义 LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 应用 LoRA 到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

### 7.2 QLoRA 配置

```python
from transformers import BitsAndBytesConfig

# 4-bit 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 7.3 训练与保存

```python
# 训练
trainer.train()

# 保存 LoRA 适配器
model.save_pretrained("lora_adapter")

# 加载适配器
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("base_model")
model = PeftModel.from_pretrained(base_model, "lora_adapter")
```

---

## 8. 参考文献

1. **Hu, E. et al. (2021).** "LoRA: Low-Rank Adaptation of Large Language Models". arXiv:2106.09685

2. **Dettmers, T. et al. (2023).** "QLoRA: Efficient Finetuning of Quantized LLMs". arXiv:2305.14314

3. **Li, X. et al. (2021).** "Prefix-Tuning: Optimizing Continuous Prompts for Generation". ACL.

4. **Liu, X. et al. (2022).** "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks". ACL.

5. **Houlsby, N. et al. (2019).** "Parameter-Efficient Transfer Learning for NLP". ICML.

6. **Pfeiffer, J. et al. (2020).** "AdapterHub: A Framework for Adapting Transformers". EMNLP.

7. **Mao, S. et al. (2023).** "Compacter: Efficient Low-Rank Hypercomplex Adapter Layers". NeurIPS.

8. **Liu, H. et al. (2023).** "DoRA: Weight-Decomposed Low-Rank Adaptation". arXiv:2402.09353

9. **HuggingFace PEFT Documentation.** https://huggingface.co/docs/peft

---

*本章完 - 高效微调技术使大模型微调变得触手可及*
