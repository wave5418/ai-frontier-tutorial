# 第 10 章 扩散模型与 Stable Diffusion

## 10.1 扩散模型背景与发展

### 10.1.1 生成模型的演进

生成模型的发展经历了几个重要阶段：

1. **VAE (Variational Autoencoder, 2013)**: 通过变分推断学习数据的潜在表示，但生成质量有限
2. **GAN (Generative Adversarial Network, 2014)**: 通过生成器和判别器的对抗训练获得高质量生成，但训练不稳定
3. **Flow-based Models (2018)**: 通过可逆变换建模数据分布，计算精确似然但架构复杂
4. **Diffusion Models (2020)**: 通过渐进式去噪过程生成数据，训练稳定且生成质量优异

### 10.1.2 扩散模型的突破

扩散模型的核心思想来源于**非平衡热力学**。2020 年，Ho 等人提出的**DDPM (Denoising Diffusion Probabilistic Models)** 在图像生成质量上首次超越了 GAN，引发了研究热潮。

关键优势：
- **训练稳定**: 不需要对抗训练，优化目标简单明确
- **生成质量高**: 能够生成高分辨率、高保真度的图像
- **模式覆盖好**: 不易出现模式坍塌问题
- **理论优美**: 与得分匹配 (Score Matching) 和随机微分方程 (SDE) 有深刻联系

### 10.1.3 重要里程碑

| 时间 | 模型 | 贡献 |
|------|------|------|
| 2015 | Deep Unsupervised Learning using Nonequilibrium Thermodynamics | 扩散模型思想萌芽 |
| 2020 | DDPM | 实用化扩散模型，图像生成质量突破 |
| 2021 | DDIM | 非马尔可夫采样，加速推理 |
| 2021 | Stable Diffusion | 潜空间扩散，大幅降低计算成本 |
| 2022 | ControlNet | 条件控制，精确控制生成内容 |

---

## 10.2 前向扩散过程（加噪）

### 10.2.1 核心思想

前向扩散过程是一个**马尔可夫链**，逐步向数据中添加高斯噪声，直到数据变成纯噪声。

给定初始数据 $x_0 \sim q(x_0)$，前向过程定义为：

$$q(x_{1:T} | x_0) = \prod_{t=1}^{T} q(x_t | x_{t-1})$$

其中每一步的转移分布为：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

### 10.2.2 噪声调度

$\beta_t$ 是**方差调度 (variance schedule)**，控制每一步添加的噪声量。常用调度方式：

1. **线性调度**: $\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1)$
2. **余弦调度**: 更平滑的噪声添加过程

通常设置 $\beta_1 = 10^{-4}$, $\beta_T = 0.02$, $T = 1000$。

### 10.2.3 任意时刻的采样

利用高斯分布的性质，可以直接采样任意时刻 $t$ 的状态：

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中：
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$

这个公式非常重要，它允许我们**直接从 $x_0$ 采样 $x_t$**，无需逐步模拟。

### 10.2.4 物理直觉

可以将前向扩散想象为：
- 一滴墨水滴入清水中
- 随着时间推移，墨水分子逐渐扩散
- 最终达到均匀分布（纯噪声）

---

## 10.3 反向扩散过程（去噪）

### 10.3.1 学习目标

反向过程的目标是**学习一个模型**，能够从噪声中逐步恢复出原始数据：

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)$$

其中 $p(x_T) = \mathcal{N}(x_T; 0, I)$ 是标准高斯分布。

### 10.3.2 反向转移分布

每一步的去噪过程建模为：

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

关键问题是：如何学习 $\mu_\theta$ 和 $\Sigma_\theta$？

### 10.3.3 均值预测

理论分析表明，最优的反向均值与**噪声预测**相关：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$$

其中 $\epsilon_\theta(x_t, t)$ 是神经网络预测的噪声。

### 10.3.4 方差选择

实践中常用固定方差：
- $\Sigma_\theta(x_t, t) = \sigma_t^2 I$
- $\sigma_t^2 = \beta_t$ 或 $\sigma_t^2 = \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$

---

## 10.4 DDPM 原理详解

### 10.4.1 训练目标

DDPM 的训练目标是**预测添加的噪声**：

$$\mathcal{L}_{simple} = \mathbb{E}_{t,x_0,\epsilon}\left[||\epsilon - \epsilon_\theta(x_t, t)||^2\right]$$

其中 $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$。

### 10.4.2 训练算法

```
算法：DDPM 训练
输入：训练数据 {x_0}, 时间步 T, 网络 ε_θ
重复:
    1. 从训练集采样 x_0
    2. 均匀采样 t ~ Uniform({1, ..., T})
    3. 采样噪声 ε ~ N(0, I)
    4. 计算 x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε
    5. 计算梯度下降步：∇_θ ||ε - ε_θ(x_t, t)||²
直到收敛
```

### 10.4.3 采样算法

```
算法：DDPM 采样
输入：训练好的网络 ε_θ, 时间步 T
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    z ~ N(0, I) if t > 1 else 0
    x_{t-1} = (1/√α_t)(x_t - ((1-α_t)/√(1-ᾱ_t))ε_θ(x_t, t)) + σ_t z
返回 x_0
```

### 10.4.4 网络架构

DDPM 通常使用**U-Net**架构：
- 输入：带噪声的图像 $x_t$ 和时间嵌入 $t$
- 输出：预测的噪声 $\epsilon_\theta(x_t, t)$
- 特点：跳跃连接，多尺度特征融合

---

## 10.5 DDIM 加速采样

### 10.5.1 动机

DDPM 需要上千步采样，速度很慢。DDIM (Denoising Diffusion Implicit Models) 通过**非马尔可夫**公式实现加速。

### 10.5.2 核心思想

DDIM 观察到：只要边缘分布 $q(x_t|x_0)$ 相同，可以有不同的转移核。因此可以设计**确定性**的采样过程。

### 10.5.3 DDIM 采样公式

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\left(\frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1-\bar{\alpha}_{t-1}}\cdot\epsilon_\theta(x_t, t)$$

关键特点：
- **确定性**: 没有随机噪声项
- **一致性**: 任意子序列的采样结果一致
- **可加速**: 可以跳过中间步骤

### 10.5.4 加速采样

使用子序列 $\tau = [\tau_1, \tau_2, ..., \tau_S]$ 其中 $S \ll T$：

```
算法：DDIM 加速采样
输入：ε_θ, 子序列 τ (长度 S)
x_{τ_S} ~ N(0, I)
for i = S, S-1, ..., 1:
    t = τ_i, prev_t = τ_{i-1} (或 0 如果 i=1)
    使用 DDIM 公式从 x_t 计算 x_{prev_t}
返回 x_0
```

通常可以用 50 步甚至 20 步获得与 DDPM 1000 步相当的质量。

---

## 10.6 Stable Diffusion 架构

### 10.6.1 核心创新

Stable Diffusion 的关键创新是**在潜空间 (latent space) 进行扩散**，而非像素空间。

### 10.6.2 整体架构

```
文本提示 → CLIP Text Encoder → 文本嵌入
                              ↓
高斯噪声 → UNet (潜空间) → 去噪后的潜表示
                              ↓
                         VAE Decoder
                              ↓
                         输出图像
```

### 10.6.3 VAE (变分自编码器)

**作用**: 压缩和解压缩图像

- **Encoder**: $z = E(x)$, 将 512×512×3 图像压缩为 64×64×4 潜表示
- **Decoder**: $\hat{x} = D(z)$, 从潜表示重建图像

**优势**:
- 计算效率提升 48 倍 (64² vs 512²)
- 保留语义信息，去除高频细节
- 扩散过程在更平滑的空间进行

### 10.6.4 UNet

**作用**: 潜空间的去噪网络

架构特点：
- **Down Block**: 逐步下采样，提取特征
- **Middle Block**: 深层特征处理
- **Up Block**: 逐步上采样，恢复分辨率
- **跳跃连接**: 融合多尺度信息
- **时间嵌入**: 通过 FiLM 或 AdaGN 注入时间信息
- **交叉注意力**: 融合文本条件

### 10.6.5 CLIP Text Encoder

**作用**: 将文本转换为语义嵌入

- 使用 CLIP 的文本编码器（通常是 ViT 或 Transformer）
- 将文本提示转换为 77 个 token，每个 768 维
- 通过交叉注意力机制注入 UNet

### 10.6.6 训练过程

1. **VAE 预训练**: 在大规模图像上独立训练 VAE
2. **扩散模型训练**: 冻结 VAE，在潜空间训练 UNet
3. **条件注入**: 使用分类器自由引导 (Classifier-Free Guidance)

---

## 10.7 文生图原理

### 10.7.1 条件扩散模型

将无条件扩散扩展为条件扩散：

$$p_\theta(x_{t-1} | x_t, c) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, c), \Sigma_\theta(x_t, t))$$

其中 $c$ 是条件信息（如文本描述）。

### 10.7.2 条件注入方式

1. **交叉注意力 (Cross-Attention)**:
   - 在 UNet 的特定层添加交叉注意力模块
   - Query 来自图像特征，Key/Value 来自文本嵌入
   - 公式：$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d}})V$

2. **FiLM (Feature-wise Linear Modulation)**:
   - 通过仿射变换调制特征：$\gamma(c) \cdot x + \beta(c)$

3. **AdaGN (Adaptive Group Normalization)**:
   - 条件化的组归一化

### 10.7.3 分类器自由引导 (Classifier-Free Guidance)

**问题**: 传统分类器引导需要额外的分类器模型

**解决方案**: 训练时随机丢弃条件，推理时混合条件和无条件预测

**训练**:
- 以概率 $p_{drop}$ 将条件 $c$ 替换为空条件 $\emptyset$
- 模型同时学习 $p(x|c)$ 和 $p(x)$

**推理**:
$$\epsilon_{guided} = \epsilon_\theta(x_t, t, \emptyset) + s \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))$$

其中 $s$ 是引导强度 (guidance scale)，通常 7-15。

### 10.7.4 完整文生图流程

```
1. 文本编码：c = CLIP_Text_Encoder(prompt)
2. 采样噪声：z_T ~ N(0, I)
3. 迭代去噪：for t = T...1:
   - 预测噪声：ε = UNet(z_t, t, c)
   - 应用引导：ε_guided = ε_uncond + s(ε - ε_uncond)
   - 更新潜变量：z_{t-1} = DDIM_Step(z_t, ε_guided, t)
4. 解码图像：x = VAE_Decoder(z_0)
```

---

## 10.8 ControlNet 等扩展

### 10.8.1 ControlNet 核心思想

**问题**: Stable Diffusion 难以精确控制生成结构

**解决方案**: 复制 UNet 的编码器层，添加可训练的"控制块"

### 10.8.2 ControlNet 架构

```
输入条件（边缘图/深度图/姿态等）
         ↓
   Zero Convolution (初始化为 0)
         ↓
   控制块 (复制的编码器层)
         ↓
   注入到原始 UNet (通过加法)
```

**关键设计**:
- **Zero Convolution**: 初始化为 0，确保训练初期不影响原模型
- **锁定原模型**: 原始 SD 权重冻结，只训练控制块
- **灵活条件**: 可以接受各种空间条件（边缘、深度、姿态、分割等）

### 10.8.3 常见 ControlNet 类型

| 类型 | 条件输入 | 应用场景 |
|------|----------|----------|
| Canny | 边缘图 | 保持结构轮廓 |
| Depth | 深度图 | 控制空间层次 |
| Pose | 人体姿态 | 人物动作控制 |
| Segmentation | 语义分割 | 区域内容控制 |
| Scribble | 草图 | 手绘转精美图像 |
| Inpaint | 掩码 | 局部重绘 |

### 10.8.4 其他重要扩展

1. **LoRA (Low-Rank Adaptation)**:
   - 低秩矩阵微调，参数量小
   - 快速适配新风格/角色

2. **Textual Inversion**:
   - 学习新的文本嵌入表示
   - 个性化概念注入

3. **DreamBooth**:
   - 微调整个模型
   - 高质量个性化生成

4. **IP-Adapter**:
   - 图像提示适配器
   - 用参考图像控制风格/内容

5. **AnimateDiff**:
   - 视频生成扩展
   - 时序一致的帧生成

---

## 10.9 代码实现示例

详见 `code/diffusion.py`，包含：

1. **扩散过程实现**: 前向加噪和反向去噪
2. **UNet 网络**: 简化版 U-Net 架构
3. **DDPM 训练**: 完整训练循环
4. **DDIM 采样**: 加速采样实现
5. **条件扩散**: 文本条件注入
6. **文生图示例**: 简化版文本到图像生成

---

## 10.10 参考文献

### 核心论文

1. **DDPM**: Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
   - 扩散模型的奠基之作

2. **DDIM**: Song, J., Meng, C., & Ermon, S. (2021). "Denoising Diffusion Implicit Models." ICLR 2021.
   - 加速采样的关键工作

3. **Stable Diffusion**: Rombach, R., et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022.
   - 潜空间扩散，实用化突破

4. **Classifier-Free Guidance**: Ho, J. & Salimans, T. (2022). "Classifier-Free Diffusion Guidance."
   - 条件生成的标准方法

5. **ControlNet**: Zhang, L. & Agrawala, M. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models." ICCV 2023.
   - 精确控制生成的里程碑

### 理论深入

6. **Score SDE**: Song, Y., et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021.
   - 扩散模型的 SDE 视角

7. **Elucidating Diffusion**: Karras, T., et al. (2022). "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS 2022.
   - 扩散模型设计空间分析

### 应用扩展

8. **LoRA**: Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." (应用于扩散模型)

9. **DreamBooth**: Ruiz, N., et al. (2023). "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation." CVPR 2023.

10. **AnimateDiff**: Guo, Y., et al. (2023). "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning."

### 在线资源

- Hugging Face Diffusers 文档：https://huggingface.co/docs/diffusers
- Stable Diffusion WebUI: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- ControlNet 项目：https://github.com/lllyasviel/ControlNet

---

## 本章小结

扩散模型代表了生成式 AI 的重大突破，其核心优势在于：

1. **理论优美**: 基于热力学和得分匹配的坚实理论基础
2. **训练稳定**: 简单的 MSE 损失，无需对抗训练
3. **生成质量高**: 能够生成高分辨率、多样化的图像
4. **灵活可控**: 通过条件机制实现精确控制

Stable Diffusion 通过潜空间扩散大幅降低了计算成本，使得消费级 GPU 也能运行。ControlNet 等扩展进一步增强了控制能力，开启了创意应用的新纪元。

理解扩散模型的原理对于深入掌握现代 AIGC 技术至关重要。希望本章的理论和代码示例能够帮助你建立扎实的基础，为进一步探索和创新打下根基。
