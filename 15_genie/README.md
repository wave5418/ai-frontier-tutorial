# 第 15 章 Genie 与环境建模

## 目录结构

```
15_genie/
├── README.md          # 本章说明（本文件）
├── theory.md          # 理论内容
└── code/
    └── genie.py       # 完整代码实现
```

## 文件说明

### theory.md
包含 Genie 的完整理论讲解：
- Genie 背景与贡献
- 世界模型作为可交互环境
- 潜在空间动作模型
- 自回归视频生成
- 智能体训练与应用
- 与 Dreamer 等方法对比
- 局限性与未来方向
- 参考文献

### code/genie.py
可运行的 Python 实现，包含：
- `LatentEncoder`: 潜在空间编码器
- `ActionInference`: 动作推断网络
- `ActionConditionedGenerator`: 动作条件生成器
- `AutoregressiveDecoder`: 自回归解码器
- `Genie`: 完整模型整合
- `GenieTrainer`: 训练器
- `SimpleGridEnvironment`: 简单环境示例
- 演示函数和单元测试

## 运行代码

```bash
# 确保安装了依赖
pip install torch numpy matplotlib scipy

# 运行演示
cd code
python genie.py
```

## 依赖要求

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib（可选，用于可视化）
- SciPy（用于图像处理）

## 学习建议

1. 先阅读 `theory.md` 理解概念
2. 运行 `genie.py` 查看演示输出
3. 阅读代码中的中文注释理解实现细节
4. 尝试修改参数观察效果

---

*上一章：第 14 章 - 视频生成模型*  
*下一章：第 16 章 - 多模态世界模型与具身智能*
