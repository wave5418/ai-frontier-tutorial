"""
第 25 章：DPO 与直接偏好优化 - 代码实现
========================================

本模块实现了 DPO（Direct Preference Optimization）的核心算法，包括：
- DPOLoss：DPO 损失函数
- DPOTrainer：DPO 训练器
- 偏好数据集处理
- 完整训练流程
- 与 PPO 的对比示例

作者：AI 前沿技术教程
日期：2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
import warnings


# ============================================================
# 1. 数据类定义
# ============================================================

@dataclass
class PreferenceSample:
    """偏好样本数据类"""
    prompt: str           # 输入提示
    chosen: str          # 偏好回复（winner）
    rejected: str        # 拒绝回复（loser）
    
    def __post_init__(self):
        if not isinstance(self.prompt, str):
            raise ValueError("prompt 必须是字符串")
        if not isinstance(self.chosen, str):
            raise ValueError("chosen 必须是字符串")
        if not isinstance(self.rejected, str):
            raise ValueError("rejected 必须是字符串")


@dataclass
class DPODataCollator:
    """DPO 数据整理器 - 用于批处理偏好数据"""
    
    tokenizer: PreTrainedTokenizer
    max_length: int = 512
    padding: bool = True
    return_tensors: str = "pt"
    
    def __call__(self, samples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """
        将样本列表整理为批处理数据
        
        Args:
            samples: 样本列表，每个样本包含 prompt, chosen, rejected
            
        Returns:
            包含输入 IDs 和注意力掩码的字典
        """
        # 提取所有文本
        prompts = [s['prompt'] for s in samples]
        chosens = [s['chosen'] for s in samples]
        rejecteds = [s['rejected'] for s in samples]
        
        # 构建完整的输入序列
        chosen_texts = [p + c for p, c in zip(prompts, chosens)]
        rejected_texts = [p + r for p, r in zip(prompts, rejecteds)]
        
        # 分词
        chosen_encoded = self.tokenizer(
            chosen_texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors=self.return_tensors
        )
        
        rejected_encoded = self.tokenizer(
            rejected_texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors=self.return_tensors
        )
        
        return {
            'chosen_input_ids': chosen_encoded['input_ids'],
            'chosen_attention_mask': chosen_encoded['attention_mask'],
            'rejected_input_ids': rejected_encoded['input_ids'],
            'rejected_attention_mask': rejected_encoded['attention_mask'],
            'prompts': prompts  # 保留原始 prompt 用于日志
        }


# ============================================================
# 2. DPO 损失函数实现
# ============================================================

class DPOLoss(nn.Module):
    """
    DPO（Direct Preference Optimization）损失函数
    
    核心公式：
    L_DPO = -E[log(σ(β * log(π(y_w|x)/π_ref(y_w|x)) - β * log(π(y_l|x)/π_ref(y_l|x))))]
    
    其中：
    - y_w: 偏好回复（winner）
    - y_l: 拒绝回复（loser）
    - π: 当前策略模型
    - π_ref: 参考策略模型
    - β: KL 约束强度
    - σ: sigmoid 函数
    """
    
    def __init__(self, beta: float = 0.1, reduction: str = 'mean'):
        """
        初始化 DPO 损失
        
        Args:
            beta: KL 约束强度，控制与参考策略的偏离程度
                  - 较小值（0.1）：允许更多偏离，优化更强
                  - 较大值（0.5）：更保守，接近参考策略
            reduction: 损失归约方式 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.sigmoid = nn.Sigmoid()
        
    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 DPO 损失
        
        Args:
            policy_chosen_logps: 策略模型对偏好回复的对数概率 [batch_size]
            policy_rejected_logps: 策略模型对拒绝回复的对数概率 [batch_size]
            reference_chosen_logps: 参考模型对偏好回复的对数概率 [batch_size]
            reference_rejected_logps: 参考模型对拒绝回复的对数概率 [batch_size]
            
        Returns:
            loss: DPO 损失标量
            metrics: 包含额外指标的字典
        """
        # 计算隐式奖励差值
        # r_pi(x, y) = β * (log π(y|x) - log π_ref(y|x))
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        
        # DPO 的核心：偏好回复的隐式奖励 - 拒绝回复的隐式奖励
        logits = self.beta * (chosen_logratios - rejected_logratios)
        
        # 应用 sigmoid 并计算负对数似然
        # 等价于：-log(σ(logits))
        losses = -F.logsigmoid(logits)
        
        # 归约
        if self.reduction == 'mean':
            loss = losses.mean()
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses
        
        # 计算额外指标用于监控
        with torch.no_grad():
            metrics = {
                'loss': loss.item(),
                'chosen_rewards': (self.beta * chosen_logratios).mean().item(),
                'rejected_rewards': (self.beta * rejected_logratios).mean().item(),
                'reward_margin': (self.beta * (chosen_logratios - rejected_logratios)).mean().item(),
                'accuracy': (logits > 0).float().mean().item(),  # 偏好样本被正确分类的比例
            }
        
        return loss, metrics


# ============================================================
# 3. 辅助函数：计算对数概率
# ============================================================

def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    average_log_prob: bool = False
) -> torch.Tensor:
    """
    计算序列的对数概率
    
    Args:
        logits: 模型输出的 logits [batch_size, seq_len, vocab_size]
        labels: 目标 token IDs [batch_size, seq_len]
        attention_mask: 注意力掩码 [batch_size, seq_len]
        average_log_prob: 是否使用平均对数概率（而非总和）
        
    Returns:
        每个序列的对数概率 [batch_size]
    """
    # 移动 labels 以匹配 logits（预测下一个 token）
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    shift_mask = attention_mask[:, 1:]
    
    # 计算每个 token 的对数概率
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logps = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # 应用注意力掩码
    token_logps = token_logps * shift_mask
    
    # 聚合为序列级别的对数概率
    if average_log_prob:
        # 平均对数概率（SimPO 风格）
        lengths = shift_mask.sum(dim=1).clamp(min=1)
        sequence_logps = token_logps.sum(dim=1) / lengths
    else:
        # 总和对数概率（标准 DPO）
        sequence_logps = token_logps.sum(dim=1)
    
    return sequence_logps


# ============================================================
# 4. DPO Trainer 实现
# ============================================================

class DPOTrainer:
    """
    DPO 训练器
    
    封装了完整的 DPO 训练流程，包括：
    - 模型管理（策略模型和参考模型）
    - 训练循环
    - 评估和日志
    - 检查点保存
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: Optional[PreTrainedModel],
        tokenizer: PreTrainedTokenizer,
        beta: float = 0.1,
        learning_rate: float = 5e-6,
        max_length: int = 512,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        warmup_ratio: float = 0.1,
        logging_steps: int = 10,
        save_steps: int = 100,
        output_dir: str = "./dpo_output",
        device: Optional[str] = None
    ):
        """
        初始化 DPO 训练器
        
        Args:
            model: 策略模型（将被优化）
            ref_model: 参考模型（冻结，用于 KL 约束）
                       如果为 None，使用 model 的初始状态作为参考
            tokenizer: 分词器
            beta: DPO 损失中的 KL 约束强度
            learning_rate: 学习率
            max_length: 最大序列长度
            batch_size: 批大小
            gradient_accumulation_steps: 梯度累积步数
            warmup_ratio: 学习率预热比例
            logging_steps: 日志记录间隔
            save_steps: 保存检查点间隔
            output_dir: 输出目录
            device: 训练设备
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.output_dir = output_dir
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            self.ref_model.eval()  # 参考模型设为评估模式
        
        # 初始化损失函数
        self.loss_fn = DPOLoss(beta=beta)
        
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.training_history = []
        
        # 创建输出目录
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def _compute_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算模型对输入序列的对数概率
        
        Args:
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码
            
        Returns:
            序列的对数概率
        """
        # 获取模型输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        logits = outputs.logits
        
        # 计算对数概率
        logps = get_batch_logps(logits, input_ids, attention_mask)
        return logps
    
    def _compute_reference_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算参考模型对输入序列的对数概率
        
        如果 ref_model 为 None，使用 model 的初始状态（需要预先保存）
        """
        if self.ref_model is None:
            raise ValueError("参考模型未提供")
        
        with torch.no_grad():
            outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits
            logps = get_batch_logps(logits, input_ids, attention_mask)
        
        return logps
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict]:
        """
        执行单个训练步骤
        
        Args:
            batch: 批处理数据
            
        Returns:
            loss: 损失值
            metrics: 指标字典
        """
        self.model.train()
        
        # 将数据移动到设备
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        # 计算策略模型的对数概率
        policy_chosen_logps = self._compute_logps(
            batch['chosen_input_ids'],
            batch['chosen_attention_mask']
        )
        policy_rejected_logps = self._compute_logps(
            batch['rejected_input_ids'],
            batch['rejected_attention_mask']
        )
        
        # 计算参考模型的对数概率
        with torch.no_grad():
            reference_chosen_logps = self._compute_reference_logps(
                batch['chosen_input_ids'],
                batch['chosen_attention_mask']
            )
            reference_rejected_logps = self._compute_reference_logps(
                batch['rejected_input_ids'],
                batch['rejected_attention_mask']
            )
        
        # 计算 DPO 损失
        loss, metrics = self.loss_fn(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )
        
        # 反向传播
        loss.backward()
        
        return loss.item(), metrics
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        num_epochs: int = 3,
        collate_fn: Optional[Any] = None
    ) -> Dict[str, List[float]]:
        """
        执行完整训练
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 验证数据集（可选）
            num_epochs: 训练轮数
            collate_fn: 数据整理函数
            
        Returns:
            训练历史
        """
        # 创建数据加载器
        if collate_fn is None:
            collate_fn = DPODataCollator(
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        # 计算总步数和预热步数
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        print(f"开始 DPO 训练")
        print(f"设备：{self.device}")
        print(f"训练样本数：{len(train_dataset)}")
        print(f"总步数：{total_steps}")
        print(f"学习率：{self.learning_rate}")
        print(f"Beta: {self.beta}")
        print("-" * 50)
        
        training_history = {
            'loss': [],
            'accuracy': [],
            'eval_loss': []
        }
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            # 梯度累积
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(train_loader):
                # 训练步骤
                loss, metrics = self.train_step(batch)
                
                # 梯度累积
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                # 累积指标
                epoch_loss += loss
                epoch_accuracy += metrics.get('accuracy', 0.0)
                num_batches += 1
                
                # 日志记录
                if self.global_step % self.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches
                    avg_accuracy = epoch_accuracy / num_batches
                    print(f"Step {self.global_step}: Loss={avg_loss:.4f}, "
                          f"Accuracy={avg_accuracy:.4f}")
                    
                    training_history['loss'].append(avg_loss)
                    training_history['accuracy'].append(avg_accuracy)
                
                # 保存检查点
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")
            
            # epoch 结束
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            avg_epoch_accuracy = epoch_accuracy / max(num_batches, 1)
            print(f"\nEpoch {epoch + 1}/{num_epochs} 完成")
            print(f"平均损失：{avg_epoch_loss:.4f}")
            print(f"平均准确率：{avg_epoch_accuracy:.4f}")
            
            # 验证
            if eval_dataset is not None:
                eval_loss = self.evaluate(eval_dataset, collate_fn)
                training_history['eval_loss'].append(eval_loss)
                print(f"验证损失：{eval_loss:.4f}")
        
        # 保存最终模型
        self.save_checkpoint("final")
        
        self.training_history = training_history
        return training_history
    
    def evaluate(
        self,
        eval_dataset: Dataset,
        collate_fn: Optional[Any] = None
    ) -> float:
        """
        在验证集上评估
        
        Args:
            eval_dataset: 验证数据集
            collate_fn: 数据整理函数
            
        Returns:
            平均验证损失
        """
        self.model.eval()
        
        if collate_fn is None:
            collate_fn = DPODataCollator(
                tokenizer=self.tokenizer,
                max_length=self.max_length
            )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
                
                policy_chosen_logps = self._compute_logps(
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask']
                )
                policy_rejected_logps = self._compute_logps(
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask']
                )
                
                reference_chosen_logps = self._compute_reference_logps(
                    batch['chosen_input_ids'],
                    batch['chosen_attention_mask']
                )
                reference_rejected_logps = self._compute_reference_logps(
                    batch['rejected_input_ids'],
                    batch['rejected_attention_mask']
                )
                
                loss, _ = self.loss_fn(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()  # 恢复训练模式
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, name: str):
        """
        保存检查点
        
        Args:
            name: 检查点名称
        """
        import os
        checkpoint_path = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # 保存训练状态
        import json
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer_state': self.optimizer.state_dict(),
            'training_history': self.training_history
        }
        with open(os.path.join(checkpoint_path, 'training_state.json'), 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"检查点已保存到：{checkpoint_path}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        使用训练后的模型生成文本
        
        Args:
            prompt: 输入提示
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: Top-p 采样参数
            
        Returns:
            生成的文本
        """
        self.model.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


# ============================================================
# 5. 示例：创建偏好数据集
# ============================================================

class PreferenceDataset(Dataset):
    """偏好数据集"""
    
    def __init__(self, samples: List[PreferenceSample]):
        """
        初始化数据集
        
        Args:
            samples: 偏好样本列表
        """
        self.samples = samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        sample = self.samples[idx]
        return {
            'prompt': sample.prompt,
            'chosen': sample.chosen,
            'rejected': sample.rejected
        }


def create_example_dataset() -> List[PreferenceSample]:
    """
    创建示例偏好数据集
    
    Returns:
        偏好样本列表
    """
    samples = [
        PreferenceSample(
            prompt="请解释什么是机器学习。",
            chosen="机器学习是人工智能的一个分支，它使计算机能够从数据中学习并做出决策或预测，而无需明确编程。",
            rejected="机器学习就是让机器变得聪明，可以像人一样思考。"
        ),
        PreferenceSample(
            prompt="写一首关于春天的短诗。",
            chosen="春风拂面花自开，\n绿柳垂丝燕归来。\n万物复苏生机现，\n一年之计在此时。",
            rejected="春天来了，花开了，树绿了，鸟儿叫了，天气暖和了。"
        ),
        PreferenceSample(
            prompt="如何学习编程？",
            chosen="学习编程的建议：1) 选择一门入门语言如 Python；2) 理解基本概念如变量、循环、函数；3) 通过项目实践；4) 阅读他人代码；5) 持续学习和练习。",
            rejected="多写代码就行了，熟能生巧。"
        ),
        PreferenceSample(
            prompt="解释量子纠缠。",
            chosen="量子纠缠是量子力学中的现象，两个或多个粒子形成关联状态，即使相隔很远，一个粒子的状态变化会瞬间影响另一个粒子的状态。爱因斯坦称之为'鬼魅般的超距作用'。",
            rejected="量子纠缠就是两个粒子之间有神秘联系，不管多远都能互相影响。"
        ),
        PreferenceSample(
            prompt="推荐几本好书。",
            chosen="推荐书籍：1)《人类简史》- 尤瓦尔·赫拉利；2)《思考，快与慢》- 丹尼尔·卡尼曼；3)《原则》- 瑞·达利欧；4)《穷查理宝典》- 查理·芒格。",
            rejected="看书挺好的，我推荐一些书，你可以去看看。"
        )
    ]
    return samples


# ============================================================
# 6. DPO 与 PPO 对比示例
# ============================================================

def compare_dpo_ppo():
    """
    DPO 与 PPO 的对比分析
    
    这个函数展示了两种方法的关键差异
    """
    print("=" * 60)
    print("DPO vs PPO 对比")
    print("=" * 60)
    
    comparison = """
    【模型数量】
    DPO: 2 个（策略模型 + 参考模型）
    PPO: 4 个（策略模型 + 参考模型 + 奖励模型 + 价值模型）
    
    【优化方式】
    DPO: 监督学习（交叉熵损失）
    PPO: 强化学习（策略梯度）
    
    【训练稳定性】
    DPO: 高（类似 SFT）
    PPO: 中低（需要仔细调参）
    
    【计算开销】
    DPO: 中等
    PPO: 高（需要多次 rollout）
    
    【实现复杂度】
    DPO: 低（只需修改损失函数）
    PPO: 高（需要完整的 RL 基础设施）
    
    【内存占用】
    DPO: 中等（2 个模型）
    PPO: 高（4 个模型 + 额外缓冲区）
    
    【效果】
    多项研究表明 DPO 能达到与 PPO 相当甚至更好的效果，
    同时训练更稳定、更高效。
    """
    print(comparison)
    print("=" * 60)


# ============================================================
# 7. 完整训练示例
# ============================================================

def run_dpo_training_example():
    """
    完整的 DPO 训练示例
    
    注意：这是一个示例，实际使用需要：
    1. 加载预训练模型
    2. 准备真实偏好数据集
    3. 调整超参数
    """
    print("\n" + "=" * 60)
    print("DPO 训练示例")
    print("=" * 60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # 配置
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 示例模型
        beta = 0.1
        learning_rate = 5e-6
        batch_size = 4  # 小批量用于示例
        num_epochs = 1
        
        print(f"加载模型：{model_name}")
        
        # 加载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # 参考模型（使用相同的初始权重）
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        ref_model.eval()
        
        # 创建数据集
        samples = create_example_dataset()
        dataset = PreferenceDataset(samples)
        
        print(f"数据集大小：{len(dataset)} 样本")
        
        # 创建训练器
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            beta=beta,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_length=256,
            output_dir="./dpo_example_output"
        )
        
        # 训练
        history = trainer.train(
            train_dataset=dataset,
            num_epochs=num_epochs
        )
        
        # 测试生成
        test_prompt = "请用一句话总结 DPO 的核心思想。"
        generated = trainer.generate(test_prompt, max_new_tokens=100)
        print(f"\n测试生成:\n{generated}")
        
        print("\n训练完成！")
        
    except ImportError:
        print("需要安装 transformers 库：pip install transformers")
        print("\n以下是简化的伪代码示例：")
        print("""
        # 伪代码示例
        model = load_pretrained_model("your-model")
        ref_model = load_pretrained_model("your-model")
        tokenizer = load_tokenizer("your-model")
        
        dataset = PreferenceDataset(your_samples)
        
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            beta=0.1,
            learning_rate=5e-6
        )
        
        trainer.train(dataset, num_epochs=3)
        """)


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    print("第 25 章：DPO 与直接偏好优化 - 代码示例")
    print("=" * 60)
    
    # 显示 DPO 与 PPO 对比
    compare_dpo_ppo()
    
    # 运行训练示例（需要 transformers 库）
    # 取消注释以实际运行
    # run_dpo_training_example()
    
    print("\n提示：实际使用请确保安装以下依赖：")
    print("  pip install transformers torch accelerate")
    print("\n完整教程请参考 theory.md")
