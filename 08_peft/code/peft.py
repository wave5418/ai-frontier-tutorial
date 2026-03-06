#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 08 章 高效微调技术代码实现
PEFT (Parameter-Efficient Fine-Tuning) Code Implementation

包含：
1. LoRA 实现（从 0 开始）
2. QLoRA 量化示例
3. P-Tuning 实现
4. 完整 PEFT 微调流程
5. 使用 HuggingFace PEFT 库示例
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from typing import Dict, Optional


# ============================================================
# 1. LoRA 实现（从 0 开始）
# ============================================================

class LoRALinear(nn.Module):
    """
    LoRA 线性层实现
    
    核心公式：h = Wx + (alpha/r) * BAx
    """
    def __init__(self, in_features: int, out_features: int, 
                 rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # 冻结的主权重
        self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)
        
        # 可训练的 LoRA 矩阵
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 缩放因子
        self.scaling = alpha / rank
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主路径（冻结）
        main = nn.functional.linear(x, self.weight)
        
        # LoRA 路径
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        
        return main + lora


class LoRAModel:
    """
    LoRA 模型包装器
    将 LoRA 适配器应用到预训练模型
    """
    def __init__(self, model: nn.Module, rank: int = 8, alpha: float = 16.0,
                 target_modules: Optional[list] = None):
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.target_modules = target_modules or ["q_proj", "v_proj"]
        
        # 冻结所有原始参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 应用 LoRA 到目标模块
        self._apply_lora()
    
    def _apply_lora(self):
        """递归应用 LoRA 到目标模块"""
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    # 创建 LoRA 层
                    lora_layer = LoRALinear(
                        module.in_features,
                        module.out_features,
                        self.rank,
                        self.alpha
                    )
                    # 复制原始权重
                    lora_layer.weight.data = module.weight.data
                    
                    # 替换模块
                    parent_name = ".".join(name.split(".")[:-1])
                    module_name = name.split(".")[-1]
                    parent = dict(self.model.named_modules())[parent_name] if parent_name else self.model
                    setattr(parent, module_name, lora_layer)
    
    def print_trainable_parameters(self):
        """打印可训练参数统计"""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


# ============================================================
# 2. QLoRA 量化示例
# ============================================================

def create_qlora_config():
    """创建 QLoRA 量化配置"""
    try:
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        return quantization_config
    except ImportError:
        print("bitsandbytes not installed. Install with: pip install bitsandbytes")
        return None


def load_qlora_model(model_name: str, quantization_config):
    """加载 QLoRA 模型"""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return model


# ============================================================
# 3. P-Tuning 实现
# ============================================================

class PromptEncoder(nn.Module):
    """
    P-Tuning 提示编码器
    将离散提示编码为连续向量
    """
    def __init__(self, hidden_size: int, num_virtual_tokens: int = 20):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_virtual_tokens = num_virtual_tokens
        
        # 提示嵌入
        self.prompt_embeddings = nn.Embedding(num_virtual_tokens, hidden_size)
        nn.init.uniform_(self.prompt_embeddings.weight, -0.1, 0.1)
        
        # 编码器（可选，用于 P-Tuning v2）
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(self, batch_size: int) -> torch.Tensor:
        # 生成提示
        prompt = self.prompt_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 通过编码器
        prompt = self.encoder(prompt)
        
        return prompt


class P_Tuning_Model:
    """
    P-Tuning 模型包装器
    """
    def __init__(self, model: nn.Module, num_virtual_tokens: int = 20):
        self.model = model
        self.num_virtual_tokens = num_virtual_tokens
        self.hidden_size = model.config.hidden_size
        
        # 创建提示编码器
        self.prompt_encoder = PromptEncoder(self.hidden_size, num_virtual_tokens)
        
        # 冻结模型参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 仅训练提示编码器
        for param in self.prompt_encoder.parameters():
            param.requires_grad = True
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 获取提示
        batch_size = input_ids.shape[0]
        prompts = self.prompt_encoder(batch_size).to(input_ids.device)
        
        # 获取输入嵌入
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # 拼接提示和输入
        embeddings = torch.cat([prompts, inputs_embeds], dim=1)
        
        # 扩展 attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(batch_size, self.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        # 前向传播
        outputs = self.model(inputs_embeds=embeddings, attention_mask=attention_mask, labels=labels)
        
        return outputs
    
    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.prompt_encoder.parameters())
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")


# ============================================================
# 4. 完整 PEFT 微调流程（使用 HuggingFace PEFT）
# ============================================================

def peft_finetune(model_name: str, train_dataset, output_dir: str,
                  peft_type: str = "LORA", rank: int = 16):
    """
    完整的 PEFT 微调流程
    
    Args:
        model_name: 预训练模型名称
        train_dataset: 训练数据集
        output_dir: 输出目录
        peft_type: PEFT 类型 (LORA/QLORA/P_TUNING)
        rank: LoRA rank
    """
    try:
        from peft import (
            LoraConfig, 
            PrefixTuningConfig,
            PromptEncoderConfig,
            get_peft_model,
            TaskType,
            prepare_model_for_kbit_training
        )
    except ImportError:
        print("PEFT not installed. Install with: pip install peft")
        return
    
    print(f"加载模型：{model_name}")
    
    # 根据 PEFT 类型加载模型
    if peft_type == "QLORA":
        quantization_config = create_qlora_config()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 配置 PEFT
    print(f"配置 {peft_type}...")
    
    if peft_type in ["LORA", "QLORA"]:
        peft_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
    elif peft_type == "P_TUNING":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,
            encoder_hidden_size=512
        )
    elif peft_type == "PREFIX":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20
        )
    else:
        raise ValueError(f"Unknown PEFT type: {peft_type}")
    
    # 应用 PEFT
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 训练配置
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )
    
    # 训练
    print("开始训练...")
    trainer.train()
    
    # 保存
    print(f"保存模型到：{output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer


# ============================================================
# 5. 使用示例
# ============================================================

def example_lora_usage():
    """LoRA 使用示例"""
    print("=" * 50)
    print("LoRA 使用示例")
    print("=" * 50)
    
    # 加载预训练模型
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 应用 LoRA
    lora_model = LoRAModel(model, rank=8, alpha=16)
    lora_model.print_trainable_parameters()
    
    return lora_model


def example_peft_usage():
    """HuggingFace PEFT 使用示例"""
    print("=" * 50)
    print("HuggingFace PEFT 使用示例")
    print("=" * 50)
    
    from peft import LoraConfig, get_peft_model, TaskType
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 配置 LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # 应用 PEFT
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def main():
    """主函数 - 完整示例"""
    print("=" * 60)
    print("第 08 章 PEFT 高效微调技术 - 完整示例")
    print("=" * 60)
    
    # 示例 1: 自定义 LoRA
    print("\n[1] 自定义 LoRA 实现")
    example_lora_usage()
    
    # 示例 2: HuggingFace PEFT
    print("\n[2] HuggingFace PEFT 库")
    example_peft_usage()
    
    # 示例 3: 完整微调流程（需要数据集）
    print("\n[3] 完整微调流程")
    print("注意：需要提供训练数据集才能运行")
    print("调用方式：peft_finetune(model_name, dataset, output_dir, peft_type='LORA')")
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
