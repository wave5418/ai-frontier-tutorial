#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 06 章 指令微调代码实现
Instruction Tuning Code Implementation
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from typing import List, Dict

class InstructionDataset(Dataset):
    """指令微调数据集类"""
    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 512):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def _load_data(self, data_path: str) -> List[Dict]:
        data = []
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        return data
    
    def _format_alpaca(self, item: Dict) -> str:
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        parts = ['### Instruction:', instruction, '', '### Response:', output]
        if input_text:
            parts.insert(2, '### Input:')
            parts.insert(3, input_text)
        return chr(10).join(parts)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = self._format_alpaca(item)
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)
        return {'input_ids': input_ids, 'attention_mask': encoding['attention_mask'].squeeze(0), 'labels': input_ids.clone()}


def create_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """加载预训练模型和 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device == "cuda" else torch.float32, device_map="auto" if device == "cuda" else None)
    return model, tokenizer


def train_sft(model, tokenizer, train_dataset, output_dir: str, batch_size: int = 4, num_epochs: int = 3, learning_rate: float = 2e-5):
    """监督微调训练"""
    training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=num_epochs, per_device_train_batch_size=batch_size, gradient_accumulation_steps=4, learning_rate=learning_rate, warmup_ratio=0.03, logging_steps=10, save_strategy="epoch", fp16=True, report_to="none")
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return trainer


class ChatBot:
    """多轮对话机器人"""
    def __init__(self, model, tokenizer, system_prompt: str = ""):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.history = []
    
    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
    
    def clear_history(self):
        self.history = []
    
    def generate_response(self, user_input: str, max_new_tokens: int = 256) -> str:
        """生成回复"""
        self.add_message("user", user_input)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.history)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=self.tokenizer.pad_token_id)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取助手回复 - 查找最后一个换行后的内容
        lines = response.split(chr(10))
        if len(lines) > 1:
            response = lines[-1].strip()
        self.add_message("assistant", response)
        return response


# ============================================================
# 4. 完整微调流程示例
# ============================================================

def main():
    """完整微调流程示例"""
    # 配置
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    data_path = "data/instructions.jsonl"
    output_dir = "outputs/instruction_tuned"
    
    print("加载模型和 tokenizer...")
    model, tokenizer = create_model_and_tokenizer(model_name)
    
    print("准备数据集...")
    train_dataset = InstructionDataset(data_path, tokenizer, max_length=512)
    
    print("开始训练...")
    train_sft(model, tokenizer, train_dataset, output_dir, batch_size=2, num_epochs=3)
    
    print("训练完成！模型已保存到:", output_dir)
    
    # 测试对话
    print("测试对话模式 (输入 quit 退出):")
    # 测试对话
    print("Loading chatbot...")
    bot = ChatBot(model, tokenizer, system_prompt="You are a helpful assistant.")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        response = bot.generate_response(user_input)
        print("Assistant:", response)


if __name__ == "__main__":
    main()
