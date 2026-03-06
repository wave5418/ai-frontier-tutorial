# -*- coding: utf-8 -*-
"""
第一章代码实现：预训练语言模型

包含：
1. Word2Vec 实现
2. LSTM 语言模型
3. 简单预训练语言模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ============================================
# 1. Word2Vec 实现
# ============================================

class Word2Vec(nn.Module):
    """Word2Vec 模型 (Skip-gram)"""
    
    def __init__(self, vocab_size, embed_dim):
        super(Word2Vec, self).__init__()
        # 输入嵌入（中心词）
        self.W_in = nn.Embedding(vocab_size, embed_dim)
        # 输出嵌入（上下文词）
        self.W_out = nn.Embedding(vocab_size, embed_dim)
        
        # 初始化
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
    
    def forward(self, center_word):
        """
        Args:
            center_word: [batch_size] 中心词索引
        Returns:
            center_embed: [batch_size, embed_dim] 中心词嵌入
        """
        center_embed = self.W_in(center_word)
        return center_embed
    
    def get_word_embedding(self, word_idx):
        """获取词的向量表示"""
        with torch.no_grad():
            return self.W_in.weight[word_idx].cpu().numpy()


class CBOW(nn.Module):
    """CBOW 模型"""
    
    def __init__(self, vocab_size, embed_dim, context_size=2):
        super(CBOW, self).__init__()
        self.context_size = context_size
        self.W_in = nn.Embedding(vocab_size, embed_dim)
        self.W_out = nn.Linear(embed_dim, vocab_size)
        
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
    
    def forward(self, context_words):
        """
        Args:
            context_words: [batch_size, context_size*2] 上下文词索引
        Returns:
            output: [batch_size, vocab_size] 预测的中心词概率分布
        """
        # 获取上下文词嵌入并平均
        embed = self.W_in(context_words)  # [batch_size, context_size*2, embed_dim]
        embed = embed.mean(dim=1)  # [batch_size, embed_dim]
        
        # 预测中心词
        output = self.W_out(embed)  # [batch_size, vocab_size]
        return output


# ============================================
# 2. LSTM 语言模型
# ============================================

class LSTMLanguageModel(nn.Module):
    """基于 LSTM 的语言模型"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTMLanguageModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM 层
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def init_hidden(self, batch_size, device):
        """初始化隐藏状态"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: [batch_size, seq_len] 输入序列
            hidden: (h0, c0) 初始隐藏状态
        Returns:
            logits: [batch_size, seq_len, vocab_size] 预测 logits
            hidden: (hn, cn) 最终隐藏状态
        """
        batch_size = x.size(0)
        device = x.device
        
        # 初始化隐藏状态
        if hidden is None:
            hidden = self.init_hidden(batch_size, device)
        
        # 嵌入
        embed = self.dropout(self.embedding(x))  # [batch_size, seq_len, embed_dim]
        
        # LSTM
        out, hidden = self.lstm(embed, hidden)  # [batch_size, seq_len, hidden_dim]
        out = self.dropout(out)
        
        # 输出层
        logits = self.fc(out)  # [batch_size, seq_len, vocab_size]
        
        return logits, hidden
    
    def generate(self, start_token, max_len=50, temperature=1.0):
        """文本生成"""
        self.eval()
        device = next(self.parameters()).device
        
        # 初始化
        tokens = [start_token]
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_len):
                # 准备输入
                x = torch.tensor([[tokens[-1]]], device=device)
                
                # 前向传播
                logits, hidden = self.forward(x, hidden)
                
                # 采样下一个词
                logits = logits[0, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # 检查是否结束
                if next_token == 0:  # 假设 0 是<EOS>
                    break
                
                tokens.append(next_token)
        
        return tokens


# ============================================
# 3. 预训练语言模型
# ============================================

class PretrainingLM(nn.Module):
    """用于预训练的语言模型"""
    
    def __init__(self, vocab_size, embed_dim=512, hidden_dim=512, 
                 num_layers=4, max_seq_len=512):
        super(PretrainingLM, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 位置嵌入
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        # LSTM 层
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids, position_ids=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            position_ids: [batch_size, seq_len] 或 None
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 位置编码
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # 词嵌入 + 位置嵌入
        token_embed = self.token_embedding(input_ids)
        pos_embed = self.position_embedding(position_ids)
        embed = token_embed + pos_embed
        
        # LSTM
        lstm_out, _ = self.lstm(embed)
        
        # 输出
        logits = self.fc(lstm_out)
        
        return logits
    
    def mlm_loss(self, logits, labels, mask):
        """
        计算 MLM loss
        
        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len] 真实标签
            mask: [batch_size, seq_len] 1 表示需要预测的位置
        Returns:
            loss: scalar
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # 只计算 mask 位置的 loss
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
        loss = loss.view(batch_size, seq_len)
        
        # 应用 mask
        loss = (loss * mask).sum() / mask.sum()
        
        return loss


# ============================================
# 4. 数据集
# ============================================

class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 分词
        tokens = self.tokenizer.encode(text)
        
        # 截断或填充
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [0] * (self.max_len - len(tokens))
        
        # 转换为 tensor
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, labels


# ============================================
# 5. 训练函数
# ============================================

def train_word2vec(model, dataloader, epochs=5, lr=0.01, device='cpu'):
    """训练 Word2Vec"""
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            context, target = context.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            if isinstance(model, Word2Vec):
                # Skip-gram: 中心词→上下文
                embed = model(context)
                # 简化：直接计算与目标词的相似度
                output = model.W_out.weight[target].sum(dim=1)
                loss = -output.mean()
            else:
                # CBOW
                output = model(context)
                loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


def train_language_model(model, dataloader, epochs=10, lr=1e-3, device='cpu'):
    """训练语言模型"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids, labels = input_ids.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            logits, _ = model(input_ids)
            
            # 计算 loss
            batch_size, seq_len, vocab_size = logits.shape
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_lm.pth')
            print("Saved best model")


# ============================================
# 6. 使用示例
# ============================================

if __name__ == "__main__":
    # 配置
    vocab_size = 10000
    embed_dim = 300
    hidden_dim = 512
    num_layers = 2
    batch_size = 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. 训练 Word2Vec
    print("\n=== Training Word2Vec ===")
    w2v_model = Word2Vec(vocab_size, embed_dim)
    # 需要准备 (center_word, context_word) 数据对
    # train_word2vec(w2v_model, w2v_dataloader)
    
    # 2. 训练 LSTM 语言模型
    print("\n=== Training LSTM Language Model ===")
    lm_model = LSTMLanguageModel(vocab_size, embed_dim, hidden_dim, num_layers)
    # 需要准备 (input_ids, labels) 数据
    # train_language_model(lm_model, lm_dataloader)
    
    # 3. 预训练模型
    print("\n=== Pretraining Language Model ===")
    pretrain_model = PretrainingLM(vocab_size, embed_dim, hidden_dim, num_layers)
    # 需要准备 MLM 格式的数据
    # train_pretraining(pretrain_model, pretrain_dataloader)
    
    print("\nTraining complete!")
