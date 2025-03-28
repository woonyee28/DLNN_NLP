import torch.nn as nn
import torch
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)  # [batch_size, num_queries, d_k]
        self.W_k = nn.Linear(d_model, d_model)  # [batch_size, num_keys, d_k]
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
    
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        # x.view will reshape [batch_size, seq_length, d_model] to [batch_size, seq_length, num_heads, d_k]
        # transpose will change [batch_size, seq_length, num_heads, d_k] to [batch_size, num_heads, seq_length, d_k]
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        # transpose will change the shape from [batch_size, num_heads, seq_length, d_k] to [batch_size, seq_length, num_heads, d_k]
        # contiguous ensures the tensor is stored in a contiguous block of memory, required before view
        # view will combine the last 2 dimention to form self.d_model where self.d_model = num_heads × d_k
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output




