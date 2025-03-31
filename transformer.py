import torch.nn as nn
import torch
import math
from common_utils import MAX_SEQ_LENGTH, BATCH_SIZE

########################################## Multi Head Attention ##################################################

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

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
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
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EncoderLayerMHA(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayerMHA, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    

class TransformerClassifierMHA(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(TransformerClassifierMHA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            EncoderLayerMHA(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)  
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
        
    def forward(self, src):
        src_mask = self.generate_mask(src)
        src_embedded = self.dropout(self.positional_encoding(self.embedding(src)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask) 
        
        pooled = self.pool(enc_output.transpose(1, 2)).squeeze(2)
        output = self.classifier(pooled)
        return output


########################################## Multi Query Attention ##################################################

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiQueryAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model) 
        self.W_k = nn.Linear(d_model, self.d_k)
        self.W_v = nn.Linear(d_model, self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
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
        batch_size = Q.size(0)
        Q = self.split_heads(self.W_q(Q))
        K = self.W_k(K)
        V = self.W_v(V)

        K = K.unsqueeze(1).expand(batch_size, self.num_heads, MAX_SEQ_LENGTH, self.d_k)  
        V = V.unsqueeze(1).expand(batch_size, self.num_heads, MAX_SEQ_LENGTH, self.d_k) 
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class EncoderLayerMQA(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayerMQA, self).__init__()
        self.self_attn = MultiQueryAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    

class TransformerClassifierMQA(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(TransformerClassifierMQA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            EncoderLayerMQA(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)  
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
        
    def forward(self, src):
        src_mask = self.generate_mask(src)
        src_embedded = self.dropout(self.positional_encoding(self.embedding(src)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask) 
        
        pooled = self.pool(enc_output.transpose(1, 2)).squeeze(2)
        output = self.classifier(pooled)
        return output
    
########################################## Group Query Attention ##################################################

class GroupQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_groups):
        super(GroupQueryAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.num_kv_groups = num_kv_groups
        self.heads_per_group = num_heads // num_kv_groups

        self.W_q = nn.Linear(d_model, d_model) 
        self.W_k = nn.Linear(d_model, self.d_k * self.num_kv_groups)
        self.W_v = nn.Linear(d_model, self.d_k * self.num_kv_groups)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
    
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        # x.view will reshape [batch_size, seq_length, d_model] to [batch_size, seq_length, num_heads, d_k]
        # transpose will change [batch_size, seq_length, num_heads, d_k] to [batch_size, num_heads, seq_length, d_k]
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def split_kv_groups(self, x):
        batch_size, seq_length, d_kv = x.size()
        return x.view(batch_size, seq_length, self.num_kv_groups, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        # transpose will change the shape from [batch_size, num_heads, seq_length, d_k] to [batch_size, seq_length, num_heads, d_k]
        # contiguous ensures the tensor is stored in a contiguous block of memory, required before view
        # view will combine the last 2 dimention to form self.d_model where self.d_model = num_heads × d_k
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_kv_groups(self.W_k(K))
        V = self.split_kv_groups(self.W_v(V))

        K_expanded = []
        V_expanded = []

        for i in range(self.num_kv_groups):
            K_group = K[:, i:i+1]  # [batch_size, 1, seq_length_k, d_k]
            V_group = V[:, i:i+1]  
            
            K_group = K_group.expand(-1, self.heads_per_group, -1, -1)  # [batch_size, heads_per_group, seq_length_k, d_k]
            V_group = V_group.expand(-1, self.heads_per_group, -1, -1) 
            
            K_expanded.append(K_group)
            V_expanded.append(V_group)
        
        K = torch.cat(K_expanded, dim=1) # [batch_size, num_heads, seq_length_k, d_k]
        V = torch.cat(V_expanded, dim=1) 
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class EncoderLayerGQA(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_kv_groups, dropout):
        super(EncoderLayerGQA, self).__init__()
        self.self_attn = GroupQueryAttention(d_model, num_heads, num_kv_groups)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    

class TransformerClassifierGQA(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model, num_heads, num_layers, d_ff, max_seq_length, num_kv_groups, dropout):
        super(TransformerClassifierGQA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            EncoderLayerGQA(d_model, num_heads, d_ff, num_kv_groups, dropout) 
            for _ in range(num_layers)
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)  
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
        
    def forward(self, src):
        src_mask = self.generate_mask(src)
        src_embedded = self.dropout(self.positional_encoding(self.embedding(src)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask) 
        
        pooled = self.pool(enc_output.transpose(1, 2)).squeeze(2)
        output = self.classifier(pooled)
        return output
    

########################################## MultiHead Latent Attention ##################################################

class MultiheadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, latent_dim=576):
        super(MultiheadLatentAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.latent_dim = latent_dim
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_uk = nn.Linear(d_model, latent_dim)
        
        # key-value latent
        self.W_dkv = nn.Linear(d_model, latent_dim)
        self.W_o = nn.Linear(latent_dim, d_model)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        # key & value latents (L_KV)
        L_kv = self.W_dkv(x)  # [batch_size, seq_length, latent_dim]
        
        # project queries to latent space - simulating X(W_Q*W_UK^T)
        q_projected = self.W_q(x) 
        queries_projected = self.W_uk(q_projected)  # [batch_size, seq_length, latent_dim]
        
        # transpose L_KV for attention computation
        L_kv_t = L_kv.transpose(-1, -2) 
        
        # Compute attention scores: Q*K^T / sqrt(d_k)
        attn_scores = torch.matmul(queries_projected, L_kv_t) / self.scale  # [batch_size, seq_length, seq_length]
        
        if mask is not None:
            if mask.dim() == 4:
                mask = mask.squeeze(1)  
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch_size, seq_length, seq_length]
        
        weighted_latents = torch.matmul(attn_weights, L_kv)  # [batch_size, seq_length, latent_dim]
        
        output = self.W_o(weighted_latents)  # [batch_size, seq_length, d_model]
        
        return output
    

class EncoderLayerMLA(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, latent_dim, dropout):
        super(EncoderLayerMLA, self).__init__()
        self.self_attn = MultiheadLatentAttention(d_model, num_heads, latent_dim)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    

class TransformerClassifierMLA(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model, num_heads, num_layers, d_ff, max_seq_length, latent_dim, dropout):
        super(TransformerClassifierMLA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            EncoderLayerMLA(d_model, num_heads, d_ff, latent_dim, dropout) 
            for _ in range(num_layers)
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)  
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
        
    def forward(self, src):
        src_mask = self.generate_mask(src)
        src_embedded = self.dropout(self.positional_encoding(self.embedding(src)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask) 
        
        if enc_output.dim() == 4:
            batch_size, dim1, seq_length, features = enc_output.shape
            enc_output = enc_output.reshape(batch_size, seq_length, -1)
        
        pooled = self.pool(enc_output.transpose(1, 2)).squeeze(2)
        output = self.classifier(pooled)
        return output
    