"""
Decoder-Only Transformer Model Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from config import Config


class EmbeddingLayer(nn.Module):
    """Token embedding with sinusoidal positional encoding"""

    def __init__(self, vocab_size, embed_dim, max_seq_len, pretrained_embeddings=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=Config.PAD_IDX)

        if pretrained_embeddings is not None:
            self.token_embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        pe = self._create_positional_encoding(max_seq_len, embed_dim)
        self.register_buffer('pos_encoding', pe)

    def _create_positional_encoding(self, max_seq_len, embed_dim):
        """Sinusoidal positional encoding as per Vaswani et al."""
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) *
                           -(np.log(10000.0) / embed_dim))

        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        token_emb = self.token_embedding(x)
        pos_emb = self.pos_encoding[:, :seq_len, :]
        return token_emb + pos_emb


class LayerNorm(nn.Module):
    """Layer normalization"""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        return self.gamma * (x - mean) / std + self.beta


class MultiHeadSelfAttention(nn.Module):
    """Masked multi-head self-attention with optional KV caching"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('causal_mask', None)

    def _create_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
            self.causal_mask = self._create_causal_mask(seq_len, x.device)
        scores = scores + self.causal_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(attn_output)

    def forward_with_cache(self, x, mask=None, kv_cache=None, use_cache=False):
        """Forward pass with KV caching for faster inference"""
        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if use_cache and kv_cache is not None:
            past_K, past_V = kv_cache
            K = torch.cat([past_K, K], dim=2)
            V = torch.cat([past_V, V], dim=2)

        if use_cache:
            new_cache = (K, V)
        else:
            new_cache = None

        full_seq_len = K.size(2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is None:
            if self.causal_mask is None or self.causal_mask.size(0) != full_seq_len:
                self.causal_mask = self._create_causal_mask(full_seq_len, x.device)
            mask_to_use = self.causal_mask[-seq_len:, :full_seq_len].unsqueeze(0).unsqueeze(0)
        else:
            mask_to_use = mask

        scores = scores + mask_to_use
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)

        return output, new_cache


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class CheckpointFunction(torch.autograd.Function):
    """Manual gradient checkpointing implementation"""

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state

        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            if torch.cuda.is_available():
                ctx.fwd_gpu_state = torch.cuda.get_rng_state()

        ctx.save_for_backward(*args)

        with torch.no_grad():
            outputs = run_function(*args)

        return outputs

    @staticmethod
    def backward(ctx, *output_grads):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad()")

        inputs = ctx.saved_tensors

        if ctx.preserve_rng_state:
            rng_state = torch.get_rng_state()
            torch.set_rng_state(ctx.fwd_cpu_state)
            if torch.cuda.is_available():
                cuda_rng_state = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(ctx.fwd_gpu_state)

        detached_inputs = tuple(x.detach().requires_grad_(True) for x in inputs)

        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        if ctx.preserve_rng_state:
            torch.set_rng_state(rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        torch.autograd.backward(outputs, output_grads)

        return (None, None) + tuple(x.grad for x in detached_inputs)


def checkpoint(function, *args, use_reentrant=True, **kwargs):
    """Apply checkpointing to a function"""
    if kwargs:
        raise ValueError("Keyword arguments are not supported")
    return CheckpointFunction.apply(function, use_reentrant, *args)


class TransformerBlock(nn.Module):
    """Transformer decoder block"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_checkpoint=False):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x):
        normed = self.norm1(x)
        attn_out = self.attention(normed)
        attn_out = self.dropout1(attn_out)
        x = x + attn_out

        normed = self.norm2(x)
        ff_out = self.feed_forward(normed)
        ff_out = self.dropout2(ff_out)
        x = x + ff_out

        return x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)

    def forward_with_cache(self, x, mask=None, kv_cache=None, use_cache=False):
        """Forward with KV caching"""
        normed_x = self.norm1(x)
        attn_output, new_cache = self.attention.forward_with_cache(
            normed_x, mask=mask, kv_cache=kv_cache, use_cache=use_cache
        )
        attn_output = self.dropout1(attn_output)
        x = x + attn_output

        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        ff_output = self.dropout2(ff_output)
        x = x + ff_output

        return x, new_cache


class DecoderOnlyTransformer(nn.Module):
    """Decoder-only transformer model"""

    def __init__(self, vocab_size, d_model=300, num_heads=6, num_layers=6,
                 d_ff=1200, max_seq_len=64, dropout=0.1, pad_idx=0,
                 pretrained_embeddings=None, use_checkpoint=False):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

        self.embedding = EmbeddingLayer(vocab_size, d_model, max_seq_len, pretrained_embeddings)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, use_checkpoint)
            for _ in range(num_layers)
        ])

        self.final_norm = LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nDecoder-Only Transformer")
        print(f"Vocab: {vocab_size:,}, Layers: {num_layers}, Heads: {num_heads}")
        print(f"Parameters: {total_params:,}\n")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        assert seq_len <= self.max_seq_len

        x = self.embedding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.output_projection(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, self.vocab_size)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=self.pad_idx)

        return logits, loss
