# %%
import os
import re
import math
import random
import functools
import torch
import torch.nn as nn
from torch.nn import functional as F
import triton
import triton.language as tl
from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime

load_dotenv()

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team.
#
# This file contains an adapted flex-attention sink kernel derived from Unsloth.
# Attribution required by the request: https://github.com/unslothai/unsloth

FLEX_ATTENTION_KV_INCREMENT = 512


def _torch_compile(fn):
    if hasattr(torch, "compile"):
        try:
            return torch.compile(fn, fullgraph=False, dynamic=True)
        except TypeError:
            return torch.compile(fn, fullgraph=False)
    return fn


try:
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as FLEX_ATTENTION_BLOCK_SIZE
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask as _create_block_mask,
    )
    from torch.nn.attention.flex_attention import AuxRequest, _score_mod_signature, _mask_mod_signature
    HAS_FLEX_ATTENTION = True
except Exception:
    HAS_FLEX_ATTENTION = False
    FLEX_ATTENTION_BLOCK_SIZE = None
    _flex_attention = None
    _create_block_mask = None


if HAS_FLEX_ATTENTION:
    try:
        import torch._dynamo as _dynamo
    except Exception:
        pass
    vram_of_gpu = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        vram_of_gpu = min(
            torch.cuda.memory.mem_get_info(i)[-1] / 1024 / 1024 / 1024
            for i in range(torch.cuda.device_count())
        )
    kernel_options = None
    if vram_of_gpu is not None and vram_of_gpu <= 16:
        kernel_options = {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "BLOCK_M1": 32,
            "BLOCK_N1": 32,
            "BLOCK_M2": 32,
            "BLOCK_N2": 32,
        }
    elif vram_of_gpu is not None and vram_of_gpu <= 24:
        kernel_options = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_M1": 32,
            "BLOCK_N1": 64,
            "BLOCK_M2": 64,
            "BLOCK_N2": 32,
        }
    if kernel_options is not None:
        _flex_attention = functools.partial(_flex_attention, kernel_options=kernel_options)

    uncompiled_flex_attention = _flex_attention
    flex_attention = _torch_compile(_flex_attention)
    _compiled_create_block_mask = _torch_compile(_create_block_mask)

    @functools.lru_cache
    def create_block_mask_cached(mask_mod, M, N, device="cuda"):
        return _create_block_mask(mask_mod, None, None, M, N, device=device)

    @functools.lru_cache
    def create_block_mask(mask_mod, bsz, head, M, N, device="cuda"):
        return _create_block_mask(mask_mod, bsz, head, M, N, device=device)

    def compiled_create_block_mask_cached(mask_mod, M, N, device="cuda"):
        return _compiled_create_block_mask(mask_mod, None, None, M, N, device=device)

    def compiled_create_block_mask(mask_mod, bsz, head, M, N, device="cuda"):
        return _compiled_create_block_mask(mask_mod, bsz, head, M, N, device=device)

    def causal_mask(batch_idx, head_idx, q_idx, kv_idx):
        return q_idx >= kv_idx

    def generate_causal_mask_with_padding(padding_start_idx=None):
        assert padding_start_idx is not None and type(padding_start_idx) is torch.Tensor
        assert padding_start_idx.dim() == 1
        assert padding_start_idx.shape[0] >= 1

        def _mask(batch_idx, head_idx, q_idx, kv_idx):
            q_start = q_idx >= padding_start_idx[batch_idx]
            k_start = kv_idx >= padding_start_idx[batch_idx]
            return q_start & k_start & (q_idx >= kv_idx)

        _mask.__name__ = _mask.__doc__ = "causal_mask_with_left_padding"
        return _mask

    def generate_decoding_causal_mask_with_padding(padding_start_idx=None):
        assert padding_start_idx is not None and type(padding_start_idx) is torch.Tensor
        assert padding_start_idx.dim() == 1
        assert padding_start_idx.shape[0] >= 1

        def _mask(batch_idx, head_idx, q_idx, kv_idx):
            k_start = kv_idx >= padding_start_idx[batch_idx]
            return k_start & (q_idx >= kv_idx)

        _mask.__name__ = _mask.__doc__ = "decoding_causal_mask_with_left_padding"
        return _mask

    @functools.lru_cache
    def generate_sliding_window_mask(window_size: int):
        def sliding_window(batch_idx, head_idx, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            windowed = q_idx - kv_idx < window_size
            return causal & windowed

        sliding_window.__name__ = sliding_window.__doc__ = f"sliding_window_{window_size}"
        return sliding_window

    def generate_sliding_window_mask_with_padding(window_size: int, padding_start_idx=None):
        assert padding_start_idx is not None and type(padding_start_idx) is torch.Tensor
        assert padding_start_idx.dim() == 1
        assert padding_start_idx.shape[0] >= 1

        def sliding_window(batch_idx, head_idx, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            windowed = q_idx - kv_idx < window_size
            q_padded = q_idx >= padding_start_idx[batch_idx]
            k_padded = kv_idx >= padding_start_idx[batch_idx]
            return q_padded & k_padded & causal & windowed

        sliding_window.__name__ = sliding_window.__doc__ = f"sliding_window_with_left_padding_{window_size}"
        return sliding_window

    def generate_decoding_sliding_window_mask_with_padding(window_size: int, padding_start_idx=None):
        return generate_sliding_window_mask(window_size)

    def get_score_mod_w_offset(score_mod: _score_mod_signature, _offset: torch.tensor):
        def _score_mod(score, b, h, q, kv):
            return score_mod(score, b, h, q + _offset, kv)

        return _score_mod

    def get_mask_mod_w_offset(mask_mod: _mask_mod_signature, _offset: torch.tensor):
        def _mask_mod(b, h, q, kv):
            return mask_mod(b, h, q + _offset, kv)

        return _mask_mod

    class FlexAttentionCache:
        __slots__ = (
            "offset",
            "offset_tensor",
            "mask_mod_with_offset",
            "block_mask",
            "mask_mod",
            "max_length",
            "block_size",
            "sliding_window",
            "block_mask_slice",
        )

        def __init__(self, key, mask_mod, sliding_window):
            bsz, heads_kv, qlen_kv, dim = key.shape
            if sliding_window is None:
                div, mod = divmod(qlen_kv, FLEX_ATTENTION_KV_INCREMENT)
                n = FLEX_ATTENTION_KV_INCREMENT * div + (FLEX_ATTENTION_KV_INCREMENT if mod != 0 else 0)
                self.offset = qlen_kv - 2
                if self.offset <= -2:
                    self.offset = -1
                self.sliding_window = None
            else:
                n = sliding_window
                self.offset = min(sliding_window, qlen_kv) - 2
                if self.offset <= -2:
                    self.offset = -1
                self.sliding_window = sliding_window - 1
            self.offset_tensor = torch.tensor(self.offset, device=key.device, dtype=torch.int32)
            self.block_mask = compiled_create_block_mask(mask_mod, bsz, heads_kv, n, n, device=key.device)
            self.mask_mod = mask_mod
            self.max_length = n
            self.block_size = self.block_mask.BLOCK_SIZE[0]
            self.mask_mod_with_offset = get_mask_mod_w_offset(self.mask_mod, self.offset_tensor)
            self.block_mask_slice = None

        def __call__(self, key):
            bsz, heads_kv, qlen_kv, dim = key.shape
            if (self.sliding_window is None) or (self.offset < self.sliding_window):
                self.offset += 1
                self.offset_tensor.add_(1)
            elif self.sliding_window is not None:
                return self.block_mask_slice
            if self.offset >= self.max_length:
                self.max_length += FLEX_ATTENTION_KV_INCREMENT
                self.block_mask = compiled_create_block_mask(
                    self.mask_mod, bsz, heads_kv, self.max_length, self.max_length, device=key.device
                )
                self.block_size = self.block_mask.BLOCK_SIZE[0]
            block_offset = self.offset // self.block_size
            block_mask_slice = self.block_mask[:, :, block_offset]
            block_mask_slice.mask_mod = self.mask_mod_with_offset
            block_mask_slice.seq_lengths = (1, qlen_kv)
            self.block_mask_slice = block_mask_slice
            return block_mask_slice

    def causal_mask_with_sink(batch, head, q_idx, kv_idx):
        causal = (q_idx + 1) >= kv_idx
        sink_first_column = kv_idx == 0
        return causal | sink_first_column

    @functools.lru_cache
    def generate_sliding_window_with_sink(window_size: int):
        def sliding_window(batch, head, q_idx, kv_idx):
            causal = (q_idx + 1) >= kv_idx
            windowed = (q_idx + 1) - kv_idx < window_size
            sink_first_column = kv_idx == 0
            return (causal & windowed) | sink_first_column

        sliding_window.__name__ = sliding_window.__doc__ = f"sliding_window_{window_size}_sink"
        return sliding_window

    @functools.lru_cache
    def generate_sink_score_mod(sink_weights: torch.Tensor):
        def sink_score_mod(score, batch, head, q_idx, kv_idx):
            return torch.where(
                kv_idx == 0,
                sink_weights[head].to(score.dtype) + 0.0,
                score,
            )

        return sink_score_mod

    def old_flex_attention_with_sink(
        self_attn,
        query,
        key,
        value,
        attention_mask=None,
        scale=None,
        sliding_window=None,
        compile=True,
    ):
        if not self_attn.training:
            raise NotImplementedError("flex attention sink only supports training in this mode")
        assert getattr(self_attn, "sinks", None) is not None, "self_attn must have sinks"
        sink_weights = self_attn.sinks
        enable_gqa = getattr(self_attn, "num_key_value_groups", 1) != 1
        scale = getattr(self_attn, "scaling", None) or getattr(self_attn, "scale", None) or scale

        bsz, heads_q, qlen_q, dim = query.shape
        _, heads_kv, qlen_kv, _ = key.shape

        key_padded = torch.cat([key.new_zeros(bsz, heads_kv, 1, dim), key], dim=2)
        value_padded = torch.cat([value.new_zeros(bsz, heads_kv, 1, dim), value], dim=2)

        sliding_window = sliding_window or getattr(self_attn, "sliding_window", None)
        mask_mod = (
            generate_sliding_window_with_sink(sliding_window)
            if type(sliding_window) is int and sliding_window != 0
            else causal_mask_with_sink
        )
        score_mod = generate_sink_score_mod(sink_weights)
        block_mask = compiled_create_block_mask(mask_mod, qlen_q, qlen_kv + 1, device=key.device)
        attn_output = (flex_attention if compile else uncompiled_flex_attention)(
            query,
            key_padded,
            value_padded,
            block_mask=block_mask,
            score_mod=score_mod,
            enable_gqa=enable_gqa,
            scale=scale,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output

    def is_flex_attention_decoding(self_attn, query):
        if query.dim() == 4:
            bsz, heads_q, qlen_q, dim = query.shape
        else:
            bsz, qlen_q, dim = query.shape
        is_training = self_attn.training
        has_flex_cache = hasattr(self_attn, "_flex_attention_cache")
        if is_training or (not is_training and (not has_flex_cache or qlen_q != 1)):
            return False
        return True

    def flex_attention_with_sink(
        self_attn,
        query,
        key,
        value,
        attention_mask=None,
        scale=None,
        sliding_window=None,
        compile=True,
        has_static_cache=True,
    ):
        assert getattr(self_attn, "sinks", None) is not None, "self_attn must have sinks"
        sink_weights = self_attn.sinks
        enable_gqa = getattr(self_attn, "num_key_value_groups", 1) != 1
        scale = getattr(self_attn, "scaling", None) or getattr(self_attn, "scale", None) or scale

        bsz, heads_q, qlen_q, dim = query.shape
        _, heads_kv, qlen_kv, _ = key.shape

        sliding_window = sliding_window or getattr(self_attn, "sliding_window", None)
        is_training = self_attn.training
        mask_mod = None
        block_mask = None
        has_flex_cache = hasattr(self_attn, "_flex_attention_cache")
        if attention_mask is not None and has_static_cache:
            if is_training or (not is_training and (not has_flex_cache or qlen_q != 1)):
                if is_training:
                    if has_flex_cache:
                        del self_attn._flex_attention_cache
                else:
                    assert attention_mask is not None
                    assert attention_mask.dim() == 2, f"attention_mask has dim = {attention_mask.dim()}"
                    padding_start_idx = attention_mask.argmax(1).to(query.device)
                    do_padding = (
                        torch.arange(max(qlen_q, qlen_kv), device=query.device)
                        .repeat((bsz, 1))
                        .lt(padding_start_idx.unsqueeze(0).T)
                    )
                    query.transpose(2, 1)[do_padding[:, :qlen_q]] = 1
                    key.transpose(2, 1)[do_padding[:, :qlen_kv]] = -torch.inf
                    value.transpose(2, 1)[do_padding[:, :qlen_kv]] = 0
                    mask_mod = prefill_mask_mod = (
                        generate_sliding_window_mask_with_padding(sliding_window, padding_start_idx)
                        if type(sliding_window) is int and sliding_window != 0
                        else generate_causal_mask_with_padding(padding_start_idx)
                    )
                    decoding_mask_mod = (
                        generate_decoding_sliding_window_mask_with_padding(sliding_window, padding_start_idx)
                        if type(sliding_window) is int and sliding_window != 0
                        else generate_decoding_causal_mask_with_padding(padding_start_idx)
                    )
                    self_attn._flex_attention_cache = FlexAttentionCache(key, decoding_mask_mod, sliding_window)
            else:
                block_mask = self_attn._flex_attention_cache(key)
        if mask_mod is None:
            mask_mod = (
                generate_sliding_window_mask(sliding_window)
                if type(sliding_window) is int and sliding_window != 0
                else causal_mask
            )
        if block_mask is None:
            block_mask = compiled_create_block_mask(mask_mod, bsz, heads_q, qlen_q, qlen_kv, device=key.device)

        if compile:
            out = flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                score_mod=None,
                enable_gqa=enable_gqa,
                scale=scale,
                return_aux=AuxRequest(lse=True),
            )
        else:
            out = uncompiled_flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                score_mod=None,
                enable_gqa=enable_gqa,
                scale=scale,
                return_aux=AuxRequest(lse=True),
            )
        attn_output, aux = out
        logsumexp = aux.lse

        sink_scale = torch.sigmoid(logsumexp - sink_weights.unsqueeze(1))
        attn_output = attn_output * sink_scale.unsqueeze(-1).to(attn_output.dtype)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output

    def flex_attention_with_sink_decoding(
        self_attn,
        query,
        key,
        value,
        scale=None,
    ):
        assert getattr(self_attn, "sinks", None) is not None, "self_attn must have sinks"
        enable_gqa = getattr(self_attn, "num_key_value_groups", 1) != 1
        scale = getattr(self_attn, "scaling", None) or getattr(self_attn, "scale", None) or scale
        block_mask = self_attn._flex_attention_cache(key)
        out = flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            score_mod=None,
            enable_gqa=enable_gqa,
            scale=scale,
            return_aux=AuxRequest(lse=True),
        )
        attn_output, aux = out
        return attn_output, aux.lse

    def flex_attention_add_sinks(
        self_attn,
        attn_output,
        logsumexp,
    ):
        logsumexp -= self_attn.sinks.unsqueeze(1)
        sink_scale = torch.sigmoid(logsumexp, out=logsumexp)
        attn_output *= sink_scale.unsqueeze(-1).to(attn_output.dtype)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output

    def flash_attention_left_padded(
        self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        is_causal=True,
        window_size_left=None,
        dropout_p=0.0,
        scale=None,
    ):
        assert attention_mask.dtype in (torch.int32, torch.int64, torch.bool)
        device = query_states.device

        bsz, qlen = attention_mask.shape
        n_heads = self_attn.config.num_attention_heads
        n_kv_heads = getattr(self_attn.config, "num_key_value_heads", n_heads)
        head_dim = self_attn.head_dim

        bsz, heads_q, qlen_q, dim = query_states.shape
        _, heads_kv, qlen_kv, _ = key_states.shape

        q = query_states.transpose(1, 2)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)

        seqlens = attention_mask.to(dtype=torch.int32, device=device).sum(dim=1)
        cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0))
        max_seqlen = int(seqlens.max().item())

        flat_mask = attention_mask.reshape(-1).to(device=device)
        keep = flat_mask.nonzero(as_tuple=False).squeeze(-1)

        q_flat = q.reshape(bsz * qlen_q, n_heads, head_dim)
        k_flat = k.reshape(bsz * qlen_kv, n_kv_heads, head_dim)
        v_flat = v.reshape(bsz * qlen_kv, n_kv_heads, head_dim)

        q_unpad = q_flat.index_select(0, keep).contiguous()
        k_unpad = k_flat.index_select(0, keep).contiguous()
        v_unpad = v_flat.index_select(0, keep).contiguous()

        if scale is None:
            scale = 1.0 / math.sqrt(head_dim)

        kwargs = dict(scale=scale)
        if window_size_left is not None:
            kwargs["window_size_left"] = int(window_size_left)
            kwargs["window_size_right"] = 0

        attn_output, logsumexp, rng_state, _, _ = torch.ops.aten._flash_attention_forward(
            query=q_unpad,
            key=k_unpad,
            value=v_unpad,
            cum_seq_q=cu_seqlens,
            cum_seq_k=cu_seqlens,
            max_q=max_seqlen,
            max_k=max_seqlen,
            dropout_p=float(dropout_p),
            is_causal=bool(is_causal),
            return_debug_mask=False,
            **kwargs,
        )
        sink_scale = torch.sigmoid(logsumexp - self_attn.sinks.unsqueeze(1))
        attn_output = attn_output * sink_scale.unsqueeze(-1).transpose(0, 1).to(attn_output.dtype)

        out_flat = q_flat.new_zeros((bsz * qlen_q, n_heads, head_dim))
        out_flat[keep] = attn_output
        attn_output = out_flat.view(bsz, qlen_q, n_heads, head_dim)

        attn_output = attn_output.contiguous()
        return attn_output
else:
    def flex_attention_with_sink(*args, **kwargs):
        raise RuntimeError("flex_attention is not available in this PyTorch build")

# %%
use_wandb = True  # set to False to disable wandb logging
use_compiled_autograd = False  # enable only if backward compile is stable
use_model_compile = False  # keep False to avoid long/stuck full-model compilation
wandb = None
if use_wandb:
    try:
        import wandb as _wandb
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if wandb_api_key:
            _wandb.login(key=wandb_api_key)
            wandb = _wandb
            print("wandb initialized successfully")
        else:
            use_wandb = False
            print("WANDB_API_KEY not found - wandb logging disabled")
    except ImportError:
        use_wandb = False
        print("wandb not installed - wandb logging disabled")

torch.manual_seed(69)
torch.set_printoptions(profile="short", sci_mode=False, linewidth=100000)
torch.set_float32_matmul_precision('high')
# this script is configured to run on a RTX 3060 12GB GPU. you'll want to adjust the model sizes and batch sizes for other devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
plot_losses = False  # set True to save loss plots during training
plot_path = None  # set a filepath like "training_loss.png" when plot_losses is True
plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 50
plt.rcParams['axes.grid'] = True
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

# %% [markdown]
# # Data Prep

# %%
# we use this 40mb file of concatenated anime subtitles as our dataset
# just the right size for toy experiments like this I think
with open('animesubs.txt', 'r', encoding='latin') as f:
    text = f.read()
print("length of dataset in characters: ", len(text))

# %%
print(text[:500])

# %%
# remove japanese characters
text = ''.join(filter(lambda character:ord(character) < 0x3000, text))

# %%
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("unique characters:", vocab_size, ''.join(chars))

# %%
# yes, all language models will be character level, which isn't ideal but it's good for simplicity
# very simple tokenizer
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
# add special token for padding
stoi[''] = len(stoi)
itos[len(itos)] = ''
print(stoi)
print(itos)
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])
print("encoded:", encode(text[:20]))
print("decoded:", decode(encode(text[:20])))
vocab_size = len(itos)
print("vocab size:", vocab_size)

# %%
data = torch.tensor(encode(text), dtype=torch.int64)
data.shape

# %%
data[:100]

# %%
n = int(len(data) * 0.99)
train_data = data[:n]
val_data = data[n:]
print(train_data.shape, val_data.shape)

# %%
seq_len = 8
train_data[:seq_len+1]

# %%
def get_batch(split, seq_len, batch_size=4):
    # generate a small batch of data of inputs x and targets y
    # targets are just inputs shifted by 1
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

xb, yb = get_batch('train', 64, 2)
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

# %% [markdown]
# # Training Prep

# %%
# Make all steps, sequence lengths, and batch size the same
total_steps = 5000
seq_len = 256
batch_size = 256 # these are small models so we can use large batch sizes to fully utilize the GPU
# should cover around 2x the dataset
total_steps * seq_len * batch_size

# %%
# Test forward pass
torch_compile_options = {
    'epilogue_fusion': True, 
    'max_autotune': False, 
    'shape_padding': True, 
    'trace.enabled': False, 
    'triton.cudagraphs': False, 
    'debug': False, 
    'dce': True, 
    'memory_planning': True, 
    'coordinate_descent_tuning': False, 
    'trace.graph_diagram': False, 
    'compile_threads': 32, 
    'group_fusion': True, 
    'disable_progress': False, 
    'verbose_progress': False, 
    'triton.multi_kernel': 0, 
    'triton.use_block_ptr': False, 
    'triton.enable_persistent_tma_matmul': False, 
    'triton.autotune_at_compile_time': False, 
    'triton.cooperative_reductions': False, 
    'cuda.compile_opt_level': '-O2', 
    'cuda.enable_cuda_lto': True, 
    'combo_kernels': True, 
    'benchmark_combo_kernel': True, 
    'combo_kernel_foreach_dynamic_shapes': True
}

@torch.compile(fullgraph=False, options=torch_compile_options)
def compile_optimizer_lr(opt, scheduler):
    opt.step()
    scheduler.step()


def save_checkpoint(model, optimizer, scheduler, step, losses, val_losses, seq_len, batch_size, total_steps, save_dir='checkpoints'):
    """Save a complete checkpoint including model, optimizer, scheduler states and training metrics"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': losses,
        'val_losses': val_losses,
        'config': {
            'seq_len': seq_len,
            'batch_size': batch_size,
            'total_steps': total_steps,
            'vocab_size': model.token_embedding_table.num_embeddings,
            'embed_size': model.blocks[0].sa_heads.head_size * model.head_num,
            'head_num': model.head_num,
            'layer_num': model.layer_num
        }
    }
    
    checkpoint_path = os.path.join(save_dir, f'checkpoint_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)
    if use_wandb and wandb is not None:
        wandb.save(checkpoint_path)
    print(f"Checkpoint saved at step {step}: {checkpoint_path}")
    
    return checkpoint_path

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load a checkpoint and restore model, optimizer, scheduler states"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from step {checkpoint['step']}")
    return checkpoint['step'], checkpoint['train_losses'], checkpoint['val_losses']

def train(model, optimizer, scheduler, seq_len, batch_size, total_steps, val_steps=10, val_interval=50, checkpoint_interval=500, save_dir='checkpoints', plot_losses=False, plot_path=None):
    losses = []
    val_losses = []
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = (None, None)
    if plot_losses:
        fig, ax = plt.subplots()
    
    for step in (bar := tqdm(range(total_steps))):
        # sample a batch of data
        xb, yb = get_batch('train', seq_len=seq_len, batch_size=batch_size)

        # evaluate the loss
        logits, loss = model(xb, yb)
        # backprop
        optimizer.zero_grad(set_to_none=True)

        if use_compiled_autograd:
            with torch._dynamo.compiled_autograd._enable(torch.compile()):
                loss.backward()
        else:
            loss.backward()
        compile_optimizer_lr(optimizer, scheduler)

        bar.set_description(f"loss: {loss.item():.2f}, val loss: {val_losses[-1] if val_losses else 0:.2f}, lr: {scheduler.get_last_lr()[0]:.2e}")
        losses.append(loss.item())
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': scheduler.get_last_lr()[0],
                'step': step
            })
        
        if step % val_interval == 0:
            # Calculate validation loss
            with torch.no_grad():
                val_loss = 0
                for _ in range(val_steps):
                    xb, yb = get_batch('val', seq_len=seq_len, batch_size=batch_size)
                    _, loss = model(xb, yb)
                    val_loss += loss.item()
                val_loss /= val_steps
                val_losses.append(val_loss)
                
            # Log validation loss to wandb
            if use_wandb:
                wandb.log({
                    'val_loss': val_loss,
                    'step': step
                })
            if plot_losses:
                ax.clear()
                ax.plot(losses, color='blue', label='train loss', alpha=0.7)
                ax.plot(range(0, len(losses), val_interval), val_losses, color='red', label='val loss', alpha=0.7)
                ax.set_ylim(1, 4)
                ax.legend()
                if plot_path:
                    fig.savefig(plot_path, bbox_inches="tight")
            
        # Save checkpoint
        if step % checkpoint_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, scheduler, step, losses, val_losses, seq_len, batch_size, total_steps, save_dir)
            
    print('final loss:', loss.item(), 'final val loss:', val_loss)
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, total_steps, losses, val_losses, seq_len, batch_size, total_steps, save_dir)
    
    return losses, val_losses

# %%
# Measure post training perplexity on validation set
# Create function that receives a model, context length, and PPL sequence length, and returns the perplexity
# The PPL sequence length is the number of characters the function uses to calculate the perplexity
# We take the logits and calculate the cross entropy loss from scratch, then exponentiate it to get the perplexity
# not only that, but we want the models to do this in actual inference
def perplexity(model, seq_len, ppl_seq_len, batch_size=128, val_steps=1000):
    with torch.no_grad():
        val_loss = 0
        for _ in tqdm(range(val_steps)):
            xb, yb = get_batch('val', seq_len=seq_len, batch_size=batch_size)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _ = model(xb, yb)
            logits = logits.reshape(batch_size, seq_len, vocab_size)
            logits = logits[:, :ppl_seq_len]
            yb = yb[:, :ppl_seq_len]
            # flatten logits and targets
            logits = logits.reshape(batch_size*ppl_seq_len, vocab_size)
            yb = yb.reshape(batch_size*ppl_seq_len)
            # calculate cross entropy loss from scratch
            loss = F.cross_entropy(logits, yb)
            val_loss += loss.item()
        val_loss /= val_steps
        ppl = torch.exp(torch.tensor(val_loss))
        return ppl.item(), val_loss

# %% [markdown]
# # Transformers

# %% [markdown]
# ## Classic Transformer

# %%
class TransformerConfig:
    def __init__(self, vocab_size, seq_len, embed_size, head_num, layer_num):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.head_num = head_num
        self.layer_num = layer_num

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis.to(device)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    q_shape = [d if i == xq_.ndim - 2 or i == xq_.ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    k_shape = [d if i == xq_.ndim - 2 or i == xk_.ndim - 1 else 1 for i, d in enumerate(xk_.shape)]
    T_q = xq_.shape[-2] 
    q_freqs_cis = freqs_cis[-T_q:].view(*q_shape)
    k_freqs_cis = freqs_cis.view(*k_shape)
    xq_out = torch.view_as_real(xq_ * q_freqs_cis).flatten(xq.dim() - 1)
    xk_out = torch.view_as_real(xk_ * k_freqs_cis).flatten(xq.dim() - 1)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin_1 = nn.Linear(config.embed_size, config.embed_size*4)
        self.lin_2 = nn.Linear(config.embed_size*4, config.embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.relu(x)
        x = self.lin_2(x)
        return x

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention with AliBi in parallel """
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.head_num = config.head_num
        self.head_size = config.embed_size // config.head_num
        self.key = nn.Linear(config.embed_size, config.embed_size, bias=False)
        self.query = nn.Linear(config.embed_size, config.embed_size, bias=False)
        self.value = nn.Linear(config.embed_size, config.embed_size, bias=False)
        self.o = nn.Linear(config.embed_size, config.embed_size)
        self.sinks = nn.Parameter(torch.zeros(config.head_num))
        self.scaling = self.head_size**-0.5
        self.num_key_value_groups = 1
        self.sliding_window = None
        self.freqs_cis = precompute_freqs_cis(config.embed_size//config.head_num, config.seq_len)

    def forward(self, x, kv_cache=None):
        if not HAS_FLEX_ATTENTION:
            raise RuntimeError("flex_attention is not available in this PyTorch build")
        B, T, C = x.shape
        _, _, T_past, _ = kv_cache[0].shape if kv_cache is not None and kv_cache[0] is not None else (0, 0, 0, 0)
        q = self.query(x) # (B,T,C)
        k = self.key(x)   # (B,T,C)
        v = self.value(x) # (B,T,C)

        # Split into heads
        q = q.view(B, T, self.head_num, self.head_size).transpose(1, 2) # (B, H, T, C/H)
        k = k.view(B, T, self.head_num, self.head_size).transpose(1, 2) # (B, H, T, C/H)
        v = v.view(B, T, self.head_num, self.head_size).transpose(1, 2) # (B, H, T, C/H)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            if k_past is not None:
                k = torch.cat((k_past, k), dim=2)
                v = torch.cat((v_past, v), dim=2)
            if k.shape[-2] > self.seq_len:
                k = k[:, :, -self.seq_len:]
                v = v[:, :, -self.seq_len:]
            kv_cache = (k, v)
        T_k = k.shape[-2]
        q, k = apply_rotary_emb(q, k, self.freqs_cis[:T_k])

        out = flex_attention_with_sink(
            self,
            q,
            k,
            v,
            attention_mask=None,
            scale=self.scaling,
            sliding_window=self.sliding_window,
            compile=True,
            has_static_cache=True,
        )
        out = out.view(B, T, C) # (B, H, T, C/H) -> (B, T, H, C/H) -> (B, T, C)
        out = self.o(out)
        return out, kv_cache

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa_heads = MultiHeadAttention(config)
        self.ff_layer = FeedForward(config)
        self.sa_norm = RMSNorm(config.embed_size)
        self.ff_norm = RMSNorm(config.embed_size)
    
    def forward(self, x, kv_cache=None):
        a, kv_cache = self.sa_heads(self.sa_norm(x), kv_cache)
        h = x + a
        o = h + self.ff_layer(self.ff_norm(h))
        return o, kv_cache
    
class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_num = config.layer_num
        self.head_num = config.head_num
        self.seq_len = config.seq_len
        # embed raw tokens to a lower dimensional embedding with embed_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embed_size)
        # Language Modelling (?) Head is a standard linear layer to go from 
        # embeddings back to logits of vocab_size
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size)
        # transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layer_num)])

    def forward(self, idx, targets=None, kv_cache=None):
        B, T = idx.shape
        _, _, T_past, _ = kv_cache[0][0].shape if kv_cache is not None and kv_cache[0][0] is not None else (0, 0, 0, 0)
        # idx and targets are both (B,T) tensor of integers
        tok_embd = self.token_embedding_table(idx) # (B,T,C)
        x = tok_embd
        # go through blocks
        for i, block in enumerate(self.blocks):
            x, cache = block(x, None if kv_cache is None else kv_cache[i])
            if kv_cache is not None:
                kv_cache[i] = cache
        # get logits with linear layer
        logits = self.lm_head(x) # (B,T,V)
        
        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1, use_cache=True):
        if use_cache:
            # initialize key-value cache
            kv_cache = [(None, None) for _ in range(self.layer_num)]
            # idx is (B, T) array of indices in the current context
            # crop idx to the last seq_len tokens
            idx_context = idx[:, -self.seq_len:]
            for _ in range(max_new_tokens):
                # get the predictions
                logits, loss = self(idx_context, kv_cache=kv_cache)
                # focus only on the last time step
                logits = logits[:, -1, :] # becomes (B, C)
                # apply temperature
                logits = logits / temperature if temperature > 0 else logits
                # apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1) # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) if temperature > 0 else torch.argmax(probs, dim=-1, keepdim=True) # (B, 1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
                # since we have kv cache, only need to pass new token
                idx_context = idx_next
            return idx
        else:
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                #crop idx to the last seq_len tokens
                idx_context = idx[:, -self.seq_len:]
                # get the predictions
                logits, loss = self(idx_context)
                # focus only on the last time step
                logits = logits[:, -1, :] # becomes (B, C)
                # apply temperature
                logits = logits / temperature if temperature > 0 else logits
                # apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1) # (B, C)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1) if temperature > 0 else torch.argmax(probs, dim=-1, keepdim=True) # (B, 1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            return idx

# %%

config = TransformerConfig(
    vocab_size=vocab_size,
    seq_len=seq_len,
    embed_size=256,
    head_num=4,
    layer_num=6
)
m = TransformerLM(config)
if use_model_compile:
    m = torch.compile(m, options=torch_compile_options, fullgraph=False)
m.to(device)
xb, yb = get_batch('train', 5, 1)
logits, loss = m(xb, yb)
total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
total_params

def get_wandb_config(model, optimizer, scheduler, seq_len, batch_size, total_steps):
    """Automatically extract wandb config from model and training parameters"""
    config = {
        "architecture": "Transformer",
        "dataset": "anime-subs",
        "seq_len": seq_len,
        "batch_size": batch_size,
        "total_steps": total_steps,
        "attention_sink": "gpt_oss_logit_flex_attention",
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
        "initial_lr": optimizer.param_groups[0]['lr'],
    }
    
    # Add model configuration
    config.update({
        "vocab_size": model.token_embedding_table.num_embeddings,
        "embed_size": model.blocks[0].sa_heads.head_size * model.head_num,
        "head_num": model.head_num,
        "layer_num": model.layer_num,
        "total_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
    })
    
    # Add scheduler-specific config
    if hasattr(scheduler, 'T_max'):
        config['scheduler_T_max'] = scheduler.T_max
    if hasattr(scheduler, 'eta_min'):
        config['scheduler_eta_min'] = scheduler.eta_min
    
    return config

# Initialize wandb with automated config
model = TransformerLM(config)
if use_model_compile:
    model = torch.compile(model, options=torch_compile_options, fullgraph=False)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, fused=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

# warmup
for _ in range(10):
    model(*get_batch('train', seq_len=seq_len, batch_size=batch_size))

if use_wandb:
    wandb_config = get_wandb_config(model, optimizer, scheduler, seq_len, batch_size, total_steps)
    wandb.init(
        project="transformers-playground",
        config=wandb_config,
        name="gpt_oss_sink_flex_attention-lm-animesubs-256seq-256embed-4head-6layer.AMD"
    )

losses, val_losses = train(
    model,
    optimizer,
    scheduler,
    seq_len,
    batch_size,
    total_steps,
    plot_losses=plot_losses,
    plot_path=plot_path,
)

if use_wandb:
    wandb.finish()

# %%
# Save
try:
    os.mkdir(os.path.join(os.getcwd(), "models"))
except:
    pass
torch.save(model.state_dict(), 'models/TransformerLM.pt')

# %%
# Load model
model = TransformerLM(config)
if use_model_compile:
    model = torch.compile(model, options=torch_compile_options, fullgraph=False)
model.load_state_dict(torch.load('models/TransformerLM.pt'))
model.to(device)

# %%
# calculate perplexity
ppl, loss = perplexity(model, seq_len, seq_len)
print("perplexity:", ppl, "loss:", loss)

# %%
model.eval()
idx = encode("You will never")
print(torch.tensor([idx]))
print(decode(model.generate(idx=torch.tensor([idx], dtype=torch.long).to(device), max_new_tokens=1000, temperature=0.5, use_cache=True)[0].tolist()))
