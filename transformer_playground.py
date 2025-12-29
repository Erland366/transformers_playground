# %%
from __future__ import annotations
import contextlib
import functools
import inspect
import os
import re
import math
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import triton
import triton.language as tl
from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime

load_dotenv()

# %%
# Softpick function for gated attention experiment
# Gating normalized over sequence dimension (dim=seq, i.e., dim=1 after reshaping)
def softpick(x, dim: int = -1, eps: float = 1e-8):
    """
    Softpick function: relu(exp(x)-1) / sum(abs(exp(x)-1))

    Unlike sigmoid, softpick normalizes over a dimension, creating competition
    across that dimension. When dim=1 (seq dimension), gates compete across
    sequence positions - different tokens compete for influence.

    Key properties:
    - Zeros out values where exp(x) <= 1 (x <= 0) via relu
    - Normalizes over the specified dimension
    - Creates competition/selection across that dimension
    """
    x_m = torch.max(x, dim=dim, keepdim=True).values
    x_m_e_m = torch.exp(-x_m)
    x_e_1 = torch.exp(x - x_m) - x_m_e_m
    r_x_e_1 = F.relu(x_e_1)
    a_x_e_1 = torch.where(x.isfinite(), torch.abs(x_e_1), 0)
    return r_x_e_1 / (torch.sum(a_x_e_1, dim=dim, keepdim=True) + eps)

use_wandb = True  # set to False to disable wandb logging
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
use_torch_compile = True  # enable for full perf
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
seq_len = 128
batch_size = 64 # these are small models so we can use large batch sizes to fully utilize the GPU
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

if use_torch_compile:
    @torch.compile(fullgraph=False, options=torch_compile_options)
    def compile_optimizer_lr(opt, scheduler):
        opt.step()
        scheduler.step()
else:
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
            'embed_size': model.token_embedding_table.embedding_dim,
            'head_num': model.head_num,
            'layer_num': model.layer_num,
            'attn_type': model.attn_type,
            'kda_expand_v': getattr(model.blocks[0].sa_heads, "expand_v", None),
            'kda_num_v_heads': getattr(model.blocks[0].sa_heads, "num_v_heads", None),
            'kda_allow_neg_eigval': getattr(model.blocks[0].sa_heads, "allow_neg_eigval", None),
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
    def __init__(
        self,
        vocab_size,
        seq_len,
        embed_size,
        head_num,
        layer_num,
        attn_type="kda",
        kda_expand_v=1.0,
        kda_num_v_heads=None,
        kda_allow_neg_eigval=False,
        kda_norm_eps=1e-5,
        kda_mode="chunk",
        kda_use_short_conv=True,
        kda_conv_size=4,
        kda_conv_bias=False,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_size = embed_size
        self.head_num = head_num
        self.layer_num = layer_num
        if attn_type not in ("kda", "softpick"):
            raise ValueError("attn_type must be 'kda' or 'softpick'.")
        self.attn_type = attn_type
        self.kda_expand_v = kda_expand_v
        self.kda_num_v_heads = kda_num_v_heads
        self.kda_allow_neg_eigval = kda_allow_neg_eigval
        self.kda_norm_eps = kda_norm_eps
        self.kda_mode = kda_mode
        self.kda_use_short_conv = kda_use_short_conv
        self.kda_conv_size = kda_conv_size
        self.kda_conv_bias = kda_conv_bias

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


# === Inline flash-linear-attention utilities (minimal, local-only) ===
FLA_CACHE_RESULTS = os.getenv('FLA_CACHE_RESULTS', '1') == '1'
FLA_DISABLE_TENSOR_CACHE = os.getenv('FLA_DISABLE_TENSOR_CACHE', '0') == '1'
SUPPORTS_AUTOTUNE_CACHE = 'cache_results' in inspect.signature(triton.autotune).parameters

autotune_cache_kwargs = {'cache_results': FLA_CACHE_RESULTS} if SUPPORTS_AUTOTUNE_CACHE else {}

IS_AMD = torch.version.hip is not None
IS_NVIDIA = torch.version.cuda is not None and not IS_AMD
if torch.cuda.is_available() and IS_NVIDIA:
    _CUDA_NAME = torch.cuda.get_device_name(0)
    _CUDA_CAP = torch.cuda.get_device_capability(0)
else:
    _CUDA_NAME = ""
    _CUDA_CAP = (0, 0)

IS_NVIDIA_HOPPER = IS_NVIDIA and ("NVIDIA H" in _CUDA_NAME or _CUDA_CAP[0] >= 9)
IS_NVIDIA_BLACKWELL = IS_NVIDIA and _CUDA_CAP[0] == 10
USE_CUDA_GRAPH = IS_NVIDIA and os.environ.get('FLA_USE_CUDA_GRAPH', '0') == '1'
IS_TF32_SUPPORTED = IS_NVIDIA and _CUDA_CAP[0] >= 8
IS_GATHER_SUPPORTED = hasattr(tl, 'gather')

FLA_DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=FLA_DEVICE_TYPE)
autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=FLA_DEVICE_TYPE)


def custom_device_ctx(index: int):
    if torch.cuda.is_available():
        return torch.cuda.device(index)
    return contextlib.nullcontext()


def input_guard(fn):
    """Ensure input tensors are contiguous and run on the active device."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contig_args = [a.contiguous() if isinstance(a, torch.Tensor) else a for a in args]
        contig_kwargs = {k: (v.contiguous() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}

        tensor = next((a for a in contig_args if isinstance(a, torch.Tensor)), None)
        if tensor is None:
            for value in contig_kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None and tensor.is_cuda:
            with custom_device_ctx(tensor.device.index):
                return fn(*contig_args, **contig_kwargs)
        return fn(*contig_args, **contig_kwargs)

    return wrapper


def tensor_cache(fn):
    last_args = None
    last_kwargs = None
    last_result = None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal last_args, last_kwargs, last_result

        if FLA_DISABLE_TENSOR_CACHE:
            return fn(*args, **kwargs)

        if last_args is not None and last_kwargs is not None:
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args, strict=False)) and                         all(k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()):
                    return last_result

        result = fn(*args, **kwargs)
        last_args, last_kwargs, last_result = args, kwargs, result
        return result

    return wrapper


def check_shared_mem(arch: str = 'none', tensor_idx: int = 0) -> bool:
    try:
        props = triton.runtime.driver.active.utils.get_device_properties(tensor_idx)
        max_shared = props.get('max_shared_mem', 0)
        if arch.lower() == 'hopper':
            return max_shared >= 232448
        if arch.lower() == 'ampere':
            return max_shared >= 166912
        if arch.lower() == 'ada':
            return max_shared >= 101376
        return max_shared >= 102400
    except Exception:
        return False


@functools.cache
def get_multiprocessor_count(tensor_idx: int = 0) -> int:
    try:
        return triton.runtime.driver.active.utils.get_device_properties(tensor_idx)['multiprocessor_count']
    except Exception:
        return 1

# === End inline utilities ===


# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Approximate value of 1/ln(2), used for log/exp base conversion
# Best FP32 approximation: 1.4426950216 (hex 0x3FB8AA3B)
RCP_LN2 = 1.4426950216

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import os

import triton
import triton.language as tl
import triton.language.extra.libdevice as tldevice


if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    exp = tldevice.fast_expf
    exp2 = tldevice.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    exp = tl.exp
    exp2 = tl.math.exp2
    log = tl.log
    log2 = tl.log2


if not IS_GATHER_SUPPORTED:
    @triton.jit
    def gather(src, index, axis, _builder=None):
        """
        Gather operation that works when tl.gather is not supported.
        This is a fallback implementation that returns None.
        Just to make triton compiler happy.
        """
        return None
else:
    gather = tl.gather


if hasattr(triton.language, '_experimental_make_tensor_descriptor'):
    # For Triton 3.3.x
    make_tensor_descriptor = triton.language._experimental_make_tensor_descriptor
elif hasattr(triton.language, 'make_tensor_descriptor'):
    # For Triton 3.4.x and later
    make_tensor_descriptor = triton.language.make_tensor_descriptor
else:
    """
    Fallback implementation when TMA is not supported.
    Returns None to indicate TMA descriptors are unavailable.
    Just make triton compiler happy.
    """
    @triton.jit
    def make_tensor_descriptor(
        base,
        shape,
        strides,
        block_shape,
        _builder=None,
    ):
        return None

# REVISED FROM
# https://github.com/shawntan/stickbreaking-attention/blob/main/stickbreaking_attention/sb_varlen/softplus.py

import triton
from triton import language as tl



def _generate_softplus(num_pack):
    template = """
        .reg .pred p;
        setp.gt.f32  p, ${in_reg}, 20.;
        @p  mov.f32  ${out_reg}, ${in_reg};
        @!p mul.f32            ${out_reg}, ${in_reg}, 1.4426950408889634;
        @!p ex2.approx.ftz.f32 ${out_reg}, ${out_reg};
        @!p add.f32            ${out_reg}, ${out_reg}, 1.0;
        @!p lg2.approx.ftz.f32 ${out_reg}, ${out_reg};
        @!p mul.f32            ${out_reg}, ${out_reg}, 0.6931471805599453;
    """
    out_str = ""

    for i in range(num_pack):
        inner_str = template.format(out_reg=i, in_reg=i + num_pack)
        out_str += "{" + inner_str + "}\n"
    # flatten out because torch.compile doesn't like newlines
    out_str = " ".join(out_str.split("\n"))
    return out_str


def _generate_softplus2(num_pack):
    template = """
        .reg .pred p;
        setp.gt.f32  p, ${in_reg}, 15.;
        @p  mov.f32  ${out_reg}, ${in_reg};
        @!p ex2.approx.ftz.f32 ${out_reg}, ${in_reg};
        @!p add.f32            ${out_reg}, ${out_reg}, 1.0;
        @!p lg2.approx.ftz.f32 ${out_reg}, ${out_reg};
    """
    out_str = ""

    for i in range(num_pack):
        inner_str = template.format(out_reg=i, in_reg=i + num_pack)
        out_str += "{" + inner_str + "}\n"
    # flatten out because torch.compile doesn't like newlines
    out_str = " ".join(out_str.split("\n"))
    return out_str


def _generate_constraints(num_pack):
    return ",".join("=r" for i in range(num_pack)) + "," + ",".join("r" for i in range(num_pack))


_NUM_REG = 1
s_softplus: tl.constexpr = tl.constexpr(_generate_softplus(_NUM_REG))
s_softplus2: tl.constexpr = tl.constexpr(_generate_softplus2(_NUM_REG))
s_constraints: tl.constexpr = tl.constexpr(_generate_constraints(_NUM_REG))
NUM_REG: tl.constexpr = tl.constexpr(_NUM_REG)


@triton.jit
def softplus_nv(x):
    # equivalent to:
    # return tl.where(x < 20.0, tl.math.log(1 + tl.math.exp(x)), x)
    return tl.inline_asm_elementwise(
        asm=s_softplus,
        constraints=s_constraints,
        pack=NUM_REG,
        args=[
            x,
        ],
        dtype=tl.float32,
        is_pure=True,
    )

@triton.jit
def softplus_triton(x):
    return tl.where(x < 20.0, tl.math.log(1 + tl.math.exp(x)), x)

@triton.jit
def softplus2_nv(x):
    # equivalent to:
    # return tl.where(x < 15.0, tl.math.log2(1 + tl.math.exp2(x)), x)
    return tl.inline_asm_elementwise(
        asm=s_softplus2,
        constraints=s_constraints,
        pack=NUM_REG,
        args=[
            x,
        ],
        dtype=tl.float32,
        is_pure=True,
    )

@triton.jit
def softplus2_triton(x):
    return tl.where(x < 15.0, tl.math.log2(1 + tl.math.exp2(x)), x)

if IS_NVIDIA:
    softplus = softplus_nv
    softplus2 = softplus2_nv
else:
    softplus = softplus_triton
    softplus2 = softplus2_triton

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import torch
import torch.nn.functional as F
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [4, 8, 16, 32]
    ],
    key=['B'],
    **autotune_cache_kwargs,
)
@triton.jit
def prepare_position_ids_kernel(
    y,
    cu_seqlens,
    B: tl.constexpr,
):
    i_n = tl.program_id(0)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    T = eos - bos

    o = tl.arange(0, B)
    for i in range(0, tl.cdiv(T, B) * B, B):
        o_i = o + i
        tl.store(y + bos + o_i, o_i, o_i < T)


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_lens_from_mask(mask: torch.BoolTensor) -> torch.LongTensor:
    return mask.sum(dim=-1, dtype=torch.int32)


@tensor_cache
def prepare_cu_seqlens_from_lens(
    lens: torch.LongTensor,
    dtype: torch.dtype | None = torch.int32,
) -> torch.LongTensor:
    return F.pad(lens.cumsum(dim=0, dtype=dtype), (1, 0))


@tensor_cache
def prepare_cu_seqlens_from_mask(
    mask: torch.BoolTensor,
    dtype: torch.dtype | None = torch.int32,
) -> torch.LongTensor:
    return prepare_cu_seqlens_from_lens(prepare_lens_from_mask(mask), dtype)


@tensor_cache
def prepare_split_cu_seqlens(
    batch_size: int,
    seq_len: int,
    split_size: int,
    cu_seqlens: torch.LongTensor | None = None,
    dtype: torch.dtype | None = torch.int32,
    device: torch.device | None = torch.device('cpu'),
) -> torch.LongTensor:
    if cu_seqlens is None:
        total_tokens = batch_size * seq_len
        cu_seqlens = list(range(0, total_tokens, seq_len)) + [total_tokens]
    else:
        cu_seqlens = cu_seqlens.tolist()
    return torch.tensor(
        [
            i
            for bos, eos in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=False)
            for i in range(bos, eos, split_size)
        ] + [cu_seqlens[-1]],
        dtype=dtype,
        device=device,
    )


@tensor_cache
def prepare_position_ids(cu_seqlens: torch.LongTensor, cu_seqlens_cpu: torch.LongTensor | None = None) -> torch.LongTensor:
    if cu_seqlens_cpu is not None:
        return torch.cat([
            torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
            for n in prepare_lens(cu_seqlens_cpu).unbind()
        ])
    return torch.cat([
        torch.arange(n, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        for n in prepare_lens(cu_seqlens).unbind()
    ])


@tensor_cache
def prepare_sequence_ids(cu_seqlens: torch.LongTensor, cu_seqlens_cpu: torch.LongTensor | None = None) -> torch.LongTensor:
    return prepare_position_ids(cu_seqlens, cu_seqlens_cpu).eq(0).cumsum(0) - 1


@tensor_cache
def prepare_token_indices(cu_seqlens: torch.LongTensor, cu_seqlens_cpu: torch.LongTensor | None = None) -> torch.LongTensor:
    position_ids = prepare_position_ids(cu_seqlens, cu_seqlens_cpu)
    return torch.stack([prepare_sequence_ids(cu_seqlens, cu_seqlens_cpu), position_ids], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None,
) -> torch.LongTensor:
    if cu_seqlens_cpu is not None:
        indices = torch.cat([torch.arange(n, device=cu_seqlens.device)
                            for n in triton.cdiv(prepare_lens(cu_seqlens_cpu), chunk_size).tolist()])
        return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
    indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    return F.pad(triton.cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)


@tensor_cache
def get_max_num_splits(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.LongTensor | None = None
) -> int:
    if cu_seqlens_cpu is not None:
        return triton.cdiv(int(max(prepare_lens(cu_seqlens_cpu))), chunk_size)
    return triton.cdiv(int(max(prepare_lens(cu_seqlens))), chunk_size)

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import torch
import triton
import triton.language as tl


BS_LIST = [32, 64] if check_shared_mem() else [16, 32]


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['B', 'H', 'BT', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    else:
        p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_o = tl.make_block_ptr(o + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    # [BT]
    b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0)
    if REVERSE:
        b_z = tl.sum(b_s, axis=0)
        b_o = -b_o + b_z[None] + b_s
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BS': BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=['B', 'H', 'S', 'BT', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_s, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        p_s = tl.make_block_ptr(s + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    else:
        p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
    # [BT, BS]
    b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
    if REVERSE:
        b_o = tl.cumsum(b_s, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_s, axis=0)
    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))



@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [32, 64, 128, 256]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['B', 'H', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos

    b_z = tl.zeros([], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + bos*H + i_h*T, (T,), (1,), (i_t * BT,), (BT,), (0,))
        else:
            p_s = tl.make_block_ptr(s + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            p_o = tl.make_block_ptr(o + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_s = tl.load(p_s, boundary_check=(0,)).to(tl.float32)
        b_o = tl.cumsum(b_s, axis=0)
        b_ss = tl.sum(b_s, 0)
        if REVERSE:
            b_o = -b_o + b_ss + b_s
        b_o += b_z
        if i_c >= 0:
            b_z += b_ss
        if HAS_SCALE:
            b_o *= scale
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in [16, 32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['B', 'H', 'S', 'IS_VARLEN', 'REVERSE'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_global_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_s, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
    T = eos - bos

    b_z = tl.zeros([BS], dtype=tl.float32)
    NT = tl.cdiv(T, BT)
    for i_c in range(NT):
        i_t = NT - 1 - i_c if REVERSE else i_c
        if HEAD_FIRST:
            p_s = tl.make_block_ptr(s + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_o = tl.make_block_ptr(o + (bos * H + i_h*T)*S, (T, S), (S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        else:
            p_s = tl.make_block_ptr(s + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
            p_o = tl.make_block_ptr(o + (bos * H + i_h) * S, (T, S), (H*S, 1), (i_t * BT, i_s * BS), (BT, BS), (1, 0))
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)
        if REVERSE:
            b_c = b_z[None, :] + tl.cumsum(b_s, axis=0, reverse=True)
        else:
            b_c = b_z[None, :] + tl.cumsum(b_s, axis=0)
        if HAS_SCALE:
            b_c *= scale
        tl.store(p_o, b_c.to(p_o.dtype.element_ty), boundary_check=(0, 1))
        b_z += tl.sum(b_s, 0)


def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (NT, B * H)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return g


def chunk_local_cumsum_vector(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    assert chunk_size == 2**(chunk_size.bit_length()-1), "chunk_size must be a power of 2"

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    def grid(meta): return (triton.cdiv(meta['S'], meta['BS']), NT, B * H)
    # keep cummulative normalizer in fp32
    # this kernel is equivalent to
    # g = g.view(B, H, NT, BT, -1).cumsum(-2).view(B, H, T, -1)
    chunk_local_cumsum_vector_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        B=B,
        H=H,
        S=S,
        BT=BT,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return g


@input_guard
def chunk_global_cumsum_scalar(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T = s.shape
    else:
        B, T, H = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    grid = (N * H,)
    chunk_global_cumsum_scalar_kernel[grid](
        s=s,
        o=z,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        B=B,
        H=H,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return z


@input_guard
def chunk_global_cumsum_vector(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if head_first:
        B, H, T, S = s.shape
    else:
        B, T, H, S = s.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BS = min(32, triton.next_power_of_2(S))

    z = torch.empty_like(s, dtype=output_dtype or s.dtype)
    grid = (triton.cdiv(S, BS), N * H)
    chunk_global_cumsum_vector_kernel[grid](
        s=s,
        o=z,
        scale=scale,
        cu_seqlens=cu_seqlens,
        T=T,
        B=B,
        H=H,
        S=S,
        BS=BS,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )
    return z


@input_guard
def chunk_global_cumsum(
    s: torch.Tensor,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    scale: float = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert s.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(s.shape) == 3:
        return chunk_global_cumsum_scalar(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    elif len(s.shape) == 4:
        return chunk_global_cumsum_vector(
            s=s,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
            scale=scale,
            head_first=head_first,
            output_dtype=output_dtype,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {s.shape}, "
            f"which should be [B, T, H]/[B, T, H, D] if `head_first=False` "
            f"or [B, H, T]/[B, H, T, D] otherwise",
        )


@input_guard
def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    chunk_indices: torch.LongTensor | None = None,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    elif len(g.shape) == 4:
        return chunk_local_cumsum_vector(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            head_first=head_first,
            output_dtype=output_dtype,
            chunk_indices=chunk_indices,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise",
        )

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import torch.nn as nn
import triton
import triton.language as tl


BT_LIST = [8, 16, 32, 64, 128]
NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16] if IS_AMD else [1, 2, 4, 8, 16, 32]


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    rstd,
    eps,
    D,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D

    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)
    tl.store(rstd + i_t, b_rstd)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D'],
    **autotune_cache_kwargs,
)
@triton.jit
def l2norm_bwd_kernel1(
    y,
    rstd,
    dy,
    dx,
    eps,
    D,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    y += i_t * D
    dx += i_t * D
    dy += i_t * D

    cols = tl.arange(0, BD)
    mask = cols < D
    b_y = tl.load(y + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd + i_t).to(tl.float32)
    b_dy = tl.load(dy + cols, mask=mask, other=0.0).to(tl.float32)
    b_dx = b_dy * b_rstd - tl.sum(b_dy * b_y) * b_y * b_rstd
    tl.store(dx + cols, b_dx, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for BT in BT_LIST
    ],
    key=['D', 'NB'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def l2norm_fwd_kernel(
    x,
    y,
    rstd,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))

    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]

    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
        for BT in BT_LIST
    ],
    key=['D', 'NB'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def l2norm_bwd_kernel(
    y,
    rstd,
    dy,
    dx,
    eps,
    T,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    p_dy = tl.make_block_ptr(dy, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_dx = tl.make_block_ptr(dx, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))

    b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = tl.load(p_rstd, boundary_check=(0,)).to(tl.float32)
    b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
    b_dx = b_dy * b_rstd[:, None] - tl.sum(b_dy * b_y, 1)[:, None] * b_y * b_rstd[:, None]
    tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
):
    x_shape_og = x.shape
    x = x.view(-1, x.shape[-1])
    # allocate output
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    assert y.stride(-1) == 1
    T, D = x.shape[0], x.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer doesn't support feature dim >= 64KB.")

    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    if D <= 512:
        NB = triton.cdiv(T, 2048)
        def grid(meta): return (triton.cdiv(T, meta['BT']), )
        l2norm_fwd_kernel[grid](
            x=x,
            y=y,
            rstd=rstd,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            NB=NB,
        )
    else:
        l2norm_fwd_kernel1[(T,)](
            x=x,
            y=y,
            rstd=rstd,
            eps=eps,
            D=D,
            BD=BD,
        )
    return y.view(x_shape_og), rstd.view(x_shape_og[:-1])


def l2norm_bwd(
    y: torch.Tensor,
    rstd: torch.Tensor,
    dy: torch.Tensor,
    eps: float = 1e-6,
):
    y_shape_og = y.shape
    y = y.view(-1, dy.shape[-1])
    dy = dy.view(-1, dy.shape[-1])
    assert dy.shape == y.shape
    # allocate output
    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // y.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    if D <= 512:
        NB = triton.cdiv(T, 2048)
        def grid(meta): return (triton.cdiv(T, meta['BT']), )
        l2norm_bwd_kernel[grid](
            y=y,
            rstd=rstd,
            dy=dy,
            dx=dx,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            NB=NB,
        )
    else:
        l2norm_bwd_kernel1[(T,)](
            y=y,
            rstd=rstd,
            dy=dy,
            dx=dx,
            eps=eps,
            D=D,
            BD=BD,
        )

    return dx.view(y_shape_og)


class L2NormFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x,
        eps=1e-6,
        output_dtype=None,
    ):
        y, rstd = l2norm_fwd(x, eps, output_dtype)
        ctx.eps = eps
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(y, rstd)
        return y

    @staticmethod
    @input_guard
    def backward(ctx, dy):
        y, rstd = ctx.saved_tensors
        dx = l2norm_bwd(y, rstd, dy, ctx.eps)
        return dx, None, None


def l2norm(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)


l2_norm = l2norm


class L2Norm(nn.Module):

    def __init__(
        self,
        eps: float = 1e-6,
        output_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.output_dtype = output_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2norm(x, self.eps, self.output_dtype)

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


BT_LIST_AUTOTUNE = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [4, 8, 16, 32]


def naive_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Torch reference implementation for KDA gate computation.

    Computes: g = -A_log.exp().unsqueeze(-1) * softplus(g + dt_bias.view(g.shape[-2:]))

    Args:
        g (torch.Tensor):
            Input tensor of shape `[..., H, K]`.
        A_log (torch.Tensor):
            Parameter tensor with `H` elements.
        dt_bias (torch.Tensor | None):
            Optional bias tensor added to `g` before activation, shape `[H * K]`.

    Returns:
        Output tensor of shape `[..., H, K]` .
    """
    H, _ = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)

    g = (-A_log.view(H, 1).float().exp() * F.softplus(g.float())).to(output_dtype)
    return g


@triton.heuristics({
    "HAS_BIAS": lambda args: args["dt_bias"] is not None,
    "HAS_BETA": lambda args: args["beta"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps, num_stages=num_stages)
        for BT in BT_LIST_AUTOTUNE
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3]
    ],
    key=["H", "D"],
    **autotune_cache_kwargs,
)
@triton.jit
def kda_gate_fwd_kernel(
    g,
    A_log,
    dt_bias,
    beta,
    yg,
    yb,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)

    b_A = tl.load(A_log + i_h).to(tl.float32)

    p_g = tl.make_block_ptr(g + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_yg = tl.make_block_ptr(yg + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    # [BT, BD]
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        p_b = tl.make_block_ptr(dt_bias, (H * D,), (1,), (i_h * D,), (BD,), (0,))
        b_g = b_g + tl.load(p_b, boundary_check=(0,)).to(tl.float32)
    b_yg = -tl.exp(b_A) * softplus(b_g)
    tl.store(p_yg, b_yg.to(p_yg.dtype.element_ty), boundary_check=(0, 1))

    if HAS_BETA:
        p_b = tl.make_block_ptr(beta + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_yb = tl.make_block_ptr(yb + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_yb = tl.sigmoid(tl.load(p_b, boundary_check=(0,)).to(tl.float32))
        tl.store(p_yb, b_yb.to(p_yb.dtype.element_ty), boundary_check=(0,))


@triton.heuristics({
    "HAS_BIAS": lambda args: args["dt_bias"] is not None,
    "HAS_BETA": lambda args: args["beta"] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS_AUTOTUNE
        for num_stages in [2, 3]
    ],
    key=["H", "D"],
    **autotune_cache_kwargs,
)
@triton.jit
def kda_gate_bwd_kernel(
    g,
    A_log,
    dt_bias,
    beta,
    dyg,
    dyb,
    dg,
    dA,
    dbeta,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)

    b_A = tl.load(A_log + i_h).to(tl.float32)
    b_A = -tl.exp(b_A)

    p_g = tl.make_block_ptr(g + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_dg = tl.make_block_ptr(dg + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_dyg = tl.make_block_ptr(dyg + i_h * D, (T, D), (H * D, 1), (i_t * BT, 0), (BT, BD), (1, 0))

    # [BT, BD]
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_dyg = tl.load(p_dyg, boundary_check=(0, 1)).to(tl.float32)

    if HAS_BIAS:
        p_b = tl.make_block_ptr(dt_bias, (H * D,), (1,), (i_h * D,), (BD,), (0,))
        b_g = b_g + tl.load(p_b, boundary_check=(0,)).to(tl.float32)

    # [BT, BD]
    b_yg = b_A * softplus(b_g)
    b_dg = b_A * (b_dyg * tl.sigmoid(b_g))
    b_dA = tl.sum(tl.sum(b_dyg * b_yg, 1), 0)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dA + i_t * H + i_h, b_dA)

    if HAS_BETA:
        p_b = tl.make_block_ptr(beta + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_db = tl.make_block_ptr(dbeta + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
        p_dyb = tl.make_block_ptr(dyb + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))

        b_b = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
        b_db = tl.load(p_dyb, boundary_check=(0,)).to(tl.float32) * b_b * (1.0 - b_b)
        tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


def kda_gate_fwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    H, K = g.shape[-2:]
    T = g.numel() // (H * K)

    yg = torch.empty_like(g, dtype=output_dtype)

    def grid(meta):
        return (triton.cdiv(T, meta["BT"]), H)

    kda_gate_fwd_kernel[grid](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        beta=None,
        yg=yg,
        yb=None,
        T=T,
        H=H,
        D=K,
        BD=triton.next_power_of_2(K),
    )
    return yg


def kda_gate_bwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    dyg: torch.Tensor | None = None,
    dyb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    H, K = g.shape[-2:]
    T = g.numel() // (H * K)
    BT = 32
    NT = triton.cdiv(T, BT)

    dg = torch.empty_like(g, dtype=torch.float32)
    dA = A_log.new_empty(NT, H, dtype=torch.float32)

    grid = (triton.cdiv(T, BT), H)
    kda_gate_bwd_kernel[grid](
        g=g,
        A_log=A_log,
        dt_bias=dt_bias,
        beta=None,
        dyg=dyg,
        dyb=dyb,
        dg=dg,
        dA=dA,
        dbeta=None,
        T=T,
        H=H,
        D=K,
        BT=BT,
        BD=triton.next_power_of_2(K),
    )

    dg = dg.view_as(g)
    dA = dA.sum(0).view_as(A_log)
    dbias = dg.view(-1, H * K).sum(0) if dt_bias is not None else None

    return dg, dA, dbias


class KDAGateFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        g: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        yg = kda_gate_fwd(
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            output_dtype=output_dtype
        )
        ctx.save_for_backward(g, A_log, dt_bias)
        return yg

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, dyg: torch.Tensor):
        g, A_log, dt_bias = ctx.saved_tensors
        dg, dA, dbias = kda_gate_bwd(
            g=g,
            A_log=A_log,
            dt_bias=dt_bias,
            dyg=dyg,
        )
        if dt_bias is not None:
            dbias = dbias.to(dt_bias)
        return dg.to(g), dA.to(A_log), dbias, None


def fused_kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Fused KDA gate computation with autograd support.

    Computes: g = -A_log.exp().unsqueeze(-1) * softplus(g + dt_bias.view(g.shape[-2:]))

    Args:
        g (torch.Tensor):
            Input tensor of shape `[..., H, K]`.
        A_log (torch.Tensor):
            Parameter tensor with `H` elements.
        dt_bias (torch.Tensor | None):
            Optional bias tensor added to `g` before activation, shape `[H * K]`.

    Returns:
        Output tensor of shape `[..., H, K]`.
    """
    return KDAGateFunction.apply(g, A_log, dt_bias, output_dtype)

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl


NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8, 16]


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'SAVE_NEW_VALUE': lambda args: args['v_new'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=['H', 'K', 'V', 'BT', 'USE_EXP2'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * H + i_h).to(tl.int64) * K*V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K*V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K*V

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            p_h0_3 = tl.make_block_ptr(h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            p_h0_4 = tl.make_block_ptr(h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(h + i_t * H*K*V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(h + i_t * H*K*V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_h3 = tl.make_block_ptr(h + i_t * H*K*V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_h4 = tl.make_block_ptr(h + i_t * H*K*V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(w, (T, K), (H*K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(v_new, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            if USE_EXP2:
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k1, mask=(o_k1 < K), other=0.)
            if USE_EXP2:
                b_h1 *= exp2(b_gk_last1)[:, None]
            else:
                b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k2, mask=(o_k2 < K), other=0.)
                if USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[:, None]
                else:
                    b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k3, mask=(o_k3 < K), other=0.)
                if USE_EXP2:
                    b_h3 *= exp2(b_gk_last3)[:, None]
                else:
                    b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + (bos + last_idx) * H*K + i_h * K + o_k4, mask=(o_k4 < K), other=0.)
                if USE_EXP2:
                    b_h4 *= exp2(b_gk_last4)[:, None]
                else:
                    b_h4 *= exp(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (64, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (128, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (192, i_t * BT), (64, BT), (0, 1))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE_GRADIENT': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in ([4, 3, 2] if check_shared_mem('ampere') else [1])
        for BV in [64, 32]
    ],
    key=['H', 'K', 'V', 'BT', 'BV', 'USE_G', 'USE_EXP2'],
    use_cuda_graph=USE_CUDA_GRAPH,
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64(
    q,
    k,
    w,
    g,
    gk,
    dht,
    dh0,
    do,
    dh,
    dv,
    dv2,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_dh1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_dh2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_dh3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_dh4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    q += (bos * H + i_h).to(tl.int64) * K
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    do += (bos * H + i_h).to(tl.int64) * V
    dv += (bos * H + i_h).to(tl.int64) * V
    dv2 += (bos * H + i_h).to(tl.int64) * V
    dh += (boh * H + i_h).to(tl.int64) * K*V
    if USE_GK:
        gk += (bos * H + i_h).to(tl.int64) * K

    if USE_INITIAL_STATE:
        dh0 += i_nh * K*V
    if USE_FINAL_STATE_GRADIENT:
        dht += i_nh * K*V

    if USE_FINAL_STATE_GRADIENT:
        p_dht1 = tl.make_block_ptr(dht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_dh1 += tl.load(p_dht1, boundary_check=(0, 1))
        if K > 64:
            p_dht2 = tl.make_block_ptr(dht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            b_dh2 += tl.load(p_dht2, boundary_check=(0, 1))
        if K > 128:
            p_dht3 = tl.make_block_ptr(dht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            b_dh3 += tl.load(p_dht3, boundary_check=(0, 1))
        if K > 192:
            p_dht4 = tl.make_block_ptr(dht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            b_dh4 += tl.load(p_dht4, boundary_check=(0, 1))

    for i_t in range(NT - 1, -1, -1):
        p_dh1 = tl.make_block_ptr(dh + i_t*H*K*V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh1, b_dh1.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh2 = tl.make_block_ptr(dh + i_t*H*K*V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh2.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh3 = tl.make_block_ptr(dh + i_t*H*K*V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh3.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh4 = tl.make_block_ptr(dh + i_t*H*K*V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh4, b_dh4.to(p_dh4.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            bg_last = tl.load(g + (bos + last_idx) * H + i_h)
            p_g = tl.make_block_ptr(g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
            b_g = tl.load(p_g, boundary_check=(0,))
            if USE_EXP2:
                bg_last_exp = exp2(bg_last)
                b_g_exp = exp2(b_g)
            else:
                bg_last_exp = exp(bg_last)
                b_g_exp = exp(b_g)

        p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv2 = tl.make_block_ptr(dv2, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_do = tl.load(p_do, boundary_check=(0, 1))

        # Update dv
        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(gk + last_idx * H*K + o_k1, mask=(o_k1 < K), other=0.)
        b_dv = tl.dot(b_k, b_dh1.to(b_k.dtype))

        if K > 64:
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 64), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(gk + last_idx * H*K + o_k2, mask=(o_k2 < K), other=0.)
            b_dv += tl.dot(b_k, b_dh2.to(b_k.dtype))

        if K > 128:
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 128), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(gk + last_idx * H*K + o_k3, mask=(o_k3 < K), other=0.)
            b_dv += tl.dot(b_k, b_dh3.to(b_k.dtype))

        if K > 192:
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, 192), (BT, 64), (1, 0))
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if USE_GK:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(gk + last_idx * H*K + o_k4, mask=(o_k4 < K), other=0.)
            b_dv += tl.dot(b_k, b_dh4.to(b_k.dtype))

        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            if USE_EXP2:
                b_dv *= tl.where(m_t, exp2(bg_last - b_g), 0)[:, None]
            else:
                b_dv *= tl.where(m_t, exp(bg_last - b_g), 0)[:, None]
        b_dv += tl.load(p_dv, boundary_check=(0, 1))

        tl.store(p_dv2, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
        # Update dh
        p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (0, i_t * BT), (64, BT), (0, 1))
        p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (0, i_t * BT), (64, BT), (0, 1))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        if USE_G:
            b_dh1 *= bg_last_exp
            b_q = b_q * b_g_exp[None, :]
        if USE_GK:
            if USE_EXP2:
                b_dh1 *= exp2(b_gk_last1[:, None])
            else:
                b_dh1 *= exp(b_gk_last1[:, None])
        b_dh1 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 64:
            p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (64, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (64, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh2 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if USE_EXP2:
                    b_dh2 *= exp2(b_gk_last2[:, None])
                else:
                    b_dh2 *= exp(b_gk_last2[:, None])
            b_dh2 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 128:
            p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (128, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (128, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh3 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if USE_EXP2:
                    b_dh3 *= exp2(b_gk_last3[:, None])
                else:
                    b_dh3 *= exp(b_gk_last3[:, None])
            b_dh3 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))
        if K > 192:
            p_q = tl.make_block_ptr(q, (K, T), (1, H*K), (192, i_t * BT), (64, BT), (0, 1))
            p_w = tl.make_block_ptr(w, (K, T), (1, H*K), (192, i_t * BT), (64, BT), (0, 1))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if USE_G:
                b_dh4 *= bg_last_exp
                b_q = b_q * b_g_exp[None, :]
            if USE_GK:
                if USE_EXP2:
                    b_dh4 *= exp2(b_gk_last4[:, None])
                else:
                    b_dh4 *= exp(b_gk_last4[:, None])
            b_dh4 += tl.dot(b_q.to(b_q.dtype), b_do.to(b_q.dtype)) * scale - tl.dot(b_w, b_dv.to(b_w.dtype))

    if USE_INITIAL_STATE:
        p_dh0 = tl.make_block_ptr(dh0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_dh0, b_dh1.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_dh1 = tl.make_block_ptr(dh0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh1, b_dh2.to(p_dh1.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            p_dh2 = tl.make_block_ptr(dh0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh2, b_dh3.to(p_dh2.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            p_dh3 = tl.make_block_ptr(dh0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0))
            tl.store(p_dh3, b_dh4.to(p_dh3.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V = *k.shape, u.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
    )
    return h, v_new, final_state


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *q.shape, do.shape[-1]
    # N: the actual number of sequences in the batch with either equal or variable lengths
    BT = 64
    assert K <= 256, "current kernel does not support head dimension being larger than 256."

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    dh = q.new_empty(B, NT, H, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    def grid(meta): return (triton.cdiv(V, meta['BV']), N*H)
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
    )
    return dh, dh0, dv2

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl


BK_LIST = [32, 64] if check_shared_mem() else [16, 32]
BV_LIST = [64, 128] if check_shared_mem('ampere') else [16, 32]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q,
    k,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_i, i_j = i_c // NC, i_c % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_q = tl.make_block_ptr(q + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + (bos*H+i_h)*K, (K, T), (1, H*K), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = g + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k

        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        # [BC, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))
        b_qg = b_q * exp(b_g - b_gn[None, :]) * scale
        # [BK, BC]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kg = b_k * exp(b_gn[:, None] - b_gk)

        # [BC, BC] using tf32 to improve precision here.
        b_A += tl.dot(b_qg, b_kg)

    p_A = tl.make_block_ptr(A + (bos*H + i_h)*BT, (T, BT), (H*BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3]
    ],
    key=["BK", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q,
    k,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_j = i_i
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BT + i_j * BC
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += (bos * H + i_h) * BT

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_k = k + (i_t * BT + i_j * BC) * H*K + o_k
    p_gk = g + (i_t * BT + i_j * BC) * H*K + o_k

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_q * b_k[None, :] * exp(b_g - b_gk[None, :]), 1) * scale

        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += H*K
        p_gk += H*K

    tl.debug_barrier()
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    tl.store(A + o_A[:, None] + o_i, b_A, mask=m_A[:, None] & (o_i[:, None] < o_i))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8]
    ],
    key=['BC', 'BK'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_A_kernel_intra_sub_intra_split(
    q,
    k,
    g,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_tc, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_t, i_i = i_tc // NC, i_tc % NC
    i_j = i_i
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    o_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BC
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    A += ((i_k * all + bos) * H + i_h) * BC

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_k = k + (i_t * BT + i_j * BC) * H*K + o_k
    p_gk = g + (i_t * BT + i_j * BC) * H*K + o_k
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_q * b_k[None, :] * exp(b_g - b_gk[None, :]), 1) * scale
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += H*K
        p_gk += H*K

    tl.debug_barrier()
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    tl.store(A + o_A[:, None] + o_i, b_A, mask=m_A[:, None] & (o_i[:, None] < o_i))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['BC'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge(
    A,
    A2,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        all = T
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
        all = B * T

    if i_t * BT + i_c * BC >= T:
        return

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(0, NK):
        p_A = tl.make_block_ptr(A + (i_k*all+bos)*H*BC+i_h*BC, (T, BC), (H*BC, 1), (i_t*BT + i_c*BC, 0), (BC, BC), (1, 0))
        b_A += tl.load(p_A, boundary_check=(0, 1))
    p_A2 = tl.make_block_ptr(A2 + (bos*H+i_h)*BT, (T, BT), (H*BT, 1), (i_t * BT + i_c * BC, i_c * BC), (BC, BC), (1, 0))
    tl.store(p_A2, b_A.to(A2.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + (i_tg * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1))
        # [BT, BK]
        b_g = tl.load(p_g, boundary_check=(0, 1))
        # [BT, BK]
        if USE_EXP2:
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        else:
            b_qg = (b_q * exp(b_g)).to(b_q.dtype)
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        # works but dkw, owing to divine benevolence
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
    b_o *= scale
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    # [BT, BV]
    b_v = tl.load(p_v, boundary_check=(0, 1))
    # [BT, BT]
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_A = tl.where(m_s, b_A, 0.).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BK', 'NC', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_bwd_kernel_intra(
    q,
    k,
    g,
    dA,
    dq,
    dk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos
    if i_t * BT + i_i * BC >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    p_g = tl.make_block_ptr(g + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    # [BC, BK]
    b_g = tl.load(p_g, boundary_check=(0, 1))

    b_dq = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + (bos + i_t * BT + i_i * BC) * H*K + i_h*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g+(bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k * BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA+(bos*H+i_h)*BT, (T, BT), (H*BT, 1), (i_t*BT+i_i*BC, i_j * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * exp(b_gn[None, :] - b_gk)
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))

            b_dq += tl.dot(b_dA, b_kg)
        b_dq *= exp(b_g - b_gn[None, :])

    o_i = tl.arange(0, BC)
    m_dA = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T
    o_dA = bos*H*BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * H*BT + i_h * BT + i_i * BC
    p_kj = k + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    p_gkj = g + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    p_dq = tl.make_block_ptr(dq + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j, mask=m_dA, other=0)
        # [BK,]
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] >= j
        # [BC, BK]
        # (SY 09/17) important to not use bf16 here to have a good precision.
        b_dq += tl.where(m_i, b_dA[:, None] * b_kj[None, :] * exp(b_g - b_gkj[None, :]), 0.)
        p_kj += H*K
        p_gkj += H*K
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))

    tl.debug_barrier()
    # [BC, BK]
    b_dk = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (bos + min(i_t * BT + i_i * BC + BC, T) - 1) * H*K + i_h * K + o_k

        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k*BK), (BC, BK), (1, 0))
            p_gq = tl.make_block_ptr(g + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k*BK), (BC, BK), (1, 0))
            p_dA = tl.make_block_ptr(dA + (bos*H+i_h)*BT, (BT, T), (1, H*BT), (i_i*BC, i_t*BT + i_j*BC), (BC, BC), (0, 1))

            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            # [BC, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_gq = tl.load(p_gq, boundary_check=(0, 1))
            b_qg = b_q * tl.where(m_j[:, None], exp(b_gq - b_gn[None, :]), 0)
            # [BC, BC]
            b_dA = tl.load(p_dA, boundary_check=(0, 1))
            # [BC, BK]
            # (SY 09/17) important to not use bf16 here to have a good precision.
            b_dk += tl.dot(b_dA, b_qg)
        b_dk *= exp(b_gn[None, :] - b_g)
    o_dA = bos*H*BT + (i_t * BT + i_i * BC) * H*BT + i_h * BT + i_i * BC + tl.arange(0, BC)
    p_qj = q + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    p_gqj = g + (bos + i_t * BT + i_i * BC) * H*K + i_h * K + o_k
    p_dk = tl.make_block_ptr(dk + (bos*H+i_h)*K, (T, K), (H*K, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dA = tl.load(dA + o_dA + j * H*BT)
        # [BK,]
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_gqj = tl.load(p_gqj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] <= j
        b_dk += tl.where(m_i, b_dA[:, None] * b_qj[None, :] * exp(b_gqj[None, :] - b_g), 0.)
        p_qj += H*K
        p_gqj += H*K
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BV', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_bwd_kernel_dA(
    v,
    do,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_do = tl.make_block_ptr(do + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (V, T), (1, H*V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))

        b_dA += tl.dot(b_do, b_v)

    p_dA = tl.make_block_ptr(dA + (bos * H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]
    b_dA = tl.where(m_s, b_dA * scale, 0.)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BK_LIST
        for BV in BV_LIST
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_bwd_kernel_dv(
    k,
    g,
    A,
    do,
    dh,
    dv,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))
    p_do = tl.make_block_ptr(do + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))

    b_A = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], b_A, 0.)
    # (SY 09/17) important to disallow tf32 here to maintain a good precision.
    b_dv = tl.dot(b_A, b_do.to(b_A.dtype), allow_tf32=False)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gk = tl.make_block_ptr(g + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_gn = g + (bos + min(i_t * BT + BT, T) - 1)*H*K + i_h * K + o_k
        p_dh = tl.make_block_ptr(dh + (i_tg * H + i_h) * K*V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        b_gn = exp(tl.load(p_gn, mask=m_k, other=0)[None, :] - b_gk)
        b_k = (b_k * b_gn).to(b_k.dtype)
        # [BT, BV]
        # (SY 09/17) it is ok to have bf16 interchunk gradient contribution here
        b_dv += tl.dot(b_k, b_dh.to(b_k.dtype))

    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps)
        for BK in BK_LIST
        for BV in BV_LIST
        for num_warps in [2, 4, 8]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_gla_bwd_kernel_inter(
    q,
    k,
    v,
    g,
    h,
    do,
    dh,
    dq,
    dk,
    dq2,
    dk2,
    dg,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    h += (i_tg * H + i_h) * K*V
    do += (bos * H + i_h) * V
    dh += (i_tg * H + i_h) * K*V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dq2 += (bos * H + i_h) * K
    dk2 += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K

    p_gk = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_gk = tl.load(p_gk, boundary_check=(0, 1))
    p_gn = g + (min(T, i_t * BT + BT) - 1) * H*K + o_k
    b_gn = tl.load(p_gn, mask=m_k, other=0)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1))
        b_dh = tl.load(p_dh, boundary_check=(0, 1))

        # [BK]
        b_dgk += tl.sum(b_h * b_dh, axis=0)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

    b_dgk *= exp(b_gn)
    b_dq *= scale
    b_dq = b_dq * exp(b_gk)
    b_dk = b_dk * exp(b_gn[None, :] - b_gk)

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dgk += tl.sum(b_dk * b_k, axis=0)
    b_dq += tl.load(p_dq, boundary_check=(0, 1))
    b_dk += tl.load(p_dk, boundary_check=(0, 1))
    b_dg = b_q * b_dq - b_k * b_dk
    # tl.debug_barrier()
    b_dg = b_dg - tl.cumsum(b_dg, axis=0) + tl.sum(b_dg, axis=0)[None, :] + b_dgk[None, :]
    # Buggy due to strange triton compiler issue.
    # m_s = tl.where(tl.arange(0, BT)[:, None] <= tl.arange(0, BT)[None, :], 1., 0.)
    # b_dg = tl.dot(m_s, b_dg, allow_tf32=False) + b_dgk[None, :]
    p_dq = tl.make_block_ptr(dq2, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk2, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))


def chunk_gla_fwd_intra_gk(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    B, T, H, K = k.shape
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BC = min(16, BT)
    NC = triton.cdiv(BT, BC)

    A = q.new_empty(B, T, H, BT, dtype=torch.float)
    grid = (NT, NC * NC, B * H)
    chunk_gla_fwd_A_kernel_intra_sub_inter[grid](
        q=q,
        k=k,
        g=g,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
    )

    grid = (NT, NC, B * H)
    # load the entire [BC, K] blocks into SRAM at once
    if K <= 256:
        BK = max(triton.next_power_of_2(K), 16)
        chunk_gla_fwd_A_kernel_intra_sub_intra[grid](
            q=q,
            k=k,
            g=g,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
        )
    # split then merge
    else:
        BK = min(128, triton.next_power_of_2(K))
        NK = triton.cdiv(K, BK)
        A_intra = q.new_empty(NK, B, T, H, BC, dtype=torch.float)

        grid = (NK, NT * NC, B * H)
        chunk_gla_fwd_A_kernel_intra_sub_intra_split[grid](
            q=q,
            k=k,
            g=g,
            A=A_intra,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            T=T,
            B=B,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            NC=NC,
        )

        grid = (NT, NC, B * H)
        chunk_gla_fwd_A_kernel_intra_sub_intra_merge[grid](
            A=A_intra,
            A2=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            B=B,
            H=H,
            BT=BT,
            BC=BC,
            NK=NK,
        )
    return A


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = False,
):
    B, T, H, K, V = *q.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    o = torch.empty_like(v)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
    )
    return o


def chunk_gla_bwd_dA(
    v: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, V = v.shape
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BV = min(64, triton.next_power_of_2(V))

    dA = v.new_empty(B, T, H, BT, dtype=torch.float)
    grid = (NT, B * H)
    chunk_gla_bwd_kernel_dA[grid](
        v=v,
        do=do,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        V=V,
        BT=BT,
        BV=BV,
    )
    return dA


def chunk_gla_bwd_dv(
    k: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dv = torch.empty_like(do)
    def grid(meta): return (triton.cdiv(V, meta['BV']), NT, B * H)
    chunk_gla_bwd_kernel_dv[grid](
        k=k,
        g=g,
        A=A,
        do=do,
        dh=dh,
        dv=dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return dv


def chunk_gla_bwd_dqk_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    dA: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    B, T, H, K = q.shape
    BT = chunk_size
    BC = min(16, BT)
    BK = min(64, triton.next_power_of_2(K))

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    grid = (NK * NC, NT, B * H)
    chunk_gla_bwd_kernel_intra[grid](
        q=q,
        k=k,
        g=g,
        dA=dA,
        dq=dq,
        dk=dk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
    )
    return dq, dk


def chunk_gla_bwd_dqkg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dg = torch.empty_like(g)
    dq2 = torch.empty_like(dq)
    dk2 = torch.empty_like(dk)
    def grid(meta): return (triton.cdiv(K, meta['BK']), NT, B * H)
    chunk_gla_bwd_kernel_inter[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dq2=dq2,
        dk2=dk2,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    return dq2, dk2, dg


def chunk_gla_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, chunk_size, cu_seqlens=cu_seqlens)

    h, ht = chunk_fwd_h(
        k=k,
        v=v,
        g=None,
        gk=g_cumsum,
        gv=None,
        h0=initial_state,
        output_final_state=output_final_state,
        states_in_fp32=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    # the intra A is kept in fp32
    # the computation has very marginal effect on the entire throughput
    A = chunk_gla_fwd_intra_gk(
        q=q,
        k=k,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v,
        g=g_cumsum,
        A=A,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return g_cumsum, A, h, ht, o


def chunk_gla_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    g_cumsum: torch.Tensor | None,
    scale: float,
    initial_state: torch.Tensor,
    h: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    if g_cumsum is None:
        g_cumsum = chunk_local_cumsum(g, chunk_size, cu_seqlens=cu_seqlens)

    if h is None:
        h, _ = chunk_fwd_h(
            k=k,
            v=v,
            g=None,
            gk=g_cumsum,
            gv=None,
            h0=initial_state,
            output_final_state=False,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            states_in_fp32=True,
        )
    dh, dh0 = chunk_bwd_dh(
        q=q,
        k=k,
        v=v,
        g=None,
        gk=g_cumsum,
        gv=None,
        do=do,
        h0=initial_state,
        dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        states_in_fp32=True,
    )

    dv = chunk_gla_bwd_dv(
        k=k,
        g=g_cumsum,
        A=A,
        do=do,
        dh=dh,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    # dq dk in fp32
    dA = chunk_gla_bwd_dA(
        v=v,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dq, dk = chunk_gla_bwd_dqk_intra(
        q=q,
        k=k,
        g=g_cumsum,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dq, dk, dg = chunk_gla_bwd_dqkg(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g_cumsum,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return dq, dk, dv, dg, dh0


class ChunkGLAFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
    ):
        chunk_size = min(64, max(16, triton.next_power_of_2(q.shape[1])))

        g_cumsum, A, _, ht, o = chunk_gla_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_cumsum=None,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        # recompute g_cumsum in bwd pass
        if g.dtype != torch.float:
            g_cumsum = None
        else:
            g = None
        ctx.save_for_backward(q, k, v, g, g_cumsum, initial_state, A)
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.cu_seqlens = cu_seqlens
        return o, ht

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        q, k, v, g, g_cumsum, initial_state, A = ctx.saved_tensors
        chunk_size, scale, cu_seqlens = ctx.chunk_size, ctx.scale, ctx.cu_seqlens
        dq, dk, dv, dg, dh0 = chunk_gla_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            g_cumsum=g_cumsum,
            scale=scale,
            h=None,
            A=A,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )
        return dq.to(q), dk.to(k), dv.to(v), dg, None, dh0, None, None


@torch.compiler.disable
def chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: int | None = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            Forget gates of shape `[B, T, H, K]`.
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gla import chunk_gla
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = torch.randn(B, T, H, K, device='cuda')
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> g = F.logsigmoid(torch.randn(B, T, H, K, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = chunk_gla(
            q, k, v, g,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g = map(lambda x: rearrange(x, 'b t h d -> 1 (b t) h d'), (q, k, v, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gla(
            q, k, v, g,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = q.shape[-1] ** -0.5
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."
    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert v.shape == (*q.shape[:3], v.shape[-1]), "v must be of shape (batch size, seq len, num of head, head dim)."
    o, final_state = ChunkGLAFunction.apply(q, k, v, g, scale, initial_state, output_final_state, cu_seqlens)
    return o, final_state

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl



@triton.heuristics({
    'STORE_QG': lambda args: args['qg'] is not None,
    'STORE_KG': lambda args: args['kg'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'DOT_PRECISION': DOT_PRECISION}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
        for DOT_PRECISION in (["tf32x3", "ieee"] if IS_TF32_SUPPORTED else ["ieee"])
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def recompute_w_u_fwd_kernel(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STORE_QG: tl.constexpr,
    STORE_KG: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    p_b = tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_u = tl.make_block_ptr(u + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, input_precision=DOT_PRECISION)
        tl.store(p_u, b_u.to(p_u.dtype.element_ty), boundary_check=(0, 1))

    for i_k in range(tl.cdiv(K, BK)):
        p_w = tl.make_block_ptr(w + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_b[:, None]

        p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_gk = tl.load(p_gk, boundary_check=(0, 1))
        b_kb *= exp2(b_gk)
        if STORE_QG:
            p_q = tl.make_block_ptr(q + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            p_qg = tl.make_block_ptr(qg + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_qg = b_q * exp2(b_gk)
            tl.store(p_qg, b_qg.to(p_qg.dtype.element_ty), boundary_check=(0, 1))
        if STORE_KG:
            last_idx = min(i_t * BT + BT, T) - 1
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            b_gn = tl.load(gk + ((bos + last_idx) * H + i_h) * K + o_k, mask=m_k, other=0.)
            b_kg = b_k * tl.where((i_t * BT + tl.arange(0, BT) < T)[:, None], exp2(b_gn[None, :] - b_gk), 0)
            p_kg = tl.make_block_ptr(kg + (bos * H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
            tl.store(p_kg, b_kg.to(p_kg.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV', 'IS_VARLEN'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def prepare_wy_repr_bwd_kernel(
    k,
    v,
    beta,
    gk,
    A,
    dA,
    dw,
    du,
    dk,
    dk2,
    dv,
    db,
    dg,
    dg2,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_b = tl.make_block_ptr(beta + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_db = tl.make_block_ptr(db + (bos*H + i_h), (T,), (H,), (i_t * BT,), (BT,), (0,))
    p_A = tl.make_block_ptr(A + (bos*H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))

    b_b = tl.load(p_b, boundary_check=(0,))
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_A = tl.load(p_A, boundary_check=(0, 1))
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(k + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk2 = tl.make_block_ptr(dk2 + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dw = tl.make_block_ptr(dw + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg2 = tl.make_block_ptr(dg2 + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

        # [BT, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        p_gk = tl.make_block_ptr(gk + (bos*H + i_h) * K, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_gk_exp = exp2(tl.load(p_gk, boundary_check=(0, 1)))
        b_kbg = b_k * b_b[:, None] * b_gk_exp
        b_dw = tl.load(p_dw, boundary_check=(0, 1))

        b_dA += tl.dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype))
        b_dkbg = tl.dot(b_A, b_dw)
        b_dk = b_dkbg * b_gk_exp * b_b[:, None] + tl.load(p_dk, boundary_check=(0, 1))
        b_db += tl.sum(b_dkbg * b_k * b_gk_exp, 1)
        b_dg = b_kbg * b_dkbg + tl.load(p_dg, boundary_check=(0, 1))

        tl.store(p_dk2, b_dk.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg2, b_dg.to(p_dg2.dtype.element_ty), boundary_check=(0, 1))

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_du = tl.make_block_ptr(du + (bos*H + i_h) * V, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_vb = (b_v * b_b[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, boundary_check=(0, 1))
        b_dA += tl.dot(b_du, tl.trans(b_vb))
        b_dvb = tl.dot(b_A, b_du)
        b_dv = b_dvb * b_b[:, None]
        b_db += tl.sum(b_dvb * b_v, 1)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))

    b_dA = tl.where(m_A, -b_dA, 0)

    # if using gk, save dA first and handle dk in another kernel
    p_dA = tl.make_block_ptr(dA + (bos*H + i_h) * BT, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = 64
    BV = 64

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    qg = torch.empty_like(q) if q is not None else None
    kg = torch.empty_like(k) if gk is not None else None
    recompute_w_u_fwd_kernel[(NT, B*H)](
        q=q,
        k=k,
        qg=qg,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u, qg, kg


def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    gk: torch.Tensor,
    A: torch.Tensor,
    dk: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = 64
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    CONST_TILING = 64 if check_shared_mem() else 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)

    dk2 = torch.empty_like(dk, dtype=torch.float)
    dv = torch.empty_like(v)
    dg2 = torch.empty_like(gk, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)
    prepare_wy_repr_bwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        gk=gk,
        A=A,
        dA=dA,
        dw=dw,
        du=du,
        dk=dk,
        dk2=dk2,
        dv=dv,
        db=db,
        dg=dg,
        dg2=dg2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    dk = dk2
    dg = dg2
    return dk, dv, db, dg, dA

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Token-parallel implementation of KDA intra chunk kernel

import torch
import triton
import triton.language as tl



@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BH': BH}, num_warps=num_warps)
        for BH in [1, 2, 4, 8]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["K", "H"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T', 'N'])
def chunk_kda_fwd_kernel_intra_token_parallel(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_tg, i_hg = tl.program_id(0), tl.program_id(1)

    if IS_VARLEN:
        i_n = 0
        left, right = 0, N

        # Unrolled binary search (max B=2^32)
        # We can limit iterations based on expected max batch size if needed
        # 20 iterations covers B=1M, usually enough
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left

        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = (i_tg // T) * T
        i_t = i_tg % T

    if i_t >= T:
        return

    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC

    q += bos * H*K
    k += bos * H*K
    g += bos * H*K
    Aqk += bos * H*BT
    Akk += bos * H*BC
    beta += bos * H

    BK: tl.constexpr = triton.next_power_of_2(K)
    o_h = tl.arange(0, BH)
    o_k = tl.arange(0, BK)
    m_h = (i_hg * BH + o_h) < H
    m_k = o_k < K

    p_q = tl.make_block_ptr(q + i_t * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_t * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_t * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
    p_beta = tl.make_block_ptr(beta + i_t * H, (H,), (1,), (i_hg * BH,), (BH,), (0,))
    # [BH, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    b_k = b_k * tl.load(p_beta, boundary_check=(0,)).to(tl.float32)[:, None]

    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        p_kj = tl.make_block_ptr(k + j * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        p_gj = tl.make_block_ptr(g + j * H*K, (H, K), (K, 1), (i_hg * BH, 0), (BH, BK), (1, 0))
        # [BH, BK]
        b_kj = tl.load(p_kj, boundary_check=(0, 1)).to(tl.float32)
        b_gj = tl.load(p_gj, boundary_check=(0, 1)).to(tl.float32)

        b_kgj = tl.where(m_k[None, :], b_kj * exp2(b_g - b_gj), 0.0)
        # [BH]
        b_Aqk = tl.sum(b_q * b_kgj, axis=1) * scale
        b_Akk = tl.where(j < i_t, tl.sum(b_k * b_kgj, axis=1), 0.0)

        tl.store(Aqk + i_t * H*BT + (i_hg * BH + o_h) * BT + j % BT, b_Aqk.to(Aqk.dtype.element_ty), mask=m_h)
        tl.store(Akk + i_t * H*BC + (i_hg * BH + o_h) * BC + j % BC, b_Akk.to(Akk.dtype.element_ty), mask=m_h)


def chunk_kda_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
) -> None:
    """
    Token-parallel implementation: each token gets its own thread block.
    Supports both fixed-length and variable-length sequences.
    Reduces wasted computation on padding.

    Writes directly to Aqk and Akk tensors (in-place).

    Args:
        q: [B, T, H, K]
        k: [B, T, H, K]
        gk: [B, T, H, K] cumsum of gates
        beta: [B, T, H]
        Aqk: [B, T, H, BT] output tensor to write to
        Akk: [B, T, H, BC] output tensor for diagonal blocks (fp32)
        scale: attention scale
        chunk_size: BT (default 64)
        sub_chunk_size: BC (default 16)
    """
    B, T, H, K = q.shape
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = chunk_size
    BC = sub_chunk_size

    def grid(meta): return (B * T, triton.cdiv(H, meta['BH']))
    chunk_kda_fwd_kernel_intra_token_parallel[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        N=N,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
    )
    return Aqk, Akk

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl


if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32x3')
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')

################################################################################
# Fused inter + solve_tril kernel: compute off-diagonal Akk and solve in one pass
################################################################################


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps)
        for BK in [32, 64]
        for num_warps in [1, 2, 4]
    ],
    key=["H", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_fwd_kernel_inter_solve_fused(
    q,
    k,
    g,
    beta,
    Aqk,
    Akkd,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel: compute inter-subchunk Akk + solve_tril in one pass.
    Prerequisite: token_parallel has already computed diagonal Akk blocks in Akkd.

    This kernel:
    1. Computes off-diagonal Aqk blocks -> writes to global
    2. Computes off-diagonal Akk blocks -> keeps in registers
    3. Loads diagonal Akk blocks from Akkd (fp32)
    4. Does forward substitution on diagonals
    5. Computes merged Akk_inv
    6. Writes Akk_inv to Akk
    """
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    Aqk += (bos * H + i_h) * BT
    Akk += (bos * H + i_h) * BT
    Akkd += (bos * H + i_h) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    ################################################################################
    # off-diagonal blocks
    ################################################################################
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k0 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        p_g0 = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_tc0, i_k * BK), (BC, BK), (1, 0))
        b_k0 = tl.load(p_k0, boundary_check=(0, 1)).to(tl.float32)
        b_g0 = tl.load(p_g0, boundary_check=(0, 1)).to(tl.float32)

        b_k1, b_g1 = b_k0, b_g0
        b_k2, b_g2 = b_k0, b_g0
        if i_tc1 < T:
            p_q1 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_k1 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            p_g1 = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_tc1, i_k * BK), (BC, BK), (1, 0))
            # [BC, BK]
            b_q1 = tl.load(p_q1, boundary_check=(0, 1)).to(tl.float32)
            b_k1 = tl.load(p_k1, boundary_check=(0, 1)).to(tl.float32)
            b_g1 = tl.load(p_g1, boundary_check=(0, 1)).to(tl.float32)
            # [BK]
            b_gn1 = tl.load(g + i_tc1 * H*K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            # [BK, BC]
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            # [BC, BC]
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt)
            b_Akk10 += tl.dot(b_k1 * b_gqn, b_kgt)

        if i_tc2 < T:
            p_q2 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_k2 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            p_g2 = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_tc2, i_k * BK), (BC, BK), (1, 0))
            # [BC, BK]
            b_q2 = tl.load(p_q2, boundary_check=(0, 1)).to(tl.float32)
            b_k2 = tl.load(p_k2, boundary_check=(0, 1)).to(tl.float32)
            b_g2 = tl.load(p_g2, boundary_check=(0, 1)).to(tl.float32)
            # [BK]
            b_gn2 = tl.load(g + i_tc2 * H*K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
            b_qg2 = b_q2 * b_gqn2
            b_kg2 = b_k2 * b_gqn2
            # [BK, BC]
            b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
            b_Aqk20 += tl.dot(b_qg2, b_kgt)
            b_Akk20 += tl.dot(b_kg2, b_kgt)
            # [BC, BC]
            b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
            # [BC, BC]
            b_Aqk21 += tl.dot(b_qg2, b_kgt)
            b_Akk21 += tl.dot(b_kg2, b_kgt)

        if i_tc3 < T:
            p_q3 = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
            p_k3 = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
            p_g3 = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_tc3, i_k * BK), (BC, BK), (1, 0))
            # [BC, BK]
            b_q3 = tl.load(p_q3, boundary_check=(0, 1)).to(tl.float32)
            b_k3 = tl.load(p_k3, boundary_check=(0, 1)).to(tl.float32)
            b_g3 = tl.load(p_g3, boundary_check=(0, 1)).to(tl.float32)
            # [BK]
            b_gn3 = tl.load(g + i_tc3 * H*K + o_k, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
            b_qg3 = b_q3 * b_gqn3
            b_kg3 = b_k3 * b_gqn3
            # [BK, BC]
            b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
            # [BC, BC]
            b_Aqk30 += tl.dot(b_qg3, b_kgt)
            b_Akk30 += tl.dot(b_kg3, b_kgt)
            # [BK, BC]
            b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
            # [BC, BC]
            b_Aqk31 += tl.dot(b_qg3, b_kgt)
            b_Akk31 += tl.dot(b_kg3, b_kgt)
            # [BK, BC]
            b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
            # [BC, BC]
            b_Aqk32 += tl.dot(b_qg3, b_kgt)
            b_Akk32 += tl.dot(b_kg3, b_kgt)

    ################################################################################
    # save off-diagonal Aqk blocks and prepare Akk
    ################################################################################
    if i_tc1 < T:
        p_Aqk10 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
        tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b1 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc1,), (BC,), (0,))
        b_b1 = tl.load(p_b1, boundary_check=(0,)).to(tl.float32)
        b_Akk10 = b_Akk10 * b_b1[:, None]
    if i_tc2 < T:
        p_Aqk20 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
        p_Aqk21 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b2 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc2,), (BC,), (0,))
        b_b2 = tl.load(p_b2, boundary_check=(0,)).to(tl.float32)
        b_Akk20 = b_Akk20 * b_b2[:, None]
        b_Akk21 = b_Akk21 * b_b2[:, None]
    if i_tc3 < T:
        p_Aqk30 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
        p_Aqk31 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
        p_Aqk32 = tl.make_block_ptr(Aqk, (T, BT), (H*BT, 1), (i_tc3, 2*BC), (BC, BC), (1, 0))
        tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), boundary_check=(0, 1))

        p_b3 = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,), (i_tc3,), (BC,), (0,))
        b_b3 = tl.load(p_b3, boundary_check=(0,)).to(tl.float32)
        b_Akk30 = b_Akk30 * b_b3[:, None]
        b_Akk31 = b_Akk31 * b_b3[:, None]
        b_Akk32 = b_Akk32 * b_b3[:, None]

    p_Akk00 = tl.make_block_ptr(Akkd, (T, BC), (H*BC, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akkd, (T, BC), (H*BC, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akkd, (T, BC), (H*BC, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akkd, (T, BC), (H*BC, 1), (i_tc3, 0), (BC, BC), (1, 0))
    b_Ai00 = tl.load(p_Akk00, boundary_check=(0, 1)).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, boundary_check=(0, 1)).to(tl.float32)
    b_Ai22 = tl.load(p_Akk22, boundary_check=(0, 1)).to(tl.float32)
    b_Ai33 = tl.load(p_Akk33, boundary_check=(0, 1)).to(tl.float32)

    ################################################################################
    # forward substitution on diagonals
    ################################################################################
    m_A = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Ai00 = -tl.where(m_A, b_Ai00, 0)
    b_Ai11 = -tl.where(m_A, b_Ai11, 0)
    b_Ai22 = -tl.where(m_A, b_Ai22, 0)
    b_Ai33 = -tl.where(m_A, b_Ai33, 0)

    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = -tl.load(Akkd + (i_tc0 + i) * H*BC + o_i)
        b_a00 = tl.where(o_i < i, b_a00, 0.)
        b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(BC + 2, min(2*BC, T - i_tc0)):
        b_a11 = -tl.load(Akkd + (i_tc0 + i) * H*BC + o_i)
        b_a11 = tl.where(o_i < i - BC, b_a11, 0.)
        b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
    for i in range(2*BC + 2, min(3*BC, T - i_tc0)):
        b_a22 = -tl.load(Akkd + (i_tc0 + i) * H*BC + o_i)
        b_a22 = tl.where(o_i < i - 2*BC, b_a22, 0.)
        b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i - 2*BC)[:, None], b_a22, b_Ai22)
    for i in range(3*BC + 2, min(4*BC, T - i_tc0)):
        b_a33 = -tl.load(Akkd + (i_tc0 + i) * H*BC + o_i)
        b_a33 = tl.where(o_i < i - 3*BC, b_a33, 0.)
        b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i - 3*BC)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    ################################################################################
    # compute merged inverse using off-diagonals
    ################################################################################

    # we used tf32x3 to maintain matrix inverse's precision whenever possible.
    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION) +
        tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION
    )

    ################################################################################
    # store full Akk_inv to Akk
    ################################################################################

    p_Akk00 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc0, 0), (BC, BC), (1, 0))
    p_Akk10 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc1, 0), (BC, BC), (1, 0))
    p_Akk11 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc1, BC), (BC, BC), (1, 0))
    p_Akk20 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc2, 0), (BC, BC), (1, 0))
    p_Akk21 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc2, BC), (BC, BC), (1, 0))
    p_Akk22 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc2, 2*BC), (BC, BC), (1, 0))
    p_Akk30 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, 0), (BC, BC), (1, 0))
    p_Akk31 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, BC), (BC, BC), (1, 0))
    p_Akk32 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, 2*BC), (BC, BC), (1, 0))
    p_Akk33 = tl.make_block_ptr(Akk, (T, BT), (H*BT, 1), (i_tc3, 3*BC), (BC, BC), (1, 0))

    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BK', 'NC', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['B', 'T'])
def chunk_kda_bwd_kernel_intra(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dg,
    dg2,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_kc, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC

    all = B * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * H + i_h) * K
    beta += bos * H + i_h

    dAqk += (bos * H + i_h) * BT
    dAkk += (bos * H + i_h) * BT
    dq += (bos * H + i_h) * K
    dq2 += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dk2 += (bos * H + i_h) * K
    dg += (bos * H + i_h) * K
    dg2 += (bos * H + i_h) * K
    db += (i_k * all + bos) * H + i_h

    p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1))

    p_b = tl.make_block_ptr(beta, (T,), (H,), (i_ti,), (BC,), (0,))
    b_b = tl.load(p_b, boundary_check=(0,))

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        p_gn = g + i_ti * H*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(0, i_i):
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_dAqk = tl.make_block_ptr(dAqk, (T, BT), (H*BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            p_dAkk = tl.make_block_ptr(dAkk, (T, BT), (H*BT, 1), (i_ti, i_j * BC), (BC, BC), (1, 0))
            # [BC, BK]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            b_kg = b_k * exp2(b_gn[None, :] - b_gk)
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))
            # [BC, BK]
            b_dq2 += tl.dot(b_dAqk, b_kg)
            b_dk2 += tl.dot(b_dAkk, b_kg)
        b_gqn = exp2(b_g - b_gn[None, :])
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn

    o_i = tl.arange(0, BC)
    m_dA = (i_ti + o_i) < T
    o_dA = (i_ti + o_i) * H*BT + i_i * BC
    p_kj = k + i_ti * H*K + o_k
    p_gkj = g + i_ti * H*K + o_k

    p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_k = tl.load(p_k, boundary_check=(0, 1))

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC]
        b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
        b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
        # [BK]
        b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] >= j
        # [BC, BK]
        b_kgj = b_kj[None, :] * exp2(b_g - b_gkj[None, :])
        b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kgj, 0.)
        b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kgj, 0.)

        p_kj += H*K
        p_gkj += H*K
    b_db = tl.sum(b_dk2 * b_k, 1)
    b_dk2 *= b_b[:, None]

    p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dq2 = tl.make_block_ptr(dq2, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_db = tl.make_block_ptr(db, (T,), (H,), (i_ti,), (BC,), (0,))

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, boundary_check=(0, 1))
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        p_gn = g + (min(i_ti + BC, T) - 1) * H*K + o_k
        # [BK,]
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        for i_j in range(i_i + 1, NC):
            p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t*BT+i_j*BC, i_k*BK), (BC, BK), (1, 0))
            p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k * BK), (BC, BK), (1, 0))
            p_gk = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT + i_j * BC, i_k*BK), (BC, BK), (1, 0))
            p_b = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT + i_j * BC,), (BC,), (0,))
            p_dAqk = tl.make_block_ptr(dAqk, (BT, T), (1, H*BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            p_dAkk = tl.make_block_ptr(dAkk, (BT, T), (1, H*BT), (i_i * BC, i_t * BT + i_j * BC), (BC, BC), (0, 1))
            # [BC]
            b_b = tl.load(p_b, boundary_check=(0,))
            # [BC, BK]
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_kb = tl.load(p_k, boundary_check=(0, 1)) * b_b[:, None]
            b_gk = tl.load(p_gk, boundary_check=(0, 1))
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, boundary_check=(0, 1))
            b_dAkk = tl.load(p_dAkk, boundary_check=(0, 1))

            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            # [BC, BK]
            b_gkn = tl.where(m_j[:, None], exp2(b_gk - b_gn[None, :]), 0)
            b_qg = b_q * b_gkn
            b_kbg = b_kb * b_gkn
            # [BC, BK]
            b_dkt += tl.dot(b_dAqk, b_qg) + tl.dot(b_dAkk, b_kbg)
        b_dkt *= exp2(b_gn[None, :] - b_g)
    o_dA = i_ti * H*BT + i_i * BC + o_i
    p_qj = q + i_ti * H*K + o_k
    p_kj = k + i_ti * H*K + o_k
    p_gkj = g + i_ti * H*K + o_k
    p_bj = beta + i_ti * H

    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
        # [BC,]
        b_dAqk = tl.load(dAqk + o_dA + j * H*BT)
        b_dAkk = tl.load(dAkk + o_dA + j * H*BT)
        # [BK,]
        b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
        b_kbj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32) * tl.load(p_bj)
        b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
        # [BC, BK]
        m_i = o_i[:, None] <= j
        b_gkq = exp2(b_gkj[None, :] - b_g)
        b_dkt += tl.where(m_i, (b_dAkk[:, None] * b_kbj[None, :] + b_dAqk[:, None] * b_qj[None, :]) * b_gkq, 0.)

        p_qj += H*K
        p_kj += H*K
        p_gkj += H*K
        p_bj += H
    p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dk2 = tl.make_block_ptr(dk2, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))
    p_dg2 = tl.make_block_ptr(dg2, (T, K), (H*K, 1), (i_ti, i_k * BK), (BC, BK), (1, 0))

    b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, boundary_check=(0, 1))
    b_dk2 += tl.load(p_dk, boundary_check=(0, 1))
    b_dk2 += b_dkt

    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), boundary_check=(0, 1))


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K = k.shape
    BT = chunk_size
    BC = 16
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    Aqk = torch.empty(B, T, H, BT, device=k.device, dtype=k.dtype)
    # Akk must be zero-initialized - kernel only writes lower triangular
    Akk = torch.zeros(B, T, H, BT, device=k.device, dtype=k.dtype)
    # Separate fp32 buffer for diagonal 16x16 blocks (for precision in solve_tril)
    Akkd = torch.empty(B, T, H, BC, device=k.device, dtype=torch.float32)

    # Step 1: Run token_parallel first to compute diagonal blocks into Akkd (fp32)
    Aqk, Akkd = chunk_kda_fwd_intra_token_parallel(
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akkd,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        sub_chunk_size=BC,
    )
    # Step 2: Fused inter + solve_tril (works for both fixed-len and varlen)
    grid = (NT, B * H)
    chunk_kda_fwd_kernel_inter_solve_fused[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akkd=Akkd,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
    )
    w, u, _, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, kg, Aqk, Akk


def chunk_kda_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    B, T, H, K = k.shape
    BT = chunk_size
    BC = min(16, BT)
    BK = min(32, triton.next_power_of_2(K))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    dq2 = torch.empty_like(q)
    dk2 = torch.empty_like(k)
    db2 = beta.new_empty(NK, *beta.shape, dtype=torch.float)
    dg2 = torch.empty_like(dg, dtype=torch.float)
    grid = (NK * NC, NT, B * H)
    chunk_kda_bwd_kernel_intra[grid](
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dq2=dq2,
        dk=dk,
        dk2=dk2,
        dg=dg,
        dg2=dg2,
        db=db2,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
    )
    dq = dq2
    dk = dk2
    db = db2.sum(0).add_(db)
    dg = chunk_local_cumsum(
        dg2,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    return dq, dk, db, dg

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch
import triton
import triton.language as tl


BK_LIST = [32, 64] if check_shared_mem() else [16, 32]
BV_LIST = [64, 128] if check_shared_mem('ampere') else [16, 32]
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT', 'BK', 'BV'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_dAv(
    q,
    k,
    v,
    A,
    do,
    dv,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    do += (bos * H + i_h) * V
    dv += (bos * H + i_h) * V
    dA += (bos * H + i_h) * BT

    p_A = tl.make_block_ptr(A + (bos * H + i_h) * BT, (BT, T), (1, H*BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v, (V, T), (1, H*V), (i_v * BV, i_t * BT), (BV, BT), (0, 1))
        p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        # [BV, BT]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [BT, BV]
        b_do = tl.load(p_do, boundary_check=(0, 1))
        # [BT, BT]
        b_dA += tl.dot(b_do, b_v)
        # [BT, BV]
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))

    p_dA = tl.make_block_ptr(dA, (T, BT), (H*BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_dA = tl.where(o_t[:, None] >= o_t, b_dA * scale, 0.)
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BK_LIST
        for BV in BV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=['BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_kda_bwd_kernel_wy_dqkg_fused(
    q,
    k,
    v,
    v_new,
    g,
    beta,
    A,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    dv2,
    dg,
    db,
    dA,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = (o_t == min(T, i_t * BT + BT) - 1)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    v_new += (bos * H + i_h) * V
    g += (bos * H + i_h) * K
    beta += bos * H + i_h
    A += (bos * H + i_h) * BT
    h += (i_tg * H + i_h) * K*V
    do += (bos * H + i_h) * V
    dh += (i_tg * H + i_h) * K*V
    dq += (bos * H + i_h) * K
    dk += (bos * H + i_h) * K
    dv += (bos * H + i_h) * V
    dv2 += (bos * H + i_h) * V
    dg += (bos * H + i_h) * K
    db += bos * H + i_h
    dA += (bos * H + i_h) * BT

    p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
    b_beta = tl.load(p_beta, boundary_check=(0,))

    p_A = tl.make_block_ptr(A, (BT, T), (1, H * BT), (0, i_t * BT), (BT, BT), (0, 1))
    b_A = tl.load(p_A, boundary_check=(0, 1))

    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_db = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k = tl.make_block_ptr(k, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1))

        p_gn = g + (min(T, i_t * BT + BT) - 1) * H*K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0)

        b_dq = tl.zeros([BT, BK], dtype=tl.float32)
        b_dk = tl.zeros([BT, BK], dtype=tl.float32)
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        b_dgk = tl.zeros([BK], dtype=tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            p_v_new = tl.make_block_ptr(v_new, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_do = tl.make_block_ptr(do, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            p_h = tl.make_block_ptr(h, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dh = tl.make_block_ptr(dh, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            p_dv = tl.make_block_ptr(dv, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            # [BT, BV]
            b_v_new = tl.load(p_v_new, boundary_check=(0, 1))
            b_do = tl.load(p_do, boundary_check=(0, 1))
            # [BV, BK]
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dh = tl.load(p_dh, boundary_check=(0, 1))
            # [BT, BV]
            b_dv = tl.load(p_dv, boundary_check=(0, 1))

            b_dgk += tl.sum(b_h * b_dh, axis=0)
            b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
            b_dk += tl.dot(b_v_new, b_dh.to(b_v_new.dtype))
            b_dw += tl.dot(b_dv.to(b_v_new.dtype), b_h.to(b_v_new.dtype))

            tl.debug_barrier()
            if i_k == 0:
                p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                p_dv2 = tl.make_block_ptr(dv2, (T, V), (H*V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

                b_v = tl.load(p_v, boundary_check=(0, 1))

                b_dA += tl.dot(b_dv, tl.trans(b_v))

                b_dvb = tl.dot(b_A, b_dv)
                b_dv2 = b_dvb * b_beta[:, None]
                b_db += tl.sum(b_dvb * b_v, 1)

                tl.store(p_dv2, b_dv2.to(p_dv2.dtype.element_ty), boundary_check=(0, 1))

        b_gk_exp = exp2(b_g)
        b_gb = b_gk_exp * b_beta[:, None]
        b_dgk *= exp2(b_gn)
        b_dq = b_dq * b_gk_exp * scale
        b_dk = b_dk * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0)

        b_kg = b_k * b_gk_exp

        b_dw = -b_dw.to(b_A.dtype)
        b_dA += tl.dot(b_dw, tl.trans(b_kg.to(b_A.dtype)))

        b_dkgb = tl.dot(b_A, b_dw)
        b_db += tl.sum(b_dkgb * b_kg, 1)

        p_q = tl.make_block_ptr(q, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_kdk = b_k * b_dk
        b_dgk += tl.sum(b_kdk, axis=0)
        b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk + b_kg * b_dkgb * b_beta[:, None]
        b_dk = b_dk + b_dkgb * b_gb

        p_dq = tl.make_block_ptr(dq, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dk = tl.make_block_ptr(dk, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_dg = tl.make_block_ptr(dg, (T, K), (H*K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA = tl.where(m_A, -b_dA, 0)

    p_dA = tl.make_block_ptr(dA, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_db = tl.make_block_ptr(db, (T,), (H,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), boundary_check=(0,))


def chunk_kda_bwd_dAv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    A: torch.Tensor | None = None,
    scale: float = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, do.shape[-1]
    BT = chunk_size
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    # H100 can have larger block size
    if check_shared_mem('hopper', k.device.index):
        CONST_TILING = 128
    elif check_shared_mem:
        CONST_TILING = 64
    else:
        CONST_TILING = 32
    BK = min(max(triton.next_power_of_2(K), 16), CONST_TILING)
    BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dA = v.new_empty(B, T, H, BT, dtype=torch.float)
    dv = torch.empty_like(do)
    grid = (NT, B * H)
    chunk_kda_bwd_kernel_dAv[grid](
        q=q,
        k=k,
        v=v,
        A=A,
        do=do,
        dv=dv,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return dA, dv


def chunk_kda_bwd_wy_dqkg_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dq = torch.empty_like(q, dtype=torch.float)
    dk = torch.empty_like(k, dtype=torch.float)
    dv2 = torch.empty_like(v)
    dg = torch.empty_like(g, dtype=torch.float)
    db = torch.empty_like(beta, dtype=torch.float)
    dA = torch.empty_like(A, dtype=torch.float)

    grid = (NT, B * H)
    chunk_kda_bwd_kernel_wy_dqkg_fused[grid](
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=A,
        h=h,
        do=do,
        dh=dh,
        dq=dq,
        dk=dk,
        dv=dv,
        dv2=dv2,
        dg=dg,
        db=db,
        dA=dA,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        K=K,
        V=V,
        BT=BT,
    )
    dv = dv2
    return dq, dk, dv, db, dg, dA

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import torch



def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    w, u, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )

    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )
    return o, Aqk, Akk, final_state


def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
):
    # w = Akk @ (k * beta)
    # u = Akk @ (v * beta)
    w, u, qg, kg = recompute_w_u_fwd(
        q=q,
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        gk=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )
    # dAqk = do @ v.T
    # dv = A @ do
    dAqk, dv = chunk_kda_bwd_dAv(
        q=q,
        k=k,
        v=v_new,
        do=do,
        A=Aqk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=qg,
        k=kg,
        w=w,
        gk=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )
    dq, dk, dv, db, dg, dAkk = chunk_kda_bwd_wy_dqkg_fused(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=Akk,
        h=h,
        do=do,
        dh=dh,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    dq, dk, db, dg = chunk_kda_bwd_intra(
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        db=db,
        dg=dg,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
    )
    return dq, dk, dv, db, dg, dh0


class ChunkKDAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ):
        g_org = None
        if use_gate_in_kernel:
            g_org = g
            g = kda_gate_fwd(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
            )
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        chunk_size = 64
        g = chunk_local_cumsum(
            g=g,
            chunk_size=chunk_size,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices
        )
        o, Aqk, Akk, final_state = chunk_kda_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        if use_gate_in_kernel:
            g = None
        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g, g_org, beta, A_log, dt_bias, Aqk, Akk, initial_state, cu_seqlens, chunk_indices
        )
        ctx.chunk_size = chunk_size
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        (q, q_rstd, k, k_rstd, v, g, g_org, beta, A_log, dt_bias, Aqk, Akk, initial_state, cu_seqlens, chunk_indices) = (
            ctx.saved_tensors
        )
        if ctx.use_gate_in_kernel:
            g = kda_gate_fwd(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
            )
            g = chunk_local_cumsum(
                g=g,
                chunk_size=ctx.chunk_size,
                scale=RCP_LN2,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices
            )
        dq, dk, dv, db, dg, dh0 = chunk_kda_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            Aqk=Aqk,
            Akk=Akk,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        dA, dbias = None, None
        if ctx.use_gate_in_kernel:
            dg, dA, dbias = kda_gate_bwd(
                g=g_org,
                A_log=A_log,
                dt_bias=dt_bias,
                dyg=dg,
                dyb=db,
            )
            dA = dA.to(A_log)
            if dt_bias is not None:
                dbias = dbias.to(dt_bias)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), dA, dbias, None, dh0, None, None, None, None, None


@torch.compiler.disable
def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    **kwargs,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H, K]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the KDA attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q,k tensor internally. Default: `False`.
        use_gate_in_kernel (bool):
            Whether to compute the log-space KDA decay internally.
            - If `True`:
              The passed `g` acts as the raw input for `-exp(A_log).view(H, -1) * softplus(g + dt_bias.view(H, K))`.
              Note that as part of the input arguments,
              `A_log` (shape `[H]`) and the optional `dt_bias` (shape `[H * K]`) should be provided.
            - If `False`, `g` is expected to be the pre-computed decay value.
            Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        chunk_indices (torch.LongTensor):
            Chunk indices used for variable-length training,

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.kda import chunk_kda
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda')
        >>> g = torch.rand(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> A_log = torch.randn(H, dtype=torch.float32, device='cuda')
        >>> dt_bias = torch.randn(H * K, dtype=torch.float32, device='cuda')
        >>> o, ht = chunk_kda(
            q, k, v, g, beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_kda(
            q, k, v, g, beta,
            A_log=A_log,
            dt_bias=dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")

    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert k.shape[-1] <= 256, "Currently we only support key headdim <=256 for KDA :-("
    assert beta.shape == q.shape[:3], "beta must be of shape (batch size, seq len, num of head)."
    assert v.shape == (*q.shape[:3], v.shape[-1]), "v must be of shape (batch size, seq len, num of head, head dim)."

    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        cu_seqlens,
        chunk_indices,
    )
    return o, final_state

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import torch
import triton
import triton.language as tl



@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'USE_GK': lambda args: args['gk'] is not None,
    'USE_GV': lambda args: args['gv'] is not None,
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    gk,
    gv,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_GV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if USE_G:
        p_g = g + bos * HV + i_hv
    if USE_GK:
        p_gk = gk + (bos * HV + i_hv) * K + o_k
    if USE_GV:
        p_gv = gv + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

        # [BK, BV]
        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)

        if USE_GK:
            b_gk = tl.load(p_gk).to(tl.float32)
            b_h *= exp(b_gk[:, None])

        if USE_GV:
            b_gv = tl.load(p_gv).to(tl.float32)
            b_h *= exp(b_gv[None, :])

        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v

        # [BV]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H*K
        p_k += H*K
        p_v += HV*V
        if USE_G:
            p_g += HV
        if USE_GK:
            p_gk += HV*K
        if USE_GV:
            p_gv += HV*V
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV*V

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * K*V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V)) if gv is None else triton.next_power_of_2(V)
    NV = triton.cdiv(V, BV)

    o = torch.empty_like(v)
    final_state = q.new_empty(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    grid = (NV, N * HV)
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=gk,
        gv=gv,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=1,
        num_stages=3,
    )
    return o, final_state


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        gv: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        scale: float = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            gk=gk,
            gv=gv,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )

        return o, final_state

    @staticmethod
    @input_guard
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Backward pass is not implemented yet and we do not have plans to implement it "
            "because we haven't figured out how to compute dg without materializing the full "
            "hidden states for all time steps.",
        )


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV]`. Default: `None`.
        gk (torch.Tensor):
            gk (decays) of shape `[B, T, HV, K]`. Default: `None`.
        gv (torch.Tensor):
            gv (decays) of shape `[B, T, HV, V]`. Default: `None`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use L2 normalization in the kernel. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = fused_gated_recurrent_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])

    o, final_state = FusedRecurrentFunction.apply(
        q,
        k,
        v,
        g,
        gk,
        gv,
        beta,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
    )
    return o, final_state

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import torch



def fused_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor):
            g (decays) of shape `[B, T, HV, K]`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (Optional[bool]):
            Whether to use L2 normalization in the kernel. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, HV, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, HV, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.kda import fused_recurrent_kda
        # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, K, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_recurrent_kda(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = fused_recurrent_kda(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from einops import rearrange


NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [4, 8, 16, 32]
STATIC_WARPS = 32 if not IS_AMD else 16


try:
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d import causal_conv1d_update as causal_conv1d_update_cuda
    from causal_conv1d.cpp_functions import causal_conv1d_bwd_function
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update_cuda = None
    causal_conv1d_bwd_function = None


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D', 'W', 'NB'],
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_fwd_kernel(
    x,
    y,
    weight,
    bias,
    residual,
    cu_seqlens,
    initial_state,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        # [BD, BW]
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0).to(tl.float32)

    b_y = tl.zeros((BT, BD), dtype=tl.float32)
    if not USE_INITIAL_STATE:
        for i_w in tl.static_range(-W + 1, 1):
            p_yi = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            # [BT, BD]
            b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    elif i_t * BT >= W:
        # to make Triton compiler happy, we need to copy codes
        for i_w in tl.static_range(-W + 1, 1):
            p_yi = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            # [BT, BD]
            b_yi = tl.load(p_yi, boundary_check=(0, 1)).to(tl.float32)
            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi
    else:
        o_t = i_t * BT + tl.arange(0, BT)
        for i_w in tl.static_range(-W + 1, 1):
            o_x = o_t + i_w
            m_x = ((o_x >= 0) & (o_x < T))[:, None] & m_d
            m_c = ((o_x + W >= 0) & (o_x < 0))[:, None] & m_d

            b_yi = tl.load(x + bos * D + o_x[:, None] * D + o_d, mask=m_x, other=0).to(tl.float32)

            b_yi += tl.load(initial_state + i_n * D*W + o_d * W + (o_x + W)[:, None], mask=m_c, other=0).to(tl.float32)

            if HAS_WEIGHT:
                b_yi *= tl.sum(b_w * (o_w == (i_w + W - 1)), 1)
            b_y += b_yi

    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d).to(tl.float32)

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        p_residual = tl.make_block_ptr(residual + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_residual = tl.load(p_residual, boundary_check=(0, 1))
        b_y += b_residual

    p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(p_y, tl.cast(b_y, dtype=p_y.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['dw'] is not None,
    'HAS_BIAS': lambda args: args['db'] is not None,
    'USE_INITIAL_STATE': lambda args: args['dh0'] is not None,
    'USE_FINAL_STATE': lambda args: args['dht'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BD': BD}, num_warps=num_warps)
        for BD in [16, 32, 64, 128]
        for num_warps in [4, 8, 16, 32]
    ],
    key=['D', 'W', 'NB'],
    **autotune_cache_kwargs,
)
@triton.jit
def causal_conv1d_bwd_kernel(
    x,
    y,
    weight,
    initial_state,
    dh0,
    dht,
    dy,
    dx,
    dw,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BT: tl.constexpr,
    BW: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        i_tg = i_b * tl.num_programs(1) + i_t
        i_n = i_b
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0

    if HAS_WEIGHT:
        p_x = tl.make_block_ptr(x + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
        b_x = tl.load(p_x, boundary_check=(0, 1))
        # [BD, BW]
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0)

    b_dx = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS:
        b_db = tl.zeros((BD,), dtype=tl.float32)

    if not USE_FINAL_STATE:
        for i_w in tl.static_range(0, W):
            p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            # [BT, BD]
            b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y)
                b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
            b_wdy = b_dy
            if HAS_WEIGHT:
                # [BT, BD]
                b_wdy = b_wdy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
                # [BD]
                b_dw = tl.sum(b_dy * b_x, 0)
                tl.store(dw + i_tg * D*W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy, 0)
            b_dx += b_wdy
    elif i_t * BT >= W:
        # to make Triton compiler happy, we need to copy codes
        for i_w in tl.static_range(0, W):
            p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            # [BT, BD]
            b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                b_y = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y)
                b_dy = b_dy * b_ys * (1 + b_y * (1 - b_ys))
            b_wdy = b_dy
            if HAS_WEIGHT:
                # [BT, BD]
                b_wdy = b_wdy * tl.sum(b_w * (o_w == (W - i_w - 1)), 1)
                # [BD]
                b_dw = tl.sum(b_dy * b_x, 0)
                tl.store(dw + i_tg * D*W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)
            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy, 0)
            b_dx += b_wdy
    else:
        # which may use initial state
        o_t = i_t * BT + tl.arange(0, BT)
        for i_w in tl.static_range(0, W):
            p_dy = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
            b_dy_shift = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                p_y = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT + i_w, i_d * BD), (BT, BD), (1, 0))
                b_y_shift = tl.load(p_y, boundary_check=(0, 1)).to(tl.float32)
                b_ys = tl.sigmoid(b_y_shift)
                b_dy_shift = b_dy_shift * b_ys * (1 + b_y_shift * (1 - b_ys))
            if HAS_WEIGHT:
                # gradient comes from x：sum_t dy[t+i_w] * x[t]
                b_dw = tl.sum(b_dy_shift * b_x, 0)
                # index of cache：c = W - i_w + t
                if USE_INITIAL_STATE:
                    mask_head_rows = (o_t < i_w)
                    # dy_head = dy[t]
                    b_dy_head = tl.load(dy + bos * D + o_t[:, None] * D + o_d, mask=(mask_head_rows[:, None] & m_d[None, :]),
                                        other=0.0).to(tl.float32)
                    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                        # use y[t] （not y[t+i_w]）
                        b_y_head = tl.load(y + bos * D + o_t[:, None] * D + o_d,
                                           mask=(mask_head_rows[:, None] & m_d[None, :]), other=0.0).to(tl.float32)
                        b_ys_head = tl.sigmoid(b_y_head)
                        b_dy_head = b_dy_head * b_ys_head * (1 + b_y_head * (1 - b_ys_head))
                    o_c = W - i_w + o_t
                    # index 0 is padding 0
                    mask_c = (mask_head_rows & (o_c >= 1) & (o_c < W))
                    b_xc = tl.load(initial_state + i_n * D * W + o_d[None, :] * W + o_c[:, None],
                                   mask=(mask_c[:, None] & m_d[None, :]), other=0.0).to(tl.float32)
                    # add the gradient comes from initial_state
                    b_dw += tl.sum(b_dy_head * b_xc, 0)
                tl.store(dw + i_tg * D * W + o_d * W + W - i_w - 1, b_dw.to(dw.dtype.element_ty), mask=m_d)

            if HAS_BIAS and i_w == 0:
                b_db += tl.sum(b_dy_shift, 0)
            b_wdy = b_dy_shift if not HAS_WEIGHT else (b_dy_shift * tl.sum(b_w * (o_w == (W - i_w - 1)), 1))
            b_dx += b_wdy

        if USE_INITIAL_STATE:
            p_dy0 = tl.make_block_ptr(dy + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
            b_dy0 = tl.load(p_dy0, boundary_check=(0, 1)).to(tl.float32)
            if ACTIVATION == 'swish' or ACTIVATION == 'silu':
                p_y0 = tl.make_block_ptr(y + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
                b_y0 = tl.load(p_y0, boundary_check=(0, 1)).to(tl.float32)
                b_ys0 = tl.sigmoid(b_y0)
                b_dy0 = b_dy0 * b_ys0 * (1 + b_y0 * (1 - b_ys0))
            # index 0 is padding 0, skip calculation
            for i_w in tl.static_range(1, W):
                m_rows = (o_t < i_w)
                if HAS_WEIGHT:
                    # [BT]
                    w_idx_rows = i_w - 1 - o_t
                    # [BT, BW]
                    w_mask = (o_w[None, :] == w_idx_rows[:, None])
                    w_pick = tl.sum(b_w[None, :, :] * w_mask[:, None, :], 2)
                else:
                    w_pick = 1.0
                contrib = (b_dy0 * w_pick).to(tl.float32)
                contrib = tl.where(m_rows[:, None] & m_d[None, :], contrib, 0.0)
                # [BD]
                b_dh0_s = tl.sum(contrib, 0)
                # dh0: [NT, B, D, W]
                tl.store(dh0 + i_t * B * D * W + i_n * D * W + o_d * W + i_w,
                         b_dh0_s.to(dh0.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_d)

    if HAS_BIAS:
        b_db = tl.cast(b_db, dtype=db.dtype.element_ty, fp_downcast_rounding='rtne')
        tl.store(db + i_tg * D + o_d, b_db, mask=m_d)

    if USE_FINAL_STATE:
        if i_t * BT + BT >= T-W:
            start_tok = max(0, T - (W - 1))
            offset = i_t * BT + tl.arange(0, BT)
            tok_idx = offset - start_tok
            mask = (offset >= start_tok) & (offset < T)
            w_idx = 1 + tok_idx
            dht_off = i_n * D * W + o_d[None, :] * W + w_idx[:, None]
            b_dht = tl.load(dht + dht_off, mask=mask[:, None] & m_d[None, :], other=0.).to(tl.float32)
            b_dx += b_dht

    p_dx = tl.make_block_ptr(dx + bos * D, (T, D), (D, 1), (i_t * BT, i_d * BD), (BT, BD), (1, 0))
    tl.store(p_dx, tl.cast(b_dx, dtype=p_dx.dtype.element_ty, fp_downcast_rounding='rtne'), boundary_check=(0, 1))


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['cache'] is not None,
    'HAS_WEIGHT': lambda args: args['weight'] is not None,
    'HAS_BIAS': lambda args: args['bias'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
})
@triton.jit
def causal_conv1d_update_kernel(
    x,
    cache,
    residual,
    y,
    weight,
    bias,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    BW: tl.constexpr,
    ACTIVATION: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)

    o_d = i_d * BD + tl.arange(0, BD)
    o_w = tl.arange(0, BW) + W - BW
    m_d = o_d < D
    m_w = o_w >= 0
    m_c = o_w < W - 1

    # [BD]
    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=0).to(tl.float32)

    if USE_INITIAL_STATE:
        # shift the cache by 1 with the last one being discarded
        p_cache = tl.make_block_ptr(cache + i_n * D*W, (D, W), (W, 1), (i_d * BD, W - BW + 1), (BD, BW), (1, 0))
        # [BD, BW]
        b_cache = tl.load(p_cache, boundary_check=(0, 1)).to(tl.float32)
        b_cache = tl.where(m_c[None, :], b_cache, b_x[:, None])
    else:
        b_cache = tl.zeros((BD, BW), dtype=tl.float32)

    if HAS_WEIGHT:
        b_w = tl.load(weight + o_d[:, None] * W + o_w, mask=m_d[:, None] & m_w, other=0)
        b_y = tl.sum(b_cache * b_w, 1)
    else:
        b_y = tl.sum(b_cache, 1)
    if HAS_BIAS:
        b_y += tl.load(bias + o_d, mask=m_d)

    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * tl.sigmoid(b_y)

    if HAS_RESIDUAL:
        b_y += tl.load(residual + i_n * D + o_d, mask=m_d, other=0)

    tl.store(y + i_n * D + o_d, tl.cast(b_y, dtype=y.dtype.element_ty, fp_downcast_rounding='rtne'), mask=m_d)

    if USE_INITIAL_STATE:
        b_cache = tl.cast(b_cache, dtype=cache.dtype.element_ty, fp_downcast_rounding='rtne')
        # update the cache in-place
        p_cache = tl.make_block_ptr(cache + i_n * D*W, (D, W), (W, 1), (i_d * BD, W - BW), (BD, BW), (1, 0))
        tl.store(p_cache, b_cache, boundary_check=(0, 1))


@input_guard
def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    activation: str | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    B, T, D, W = *x.shape, weight.shape[1]
    BT = min(64, triton.next_power_of_2(triton.cdiv(max(16, B*T), get_multiprocessor_count(x.device.index))))
    BW = triton.next_power_of_2(W)
    if chunk_indices is None and (cu_seqlens is not None or cu_seqlens_cpu is not None):
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    NB = triton.cdiv(B*T, 1024)

    y = torch.empty_like(x)
    def grid(meta): return (triton.cdiv(D, meta['BD']), NT, B)
    causal_conv1d_fwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        residual=residual,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        NB=NB,
        ACTIVATION=activation,
    )
    final_state = None
    if output_final_state:
        final_state = causal_conv1d_update_states(
            x=x,
            state_len=W,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )
    return y.view(shape), final_state


def causal_conv1d_bwd(
    x: torch.Tensor,
    dy: torch.Tensor,
    dht: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
):
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    B, T, D = x.shape
    W = weight.shape[1] if weight is not None else None
    BT = min(64, triton.next_power_of_2(triton.cdiv(max(16, B*T), get_multiprocessor_count(x.device.index))))
    BW = triton.next_power_of_2(W)
    if chunk_indices is None and (cu_seqlens is not None or cu_seqlens_cpu is not None):
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT, cu_seqlens_cpu=cu_seqlens_cpu)
    NT = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T, BT)
    NB = triton.cdiv(B*T, 1024)

    y = None
    if activation is not None:
        y, _ = causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=initial_state,
            activation=None,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            output_final_state=False,
        )
    dx = torch.empty_like(x)
    dw = weight.new_empty(B*NT, *weight.shape, dtype=torch.float) if weight is not None else None
    db = bias.new_empty(B*NT, *bias.shape, dtype=torch.float) if bias is not None else None
    dr = dy if residual is not None else None
    dh0 = initial_state.new_zeros(min(NT, triton.cdiv(W, BT)), *initial_state.shape) if initial_state is not None else None

    def grid(meta): return (triton.cdiv(D, meta['BD']), NT, B)
    causal_conv1d_bwd_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        initial_state=initial_state,
        dh0=dh0,
        dht=dht,
        dy=dy,
        dx=dx,
        dw=dw,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        D=D,
        W=W,
        BT=BT,
        BW=BW,
        NB=NB,
        ACTIVATION=activation,
    )
    if weight is not None:
        dw = dw.sum(0).to(weight)
    if bias is not None:
        db = db.sum(0).to(bias)
    if initial_state is not None:
        dh0 = dh0.sum(0, dtype=torch.float32).to(initial_state)

    return dx.view(shape), dw, db, dr, dh0


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['initial_state'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit
def causal_conv1d_states_fwd_kernel(
    x,
    initial_state,
    final_state,
    cu_seqlens,
    T,
    D,
    W,
    BD: tl.constexpr,
    BW: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_n = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = (i_n * T).to(tl.int64), (i_n * T + T).to(tl.int64)

    o_t = eos - BW + tl.arange(0, BW)
    o_d = i_d * BD + tl.arange(0, BD)
    o_w = W - BW + tl.arange(0, BW)
    m_t = (o_t >= tl.maximum(bos, eos - W))
    m_d = o_d < D
    m_w = (o_w >= 0) & (o_w < W)

    b_x = tl.load(x + o_t * D + o_d[:, None], mask=(m_t & m_d[:, None]), other=0)
    if USE_INITIAL_STATE:
        if T < BW:
            o_c = W - (BW - T) + tl.arange(0, BW)
            m_c = (o_c >= 0) & (o_c < W)
            b_cache = tl.load(initial_state + i_n * D*W + o_d[:, None] * W + o_c, mask=m_d[:, None] & m_c, other=0)
            b_x += b_cache

    tl.store(final_state + i_n * D*W + o_d[:, None] * W + o_w, b_x, mask=m_d[:, None] & m_w)


@input_guard
def causal_conv1d_update_states(
    x: torch.Tensor,
    state_len: int,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    B, T, D, W = *x.shape, state_len
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

    final_state = torch.empty(N, D, W, dtype=x.dtype, device=x.device)
    BD = min(triton.next_power_of_2(D), 256)
    BW = triton.next_power_of_2(W)
    grid = (triton.cdiv(D, BD), N)
    causal_conv1d_states_fwd_kernel[grid](
        x=x,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        T=T,
        D=D,
        W=W,
        BW=BW,
        BD=BD,
    )
    return final_state


@input_guard
def causal_conv1d_update(
    x: torch.Tensor,
    cache: torch.Tensor,
    residual: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    shape = x.shape
    if weight is not None and x.shape[-1] != weight.shape[0]:
        x = rearrange(x, 'b t ... -> b t (...)')
    *_, D = x.shape
    N = x.numel() // D
    W = weight.shape[1] if weight is not None else None
    BD = 8
    BW = triton.next_power_of_2(W)

    y = torch.empty_like(x)
    # NOTE: autotuning is disabled as cache is updated in-place
    def grid(meta): return (triton.cdiv(D, meta['BD']), N)
    causal_conv1d_update_kernel[grid](
        x=x,
        cache=cache,
        residual=residual,
        y=y,
        weight=weight,
        bias=bias,
        D=D,
        W=W,
        BD=BD,
        BW=BW,
        ACTIVATION=activation,
        num_warps=STATIC_WARPS,
    )
    return y.view(shape), cache


class CausalConv1dFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool | None = False,
        activation: str | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
    ):
        ctx.activation = activation
        ctx.cu_seqlens = cu_seqlens
        ctx.cu_seqlens_cpu = cu_seqlens_cpu
        ctx.chunk_indices = chunk_indices
        ctx.save_for_backward(x, weight, bias, residual, initial_state)
        y, final_state = causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            output_final_state=output_final_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
        )
        return y, final_state

    @staticmethod
    @input_guard
    def backward(ctx, dy: torch.Tensor, dht: torch.Tensor | None = None):
        x, weight, bias, residual, initial_state = ctx.saved_tensors
        dx, dw, db, dr, dh0 = causal_conv1d_bwd(
            x=x,
            dy=dy,
            dht=dht,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=initial_state,
            activation=ctx.activation,
            cu_seqlens=ctx.cu_seqlens,
            cu_seqlens_cpu=ctx.cu_seqlens_cpu,
            chunk_indices=ctx.chunk_indices,
        )
        return dx, dw, db, dr, dh0, None, None, None, None, None


class FastCausalConv1dFn(torch.autograd.Function):
    """
    Mixed-mode (Mix) Causal Convolution Implementation - Combining Triton Forward and CUDA Backward Propagation

    This class implements forward propagation using FLA's Triton kernel, while using the optimized
    implementation from TriDao's causal_conv1d CUDA package for backward propagation.
    This hybrid strategy combines the advantages of both technologies:

    - Forward: Uses FLA's Triton implementation, optimized for the FLA framework
    - Backward: Uses TriDao's causal_conv1d_bwd_function CUDA implementation for faster speed

    Performance Benefits:
    - CUDA backward implementation is typically faster than the Triton version, reducing training time
    - Maintains the flexibility and compatibility of forward propagation

    Note:
    - Input/Output format is (batch, seqlen, dim)
    - Backward propagation requires causal_conv1d package: pip install causal-conv1d
    - Supports SILU/Swish activation functions
    - Current limitations (not yet supported):
        * output_final_state must be False
        * initial_states must be None
        * residual must be None
    """
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias=None,
        residual: torch.Tensor | None = None,
        initial_states=None,
        output_final_state=False,
        activation=None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        seq_idx: torch.LongTensor | None = None,
    ):
        if activation not in [None, "silu", "swish"]:
            raise NotImplementedError("activation must be None, silu, or swish")
        assert output_final_state is False, "output_final_state must be False for FastCausalConv1dFn"
        assert initial_states is None, "initial_states must be None for FastCausalConv1dFn"
        assert residual is None, "residual must be None for FastCausalConv1dFn"
        if x.stride(2) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        bias = bias.contiguous() if bias is not None else None
        if cu_seqlens is not None and seq_idx is None:
            seq_idx = prepare_sequence_ids(cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu).to(
                torch.int32).unsqueeze(0)
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None

        ctx.activation = activation in ["silu", "swish"]
        out, _ = causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=None,
            output_final_state=output_final_state,
            activation=activation,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
        )

        ctx.save_for_backward(x, weight, bias, seq_idx, initial_states)
        ctx.return_final_states = output_final_state
        ctx.return_dinitial_states = (
            initial_states is not None and initial_states.requires_grad
        )
        return out, None

    @staticmethod
    def backward(ctx, dout, *args):
        x, weight, bias, seq_idx, initial_states = ctx.saved_tensors
        x = rearrange(x, 'b t d -> b d t')
        dout = rearrange(dout, 'b t d -> b d t')
        dfinal_states = args[0] if ctx.return_final_states else None

        if dout.stride(2) != 1 and dout.stride(1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        # Here we just pass in None and dx will be allocated in the C++ code.
        dx, dweight, dbias, dinitial_states = causal_conv1d_bwd_function(
            x,
            weight,
            bias,
            dout,
            seq_idx,
            initial_states,
            dfinal_states,
            None,
            ctx.return_dinitial_states,
            ctx.activation,
        )
        dx = rearrange(dx, 'b d t -> b t d')
        return (
            dx,
            dweight,
            dbias if bias is not None else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fast_causal_conv1d_fn(
    x,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool | None = False,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    seq_idx: torch.LongTensor | None = None,
):
    """
    x: (batch, seqlen, dim)
    weight: (dim, width)
    bias: (dim,)
    seq_idx: (batch, seqlen)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1), to be written to
    activation: either None or "silu" or "swish"

    out: (batch, seqlen, dim)
    """
    return FastCausalConv1dFn.apply(
        x,
        weight,
        bias,
        residual,
        initial_state,
        output_final_state,
        activation,
        cu_seqlens,
        cu_seqlens_cpu,
        chunk_indices,
        seq_idx,
    )


@input_guard
def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool | None = False,
    activation: str | None = None,
    backend: str | None = 'triton',
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    **kwargs,
):
    """
    A causal 1D convolution implementation that powers Mamba/Mamba2 and DeltaNet architectures.

    When a residual connection is provided, this implements the Canon operation
    described in the paper at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5240330.

    Args:
        x (torch.Tensor):
            Input tensor of shape [B, T, D].
        weight (Optional[torch.Tensor]):
            Weight tensor of shape [D, W]. Default: `None`.
        bias (Optional[torch.Tensor]):
            Bias tensor of shape [D]. Default: `None`.
        residual (Optional[torch.Tensor]):
            Residual tensor of shape [B, T, D]. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state tensor of shape [N, D, W],
            where `N` is the number of sequences in the batch and `W` is the kernel size.
            If provided, the initial state is used to initialize the cache. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape [N, D, W]. Default: `False`.
        activation (Optional[str]):
            Activations applied to output, only `swish`/`silu` or `None` (i.e., no activation) are supported.
            Default: `None`.
        backend (Optional[str]):
            Specifies the backend to use for the convolution operation. Supported values are `'cuda'` 、 `'triton'` and `'mix'`.
            Default: `'triton'`.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths (optional)
        chunk_indices (Optional[torch.LongTensor]):
            Chunk indices for variable-length sequences (optional)

    Returns:
        Tuple of (output, final_state).
        If `output_final_state` is `False`, the final state is `None`.
    """

    if backend == 'triton':
        y, final_state = CausalConv1dFunction.apply(
            x,
            weight,
            bias,
            residual,
            initial_state,
            output_final_state,
            activation,
            cu_seqlens,
            cu_seqlens_cpu,
            chunk_indices,
        )
        return y, final_state
    elif backend == 'mix':
        if causal_conv1d_bwd_function is None:
            raise ImportError(
                "causal_conv1d is required for backend='mix', but it is not installed. "
                "Please install it with: pip install causal-conv1d\n"
                "For more details, see: https://github.com/Dao-AILab/causal-conv1d"
            )
        seq_idx = kwargs.get('seq_idx')
        return fast_causal_conv1d_fn(
            x,
            weight,
            bias,
            residual,
            initial_state,
            output_final_state,
            activation,
            cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            seq_idx=seq_idx,
        )
    B, _, D, W = *x.shape, weight.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    x = rearrange(x, 'b t d -> b d t')

    # check if cu_seqlens and cache are both provided
    # Sequence index for each token. Used for varlen.
    # Suppose a batch consists of two sequences with lengths 3 and 4,
    # seq_idx=[0, 0, 0, 1, 1, 1, 1] for this batch.
    # NOTE: No need to provide this arg if `cu_seqlens` is passed.
    # This arg is just for BC, and will be removed in the future.
    # [B, T]
    seq_idx = kwargs.get('seq_idx')
    if cu_seqlens is not None and seq_idx is None:
        seq_idx = prepare_sequence_ids(cu_seqlens).to(torch.int32).unsqueeze(0)

    # equivalent to:
    # y = _conv_forward(x, weight, bias)[..., :x.shape[-1]]
    # if activation is not None:
    #     y = ACT2FN[activation](x)

    cache, initial_state = initial_state, None
    if cache is not None:
        # To make causal-conv1d happy
        initial_state = (
            cache[:, :, -(W-1):]   # [N, D, W-1]
            .transpose(1, 2).contiguous()  # [N, W-1, D] and stride(2)==1
            .transpose(1, 2)               # [N, D, W-1] and stride(1)==1
        )

    result = causal_conv1d_fn(
        x=x,
        weight=weight,
        bias=bias,
        activation=activation,
        seq_idx=seq_idx,
        initial_states=initial_state,
        return_final_states=output_final_state,
    )
    y, final_state = result if output_final_state else (result, None)
    y = rearrange(y, 'b d t -> b t d')
    if output_final_state:
        cache = x.new_zeros(N, D, W)
        cache[:, :, -W+1:].copy_(final_state[:, :, -W+1:])
    if residual is not None:
        y.add_(residual)

    return y, cache


class ShortConvolution(nn.Conv1d):
    """Short convolution layer for efficient causal convolution operations.

    This class implements a depthwise separable 1D convolution with causal padding,
    designed for efficient sequence processing. It supports multiple backends (Triton/CUDA)
    and optional activation functions.

    Args:
        hidden_size (int): Number of input/output channels (must be equal for depthwise conv)
        kernel_size (int): Size of the convolution kernel
        bias (bool, optional): Whether to include learnable bias. Defaults to False.
        activation (Optional[str], optional): Activation function ('silu' or 'swish'). Defaults to 'silu'.
        backend (Optional[str], optional): Backend implementation ('triton' or 'cuda'). Defaults to 'triton'.
        device (Optional[torch.device], optional): Device to place the layer on. Defaults to None.
        dtype (Optional[torch.dtype], optional): Data type for layer parameters. Defaults to None.
        **kwargs: Additional keyword arguments (deprecated 'use_fast_conv1d' supported for compatibility)

    Attributes:
        hidden_size (int): Number of channels
        activation (Optional[str]): Selected activation function
        backend (str): Actual backend being used (may differ from input due to availability)

    Note:
        - Uses depthwise convolution (groups=hidden_size) for efficiency
        - Applies causal padding (kernel_size-1) to ensure no future information leakage
        - Falls back to Triton backend if CUDA backend is unavailable
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = 'silu',
        backend: str | None = 'triton',
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )

        self.hidden_size = hidden_size
        self.activation = None

        if activation is not None:
            assert activation in ['silu', 'swish'], f"Activation `{activation}` not supported yet."
            self.activation = activation

        if 'use_fast_conv1d' in kwargs:
            warnings.warn(
                "The `use_fast_conv1d` parameter is deprecated and will be ignored. "
                "Please use the `backend` parameter instead.",
            )
        import os
        self.backend = os.environ.get('FLA_CONV_BACKEND', backend)
        if backend not in ['cuda', 'triton']:
            raise ValueError(f"Invalid backend: {backend}, must be one of ['cuda', 'triton']")
        if backend == 'cuda':
            if causal_conv1d_fn is None:
                warnings.warn(
                    "The `backend` parameter is set to `cuda`, but `causal_conv1d_fn` is not available. "
                    "Switching to the Triton implementation instead. "
                    "Consider installing `causal_conv1d` to enable the CUDA backend.",
                )
                self.backend = 'triton'

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.activation is not None:
            s += ', activation={activation}'
        s += f', backend={self.backend}'
        return s.format(**self.__dict__)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]`. `B` must be 1 if `cu_seqlens` is provided.
            residual (`Optional[torch.Tensor]`):
                Residual tensor of shape `[B, T, D]`. Default: `None`.
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.
            cu_seqlens (Optional[torch.LongTensor]):
                Cumulative sequence lengths for each batch. Used for varlen. Default: `None`.
                Shape: [B+1]
            chunk_indices (Optional[torch.LongTensor]):
                Chunk indices for variable-length sequences. Default: `None`.

        Returns:
            Tensor of shape `[B, T, D]`.
        """

        B, T, *_ = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
            x = x.mul_(mask.unsqueeze(-1))

        # in decoding phase, the cache (if provided) is updated inplace
        if B * T == N:
            y, cache = self.step(
                x=x,
                residual=residual,
                cache=cache,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )
            return y, cache

        # cuda backend do not support:
        # 1. both `cu_seqlens` and `cache` being provided
        # 2. both `cu_seqlens` and `output_final_state` being provided
        if self.backend == 'cuda' and (
            (cu_seqlens is not None and cache is not None) or
            (cu_seqlens is not None and output_final_state)
        ):
            warnings.warn(
                "The CUDA backend does not support both `cu_seqlens` and `cache` being provided, "
                "or both `cu_seqlens` and `output_final_state` being provided. "
                "Switching to the Triton backend instead. ",
                stacklevel=2,
            )
            self.backend = 'triton'

        return causal_conv1d(
            x=x,
            weight=rearrange(self.weight, "d 1 w -> d w"),
            bias=self.bias,
            residual=residual,
            initial_state=cache,
            output_final_state=output_final_state,
            activation=self.activation,
            backend=self.backend,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            **kwargs,
        )

    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        cache: torch.Tensor,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        B, _, D, W = *x.shape, self.kernel_size[0]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if output_final_state and cache is None:
            cache = x.new_zeros(N, D, W)
        # NOTE: we follow the fast mode that updates the cache in-place
        if self.backend == 'triton':
            return causal_conv1d_update(
                x=x,
                cache=cache,
                residual=residual,
                weight=rearrange(self.weight, "d 1 w -> d w"),
                bias=self.bias,
                activation=self.activation,
            )

        shape = x.shape
        x = x.squeeze(0) if cu_seqlens is not None else x.squeeze(1)
        # equivalent to:
        # cache.copy_(cache.roll(shifts=-1, dims=-1))
        # cache[:, :, -1] = x
        # y = torch.sum(cache * rearrange(self.weight, "d 1 w -> d w"), dim=-1)
        y = causal_conv1d_update_cuda(
            x=x,
            conv_state=cache,
            weight=rearrange(self.weight, "d 1 w -> d w"),
            bias=self.bias,
            activation=self.activation,
        )
        y = y.view(shape)
        if residual is not None:
            y.add_(residual)
        return y, cache

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size


def fft_conv(u, k, dropout_mask, gelu=True, k_rev=None):
    seqlen = u.shape[-1]
    fft_size = 2 * seqlen
    k_f = torch.fft.rfft(k, n=fft_size) / fft_size
    if k_rev is not None:
        k_rev_f = torch.fft.rfft(k_rev, n=fft_size) / fft_size
        k_f = k_f + k_rev_f.conj()
    u_f = torch.fft.rfft(u.to(dtype=k.dtype), n=fft_size)

    if len(u.shape) > 3:
        k_f = k_f.unsqueeze(1)
    y = torch.fft.irfft(u_f * k_f, n=fft_size, norm="forward")[..., :seqlen]

    out = y + u
    if gelu:
        out = F.gelu(out)
    if dropout_mask is not None:
        return (out * rearrange(dropout_mask, "b H -> b H 1")).to(dtype=u.dtype)
    else:
        return out.to(dtype=u.dtype)


class LongConvolution(nn.Module):
    """
    LongConvolution applies a convolution operation on the input tensor using a fixed
    filter of length max_len.
    The filter is learned during training and is applied using FFT convolution.

    Args:
        hidden_size (int): The number of expected features in the input and output.
        max_len (int): The maximum sequence length.

    Returns:
        y: [batch_size, seq_len, hidden_size] tensor
    """

    def __init__(
        self,
        hidden_size: int,
        max_len: int,
        **kwargs,
    ):
        """
        Initializes the LongConvolution module.
        Args:
            hidden_size (int): The number of expected features in the input and output.
            max_len (int): The maximum sequence length.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.filter = nn.Parameter(torch.randn(self.hidden_size, max_len), requires_grad=True)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Applies the LongConvolution operation on the input tensor.
        Args:
            x: [batch_size, seq_len, hidden_size] tensor
        Returns:
            y: [batch_size, seq_len, hidden_size] tensor
        """
        x = x.transpose(1, 2)
        y = fft_conv(x, self.filter, dropout_mask=None, gelu=False)
        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)


class PositionalEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seq_len: int, **kwargs):
        """Complex exponential positional embeddings for implicit long convolution filters."""
        super().__init__()

        self.seq_len = seq_len
        # The time embedding fed to the filteres is normalized so that t_f = 1
        t = torch.linspace(0, 1, self.seq_len)[None, :, None]  # 1, L, 1

        if emb_dim > 1:
            bands = (emb_dim - 1) // 2
        # To compute the right embeddings we use the "proper" linspace
        t_rescaled = torch.linspace(0, seq_len - 1, seq_len)[None, :, None]
        w = 2 * math.pi * t_rescaled / seq_len  # 1, L, 1

        f = torch.linspace(1e-4, bands - 1, bands)[None, None]
        z = torch.exp(-1j * f * w)
        z = torch.cat([t, z.real, z.imag], dim=-1)
        self.z = nn.Parameter(z, requires_grad=False)

    def forward(self, L):
        return self.z[:, :L]


class ImplicitLongConvolution(nn.Module):
    """
    Long convolution with implicit filter parameterized by an MLP.

    Args:
        hidden_size (int):
            The number of expected features in the input and output.
        max_len (int):
            The maximum sequence length.
        d_emb (Optional[int]):
            The dimension of the positional embeddings. Must be odd and greater or equal to 3 (time, sine and cosine).
            Defaults to 3.
        d_hidden (Optional[int]):
            The number of features in the hidden layer of the MLP. Defaults to 16.

    Attributes:
        pos_emb (`PositionalEmbedding`): The positional embedding layer.
        mlp (`nn.Sequential`): The MLP that parameterizes the implicit filter.

    """

    def __init__(
        self,
        hidden_size: int,
        max_len: int,
        d_emb: int = 3,
        d_hidden: int = 16,
        **kwargs,
    ):
        """
        Long convolution with implicit filter parameterized by an MLP.


        """
        super().__init__()
        self.hidden_size = hidden_size
        self.d_emb = d_emb

        assert (
            d_emb % 2 != 0 and d_emb >= 3
        ), "d_emb must be odd and greater or equal to 3 (time, sine and cosine)"
        self.pos_emb = PositionalEmbedding(d_emb, max_len)

        # final linear layer
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, d_hidden),
            torch.nn.ReLU(),
            nn.Linear(d_hidden, hidden_size),
        )

    def filter(self, seq_len: int, *args, **kwargs):
        return self.mlp(self.pos_emb(seq_len)).transpose(1, 2)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x: [batch_size, seq_len, hidden_size] tensor

        Returns:
            y: [batch_size, seq_len, hidden_size] tensor
        """
        x = x.transpose(1, 2)
        k = self.filter(x.shape[-1])
        y = fft_conv(x, k, dropout_mask=None, gelu=False)

        y = y.transpose(1, 2)
        return y.to(dtype=x.dtype)

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl



@triton.heuristics({
    'STORE_RESIDUAL_OUT': lambda args: args['residual_out'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'HAS_WEIGHT': lambda args: args['w'] is not None,
    'HAS_BIAS': lambda args: args['b'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [4, 8, 16]
    ],
    key=['D', 'NB', 'IS_RMS_NORM', 'STORE_RESIDUAL_OUT', 'HAS_RESIDUAL', 'HAS_WEIGHT'],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_gated_fwd_kernel(
    x,  # pointer to the input
    g,  # pointer to the gate
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    residual,  # pointer to the residual
    residual_out,  # pointer to the residual
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    T,  # number of rows in x
    D: tl.constexpr,  # number of columns in x
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0)

    o_d = tl.arange(0, BD)
    m_d = o_d < D

    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    if HAS_RESIDUAL:
        p_res = tl.make_block_ptr(residual, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        b_x += tl.load(p_res, boundary_check=(0, 1)).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        p_res_out = tl.make_block_ptr(residual_out, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
        tl.store(p_res_out, b_x.to(p_res_out.dtype.element_ty), boundary_check=(0, 1))
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=1) / D
        p_mean = tl.make_block_ptr(mean, (T,), (1,), (i_t * BT,), (BT,), (0,))
        tl.store(p_mean, b_mean.to(p_mean.dtype.element_ty), boundary_check=(0,))
        b_xbar = tl.where(m_d[None, :], b_x - b_mean[:, None], 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    else:
        b_xbar = tl.where(m_d[None, :], b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=1) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)

    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean[:, None]) * b_rstd[:, None] if not IS_RMS_NORM else b_x * b_rstd[:, None]
    b_y = b_x_hat * b_w[None, :] if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b[None, :]

    # swish/sigmoid output gate
    p_g = tl.make_block_ptr(g, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == 'sigmoid':
        b_y = b_y * tl.sigmoid(b_g)

    # Write output
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'STORE_RESIDUAL_OUT': lambda args: args['residual_out'] is not None,
    'HAS_RESIDUAL': lambda args: args['residual'] is not None,
    'HAS_WEIGHT': lambda args: args['w'] is not None,
    'HAS_BIAS': lambda args: args['b'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8, 16]
    ],
    key=['D', 'IS_RMS_NORM', 'STORE_RESIDUAL_OUT', 'HAS_RESIDUAL', 'HAS_WEIGHT'],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_gated_fwd_kernel1(
    x,  # pointer to the input
    g,  # pointer to the gate
    y,  # pointer to the output
    w,  # pointer to the weights
    b,  # pointer to the biases
    residual,  # pointer to the residual
    residual_out,  # pointer to the residual
    mean,  # pointer to the mean
    rstd,  # pointer to the 1/std
    eps,  # epsilon to avoid division by zero
    D: tl.constexpr,  # number of columns in x
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t = tl.program_id(0)
    x += i_t * D
    y += i_t * D
    g += i_t * D
    if HAS_RESIDUAL:
        residual += i_t * D
    if STORE_RESIDUAL_OUT:
        residual_out += i_t * D

    o_d = tl.arange(0, BD)
    m_d = o_d < D
    b_x = tl.load(x + o_d, mask=m_d, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        b_x += tl.load(residual + o_d, mask=m_d, other=0.0).to(tl.float32)
    if STORE_RESIDUAL_OUT:
        tl.store(residual_out + o_d, b_x, mask=m_d)
    if not IS_RMS_NORM:
        b_mean = tl.sum(b_x, axis=0) / D
        tl.store(mean + i_t, b_mean)
        b_xbar = tl.where(m_d, b_x - b_mean, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    else:
        b_xbar = tl.where(m_d, b_x, 0.0)
        b_var = tl.sum(b_xbar * b_xbar, axis=0) / D
    b_rstd = 1 / tl.sqrt(b_var + eps)
    tl.store(rstd + i_t, b_rstd)

    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d).to(tl.float32)
    b_x_hat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
    b_y = b_x_hat * b_w if HAS_WEIGHT else b_x_hat
    if HAS_BIAS:
        b_y = b_y + b_b

    # swish/sigmoid output gate
    b_g = tl.load(g + o_d, mask=m_d, other=0.0).to(tl.float32)
    if ACTIVATION == 'swish' or ACTIVATION == 'silu':
        b_y = b_y * b_g * tl.sigmoid(b_g)
    elif ACTIVATION == 'sigmoid':
        b_y = b_y * tl.sigmoid(b_g)

    # Write output
    tl.store(y + o_d, b_y, mask=m_d)


@triton.heuristics({
    'HAS_DRESIDUAL': lambda args: args['dresidual'] is not None,
    'HAS_WEIGHT': lambda args: args['w'] is not None,
    'HAS_BIAS': lambda args: args['b'] is not None,
    'RECOMPUTE_OUTPUT': lambda args: args['y'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BT': BT}, num_warps=num_warps)
        for BT in [16, 32, 64]
        for num_warps in [4, 8, 16]
    ],
    key=['D', 'NB', 'IS_RMS_NORM', 'HAS_DRESIDUAL', 'HAS_WEIGHT'],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_gated_bwd_kernel(
    x,  # pointer to the input
    g,  # pointer to the gate
    w,  # pointer to the weights
    b,  # pointer to the biases
    y,  # pointer to the output to be recomputed
    dy,  # pointer to the output gradient
    dx,  # pointer to the input gradient
    dg,  # pointer to the gate gradient
    dw,  # pointer to the partial sum of weights gradient
    db,  # pointer to the partial sum of biases gradient
    dresidual,
    dresidual_in,
    mean,
    rstd,
    T,
    BS,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    i_s = tl.program_id(0)
    o_d = tl.arange(0, BD)
    m_d = o_d < D
    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=m_d).to(tl.float32)
        b_dw = tl.zeros((BT, BD), dtype=tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=m_d, other=0.0).to(tl.float32)
        b_db = tl.zeros((BT, BD), dtype=tl.float32)

    T = min(i_s * BS + BS, T)
    for i_t in range(i_s * BS, T, BT):
        p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
        p_g = tl.make_block_ptr(g, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
        p_dy = tl.make_block_ptr(dy, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
        p_dx = tl.make_block_ptr(dx, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
        p_dg = tl.make_block_ptr(dg, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
        # [BT, BD]
        b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_dy = tl.load(p_dy, boundary_check=(0, 1)).to(tl.float32)

        if not IS_RMS_NORM:
            p_mean = tl.make_block_ptr(mean, (T,), (1,), (i_t,), (BT,), (0,))
            b_mean = tl.load(p_mean, boundary_check=(0,))
        p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t,), (BT,), (0,))
        b_rstd = tl.load(p_rstd, boundary_check=(0,))
        # Compute dx
        b_xhat = (b_x - b_mean[:, None]) * b_rstd[:, None] if not IS_RMS_NORM else b_x * b_rstd[:, None]
        b_xhat = tl.where(m_d[None, :], b_xhat, 0.0)

        b_y = b_xhat * b_w[None, :] if HAS_WEIGHT else b_xhat
        if HAS_BIAS:
            b_y = b_y + b_b[None, :]
        if RECOMPUTE_OUTPUT:
            p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
            tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))

        b_sigmoid_g = tl.sigmoid(b_g)
        if ACTIVATION == 'swish' or ACTIVATION == 'silu':
            b_dg = b_dy * b_y * (b_sigmoid_g + b_g * b_sigmoid_g * (1 - b_sigmoid_g))
            b_dy = b_dy * b_g * b_sigmoid_g
        elif ACTIVATION == 'sigmoid':
            b_dg = b_dy * b_y * b_sigmoid_g * (1 - b_sigmoid_g)
            b_dy = b_dy * b_sigmoid_g
        b_wdy = b_dy

        if HAS_WEIGHT or HAS_BIAS:
            m_t = (i_t + tl.arange(0, BT)) < T
        if HAS_WEIGHT:
            b_wdy = b_dy * b_w
            b_dw += tl.where(m_t[:, None], b_dy * b_xhat, 0.0)
        if HAS_BIAS:
            b_db += tl.where(m_t[:, None], b_dy, 0.0)
        if not IS_RMS_NORM:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=1) / D
            b_c2 = tl.sum(b_wdy, axis=1) / D
            b_dx = (b_wdy - (b_xhat * b_c1[:, None] + b_c2[:, None])) * b_rstd[:, None]
        else:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=1) / D
            b_dx = (b_wdy - b_xhat * b_c1[:, None]) * b_rstd[:, None]
        if HAS_DRESIDUAL:
            p_dres = tl.make_block_ptr(dresidual, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
            b_dres = tl.load(p_dres, boundary_check=(0, 1)).to(tl.float32)
            b_dx += b_dres
        # Write dx
        if STORE_DRESIDUAL:
            p_dres_in = tl.make_block_ptr(dresidual_in, (T, D), (D, 1), (i_t, 0), (BT, BD), (1, 0))
            tl.store(p_dres_in, b_dx.to(p_dres_in.dtype.element_ty), boundary_check=(0, 1))

        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), boundary_check=(0, 1))
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))

    if HAS_WEIGHT:
        tl.store(dw + i_s * D + o_d, tl.sum(b_dw, axis=0), mask=m_d)
    if HAS_BIAS:
        tl.store(db + i_s * D + o_d, tl.sum(b_db, axis=0), mask=m_d)


@triton.heuristics({
    'HAS_DRESIDUAL': lambda args: args['dresidual'] is not None,
    'HAS_WEIGHT': lambda args: args['w'] is not None,
    'HAS_BIAS': lambda args: args['b'] is not None,
    'RECOMPUTE_OUTPUT': lambda args: args['y'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [2, 4, 8, 16]
    ],
    key=['D', 'IS_RMS_NORM', 'STORE_DRESIDUAL', 'HAS_DRESIDUAL', 'HAS_WEIGHT'],
    **autotune_cache_kwargs,
)
@triton.jit
def layer_norm_gated_bwd_kernel1(
    x,  # pointer to the input
    g,  # pointer to the gate
    w,  # pointer to the weights
    b,  # pointer to the biases
    y,  # pointer to the output to be recomputed
    dy,  # pointer to the output gradient
    dx,  # pointer to the input gradient
    dg,  # pointer to the gate gradient
    dw,  # pointer to the partial sum of weights gradient
    db,  # pointer to the partial sum of biases gradient
    dresidual,
    dresidual_in,
    mean,
    rstd,
    T,
    BS,
    D: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    i_s = tl.program_id(0)
    o_d = tl.arange(0, BD)
    mask = o_d < D
    x += i_s * BS * D
    g += i_s * BS * D
    if HAS_DRESIDUAL:
        dresidual += i_s * BS * D
    if STORE_DRESIDUAL:
        dresidual_in += i_s * BS * D
    dy += i_s * BS * D
    dx += i_s * BS * D
    dg += i_s * BS * D
    if RECOMPUTE_OUTPUT:
        y += i_s * BS * D
    if HAS_WEIGHT:
        b_w = tl.load(w + o_d, mask=mask).to(tl.float32)
        b_dw = tl.zeros((BD,), dtype=tl.float32)
    if HAS_BIAS:
        b_b = tl.load(b + o_d, mask=mask, other=0.0).to(tl.float32)
        b_db = tl.zeros((BD,), dtype=tl.float32)

    for i_t in range(i_s * BS, min(i_s * BS + BS, T)):
        # Load data to SRAM
        b_x = tl.load(x + o_d, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(g + o_d, mask=mask, other=0).to(tl.float32)
        b_dy = tl.load(dy + o_d, mask=mask, other=0).to(tl.float32)

        if not IS_RMS_NORM:
            b_mean = tl.load(mean + i_t)
        b_rstd = tl.load(rstd + i_t)
        # Compute dx
        b_xhat = (b_x - b_mean) * b_rstd if not IS_RMS_NORM else b_x * b_rstd
        b_xhat = tl.where(mask, b_xhat, 0.0)

        b_y = b_xhat * b_w if HAS_WEIGHT else b_xhat
        if HAS_BIAS:
            b_y = b_y + b_b
        if RECOMPUTE_OUTPUT:
            tl.store(y + o_d, b_y, mask=mask)

        b_sigmoid_g = tl.sigmoid(b_g)
        if ACTIVATION == 'swish' or ACTIVATION == 'silu':
            b_dg = b_dy * b_y * (b_sigmoid_g + b_g * b_sigmoid_g * (1 - b_sigmoid_g))
            b_dy = b_dy * b_g * b_sigmoid_g
        elif ACTIVATION == 'sigmoid':
            b_dg = b_dy * b_y * b_sigmoid_g * (1 - b_sigmoid_g)
            b_dy = b_dy * b_sigmoid_g
        b_wdy = b_dy
        if HAS_WEIGHT:
            b_wdy = b_dy * b_w
            b_dw += b_dy * b_xhat
        if HAS_BIAS:
            b_db += b_dy
        if not IS_RMS_NORM:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=0) / D
            b_c2 = tl.sum(b_wdy, axis=0) / D
            b_dx = (b_wdy - (b_xhat * b_c1 + b_c2)) * b_rstd
        else:
            b_c1 = tl.sum(b_xhat * b_wdy, axis=0) / D
            b_dx = (b_wdy - b_xhat * b_c1) * b_rstd
        if HAS_DRESIDUAL:
            b_dres = tl.load(dresidual + o_d, mask=mask, other=0).to(tl.float32)
            b_dx += b_dres
        # Write dx
        if STORE_DRESIDUAL:
            tl.store(dresidual_in + o_d, b_dx, mask=mask)
        tl.store(dx + o_d, b_dx, mask=mask)
        tl.store(dg + o_d, b_dg, mask=mask)

        x += D
        g += D
        if HAS_DRESIDUAL:
            dresidual += D
        if STORE_DRESIDUAL:
            dresidual_in += D
        if RECOMPUTE_OUTPUT:
            y += D
        dy += D
        dx += D
        dg += D
    if HAS_WEIGHT:
        tl.store(dw + i_s * D + o_d, b_dw, mask=mask)
    if HAS_BIAS:
        tl.store(db + i_s * D + o_d, b_db, mask=mask)


def layer_norm_gated_fwd(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = 'swish',
    eps: float = 1e-5,
    residual: torch.Tensor = None,
    out_dtype: torch.dtype = None,
    residual_dtype: torch.dtype = None,
    is_rms_norm: bool = False,
):
    if residual is not None:
        residual_dtype = residual.dtype
    T, D = x.shape
    if residual is not None:
        assert residual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)
    # allocate output
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(T, D, device=x.device, dtype=residual_dtype)
    else:
        residual_out = None
    mean = torch.empty((T,), dtype=torch.float, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((T,), dtype=torch.float, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps

    if D <= 512:
        NB = triton.cdiv(T, 2048)
        def grid(meta): return (triton.cdiv(T, meta['BT']),)
        layer_norm_gated_fwd_kernel[grid](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            T=T,
            D=D,
            BD=BD,
            NB=NB,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
        )
    else:
        layer_norm_gated_fwd_kernel1[(T,)](
            x=x,
            g=g,
            y=y,
            w=weight,
            b=bias,
            residual=residual,
            residual_out=residual_out,
            mean=mean,
            rstd=rstd,
            eps=eps,
            D=D,
            BD=BD,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
        )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    return y, mean, rstd, residual_out if residual_out is not None else x


def layer_norm_gated_bwd(
    dy: torch.Tensor,
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = 'swish',
    eps: float = 1e-5,
    mean: torch.Tensor = None,
    rstd: torch.Tensor = None,
    dresidual: torch.Tensor = None,
    has_residual: bool = False,
    is_rms_norm: bool = False,
    x_dtype: torch.dtype = None,
    recompute_output: bool = False,
):
    T, D = x.shape
    assert dy.shape == (T, D)
    if dresidual is not None:
        assert dresidual.shape == (T, D)
    if weight is not None:
        assert weight.shape == (D,)
    if bias is not None:
        assert bias.shape == (D,)
    # allocate output
    dx = torch.empty_like(x) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x.device)
    dg = torch.empty_like(g) if x_dtype is None else torch.empty(T, D, dtype=x_dtype, device=x.device)
    dresidual_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(T, D, dtype=dy.dtype, device=dy.device) if recompute_output else None

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BD = min(MAX_FUSED_SIZE, triton.next_power_of_2(D))
    if D > BD:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    NS = get_multiprocessor_count(x.device.index)
    BS = math.ceil(T / NS)

    dw = torch.empty((NS, D), dtype=torch.float, device=weight.device) if weight is not None else None
    db = torch.empty((NS, D), dtype=torch.float, device=bias.device) if bias is not None else None
    grid = (NS,)

    if D <= 512:
        NB = triton.cdiv(T, 2048)
        layer_norm_gated_bwd_kernel[grid](
            x=x,
            g=g,
            w=weight,
            b=bias,
            y=y,
            dy=dy,
            dx=dx,
            dg=dg,
            dw=dw,
            db=db,
            dresidual=dresidual,
            dresidual_in=dresidual_in,
            mean=mean,
            rstd=rstd,
            T=T,
            D=D,
            BS=BS,
            BD=BD,
            NB=NB,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            STORE_DRESIDUAL=dresidual_in is not None,
        )
    else:
        layer_norm_gated_bwd_kernel1[grid](
            x=x,
            g=g,
            w=weight,
            b=bias,
            y=y,
            dy=dy,
            dx=dx,
            dg=dg,
            dw=dw,
            db=db,
            dresidual=dresidual,
            dresidual_in=dresidual_in,
            mean=mean,
            rstd=rstd,
            T=T,
            D=D,
            BS=BS,
            BD=BD,
            ACTIVATION=activation,
            IS_RMS_NORM=is_rms_norm,
            STORE_DRESIDUAL=dresidual_in is not None,
        )
    dw = dw.sum(0).to(weight.dtype) if weight is not None else None
    db = db.sum(0).to(bias.dtype) if bias is not None else None
    # Don't need to compute dresidual_in separately in this case
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dg, dw, db, dresidual_in) if not recompute_output else (dx, dg, dw, db, dresidual_in, y)


class LayerNormGatedFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        activation: str,
        residual: torch.Tensor | None = None,
        eps: float = 1e-6,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        is_rms_norm: bool = False,
    ):
        x_shape_og = x.shape
        g_shape_og = g.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        g = g.reshape(-1, g.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float if residual_in_fp32 else None)
        )
        y, mean, rstd, residual_out = layer_norm_gated_fwd(
            x=x,
            g=g,
            weight=weight,
            bias=bias,
            activation=activation,
            eps=eps,
            residual=residual,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
        )
        ctx.save_for_backward(residual_out, g, weight, bias, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.g_shape_og = g_shape_og
        ctx.activation = activation
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, residual_out.reshape(x_shape_og))

    @staticmethod
    @input_guard
    def backward(ctx, dy, *args):
        x, g, weight, bias, mean, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dg, dw, db, dres_in = layer_norm_gated_bwd(
            dy=dy,
            x=x,
            g=g,
            weight=weight,
            bias=bias,
            activation=ctx.activation,
            eps=ctx.eps,
            mean=mean,
            rstd=rstd,
            dresidual=dresidual,
            has_residual=ctx.has_residual,
            is_rms_norm=ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
        )
        return (
            dx.reshape(ctx.x_shape_og),
            dg.reshape(ctx.g_shape_og),
            dw,
            db,
            None,
            dres_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )


class LayerNormGatedLinearFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(
        ctx,
        x: torch.Tensor,
        g: torch.Tensor,
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        linear_weight: torch.Tensor,
        linear_bias: torch.Tensor,
        residual: torch.Tensor | None = None,
        eps: float = 1e-6,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
        is_rms_norm: bool = False,
    ):
        x_shape_og = x.shape
        g_shape_og = g.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        g = g.reshape(-1, g.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float if residual_in_fp32 else None)
        )
        y, mean, rstd, residual_out = layer_norm_gated_fwd(
            x=x,
            g=g,
            weight=norm_weight,
            bias=norm_bias,
            eps=eps,
            residual=residual,
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
        )
        y = y.reshape(x_shape_og)
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else y.dtype
        linear_weight = linear_weight.to(dtype)
        linear_bias = linear_bias.to(dtype) if linear_bias is not None else None
        out = F.linear(y.to(linear_weight.dtype), linear_weight, linear_bias)
        # We don't store y, will be recomputed in the backward pass to save memory
        ctx.save_for_backward(residual_out, g, norm_weight, norm_bias, linear_weight, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.g_shape_og = g_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, residual_out.reshape(x_shape_og))

    @staticmethod
    @input_guard
    def backward(ctx, dout, *args):
        x, g, norm_weight, norm_bias, linear_weight, mean, rstd = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dy = F.linear(dout, linear_weight.t())
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dg, dnorm_weight, dnorm_bias, dres_in, y = layer_norm_gated_bwd(
            dy=dy,
            x=x,
            g=g,
            weight=norm_weight,
            bias=norm_bias,
            eps=ctx.eps,
            mean=mean,
            rstd=rstd,
            dresidual=dresidual,
            has_residual=ctx.has_residual,
            is_rms_norm=ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
            recompute_output=True,
        )
        dlinear_weight = torch.einsum("bo,bi->oi", dout, y)
        return (
            dx.reshape(ctx.x_shape_og),
            dg.reshape(ctx.g_shape_og),
            dnorm_weight,
            dnorm_bias,
            dlinear_weight,
            dlinear_bias,
            dres_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )


def layer_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = 'swish',
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
):
    return LayerNormGatedFunction.apply(
        x,
        g,
        weight,
        bias,
        activation,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        False,
    )


def rms_norm_gated(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    activation: str = 'swish',
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
):
    return LayerNormGatedFunction.apply(
        x,
        g,
        weight,
        bias,
        activation,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        True,
    )


def layer_norm_swish_gate_linear(
    x: torch.Tensor,
    g: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
):
    return LayerNormGatedLinearFunction.apply(
        x,
        g,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        False,
    )


def rms_norm_swish_gate_linear(
    x,
    g: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    linear_weight: torch.Tensor,
    linear_bias: torch.Tensor,
    residual: torch.Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
):
    return LayerNormGatedLinearFunction.apply(
        x,
        g,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        True,
    )


class FusedLayerNormGated(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        activation: str = 'swish',
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> FusedLayerNormGated:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation

        if self.activation not in ['swish', 'silu', 'sigmoid']:
            raise ValueError(f"Unsupported activation: {self.activation}")

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += f", activation={self.activation}"
        s += ")"
        return s

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        return layer_norm_gated(
            x,
            g,
            self.weight,
            self.bias,
            self.activation,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


class FusedRMSNormGated(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        activation: str = 'swish',
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> FusedRMSNormGated:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps
        self.activation = activation

        if self.activation not in ['swish', 'silu', 'sigmoid']:
            raise ValueError(f"Unsupported activation: {self.activation}")

        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += f", activation={self.activation}"
        s += ")"
        return s

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        return rms_norm_gated(
            x,
            g,
            self.weight,
            self.bias,
            self.activation,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


class FusedLayerNormSwishGate(FusedLayerNormGated):

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        bias: bool = False,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> FusedLayerNormSwishGate:
        super().__init__(
            hidden_size=hidden_size,
            elementwise_affine=elementwise_affine,
            bias=bias,
            eps=eps,
            device=device,
            dtype=dtype,
        )


class FusedRMSNormSwishGate(FusedRMSNormGated):

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> FusedRMSNormSwishGate:
        super().__init__(
            hidden_size=hidden_size,
            elementwise_affine=elementwise_affine,
            eps=eps,
            device=device,
            dtype=dtype,
        )


class FusedLayerNormGatedLinear(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> FusedLayerNormGatedLinear:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        return layer_norm_swish_gate_linear(
            x,
            g,
            self.weight,
            self.bias,
            weight,
            bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


class FusedLayerNormSwishGateLinear(FusedLayerNormGatedLinear):

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> FusedLayerNormSwishGateLinear:
        super().__init__(
            hidden_size=hidden_size,
            elementwise_affine=elementwise_affine,
            eps=eps,
            device=device,
            dtype=dtype,
        )


class FusedRMSNormGatedLinear(nn.Module):

    def __init__(
        self,
        hidden_size,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> FusedRMSNormGatedLinear:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.register_parameter("weight", None)
        self.register_parameter("bias", None)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> torch.Tensor:
        return rms_norm_swish_gate_linear(
            x,
            g,
            self.weight,
            self.bias,
            weight,
            bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )


class FusedRMSNormSwishGateLinear(FusedRMSNormGatedLinear):

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> FusedRMSNormSwishGateLinear:
        super().__init__(
            hidden_size=hidden_size,
            elementwise_affine=elementwise_affine,
            eps=eps,
            device=device,
            dtype=dtype,
        )

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# Code is adapted from flash-attn.bert_padding.py


import torch
from einops import rearrange, repeat



class IndexFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, indices):
        ctx.save_for_backward(indices)
        assert x.ndim >= 2
        ctx.first_axis_dim, other_shape = x.shape[0], x.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return x[indices]
        return torch.gather(
            rearrange(x, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, do):
        (indices,) = ctx.saved_tensors
        assert do.ndim >= 2
        other_shape = do.shape[1:]
        do = rearrange(do, "b ... -> b (...)")
        dx = torch.zeros(
            [ctx.first_axis_dim, do.shape[1]],
            device=do.device,
            dtype=do.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # dx[indices] = do
        dx.scatter_(0, repeat(indices, "z -> z d", d=do.shape[1]), do)
        return dx.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert x.ndim >= 2
        y = torch.zeros(first_axis_dim, *x.shape[1:], device=x.device, dtype=x.dtype)
        # TODO [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        y[indices] = x
        # y.scatter_(0, repeat(indices, 'z -> z d', d=x.shape[1]), x)
        return y

    @staticmethod
    def backward(ctx, do):
        (indices,) = ctx.saved_tensors
        # TODO [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        dx = do[indices]
        # dx = torch.gather(do, 0, repeat(indices, 'z -> z d', d=do.shape[1]))
        return dx, None, None


index_put_first_axis = IndexPutFirstAxis.apply


@tensor_cache
def get_unpad_data(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Args:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors.
            `cu_seqlens` shape is [batch_size + 1].
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    lens = prepare_lens_from_mask(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = lens.max().item()
    cu_seqlens = prepare_cu_seqlens_from_mask(attention_mask)
    return indices, cu_seqlens, max_seqlen_in_batch


def unpad_input(
    q: torch.Tensor,
    states: tuple[torch.Tensor],
    attention_mask: torch.Tensor,
    q_len: int,
    keepdim: bool = False,
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens
    even though they belong to different batches.


    Arguments:
        q (`torch.Tensor`):
            Query state with padding. Shape: [batch_size, q_len, ...].
        states (`Tuple[torch.Tensor]`):
            Attention state with padding. Shape: [batch_size, seq_len, ...].
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape [batch_size, sequence_length], 1 means valid and 0 means not valid.
        q_len (`int`):
            Target length.
        keepdim (`bool`):
            Whether to keep the batch dimension. Default: `False`.

    Return:
        q (`torch.Tensor`):
            Query state without padding.
            Shape: [1, total_target_length, ...] if `keepdim=True` else [total_target_length, ...].
        states (`Tuple[torch.Tensor]`):
            Attention state without padding.
            Shape: [1, total_source_length, ...] if `keepdim=True` else [total_source_length, ...].
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`Tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value),
            used to index into ragged (unpadded) tensors.
            `cu_seqlens` shape is [batch_size + 1].
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`Tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence
            i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = get_unpad_data(attention_mask)
    batch_size, seq_len, *_ = states[0].shape

    state = tuple(
        index_first_axis(rearrange(s, "b s ... -> (b s) ..."), indices_k)
        for s in states
    )

    if q_len == seq_len:
        q = index_first_axis(rearrange(q, "b s ... -> (b s) ..."), indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif q_len == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
        indices_q = cu_seqlens_q[:-1]
        q = q.squeeze(1)
    else:
        raise NotImplementedError("We only support either q_len == k_len (prefilling) or q_len == 1 (decoding)")

    if keepdim:
        q = q.unsqueeze(0)
        state = tuple(s.unsqueeze(0) for s in state)

    return (
        q,
        state,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Args:
        hidden_states ([total_tokens, ...]):
            where total_tokens denotes the number of tokens in selected in attention_mask.
        indices ([total_tokens]):
            the indices that represent the non-masked tokens of the original padded input sequence.
        batch_size (int):
            batch_size size for the padded sequence.
        seq_len (int):
            maximum sequence length for the padded sequence.

    Return:
        hidden_states of shape [batch_size, seq_len, ...]
    """
    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)

# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang


import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F


class KimiDeltaAttention(nn.Module):
    """
    Kimi Delta Attention (KDA) layer implementation.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 128.
        num_heads (int, Optional):
            The number of heads. Default: 16.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA (Grouped Value Attention) is applied if `num_v_heads` > `num_heads`. Default: `None`.
        mode (str, Optional):
            Which Kimi Delta Attention kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference:
            [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> KimiDeltaAttention:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
            )
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)))
        self.A_log._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.zeros(self.key_dim, dtype=torch.float32))
        self.dt_bias._no_weight_decay = True

        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens")
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        g = self.f_proj(hidden_states)
        beta = self.b_proj(hidden_states).sigmoid()

        q, k, g = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k, g))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        # for multi-value attention, we repeat the inputs for simplicity.
        if self.num_v_heads > self.num_heads:
            q, k, g = (repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads) for x in (q, k, g))
            beta = repeat(beta, "... h -> ... (h g)", g=self.num_v_heads // self.num_heads)

        if self.allow_neg_eigval:
            beta = beta * 2.0

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if mode == "chunk":
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        elif mode == "fused_recurrent":
            g = fused_kda_gate(g=g, A_log=self.A_log, dt_bias=self.dt_bias)
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values


FLAKimiDeltaAttention = KimiDeltaAttention


class KimiDeltaAttention(nn.Module):
    def __init__(self, config, layer_idx: int | None = None):
        super().__init__()
        head_dim = config.embed_size // config.head_num
        self.inner = FLAKimiDeltaAttention(
            hidden_size=config.embed_size,
            expand_v=config.kda_expand_v,
            head_dim=head_dim,
            num_heads=config.head_num,
            num_v_heads=config.kda_num_v_heads,
            mode=config.kda_mode,
            use_short_conv=config.kda_use_short_conv,
            allow_neg_eigval=config.kda_allow_neg_eigval,
            conv_size=config.kda_conv_size,
            conv_bias=config.kda_conv_bias,
            layer_idx=layer_idx,
            norm_eps=config.kda_norm_eps,
        )
        self.expand_v = self.inner.expand_v
        self.num_v_heads = self.inner.num_v_heads
        self.allow_neg_eigval = self.inner.allow_neg_eigval
        self.mode = self.inner.mode
        self.use_short_conv = self.inner.use_short_conv
        self.conv_size = self.inner.conv_size
        self.conv_bias = self.inner.conv_bias

    def forward(self, x, kv_cache=None):
        cache = kv_cache
        if isinstance(kv_cache, list):
            cache = _ListCache(kv_cache)
        o, _, cache_out = self.inner(hidden_states=x, past_key_values=cache)
        if isinstance(cache_out, _ListCache):
            cache_out = cache_out.items
        return o, cache_out


class _ListCache:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def update(self, recurrent_state, conv_state, layer_idx, offset):
        state = self.items[layer_idx]
        if state is None:
            state = {}
        state.update(
            {
                "recurrent_state": recurrent_state,
                "conv_state": conv_state,
                "offset": offset,
            }
        )
        self.items[layer_idx] = state

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
        self.query = nn.Linear(config.embed_size, config.embed_size * 2, bias=False)
        self.value = nn.Linear(config.embed_size, config.embed_size, bias=False)
        self.o = nn.Linear(config.embed_size, config.embed_size)
        # block_mask for FlexAttention
        def causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            return causal_mask
        self.causal_mask = create_block_mask(causal, B=None, H=None, Q_LEN=config.seq_len, KV_LEN=config.seq_len)
        self.freqs_cis = precompute_freqs_cis(config.embed_size//config.head_num, config.seq_len)
        self.register_buffer('tril', torch.tril(torch.ones(config.seq_len, config.seq_len)))

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        q = self.query(x) # (B,T,C)
        k = self.key(x)   # (B,T,C)
        v = self.value(x) # (B,T,C)

        # Split into heads
        # q = q.view(B, T, self.head_num, self.head_size).transpose(1, 2) # (B, H, T, C/H)
        k = k.view(B, T, self.head_num, self.head_size).transpose(1, 2) # (B, H, T, C/H)
        v = v.view(B, T, self.head_num, self.head_size).transpose(1, 2) # (B, H, T, C/H)

        q = q.view(B, T, self.head_num, -1)
        q, gate_score = torch.split(q, [self.head_size * 1, self.head_size * 1], dim=-1) # For MHA, num_key_value_groups is 1
        gate_score = gate_score.reshape(B, T, -1, self.head_size)
        q = q.reshape(B, T, -1, self.head_size).transpose(1, 2)

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

        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0,
            is_causal=True
        )

        out = out.transpose(1, 2).contiguous()
        # Softpick gating over sequence dimension (dim=1)
        # gate_score shape: [B, T, num_heads, head_size]
        # This makes gates compete across sequence positions
        out = out * softpick(gate_score, dim=1)

        out = out.view(B, T, C) # (B, H, T, C/H) -> (B, T, H, C/H) -> (B, T, C)
        out = self.o(out)
        return out, kv_cache

class Block(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        if config.attn_type == "kda":
            self.sa_heads = KimiDeltaAttention(config, layer_idx=layer_idx)
        else:
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
        self.attn_type = config.attn_type
        # embed raw tokens to a lower dimensional embedding with embed_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embed_size)
        # Language Modelling (?) Head is a standard linear layer to go from 
        # embeddings back to logits of vocab_size
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size)
        # transformer blocks
        self.blocks = nn.ModuleList([Block(config, layer_idx=i) for i in range(config.layer_num)])

    def forward(self, idx, targets=None, kv_cache=None):
        B, T = idx.shape
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
            if self.attn_type == "kda":
                kv_cache = [{"recurrent_state": None} for _ in range(self.layer_num)]
            else:
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
    layer_num=6,
    attn_type="kda",
)
m = TransformerLM(config)
if use_torch_compile:
    m = torch.compile(m, options=torch_compile_options, fullgraph=True)
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
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
        "initial_lr": optimizer.param_groups[0]['lr'],
    }
    
    # Add model configuration
    config.update({
        "vocab_size": model.token_embedding_table.num_embeddings,
        "embed_size": model.token_embedding_table.embedding_dim,
        "head_num": model.head_num,
        "layer_num": model.layer_num,
        "total_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "attn_type": model.attn_type,
        "kda_expand_v": getattr(model.blocks[0].sa_heads, "expand_v", None),
        "kda_num_v_heads": getattr(model.blocks[0].sa_heads, "num_v_heads", None),
        "kda_allow_neg_eigval": getattr(model.blocks[0].sa_heads, "allow_neg_eigval", None),
        "kda_mode": getattr(model.blocks[0].sa_heads, "mode", None),
        "kda_use_short_conv": getattr(model.blocks[0].sa_heads, "use_short_conv", None),
        "kda_conv_size": getattr(model.blocks[0].sa_heads, "conv_size", None),
        "kda_conv_bias": getattr(model.blocks[0].sa_heads, "conv_bias", None),
    })
    
    # Add scheduler-specific config
    if hasattr(scheduler, 'T_max'):
        config['scheduler_T_max'] = scheduler.T_max
    if hasattr(scheduler, 'eta_min'):
        config['scheduler_eta_min'] = scheduler.eta_min
    
    return config

# Initialize wandb with automated config
model = TransformerLM(config)
if use_torch_compile:
    model = torch.compile(model, options=torch_compile_options, fullgraph=True)
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
        name="kda-lm-animesubs-256seq-256embed-4head-6layer.AMD"
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
os.makedirs
torch.save(model.state_dict(), 'models/TransformerLM.pt')

# %%
# Load model
model = TransformerLM(config)
if use_torch_compile:
    model = torch.compile(model, options=torch_compile_options, fullgraph=True)
model.load_state_dict(torch.load('checkpoints/checkpoint_step_5000.pt'), strict=False)
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
