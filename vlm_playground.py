# %%
import os
import math
import contextlib
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.nn.attention.flex_attention import flex_attention as _flex_attention, create_block_mask
from tqdm.auto import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from IPython import display, get_ipython
load_dotenv()

use_wandb = True
try:
    import wandb
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
        print("wandb initialized successfully")
    else:
        use_wandb = False
        print("WANDB_API_KEY not found - wandb logging disabled")
except ImportError:
    use_wandb = False
    print("wandb not installed - wandb logging disabled")
    wandb = None

torch.manual_seed(69)
torch.set_printoptions(profile="short", sci_mode=False, linewidth=100000)
torch.set_float32_matmul_precision('high')

# Dynamo cache configuration for flex_attention
torch._dynamo.config.cache_size_limit = 1000

# Compile flex_attention for better performance with custom masks
flex_attention = torch.compile(_flex_attention, dynamic=False, fullgraph=True)

is_cuda = torch.cuda.is_available()
is_rocm = is_cuda and torch.version.hip is not None

def rocm_runtime_present():
    return os.path.isdir("/opt/rocm") or shutil.which("rocm-smi") is not None

if rocm_runtime_present() and torch.version.hip is None:
    raise RuntimeError(
        "ROCm runtime detected but this PyTorch build has no ROCm support. "
        "Reinstall torch/torchvision/torchaudio from the ROCm index."
    )

device = torch.device('cuda' if is_cuda else 'cpu')
device_type = device.type
use_torch_compile = device.type == 'cuda'
use_flex_attention = device.type == 'cuda' and not is_rocm
use_fused_optimizer = device.type == 'cuda' and not is_rocm

# Mixture of Modality Heads Configuration
# 40% heads for V->V, 40% heads for T->T, 20% heads for VT->VT
HEAD_PCT_VISION = 0.4
HEAD_PCT_TEXT = 0.4

def generate_mixture_of_modality_heads_mask_mod(
    total_heads: int,
    S_V: int,
    pct_heads_V: float,
    pct_heads_T: float
):
    """
    Generates a compile-friendly (single-return) mask_mod function
    based on percentages of heads and V/T split.
    """
    # Calculate the number of heads for each group
    H_V_count = int(total_heads * pct_heads_V)
    H_T_count = int(total_heads * pct_heads_T)
    H_VT_count = total_heads - H_V_count - H_T_count

    # Calculate the head index boundaries
    H_T_start_index = H_V_count
    H_VT_start_index = H_V_count + H_T_count

    # Print the allocation for verification
    print("--- Mask Generator Configuration ---")
    print(f"Total Heads: {total_heads}")
    print(f"S_V (Vision Tokens): {S_V}")
    print(f"Head Group V->V:   [0..{H_T_start_index - 1}] ({H_V_count} heads)")
    print(f"Head Group T->T:   [{H_T_start_index}..{H_VT_start_index - 1}] ({H_T_count} heads)")
    print(f"Head Group VT->VT: [{H_VT_start_index}..{total_heads - 1}] ({H_VT_count} heads)")
    print("------------------------------------")

    def mixture_of_multimodal_heads_mask_mod(b, h, q_idx, kv_idx):
        # Group 1: V -> V (active for h < H_T_start_index)
        head_V = (h < H_T_start_index) & (q_idx < S_V) & (kv_idx < S_V)

        # Group 2: T -> T (active for H_T_start_index <= h < H_VT_start_index)
        head_T = (h >= H_T_start_index) & (h < H_VT_start_index) & \
                 (q_idx >= S_V) & (kv_idx >= S_V) & (q_idx >= kv_idx)

        # Group 3: VT -> VT (active for h >= H_VT_start_index)
        head_VT = (h >= H_VT_start_index) & \
                  ((kv_idx < S_V) | (q_idx >= kv_idx))

        # Combine all masks. Only one group will be True for any given 'h'.
        return head_V | head_T | head_VT

    return mixture_of_multimodal_heads_mask_mod

@lru_cache
def create_block_mask_cached(mask_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(mask_mod, B, H, M, N, device=device, _compile=True)
    return block_mask

autocast_dtype = None
if device.type == 'cuda':
    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def in_notebook():
    try:
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False

def autocast_context():
    if device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=autocast_dtype)
    return contextlib.nullcontext()

plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['figure.dpi'] = 50
plt.rcParams['axes.grid'] = True
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True

print(f"Device: {device}")

# %% [markdown]
# # Data Loading - COCO Captions

# %%
from datasets import load_dataset, Dataset, DatasetDict

img_size = 96

def get_image_transform(img_size=96):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

image_transform = get_image_transform(img_size)

print("Loading COCO Captions dataset...")
df = load_dataset("Erland/coco_captions_small")
print(f"Dataset loaded: {df}")

# Convert to pandas for easier manipulation
if isinstance(df, DatasetDict):
    df = df["train"]
df_pandas = df.to_pandas()

# Split into train/val
n = int(0.9 * len(df_pandas))
train_df = df_pandas.iloc[:n].reset_index(drop=True)
val_df = df_pandas.iloc[n:].reset_index(drop=True)
print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

# %% [markdown]
# # Tokenizer

# %%
# Build character-level tokenizer from captions
text = "".join(df_pandas["caption"].tolist())
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Unique characters: {vocab_size}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Add special tokens
pad_idx = len(stoi)
stoi['<pad>'] = pad_idx
itos[pad_idx] = '<pad>'

bos_idx = len(stoi)
stoi['<bos>'] = bos_idx
itos[bos_idx] = '<bos>'

eos_idx = len(stoi)
stoi['<eos>'] = eos_idx
itos[eos_idx] = '<eos>'

vocab_size = len(stoi)
print(f"Vocab size (with special tokens): {vocab_size}")

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l if i in itos and itos[i] not in ['<pad>', '<bos>', '<eos>']])

# Test tokenizer
print("Encoded:", encode("A cat on a mat"))
print("Decoded:", decode(encode("A cat on a mat")))

# %% [markdown]
# # Configuration

# %%
@dataclass
class VLMConfig:
    """Configuration for Vision-Language Model"""
    # Vision Encoder Config
    img_size: int = 96
    patch_size: int = 16
    image_embed_dim: int = 512
    vit_num_layers: int = 4
    vit_num_heads: int = 8
    emb_dropout: float = 0.1
    return_all_patches: bool = True

    # Text Decoder Config
    vocab_size: int = vocab_size
    embed_size: int = 640
    seq_len: int = 256
    head_num: int = 10
    layer_num: int = 6
    dropout: float = 0.1

    # Training Config
    batch_size: int = 64
    total_steps: int = 5000
    learning_rate: float = 2e-3
    val_interval: int = 50
    val_steps: int = 10
    val_batch_size: int = 32
    checkpoint_interval: int = 500
    max_caption_len: int = 200

    # Generation Config
    max_new_tokens: int = 200

    # Perplexity evaluation
    ppl_val_steps: int = 1000
    ppl_batch_size: int = 128

    @property
    def num_visual_tokens(self):
        """Number of visual tokens produced by ViT + CLS"""
        num_patches = (self.img_size // self.patch_size) ** 2
        return num_patches + 1 if self.return_all_patches else 1

    @property
    def total_seq_len(self):
        """Total sequence length including visual tokens"""
        return self.num_visual_tokens + self.seq_len

    def to_dict(self):
        """Convert config to dict for wandb logging"""
        return {
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'image_embed_dim': self.image_embed_dim,
            'vit_num_layers': self.vit_num_layers,
            'vit_num_heads': self.vit_num_heads,
            'vocab_size': self.vocab_size,
            'embed_size': self.embed_size,
            'seq_len': self.seq_len,
            'head_num': self.head_num,
            'layer_num': self.layer_num,
            'num_visual_tokens': self.num_visual_tokens,
            'batch_size': self.batch_size,
            'total_steps': self.total_steps,
            'learning_rate': self.learning_rate,
        }

# CPU fallback settings
if device.type == 'cpu':
    print("CPU detected; using reduced settings")

# %% [markdown]
# # Torch Compile Options

# %%
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
    'combo_kernels': False,
    'benchmark_combo_kernel': False,
    'combo_kernel_foreach_dynamic_shapes': False
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

def maybe_compile(model):
    if use_torch_compile:
        return torch.compile(model, options=torch_compile_options, fullgraph=False)
    return model

# %% [markdown]
# # Core Components

# %%
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to(device)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    q_shape = [d if i == xq_.ndim - 2 or i == xq_.ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    k_shape = [d if i == xq_.ndim - 2 or i == xk_.ndim - 1 else 1 for i, d in enumerate(xk_.shape)]
    T_q = xq_.shape[-2]
    q_freqs_cis = freqs_cis[-T_q:].view(*q_shape)
    k_freqs_cis = freqs_cis.view(*k_shape)
    xq_out = torch.view_as_real(xq_ * q_freqs_cis).flatten(xq.dim() - 1)
    xk_out = torch.view_as_real(xk_ * k_freqs_cis).flatten(xq.dim() - 1)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RMSNorm(nn.Module):
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
    def __init__(self, embed_size, use_gelu=False):
        super().__init__()
        self.lin_1 = nn.Linear(embed_size, embed_size * 4)
        self.lin_2 = nn.Linear(embed_size * 4, embed_size)
        self.act = nn.GELU() if use_gelu else nn.ReLU()

    def forward(self, x):
        x = self.lin_1(x)
        x = self.act(x)
        x = self.lin_2(x)
        return x

# %% [markdown]
# # Vision Components

# %%
class PatchEmbeddings(nn.Module):
    def __init__(self, img_size=96, patch_size=16, hidden_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, X):
        X = self.conv(X)
        X = X.flatten(2)
        X = X.transpose(1, 2)
        return X

class ViTMultiHeadAttention(nn.Module):
    """Non-causal multi-head attention for ViT"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        # Non-causal attention (no mask)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return out

class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.attn = ViTMultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, use_gelu=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = PatchEmbeddings(
            img_size=config.img_size,
            patch_size=config.patch_size,
            hidden_dim=config.image_embed_dim
        )
        num_patches = (config.img_size // config.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.image_embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, config.image_embed_dim) * 0.02)
        self.dropout = nn.Dropout(config.emb_dropout)
        self.blocks = nn.ModuleList([
            ViTBlock(config.image_embed_dim, config.vit_num_heads, config.emb_dropout)
            for _ in range(config.vit_num_layers)
        ])
        self.norm = RMSNorm(config.image_embed_dim)
        self.return_all_patches = config.return_all_patches

    def forward(self, X):
        x = self.patch_embedding(X)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        if self.return_all_patches:
            return x
        return x[:, 0]

# %% [markdown]
# # MultiModal Projector

# %%
class MultiModalProjector(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.image_embed_dim, 4 * config.image_embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.image_embed_dim, config.embed_size),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

# %% [markdown]
# # VLM Decoder Components

# %%
class VLMMultiHeadAttention(nn.Module):
    """Multi-head attention with causal masking for text, full attention for visual tokens"""
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.total_seq_len = config.total_seq_len
        self.num_visual_tokens = config.num_visual_tokens
        self.head_num = config.head_num
        self.head_size = config.embed_size // config.head_num
        self.use_flex_attention = use_flex_attention

        self.key = nn.Linear(config.embed_size, config.embed_size, bias=False)
        self.query = nn.Linear(config.embed_size, config.embed_size, bias=False)
        self.value = nn.Linear(config.embed_size, config.embed_size, bias=False)
        self.o = nn.Linear(config.embed_size, config.embed_size)

        # FlexAttention block mask with Mixture of Modality Heads
        if self.use_flex_attention:
            # Generate the mixture of modality heads mask
            # 40% V->V heads, 40% T->T heads, 20% VT->VT heads
            mask_mod = generate_mixture_of_modality_heads_mask_mod(
                total_heads=self.head_num,
                S_V=self.num_visual_tokens,
                pct_heads_V=HEAD_PCT_VISION,
                pct_heads_T=HEAD_PCT_TEXT
            )
            # Use cached block mask creation for efficiency across layers
            self.vlm_mask = create_block_mask_cached(
                mask_mod,
                B=None,
                H=self.head_num,
                M=config.total_seq_len,
                N=config.total_seq_len,
                device="cuda"
            )
        else:
            self.vlm_mask = None

        self.freqs_cis = precompute_freqs_cis(config.embed_size // config.head_num, config.total_seq_len)
        self.register_buffer('tril', torch.tril(torch.ones(config.total_seq_len, config.total_seq_len)))

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        T_past = 0
        if kv_cache is not None and kv_cache[0] is not None:
            T_past = kv_cache[0].shape[2]

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(B, T, self.head_num, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.head_num, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.head_num, self.head_size).transpose(1, 2)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            if k_past is not None:
                k = torch.cat((k_past, k), dim=2)
                v = torch.cat((v_past, v), dim=2)
            if k.shape[2] > self.total_seq_len:
                k = k[:, :, -self.total_seq_len:]
                v = v[:, :, -self.total_seq_len:]
            kv_cache = (k, v)

        T_k = k.shape[2]
        q, k = apply_rotary_emb(q, k, self.freqs_cis[:T_k])

        if self.use_flex_attention and T == self.total_seq_len:
            out = flex_attention(q, k, v, block_mask=self.vlm_mask)
        else:
            wei = q @ k.transpose(-2, -1)
            wei = wei * self.head_size ** -0.5
            # Create VLM mask: visual tokens full attention, text tokens causal
            # For simplicity, we use a simple causal mask for the whole sequence
            # Visual tokens at the beginning can attend to each other
            mask = self.tril[T_k - T:T_k, :T_k]
            wei = wei.masked_fill(mask == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            out = wei @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o(out)
        return out, kv_cache

class VLMBlock(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.sa_heads = VLMMultiHeadAttention(config)
        self.ff_layer = FeedForward(config.embed_size)
        self.sa_norm = RMSNorm(config.embed_size)
        self.ff_norm = RMSNorm(config.embed_size)

    def forward(self, x, kv_cache=None):
        a, kv_cache = self.sa_heads(self.sa_norm(x), kv_cache)
        h = x + a
        o = h + self.ff_layer(self.ff_norm(h))
        return o, kv_cache

class VLMDecoder(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config
        self.layer_num = config.layer_num
        self.head_num = config.head_num
        self.seq_len = config.total_seq_len
        self.num_visual_tokens = config.num_visual_tokens

        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embed_size)
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size)
        self.blocks = nn.ModuleList([VLMBlock(config) for _ in range(config.layer_num)])
        self.final_norm = RMSNorm(config.embed_size)

    def forward(self, x, targets=None, kv_cache=None):
        for i, block in enumerate(self.blocks):
            x, cache = block(x, None if kv_cache is None else kv_cache[i])
            if kv_cache is not None:
                kv_cache[i] = cache

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None, kv_cache

        # Compute loss only on text tokens (after visual tokens)
        text_logits = logits[:, self.num_visual_tokens:, :]
        B, T, V = text_logits.shape

        loss = F.cross_entropy(
            text_logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=-100
        )

        return logits, loss, kv_cache

# %% [markdown]
# # Vision Language Model

# %%
class VisionLanguageModel(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config
        self.num_visual_tokens = config.num_visual_tokens
        self.layer_num = config.layer_num
        self.head_num = config.head_num

        self.vision_encoder = ViTEncoder(config)
        self.projector = MultiModalProjector(config)
        self.decoder = VLMDecoder(config)

        # For checkpoint compatibility
        self.token_embedding_table = self.decoder.token_embedding_table
        self.blocks = self.decoder.blocks

    def forward(self, images, idx, targets=None):
        image_embeds = self.vision_encoder(images)
        projected_image = self.projector(image_embeds)
        tok_emb = self.decoder.token_embedding_table(idx)
        x = torch.cat([projected_image, tok_emb], dim=1)
        logits, loss, _ = self.decoder(x, targets)
        return logits, loss

    def generate(self, images, idx=None, max_new_tokens=100, temperature=1.0, use_cache=True):
        B = images.shape[0]

        if idx is None:
            idx = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)

        with torch.no_grad():
            image_embeds = self.vision_encoder(images)
            projected_image = self.projector(image_embeds)

        if use_cache:
            kv_cache = [(None, None) for _ in range(self.decoder.layer_num)]

            # First pass: process visual tokens
            x = projected_image
            for i, block in enumerate(self.decoder.blocks):
                x, kv_cache[i] = block(x, kv_cache[i])

            generated = idx
            for _ in range(max_new_tokens):
                tok_emb = self.decoder.token_embedding_table(generated[:, -1:])
                x = tok_emb
                for i, block in enumerate(self.decoder.blocks):
                    x, kv_cache[i] = block(x, kv_cache[i])

                x = self.decoder.final_norm(x)
                logits = self.decoder.lm_head(x)[:, -1, :]

                if temperature > 0:
                    logits = logits / temperature
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, idx_next], dim=1)

                if (idx_next == eos_idx).all():
                    break

            return generated
        else:
            generated = idx
            for _ in range(max_new_tokens):
                tok_emb = self.decoder.token_embedding_table(generated)
                x = torch.cat([projected_image, tok_emb], dim=1)
                logits, _, _ = self.decoder(x)
                logits = logits[:, -1, :]

                if temperature > 0:
                    logits = logits / temperature
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, idx_next], dim=1)

                if (idx_next == eos_idx).all():
                    break

            return generated

# %% [markdown]
# # Data Batching

# %%
def process_image(img):
    """Process image from various formats (PIL, dict with bytes, path)"""
    import io
    if hasattr(img, 'convert'):
        # Already a PIL image
        return img.convert('RGB')
    elif isinstance(img, dict) and 'bytes' in img:
        # HuggingFace format with bytes
        return Image.open(io.BytesIO(img['bytes'])).convert('RGB')
    elif isinstance(img, str):
        # File path
        return Image.open(img).convert('RGB')
    else:
        raise ValueError(f"Unknown image format: {type(img)}")

def get_batch(split, batch_size, config):
    data = train_df if split == 'train' else val_df
    replace = len(data) < batch_size
    batch = data.sample(n=batch_size, replace=replace)

    # Process images
    images = torch.stack([
        image_transform(process_image(img))
        for img in batch['image']
    ]).to(device)

    # Process captions
    encoded_captions = []
    for caption in batch['caption']:
        enc = [bos_idx] + encode(caption)[:config.max_caption_len - 2] + [eos_idx]
        encoded_captions.append(enc)

    max_len = min(max(len(c) for c in encoded_captions), config.seq_len)

    input_ids = torch.full((batch_size, max_len), pad_idx, dtype=torch.long, device=device)
    targets = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)

    for i, caption in enumerate(encoded_captions):
        cap_len = min(len(caption), max_len)
        input_ids[i, :cap_len] = torch.tensor(caption[:cap_len])
        if cap_len > 1:
            targets[i, :cap_len - 1] = torch.tensor(caption[1:cap_len])

    return images, input_ids, targets

# %% [markdown]
# # Training Utilities

# %%
def save_vlm_checkpoint(model, optimizer, scheduler, step, losses, val_losses, config, save_dir='checkpoints'):
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': losses,
        'val_losses': val_losses,
        'config': config.to_dict(),
        'tokenizer': {
            'stoi': stoi,
            'itos': itos,
            'vocab_size': vocab_size,
            'pad_idx': pad_idx,
            'bos_idx': bos_idx,
            'eos_idx': eos_idx,
        }
    }

    checkpoint_path = os.path.join(save_dir, f'vlm_checkpoint_step_{step}.pt')
    torch.save(checkpoint, checkpoint_path)

    if use_wandb:
        wandb.save(checkpoint_path)

    print(f"VLM Checkpoint saved at step {step}: {checkpoint_path}")
    return checkpoint_path

def load_vlm_checkpoint(checkpoint_path, config):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = VisionLanguageModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"Checkpoint loaded from step {checkpoint['step']}")
    return model, checkpoint

def evaluate_vlm(model, config, val_steps=None):
    val_steps = val_steps or config.val_steps
    model.eval()

    with torch.no_grad():
        total_loss = 0
        for _ in range(val_steps):
            images, input_ids, targets = get_batch('val', config.val_batch_size, config)
            with autocast_context():
                _, loss = model(images, input_ids, targets)
            total_loss += loss.item()

    model.train()
    return total_loss / val_steps

def perplexity_vlm(model, config, val_steps=None):
    val_steps = val_steps or config.ppl_val_steps
    model.eval()

    with torch.no_grad():
        total_loss = 0
        total_tokens = 0

        for _ in tqdm(range(val_steps), desc="Computing perplexity"):
            images, input_ids, targets = get_batch('val', config.ppl_batch_size, config)

            with autocast_context():
                logits, _ = model(images, input_ids, targets)

            text_logits = logits[:, model.num_visual_tokens:, :]
            B, T, V = text_logits.shape
            text_logits = text_logits.reshape(B * T, V)
            targets_flat = targets.reshape(B * T)

            valid_mask = targets_flat != -100
            if valid_mask.sum() > 0:
                loss = F.cross_entropy(
                    text_logits[valid_mask],
                    targets_flat[valid_mask],
                    reduction='sum'
                )
                total_loss += loss.item()
                total_tokens += valid_mask.sum().item()

    model.train()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    return ppl, avg_loss

# %% [markdown]
# # Training Loop

# %%
def train_vlm(model, optimizer, scheduler, config, save_dir='checkpoints'):
    losses = []
    val_losses = []
    os.makedirs(save_dir, exist_ok=True)

    use_live_plot = in_notebook()
    fig, ax, dh = None, None, None
    if use_live_plot:
        fig, ax = plt.subplots()
        dh = display.display(fig, display_id=True)

    for step in (bar := tqdm(range(config.total_steps))):
        images, input_ids, targets = get_batch('train', config.batch_size, config)

        with autocast_context():
            logits, loss = model(images, input_ids, targets)

        optimizer.zero_grad(set_to_none=True)

        if use_torch_compile:
            with torch._dynamo.compiled_autograd._enable(torch.compile()):
                loss.backward()
        else:
            loss.backward()

        compile_optimizer_lr(optimizer, scheduler)

        bar.set_description(
            f"loss: {loss.item():.2f}, val: {val_losses[-1] if val_losses else 0:.2f}, "
            f"lr: {scheduler.get_last_lr()[0]:.2e}"
        )
        losses.append(loss.item())

        if use_wandb:
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': scheduler.get_last_lr()[0],
                'step': step
            })

        if step % config.val_interval == 0:
            val_loss = evaluate_vlm(model, config)
            val_losses.append(val_loss)

            if use_wandb:
                wandb.log({'val_loss': val_loss, 'step': step})

            if use_live_plot:
                ax.clear()
                ax.plot(losses, color='blue', label='train loss', alpha=0.7)
                ax.plot(range(0, len(losses), config.val_interval), val_losses,
                       color='red', label='val loss', alpha=0.7)
                ax.set_ylim(0, max(5, max(losses[-100:]) if losses else 5))
                ax.legend()

        if dh is not None:
            dh.update(fig)

        if step % config.checkpoint_interval == 0 and step > 0:
            save_vlm_checkpoint(model, optimizer, scheduler, step, losses, val_losses, config, save_dir)

    print(f'Final loss: {losses[-1]:.4f}, Final val loss: {val_losses[-1]:.4f}')
    save_vlm_checkpoint(model, optimizer, scheduler, config.total_steps, losses, val_losses, config, save_dir)

    return losses, val_losses

# %% [markdown]
# # Wandb Configuration

# %%
def get_vlm_wandb_config(model, optimizer, scheduler, config):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    vision_params = sum(p.numel() for p in model.vision_encoder.parameters() if p.requires_grad)
    projector_params = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)

    return {
        "architecture": "VisionLanguageModel",
        "dataset": "coco_captions_small",
        "img_size": config.img_size,
        "patch_size": config.patch_size,
        "num_visual_tokens": config.num_visual_tokens,
        "vit_layers": config.vit_num_layers,
        "vit_heads": config.vit_num_heads,
        "image_embed_dim": config.image_embed_dim,
        "vocab_size": config.vocab_size,
        "embed_size": config.embed_size,
        "seq_len": config.seq_len,
        "head_num": config.head_num,
        "layer_num": config.layer_num,
        "batch_size": config.batch_size,
        "total_steps": config.total_steps,
        "learning_rate": config.learning_rate,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
        "total_params": total_params,
        "vision_params": vision_params,
        "projector_params": projector_params,
        "decoder_params": decoder_params,
        "head_pct_vision": HEAD_PCT_VISION,
        "head_pct_text": HEAD_PCT_TEXT,
        "head_pct_shared": 1.0 - HEAD_PCT_VISION - HEAD_PCT_TEXT,
    }

# %% [markdown]
# # Main Training

# %%
# Initialize config
config = VLMConfig(
    img_size=96,
    patch_size=16,
    image_embed_dim=512,
    vit_num_layers=4,
    vit_num_heads=8,
    vocab_size=vocab_size,
    embed_size=640,
    seq_len=256,
    head_num=10,
    layer_num=6,
    batch_size=64 if device.type == 'cuda' else 8,
    total_steps=2000 if device.type == 'cuda' else 200,
    learning_rate=2e-3,
)

# Adjust for CPU
if device.type == 'cpu':
    config.val_batch_size = 8
    config.ppl_val_steps = 50
    config.ppl_batch_size = 8
    config.max_new_tokens = 50
    print(f"CPU settings: batch_size={config.batch_size}, total_steps={config.total_steps}")

print(f"Number of visual tokens: {config.num_visual_tokens}")
print(f"Total sequence length: {config.total_seq_len}")

# %%
# Initialize model
model = VisionLanguageModel(config)
model = maybe_compile(model)
model.to(device)

# Test forward pass
images, input_ids, targets = get_batch('train', 2, config)
logits, loss = model(images, input_ids, targets)
print(f"Test forward pass - logits shape: {logits.shape}, loss: {loss.item():.4f}")

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")

# %%
# Initialize optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, fused=use_fused_optimizer)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.total_steps, eta_min=1e-6)

# Warmup
warmup_steps = 10 if device.type == 'cuda' else 0
print(f"Running {warmup_steps} warmup steps...")
for _ in range(warmup_steps):
    images, input_ids, targets = get_batch('train', config.batch_size, config)
    with autocast_context():
        model(images, input_ids, targets)

# Initialize wandb
wandb_config = get_vlm_wandb_config(model, optimizer, scheduler, config)
if use_wandb:
    wandb.init(
        project="tf_chameleon",
        config=wandb_config,
        name=f"vlm-coco-{config.img_size}px-{config.embed_size}emb-{config.layer_num}L-442",
        settings=wandb.Settings(init_timeout=120),
    )

# Train
losses, val_losses = train_vlm(model, optimizer, scheduler, config)

if use_wandb:
    wandb.finish()

# %%
# Save model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/VLM.pt')
print("Model saved to models/VLM.pt")

# %%
# Load model (example)
# model = VisionLanguageModel(config)
# model = maybe_compile(model)
# model.load_state_dict(torch.load('models/VLM.pt'))
# model.to(device)

# %%
# Calculate perplexity
ppl, loss = perplexity_vlm(model, config, val_steps=config.ppl_val_steps)
print(f"Perplexity: {ppl:.2f}, Loss: {loss:.4f}")

# %%
# Generation demo
model.eval()
sample_idx = 0
sample_image = train_df.iloc[sample_idx]['image']
img_tensor = image_transform(process_image(sample_image)).unsqueeze(0).to(device)

print("Generating caption...")
with torch.no_grad():
    generated = model.generate(img_tensor, max_new_tokens=config.max_new_tokens, temperature=0.7, use_cache=True)

print(f"Generated caption: {decode(generated[0].tolist())}")
print(f"Actual caption: {train_df.iloc[sample_idx]['caption']}")
