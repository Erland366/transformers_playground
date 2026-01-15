"""Runner script for gradient accumulation experiments."""
import os
import sys
import argparse

# Parse args before imports to set CUDA device early
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--gradient_accumulation_steps', type=int, required=True)
parser.add_argument('--effective_batch_size', type=int, required=True,
                    help='Effective batch size (must be batch_size * gradient_accumulation_steps)')
parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--total_steps', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# Validate effective batch size
assert args.effective_batch_size == args.batch_size * args.gradient_accumulation_steps, \
    f"effective_batch_size ({args.effective_batch_size}) must equal batch_size ({args.batch_size}) * gradient_accumulation_steps ({args.gradient_accumulation_steps})"

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from dotenv import load_dotenv
from contextlib import nullcontext

load_dotenv()

import wandb
wandb_api_key = os.getenv('WANDB_API_KEY')
if wandb_api_key:
    wandb.login(key=wandb_api_key)

# Set seed for full reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)
torch.set_float32_matmul_precision('high')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and preprocess data
with open('animesubs.txt', 'r', encoding='latin') as f:
    text = f.read()
text = ''.join(filter(lambda character: ord(character) < 0x3000, text))

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
stoi[''] = len(stoi)
itos[len(itos)] = ''
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])
vocab_size = len(itos)
print(f"Vocab size: {vocab_size}")

data = torch.tensor(encode(text), dtype=torch.int64)
n = int(len(data) * 0.99)
train_data = data[:n]
val_data = data[n:]

# Config
seq_len = 256
batch_size = args.batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
total_steps = args.total_steps
use_compile = False

def get_batch(split, seq_len, batch_size=4):
    data_split = train_data if split == 'train' else val_data
    # Use the dedicated generator for deterministic sampling
    ix = torch.randint(len(data_split) - seq_len, (batch_size,), generator=data_rng)
    x = torch.stack([data_split[i:i+seq_len] for i in ix])
    y = torch.stack([data_split[i+1:i+seq_len+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def get_batch_with_indices(split, seq_len, indices):
    """Get batch using pre-determined indices for reproducibility."""
    data_split = train_data if split == 'train' else val_data
    x = torch.stack([data_split[i:i+seq_len] for i in indices])
    y = torch.stack([data_split[i+1:i+seq_len+1] for i in indices])
    x, y = x.to(device), y.to(device)
    return x, y

# Create a dedicated generator for data sampling (isolated from model operations)
data_rng = torch.Generator()

def get_micro_batches(split, seq_len, effective_batch_size, micro_batch_size, num_micro_batches, step=None, debug=False):
    """
    Generate micro-batches that together form one effective batch.
    This ensures GA and non-GA runs see exactly the same data per optimizer step.
    """
    data_split = train_data if split == 'train' else val_data
    # Sample all indices for the effective batch at once using dedicated generator
    all_indices = torch.randint(len(data_split) - seq_len, (effective_batch_size,), generator=data_rng)
    if debug and step is not None and step < 2:
        print(f"Step {step} indices (first 10): {all_indices[:10].tolist()}")
    # Split into micro-batches
    micro_batches = []
    for i in range(num_micro_batches):
        start_idx = i * micro_batch_size
        end_idx = start_idx + micro_batch_size
        indices = all_indices[start_idx:end_idx]
        x, y = get_batch_with_indices(split, seq_len, indices)
        micro_batches.append((x, y))
    return micro_batches

# Import model classes from transformer_playground
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size: int
    seq_len: int
    embed_size: int
    head_num: int
    layer_num: int

class Head(nn.Module):
    def __init__(self, embed_size, head_size, seq_len):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(torch.tril(torch.ones(T, T, device=x.device)) == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, embed_size, head_size, seq_len):
        super().__init__()
        self.heads = nn.ModuleList([Head(embed_size, head_size, seq_len) for _ in range(head_num)])
        self.proj = nn.Linear(head_num * head_size, embed_size)
        self.head_size = head_size

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, head_num, embed_size, seq_len):
        super().__init__()
        head_size = embed_size // head_num
        self.sa_heads = MultiHeadAttention(head_num, embed_size, head_size, seq_len)
        self.ffwd = FeedForward(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.head_num = config.head_num
        self.layer_num = config.layer_num
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embed_size)
        self.position_embedding_table = nn.Embedding(config.seq_len, config.embed_size)
        self.blocks = nn.Sequential(*[Block(config.head_num, config.embed_size, config.seq_len) for _ in range(config.layer_num)])
        self.ln_f = nn.LayerNorm(config.embed_size)
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size)

    def forward(self, idx, targets=None, use_silu_softpick=False):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

# Training function
def train(model, optimizer, scheduler, seq_len, batch_size, total_steps, gradient_accumulation_steps, effective_batch_size, val_steps=10, val_interval=50):
    losses = []
    val_losses = []

    for step in (bar := tqdm(range(total_steps))):
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        # Get all micro-batches for this step (deterministic splitting of effective batch)
        micro_batches = get_micro_batches(
            'train', seq_len, effective_batch_size, batch_size, gradient_accumulation_steps
        )

        for micro_step, (xb, yb) in enumerate(micro_batches):
            logits, loss = model(xb, yb)

            scaled_loss = loss / gradient_accumulation_steps
            accumulated_loss += loss.item()
            scaled_loss.backward()

        optimizer.step()
        scheduler.step()

        avg_loss = accumulated_loss / gradient_accumulation_steps
        bar.set_description(f"loss: {avg_loss:.2f}, val loss: {val_losses[-1] if val_losses else 0:.2f}, lr: {scheduler.get_last_lr()[0]:.2e}")
        losses.append(avg_loss)

        wandb.log({
            'train_loss': avg_loss,
            'learning_rate': scheduler.get_last_lr()[0],
            'step': step
        })

        if step % val_interval == 0:
            with torch.no_grad():
                val_loss = 0
                for _ in range(val_steps):
                    # Use effective_batch_size for validation to ensure consistency
                    xb, yb = get_batch('val', seq_len=seq_len, batch_size=effective_batch_size)
                    _, loss = model(xb, yb)
                    val_loss += loss.item()
                val_loss /= val_steps
                val_losses.append(val_loss)

            wandb.log({
                'val_loss': val_loss,
                'step': step
            })

    print(f'final loss: {avg_loss}, final val loss: {val_loss}')
    return losses, val_losses

# Setup model
config = TransformerConfig(
    vocab_size=vocab_size,
    seq_len=seq_len,
    embed_size=256,
    head_num=4,
    layer_num=6
)

model = TransformerLM(config)
model.to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

# Initialize wandb
effective_batch_size = args.effective_batch_size
wandb.init(
    project="transformers-playground",
    name=args.run_name,
    config={
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "total_steps": total_steps,
        "seq_len": seq_len,
        "embed_size": 256,
        "head_num": 4,
        "layer_num": 6,
        "vocab_size": vocab_size,
        "total_params": total_params,
        "learning_rate": 2e-3,
        "seed": args.seed,
    }
)

# Train - reset seed right before training to ensure deterministic data sampling
set_seed(args.seed)
data_rng.manual_seed(args.seed)  # Seed the dedicated data sampling generator
print(f"Starting training: batch_size={batch_size}, ga_steps={gradient_accumulation_steps}, effective_batch={effective_batch_size}")
losses, val_losses = train(model, optimizer, scheduler, seq_len, batch_size, total_steps, gradient_accumulation_steps, effective_batch_size)

wandb.finish()
print("Done!")
