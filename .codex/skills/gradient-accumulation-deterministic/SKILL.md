---
name: gradient-accumulation-deterministic
description: >
  Implement gradient accumulation that produces bit-identical results to standard batching.
  Use when: comparing GA vs non-GA runs, debugging training reproducibility.
metadata:
  short-description: "Deterministic gradient accumulation"
  tags:
    - training
    - reproducibility
    - gradient-accumulation
  domain: research
  created: 2026-01-15
  author: claude
---

# Gradient Accumulation (Deterministic)

## General Description

Gradient accumulation allows training with larger effective batch sizes when GPU memory is limited.
This skill documents how to implement GA that produces mathematically equivalent results to
standard batching, enabling fair A/B comparisons.

## When to Apply

Use this knowledge when:
- Comparing training with different batch sizes
- Debugging gradient accumulation implementations
- Need reproducible training runs with GA enabled
- Validating that GA implementation is correct

## Results Summary

| Metric | Baseline (bs=64, ga=1) | GA (bs=16, ga=4) | Notes |
|--------|------------------------|------------------|-------|
| Train loss (100 steps) | 2.434814453125 | 2.434814453125 | Bit-identical |
| Val loss (100 steps) | 2.47094004 | 2.47094028 | ~2e-7 difference (FP precision) |
| Speed | ~12 it/s | ~8 it/s | GA ~3-4x slower per step |

## Recommended Practice

### 1. Use dedicated RNG for data sampling

Isolate data sampling randomness from model operations:

```python
data_rng = torch.Generator()
data_rng.manual_seed(seed)

# In get_batch or data loader:
indices = torch.randint(data_size, (batch_size,), generator=data_rng)
```

### 2. Pre-sample full effective batch, then split

Ensures GA and non-GA see identical data per optimizer step:

```python
def get_micro_batches(effective_batch_size, micro_batch_size, ga_steps):
    # Sample all indices at once
    all_indices = torch.randint(data_size, (effective_batch_size,), generator=data_rng)

    # Split into micro-batches
    micro_batches = []
    for i in range(ga_steps):
        start = i * micro_batch_size
        end = start + micro_batch_size
        micro_batches.append(get_data(all_indices[start:end]))
    return micro_batches
```

### 3. Scale loss before backward

```python
for micro_batch in micro_batches:
    loss = model(micro_batch)
    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()  # Gradients accumulate automatically
```

### 4. Use effective_batch_size for validation

Ensures validation sees identical data regardless of micro-batch size:

```python
# In validation loop
val_batch = get_batch(batch_size=effective_batch_size)  # Not micro_batch_size
```

### 5. Enable full determinism

```python
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| Independent micro-batch sampling | Each micro-batch samples different random indices | Pre-sample full effective batch, then split |
| Using global torch RNG | Model forward/backward ops consume random state between steps | Use dedicated `torch.Generator()` for data |
| Different validation batch sizes | GA used micro_batch_size, baseline used full batch_size | Always use effective_batch_size for validation |
| `wandb/` folder in project root | Python imports local `wandb/` directory instead of package | Rename to `wandb_logs/` or move outside project |

## Mathematical Equivalence

For mean-reduction cross-entropy loss:

**Baseline (single batch):**
```
loss = sum(CE_i for i in 0..N) / N
grad = d(loss)/d(w)
```

**GA (K micro-batches of size N/K):**
```
loss_k = sum(CE_i for i in batch_k) / (N/K)
scaled_loss_k = loss_k / K
total_grad = sum(d(scaled_loss_k)/d(w) for k in 0..K)
           = sum((1/K) * d(loss_k)/d(w))
           = sum((1/K) * (K/N) * sum(d(CE_i)/d(w)))
           = (1/N) * sum(d(CE_i)/d(w))
           = d(loss)/d(w)  # Identical to baseline!
```

## Complete Training Loop Example

```python
import torch

# Dedicated RNG for reproducible data sampling
data_rng = torch.Generator()

def train(model, optimizer, total_steps, batch_size, ga_steps, effective_batch_size, seed):
    # Reset seed before training
    set_seed(seed)
    data_rng.manual_seed(seed)

    for step in range(total_steps):
        optimizer.zero_grad()

        # Get all micro-batches (pre-sampled from same effective batch)
        micro_batches = get_micro_batches(
            effective_batch_size, batch_size, ga_steps
        )

        accumulated_loss = 0.0
        for xb, yb in micro_batches:
            loss = model(xb, yb)
            scaled_loss = loss / ga_steps
            accumulated_loss += loss.item()
            scaled_loss.backward()

        optimizer.step()

        avg_loss = accumulated_loss / ga_steps
        # avg_loss is now identical whether ga_steps=1 or ga_steps>1
```

## Related Files

- `run_ga_experiment.py` - Runner script implementing this pattern
- `transformer_playground.py` - Original training script with GA support
