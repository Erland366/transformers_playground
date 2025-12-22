# Gated Softpick Seq Experiment

**Worktree branch** testing softpick gating (normalized over sequence dimension) as an alternative to sigmoid gating.

## Overview

This experiment replaces standard sigmoid gating with **softpick gating** that normalizes over the sequence dimension, causing gates to **compete across token positions**.

## Key Change

In `transformer_playground.py` (~line 457):

```python
# Before (sigmoid - independent gates)
out = out * torch.sigmoid(gate_score)

# After (softpick - gates compete across sequence)
out = out * softpick(gate_score, dim=1)
```

## Softpick Function

```python
def softpick(x, dim: int = -1, eps: float = 1e-8):
    """softpick: relu(exp(x)-1) / sum(abs(exp(x)-1)) - creates competition"""
    x_m = torch.max(x, dim=dim, keepdim=True).values
    x_m_e_m = torch.exp(-x_m)
    x_e_1 = torch.exp(x - x_m) - x_m_e_m
    r_x_e_1 = F.relu(x_e_1)
    a_x_e_1 = torch.where(x.isfinite(), torch.abs(x_e_1), 0)
    return r_x_e_1 / (torch.sum(a_x_e_1, dim=dim, keepdim=True) + eps)
```

## Hypothesis

| Aspect | Sigmoid | Softpick (seq dim) |
|--------|---------|-------------------|
| Formula | `σ(x)` | `relu(exp(x)-1) / sum(abs(exp(x)-1))` |
| Normalization | None (elementwise) | Normalized over sequence |
| Competition | Independent | **Tokens compete** for attention |
| Sparsity | No explicit zeros | relu() zeros weak signals |

## WandB

- **Project:** `transformers-playground`
- **Run name:** `gated_softpick_seq-lm-animesubs-256seq-256embed-4head-6layer.AMD`

## Next Steps

If results are promising:
1. Try other dimensions: `head_dim` (features compete), `heads` (heads compete)
2. Port to `fla/ops/attn/` for Triton implementation
3. Full-scale training in `flame/` framework
