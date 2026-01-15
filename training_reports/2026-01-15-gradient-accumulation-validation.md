# Training Report: Gradient Accumulation Validation

**Date:** 2026-01-15
**Author:** Claude
**Status:** Complete

## Objective

Validate that gradient accumulation produces mathematically equivalent training to standard batching by achieving bit-identical loss values.

## Experimental Setup

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | TransformerLM (character-level) |
| Vocab size | 87 |
| Embedding size | 256 |
| Attention heads | 4 |
| Layers | 6 |
| Total parameters | 4,844,631 |
| Sequence length | 256 |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-3 |
| LR scheduler | CosineAnnealingLR (eta_min=1e-6) |
| Total steps | 100 (validation), 1000 (full runs) |
| Seed | 42 |

### Dataset

- **Source:** animesubs.txt (character-level)
- **Train/Val split:** 99% / 1%
- **Preprocessing:** Filtered non-ASCII characters (ord < 0x3000)

## Experiment Configurations

| Run | batch_size | ga_steps | effective_batch | GPU |
|-----|------------|----------|-----------------|-----|
| Baseline | 64 | 1 | 64 | RTX 4090 |
| GA | 16 | 4 | 64 | RTX 4090 |

## Results

### Final Metrics (100 steps)

| Metric | Baseline | GA | Difference |
|--------|----------|----|-----------:|
| Train loss | 2.434814453125 | 2.434814453125 | 0 (bit-identical) |
| Val loss | 2.47094004154 | 2.47094027996 | ~2e-7 |
| Time | ~8s | ~12s | 1.5x slower |

### Loss Curves

Both runs showed identical training dynamics:
- Step 0: 4.64 (initial loss)
- Step 50: ~2.8
- Step 100: ~2.43

### wandb Runs

- Baseline: https://wandb.ai/erlandpg/transformers-playground/runs/quz6sljo
- GA: https://wandb.ai/erlandpg/transformers-playground/runs/n2eop79u

## Implementation Details

### Key Requirements for Determinism

1. **Dedicated RNG for data sampling:**
   ```python
   data_rng = torch.Generator()
   data_rng.manual_seed(seed)
   indices = torch.randint(..., generator=data_rng)
   ```

2. **Pre-sample effective batch, then split:**
   ```python
   all_indices = torch.randint(size=(effective_batch_size,), generator=data_rng)
   micro_batches = [all_indices[i*bs:(i+1)*bs] for i in range(ga_steps)]
   ```

3. **Scale loss before backward:**
   ```python
   scaled_loss = loss / gradient_accumulation_steps
   scaled_loss.backward()
   ```

4. **Full determinism settings:**
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```

### Issues Encountered and Fixes

| Issue | Root Cause | Solution |
|-------|------------|----------|
| Step 1+ indices diverged | Global RNG consumed by model ops | Dedicated `torch.Generator()` |
| Val loss mismatch | Different batch sizes for validation | Use `effective_batch_size` |
| `wandb` import failed | Local `wandb/` folder shadowed package | Renamed to `wandb_logs/` |

## Conclusions

1. **Gradient accumulation is mathematically equivalent** to standard batching when implemented correctly.

2. **Bit-identical training loss** achieved with:
   - Isolated RNG for data sampling
   - Pre-sampled batch splitting
   - Proper loss scaling

3. **Validation loss** differs only at floating-point precision (~2e-7).

4. **Trade-off:** GA is ~1.5x slower per step but enables larger effective batches with limited GPU memory.

## Files Created/Modified

- `transformer_playground.py` - Added `gradient_accumulation_steps` parameter
- `run_ga_experiment.py` - Standalone runner for GA experiments
- `.codex/skills/gradient-accumulation-deterministic/SKILL.md` - Reusable skill

## Next Steps

- [ ] Test with dropout enabled
- [ ] Validate with different optimizers (SGD, AdaFactor)
- [ ] Test in distributed setting (FSDP/DDP)
- [ ] Run full 1000-step comparison for user's next experiment
