# Experiment Log

This file tracks experiment plans, decisions, and retrospectives in chronological order.

## Format

Each entry should include:
- **Date**: YYYY-MM-DD
- **Type**: Plan | Observation | Retrospective
- **General description**: One sentence for non-technical context
- **Details**: What was planned/observed/learned

---

## 2026-01-15 – Retrospective: Gradient Accumulation Determinism

**Type:** Retrospective

**General description:** Implemented and validated gradient accumulation to produce bit-identical training results compared to standard batching.

### What we tried

- Implemented gradient accumulation in `transformer_playground.py`
- Created `run_ga_experiment.py` for controlled A/B testing
- Compared: baseline (bs=64, ga=1) vs GA (bs=16, ga=4), both effective_batch=64
- Ran 100-step experiments with seed=42

### Key findings

- **Train loss bit-identical:** 2.434814453125 for both configurations
- **Val loss nearly identical:** ~2e-7 difference (floating point precision)
- Requires: dedicated RNG, pre-sampled batches, loss scaling

### What failed

1. Independent micro-batch sampling → different data per step
2. Global RNG → model ops consumed random state
3. Different val batch sizes → inconsistent validation

### Outcome

Created skill: `gradient-accumulation-deterministic`

---

<!-- New entries go above this line -->
