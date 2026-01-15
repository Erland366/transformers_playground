---
name: domain-retrospective
description: >
  Turn experiment reports and development notes into summaries and reusable skills.
  Adapts behavior based on project domain (research, unsloth, cuda) by reading registry.json.
  Triggers on <retrospective> or requests for lessons learned.
metadata:
  short-description: "Summarize findings and distill them into skills"
  tags:
    - documentation
    - retrospective
    - knowledge-capture
---

# Skill: domain-retrospective

## When to use

Use this skill when:
- The user message starts with `<retrospective>`, or
- The user requests a summary or lessons-learned across experiments/development.

## Initialization

1. Read `.codex/skills/registry.json` to determine:
   - `domain`: research | unsloth | cuda
   - `paths.reports`: where to find experiment/benchmark reports
   - `paths.experiment_log`: path to experiment log
   - `paths.troubleshooting`: path to troubleshooting guide
   - `paths.templates`: path to templates directory

2. Adapt behavior based on domain (see Domain-Specific Behavior below).

## Behavior

1. **Select inputs**
   - Use the user's description to identify relevant:
     - Reports from `paths.reports` directory
     - Sections of `paths.experiment_log`
   - If ambiguous, list candidate reports and ask the user to choose.

2. **Summarize findings**
   - For each report, extract:
     - Setup and configuration
     - Key parameters/settings
     - Metrics and results
     - What worked (successes)
     - What failed (with reasons)
   - Write a markdown summary with:
     - "What we tried"
     - "Key findings"
     - "What failed"
     - "Open questions"

3. **Update troubleshooting (if needed)**
   - If experiments reveal new error patterns and fixes:
     - Propose new entries for `paths.troubleshooting`
     - Use template from `templates/references/troubleshooting-entry-template.md`
     - Ask user for confirmation before editing.

4. **Propose or update result skills**
   - Decide what **result skills** should capture these findings.
   - For each skill:
     - If new: start from `templates/skills/result-skill-template.md`
     - If existing: identify which sections to update
   - Draft SKILL.md content including:
     - General description and context
     - When to apply this knowledge
     - Results summary with concrete numbers
     - Recommended practice
     - Failure modes to avoid
   - Use domain-appropriate terminology and focus areas.

5. **Ask before writing**
   - Present the proposed skill changes.
   - Only create or modify files under `.codex/skills/` with user approval.

6. **Log the retrospective**
   - Append a summarized entry to `paths.experiment_log`
   - Example: "2025-01-12 – Retrospective on LoRA rank experiments"
   - Include a short "General description" line for context.

---

## Domain-Specific Behavior

### Research Domain

When `domain: research`:

**What to extract from reports:**
- Model architecture details
- Training hyperparameters (lr, batch_size, epochs, warmup)
- Dataset configurations and mixtures
- Evaluation metrics (accuracy, loss, perplexity)
- Training dynamics (convergence speed, stability)

**Result skill focus:**
- Hyperparameter recommendations for specific tasks
- Dataset mixture recipes
- Model architecture insights
- Training tips and tricks

**Skill naming convention:**
- `{task}-{finding}` e.g., `colbert-chunking-optimal`, `gpt2-lr-schedule`

### Unsloth Domain

When `domain: unsloth`:

**What to extract from reports:**
- LoRA configuration (rank, alpha, target_modules)
- Quantization settings
- Memory usage and batch sizes achieved
- Fine-tuning duration and throughput
- Model-specific quirks

**Result skill focus:**
- Optimal LoRA configurations for model families
- Memory-efficient training recipes
- Quantization tradeoffs
- Common fine-tuning pitfalls

**Skill naming convention:**
- `{model}-{config}` e.g., `llama3-lora-optimal`, `mistral-4bit-recipe`

### CUDA Domain

When `domain: cuda`:

**What to extract from reports:**
- Kernel configurations (block sizes, grid dims)
- Memory access patterns
- Bandwidth and FLOPS achieved
- Occupancy and register usage
- Profiling metrics (from nsight/ncu)

**Result skill focus:**
- Optimal tiling strategies for operations
- Memory coalescing patterns
- Warp-level optimization techniques
- Triton autotuning configurations

**Skill naming convention:**
- `{operation}-{optimization}` e.g., `softmax-online`, `matmul-tiled`, `attention-flash`

---

## Result Skill Template

The generated result skill should follow this structure:

```markdown
---
name: {skill-name}
description: >
  {One-line description with trigger conditions}
  Use when: {specific scenarios}
metadata:
  short-description: "{Brief tagline}"
  tags:
    - {tag1}
    - {tag2}
  domain: {research|unsloth|cuda}
  created: {YYYY-MM-DD}
  author: {name}
---

# {Skill Name}

## General Description

{2-3 sentences on what this skill captures and why it matters}

## When to Apply

Use this knowledge when:
- {Condition 1}
- {Condition 2}

## Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| {metric1} | {value1} | {notes1} |

## Recommended Practice

{Concrete, actionable recommendations with specific values}

## Failure Modes

| What Failed | Why | Lesson |
|-------------|-----|--------|
| {attempt1} | {reason1} | {lesson1} |

## Configuration

{Copy-paste ready configuration, if applicable}
```

---

## Example Output

### Research Retrospective

```markdown
## Retrospective: Attention Head Experiments (Jan 2025)

### What we tried
- Varied attention heads from 4 to 12 on GPT-2 small architecture
- Fixed: lr=1e-4, batch_size=32, 10 epochs

### Key findings
- 6 heads achieved 91.5% accuracy (vs 92% baseline with 8 heads)
- 4 heads dropped to 87% - too aggressive
- Wider FFN (4096) partially compensated for fewer heads

### What failed
- 4 heads without FFN compensation: 87% accuracy
- 12 heads: no improvement, just slower training

### Open questions
- Would 6 heads + deeper network work better?
- Test on larger model scales

---

**Proposed skill:** `attention-head-scaling`
```

### Unsloth Retrospective

```markdown
## Retrospective: Llama-3 Fine-tuning (Jan 2025)

### What we tried
- LoRA ranks: 8, 16, 32 on Llama-3 8B
- Quantization: 4-bit vs 8-bit
- Gradient checkpointing variations

### Key findings
- rank=16 + 4-bit optimal for A100 40GB
- rank=32 needed CPU offload, 2x slower
- 8-bit gave marginal quality improvement, not worth memory cost

### What failed
- rank=8: underfitting on complex tasks
- Full fine-tune: OOM even with offload

---

**Proposed skill:** `llama3-lora-optimal`
```

### CUDA Retrospective

```markdown
## Retrospective: Softmax Kernel Optimization (Jan 2025)

### What we tried
- 1D tiling (baseline)
- 2D tiling with various block sizes
- Warp-level reduction
- Online softmax algorithm

### Key findings
- 2D tiling (64x64) achieved 95% bandwidth utilization
- Online softmax 1.5x faster for attention fusion
- Warp shuffles eliminated shared memory bank conflicts

### What failed
- BLOCK_M=128: register spilling, 30% slowdown
- Naive reduction: bank conflicts killed performance

---

**Proposed skill:** `softmax-online`
```
