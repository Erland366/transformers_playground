
---

### 4.2 `templates/reports/training-report-template.md`

```markdown
# Experiment: <short-name>

- **Date**: YYYY-MM-DD
- **Author**: your-name / Codex-assisted
- **Goal**: one-paragraph description
- **General description**: short, non-technical summary
- **Models**: list
- **Datasets**: list

---

## 1. Setup

### 1.1 Model & task

- Model(s): ...
- Task(s): ...
- Any important architecture notes

### 1.2 Data

- Dataset names + sizes
- Preprocessing / filtering / splits

### 1.3 Base hyperparameters

Common settings across runs:

- max_length, batch_size, epochs
- optimizer, scheduler, warmup, etc.
- hardware (e.g. 1×L40S)

---

## 2. Runs

### 2.1 Run table

| run_id | config_name / label | key_param_changed | job_id / link | logs_file |
|--------|---------------------|-------------------|---------------|-----------|
| r1     |                     |                   |               |           |
| r2     |                     |                   |               |           |

### 2.2 Notes per run (optional)

- r1: ...
- r2: ...

---

## 3. Results

### 3.1 Metrics

| run_id | main_metric_name | main_metric_value | other_metrics | notes |
|--------|------------------|-------------------|--------------|-------|
|        |                  |                   |              |       |

Link to detailed metrics in `logs/metrics/` as needed.

### 3.2 Plots / qualitative observations

- Describe trends or link to external dashboards/plots.

---

## 4. Analysis

- What patterns are clear?
- Which configurations are best and why?
- Any surprising or non-obvious findings?

---

## 5. Lessons learned → candidate skills

List bullet points that should eventually become or update skills:

- Candidate skill 1: ...
- Candidate skill 2: ...

Mention relevant files:
- This report: `training_reports/<filename>.md`
- Logs: `logs/raw/...`, `logs/metrics/...`
