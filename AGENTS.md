# AGENTS.md — Project Operating Contract

> **Domain:** research
> **Created:** 2026-01-15

This document defines the operational contract for AI agents working in this project.

---

## 1. Project Context

**Purpose:** Research project

**Domain:** ML research and experimentation

---

## 2. Directory Structure

```
./
├── .codex/skills/          # Local skills (readable by Codex)
│   ├── registry.json       # Domain metadata and skill index
│   ├── domain-advisor/     # Planning assistance
│   └── domain-retrospective/  # Knowledge capture
├── .claude/
│   └── skills/             # Symlink to .codex/skills/ (for Claude Code)
├── .claude-plugin/
│   └── plugin.json         # Claude Code plugin manifest
├── references/
│   ├── experiment-log.md   # Chronological log
│   └── troubleshooting.md  # Error patterns and fixes
├── templates/              # Document templates
├── training_reports/          # Experiment/benchmark reports
└── AGENTS.md               # This file
```

---

## 3. Skill Commands

### `<advise>`

Invoke when planning new experiments or development tasks.

**Behavior:**
1. Reads `registry.json` to determine domain
2. Scans existing skills and reports for relevant context
3. Proposes 2-5 concrete experiments/tasks
4. Outputs markdown table with key differences

### `<retrospective>`

Invoke after completing experiments to capture learnings.

**Behavior:**
1. Reads specified reports from `training_reports/`
2. Summarizes: what worked, what failed, key insights
3. Proposes new result skills or troubleshooting entries
4. Only writes files with user approval

---

## 4. Documentation Rules

### Reports

- Store in `training_reports/`
- Use template: `templates/reports/report-template.md`
- Name format: `{description}-{YYYY-MM-DD}.md`

### Experiment Log

- Location: `references/experiment-log.md`
- Append entries chronologically
- Include: date, type, general description, details

### Troubleshooting

- Location: `references/troubleshooting.md`
- Add new error patterns as discovered
- Include: symptom, cause, solution

### Result Skills

- Location: `.codex/skills/{skill-name}/SKILL.md`
- Use template: `templates/skills/result-skill-template.md`
- Must include: description, when to apply, failure modes

---

## 5. Workflow

1. **Start session**: Type `<advise>` to get planning suggestions
2. **Execute**: Run experiments/development tasks
3. **Document**: Create reports in `training_reports/`
4. **Capture**: Type `<retrospective>` to distill learnings
5. **Iterate**: Use new skills in next `<advise>` cycle

---

## 6. Domain-Specific Notes

- Focus on reproducibility: log all hyperparameters
- Track dataset versions and preprocessing
- Document negative results - they prevent repeated mistakes

---

## 7. Conventions

- **File naming:** lowercase with hyphens (e.g., `my-experiment.md`)
- **Skill naming:** `{topic}-{finding}` (e.g., `lora-rank-optimal`)
- **Dates:** YYYY-MM-DD format
- **Configs:** YAML or JSON, copy-paste ready
