# How to Be a Good Leader / 如何管好子 Agent

_Started 2026-03-05. Long-term document — update as I learn._

Hui's insight: managing sub-agents is essentially **leadership**. I need to learn how to:
- Give clear task briefs (downward)
- Summarize results clearly (upward, to Hui)
- Quality-check before forwarding

This is not just a technical skill — it's a craft that improves with practice.

---

## Task Brief Template

Every sub-agent task should include:

```
## Task: [clear one-line description]

### Context
[What the sub-agent needs to know. Be specific — it has no memory of our conversations.]

### Objective
[What exactly to do. Numbered steps if complex.]

### Constraints
- [What NOT to do — e.g., "READ-ONLY, do not modify files"]
- [Resource limits — e.g., "stay under 2 minutes"]

### Safety
- Before reading any file, check its size first (ls -lh or wc -l)
- Do not read files > 500 lines; note their size and skip
- [Any other warnings specific to this task]

### Output Rules
1. Write detailed results to: [specific file path]
2. Your final response must be SHORT (max 5 lines) — just a summary
3. Do NOT put full reports in your final response
```

## Lessons Learned

### 2026-03-05: grokkingRe project reorganization

**What worked:**
- Clear task brief with Safety + Output Rules → sub-agent produced clean, reviewable work
- "Analysis first, then execute" approach → Hui could approve plan before changes
- git mv preserves history → good practice for refactoring

**What to improve:**
- Sub-agent for code understanding produces summaries, but I should **read the code myself** for anything that requires judgment (e.g., "are these metrics valid?")
- Delegating understanding ≠ delegating judgment. Sub-agents are good workers, not good critics.

## Workflow: Sub-Agent Task Lifecycle

```
1. PLAN    — I understand what Hui wants, design the task brief
2. BRIEF   — Write clear task with context, constraints, output rules
3. LAUNCH  — sessions_spawn, notify Hui it's started
4. WAIT    — Stay available for Hui, monitor if needed
5. REVIEW  — Read sub-agent's output file, quality-check
6. PRESENT — Summarize findings to Hui, flag decisions needed
7. RECORD  — Note what worked/didn't for future improvement
```

## Anti-Patterns to Avoid

| Don't | Do Instead |
|-------|------------|
| Let sub-agent announce raw results | Require file output + brief summary |
| Skip quality review | Always read output before presenting |
| Give vague tasks | Use the template above |
| Forget safety warnings | Always include file-size checks, scope limits |
| Over-delegate sensitive work | Handle high-risk operations myself |

## Open Questions

- How much context to include? Too little → bad output. Too much → token waste.
- Should I create reusable task templates for common patterns?
- When to use cheaper model for sub-agents?
- How to handle partial failures gracefully?

---

_This document will grow as I get better at this. Review periodically._
