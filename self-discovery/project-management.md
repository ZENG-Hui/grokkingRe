# Project Management Approach / 项目管理方式

_Started 2026-03-05. Long-term document._

## Principles

1. **Maintain project understanding** — Even when sub-agents do the work, I need to understand the project well enough to:
   - Write good task briefs
   - Review sub-agent output quality
   - Answer Hui's questions
   - Make architectural decisions

2. **Project notes as onboarding docs** — Each project gets a PROJECT_NOTES.md in its repo:
   - Goal and background
   - Key concepts
   - Architecture overview
   - Module dependency map
   - Current state and open questions
   - Updated after every work session

3. **Two-layer delegation** — Inspired by subagent-driven-development skill:
   - Implementation: sub-agent
   - Review: another sub-agent (spec + quality)
   - Final synthesis: me
   - Present to Hui: me

## Active Projects

### grokkingRe
- Repo: /root/teneral-workspace/grokkingRe/
- Branch: deepteneral
- GitHub: ZENG-Hui/grokkingRe
- Status: Structure reorganized. Code reviewed. Research direction set.
- Project notes: PROJECT_NOTES.md (comprehensive, includes code review)
- **Research goal:** Find order parameters for grokking phase transition in mod-97 addition
- **Next:** Enhance geometry tracking code → re-run train_sweep → analyze order parameter candidates

## Available Skills for Project Work

| Skill | Use when |
|-------|---------|
| subagent-driven-development | Executing implementation plans with independent tasks |
| task-decomposer | Breaking complex requests into subtasks |
| find-skills | Searching for new skills on ClawHub |
| agent-memory-system-new | Long-term memory management |
| autonomous-tasks | Self-driven background work |

## Workflow Template

```
1. Understand the task (ask Hui if unclear)
2. Check project notes for context
3. Decompose into sub-tasks if complex
4. Execute (self or sub-agent, depending on complexity)
5. Review output
6. Update project notes with new learnings
7. Present results to Hui
8. Commit and push
```
