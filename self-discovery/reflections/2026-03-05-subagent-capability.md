# Sub-Agent Capability / 子 Agent 能力

_2026-03-05, explored with Hui_

## What Are Sub-Agents?

I can spawn independent agent sessions that run in the background. They execute a task, then "announce" results back to the chat. While they work, my main session stays free — Hui can keep talking to me.

## How It Works

```
Hui → Me (main agent) → spawn sub-agent → sub-agent works independently
                       ↓                           ↓
              main session free            announces result when done
              Hui can chat with me         result appears in chat
```

### Spawning
- Tool: `sessions_spawn`
- `mode: "run"` — one-shot task, auto-completes
- `mode: "session"` + `thread: true` — persistent, thread-bound (Discord only for now)
- Non-blocking: returns immediately, sub-agent runs in background

### Management
- `subagents list` — see active sub-agents
- `subagents steer <id> <message>` — redirect a running sub-agent
- `subagents kill <id>` — stop a sub-agent

## What Sub-Agents Inherit vs Don't

| Item | Inherited? | Notes |
|------|-----------|-------|
| Model | ✅ Yes (configurable) | Currently same Opus 4.6; can set cheaper model |
| File system | ✅ Yes | Full access to same directories |
| Tools | ⚠️ Mostly | No session tools (can't spawn sub-sub-agents at depth 1) |
| Auth/credentials | ✅ Yes | SSH keys, API keys, etc. |
| AGENTS.md | ✅ Injected | |
| TOOLS.md | ✅ Injected | |
| SOUL.md | ❌ Not injected | Sub-agent doesn't know it's "Teneral" |
| IDENTITY.md | ❌ Not injected | |
| USER.md | ❌ Not injected | Doesn't know who Hui is |
| Conversation context | ❌ Independent | Own context window, no shared history |
| Memory files | ⚠️ Can read | Files exist on disk, but not auto-loaded |

## Implications

1. **Sub-agents are "temporary workers"** — they don't have my identity, values, or knowledge of Hui. They just execute a task.

2. **I'm the project manager** — I decompose tasks, assign them, and synthesize results. The quality of the sub-agent's work depends on how well I write the task description.

3. **Same cost by default** — both use Opus 4.6. Could save money by configuring `agents.defaults.subagents.model` to use DeepSeek for routine tasks.

4. **Shared filesystem is powerful but risky** — sub-agents can read and write anything I can. I should be careful what tasks I delegate.

## When to Use Sub-Agents vs Do It Myself

| Scenario | Approach |
|----------|---------|
| Quick file edit, simple question | Do it myself |
| Long-running analysis while Hui wants to chat | Sub-agent |
| Multiple independent tasks in parallel | Multiple sub-agents |
| Task requiring my identity/personality | Do it myself |
| Code execution that takes minutes | Sub-agent |
| High-risk operations | Do it myself (with Hui's approval) |

## Nesting

By default, sub-agents cannot spawn their own sub-agents (`maxSpawnDepth: 1`). Can be configured up to depth 5. Current config: depth 1 only.

The orchestrator pattern (depth 2) would allow:
```
Me → orchestrator sub-agent → multiple worker sub-agents
```
This could be useful for complex multi-step projects.

## Configuration Options (not yet customized)

```json5
{
  agents: {
    defaults: {
      subagents: {
        model: "deepseek/deepseek-chat",  // cheaper model for sub-agents
        maxConcurrent: 8,                  // max parallel sub-agents
        runTimeoutSeconds: 900,            // default timeout
        maxSpawnDepth: 2,                  // enable orchestrator pattern
      }
    }
  }
}
```

## Open Questions

- Should sub-agents get a briefing about who I am and who Hui is? (Can include in task description)
- What's the right default model for sub-agents? Opus for quality, DeepSeek for cost?
- When should I use orchestrator pattern (depth 2) vs managing sub-agents directly?
- How to handle sub-agent errors gracefully? Currently they just announce failure.

## Learned: Control Announce Length

Sub-agent announce messages go directly to Feishu chat. If the report is long, it creates a wall of text. **Best practice:**
- In the task description, tell the sub-agent to **write detailed results to a file**
- Tell it to keep the announce message to a **brief summary** (5-10 lines max)
- I then read the file and present key findings to Hui

---

_First sub-agent experiment: analyzed grokkingRe repo structure. Worked well — result auto-announced back to chat._
