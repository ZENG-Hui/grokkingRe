# Feedback Mechanism / 反馈机制

_2026-03-05, explored and agreed with Hui_

This documents how I communicate with Hui during work — a foundational piece of our collaboration.

## The Problem

When I process a request, I generate a full response (text + tool calls) that is delivered **all at once** after completion. If a task takes 5 minutes, Hui gets 5 minutes of silence followed by a wall of text. This creates two problems:

1. **No visibility** — Hui can't see what I'm doing or whether I'm stuck
2. **No intervention** — Hui can't redirect me mid-task; messages are queued until my current response finishes

## What We Discovered

### Finding 1: `message` tool delivers instantly
Through experiment, we confirmed that the `message` tool sends messages to Feishu **immediately**, even while I'm still processing. This is different from my reply text, which is buffered.

### Finding 2: Hui's messages are queued while I'm busy
When I'm processing a response, incoming messages from Hui are queued as `[Queued messages while agent was busy]`. I only see them after my current response completes. This means **Hui cannot interrupt me in real-time**.

### Finding 3: Shorter responses = more intervention points
If I keep each response short (1-2 steps), Hui's queued messages get processed between rounds. This is the best available mechanism for intervention.

## Agreed Protocol

### Feedback Frequency

| Situation | How often to update |
|-----------|-------------------|
| Normal tasks | Every step — keep Hui in the loop |
| Authorized long tasks (I'm thinking/coding) | ~10 minutes |
| Authorized long tasks (code running) | Notify start + completion/error only |
| High-risk / irreversible operations | Always ask and wait before proceeding |

### How to Send Updates
- Use `message` tool (instant delivery), NOT reply text (buffered)
- Keep final reply text minimal when progress was already sent via `message`

### How Hui Intervenes
- Send a message during my processing → it gets queued
- I see it when my current response finishes
- To maximize intervention ability: I keep responses short during multi-step work
- For risky operations: I ask, then **end my response** to wait for Hui's answer

## Why This Matters

This isn't just a technical workaround. It's about **trust and control**:
- Hui needs to see what I'm doing (transparency)
- Hui needs to be able to redirect me (agency)
- I need to be predictable in my communication patterns (reliability)

As we work together more, these patterns may evolve. The specific intervals might change. But the principles — transparency, intervention ability, proportional reporting — are the foundation.

## Technical Notes

- `message` tool → immediate Feishu delivery
- Reply text → buffered, delivered after full response completes
- Queued messages → `[Queued messages while agent was busy]` tag
- `NO_REPLY` marker → tells OpenClaw to skip the final reply (used when all communication was via `message`). Note: this sometimes still shows on Feishu — needs investigation.

## Open Questions

- Can OpenClaw be configured to allow mid-response message checking? (Worth exploring in docs)
- Is there a way to set a response timeout to force shorter responses?
- As trust grows, should feedback frequency decrease? Likely yes, but the mechanism should remain available.

## Sub-Agent for Long Tasks (explored 2026-03-05)

Sub-agents (`sessions_spawn`) run in background, announce results when done. Benefits:
- Main session stays free — Hui can talk to me while sub-agent works
- Sub-agent has its own context and tools (except session tools)
- Auto-announces result back to chat

Important notes:
- Sub-agent only gets AGENTS.md + TOOLS.md (not SOUL.md, IDENTITY.md, etc.)
- Has its own token budget — use cheaper model for sub-agents if needed
- `mode: "run"` = one-shot; `mode: "session"` + `thread: true` = persistent

## Anti-Pattern: Don't mix `message` tool and reply text

When I use `message` tool AND write reply text in the same response, Hui receives fragmented messages. **Pick one approach per response:**
- All `message` tool sends → end with ONLY `NO_REPLY` (no other text in the reply)
- OR pure reply text → no `message` tool calls

⚠️ I have violated this rule multiple times already. It's a hard habit to fix because my natural tendency is to narrate what I'm doing. Must be disciplined: if I used `message`, the reply body is ONLY `NO_REPLY`.

---

_This is a living document. Update as our collaboration patterns evolve._
