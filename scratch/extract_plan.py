#!/usr/bin/env python3
"""Extract thesis revision plan from session transcript."""

import json
from datetime import datetime

JSONL_PATH = "/root/.openclaw/agents/main/sessions/9cc0fc24-081d-42be-a33f-8bf8ba0ede01.jsonl"

KEYWORDS = ["phase", "阶段", "计划", "plan", "步骤", "工作量", "优先", "诊断"]

def parse_timestamp(ts):
    if not ts:
        return None
    if isinstance(ts, str):
        for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"]:
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
    return None

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)

def has_keywords(text):
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in KEYWORDS)

results = []
all_messages = []

with open(JSONL_PATH, 'r') as f:
    for line_no, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        # Only process message type records
        if record.get("type") != "message":
            continue
            
        ts = record.get("timestamp")
        dt = parse_timestamp(ts)
        
        if not dt or dt.year != 2026 or dt.month != 3 or dt.day != 5:
            continue
        if not (14 <= dt.hour < 16):
            continue
            
        msg = record.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", "")
        text = extract_text(content)
        
        if not text.strip():
            continue
            
        all_messages.append({
            "line": line_no,
            "time": dt.strftime("%H:%M:%S"),
            "role": role,
            "text": text,
        })
        
        if has_keywords(text):
            results.append({
                "line": line_no,
                "time": dt.strftime("%H:%M:%S"),
                "role": role,
                "text": text,
            })

print(f"=== Total messages in 14:00-16:00 window: {len(all_messages)} ===\n")

# Show all messages briefly
print("=== ALL MESSAGES (first 300 chars) ===\n")
for msg in all_messages:
    preview = msg['text'][:300].replace('\n', '\\n')
    print(f"[L{msg['line']}] {msg['time']} [{msg['role']}]: {preview}")
    print("---")

print(f"\n=== KEYWORD MATCHES: {len(results)} ===\n")
for r in results:
    print(f"[L{r['line']}] {r['time']} [{r['role']}]:")
    print(r['text'][:3000])
    if len(r['text']) > 3000:
        print(f"... [{len(r['text'])} total chars]")
    print("=" * 60 + "\n")
