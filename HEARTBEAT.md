# HEARTBEAT.md

**⛔ 你必须按顺序执行以下每一步。不允许跳过任何步骤。不允许在执行完所有步骤之前回复 HEARTBEAT_OK。**

## 步骤 1：检查主 session 的 context 使用率

⚠️ **不要用 session_status（那是 heartbeat session 自己的 context）。** 用以下脚本读主 session 的实际 token 用量：

```bash
python3 -c "
import json
d = json.load(open('/root/.openclaw/agents/main/sessions/sessions.json'))
for k, v in d.items():
    if 'feishu' in k and isinstance(v, dict) and 'totalTokens' in v:
        total = v['totalTokens']
        limit = v.get('contextTokens', 200000)
        pct = total / limit * 100
        print(f'MAIN_SESSION_CONTEXT: {total}/{limit} ({pct:.0f}%)')
        break
"
```

- 记录百分比，如果 > 70% 需要在💓消息中标注预警（但 context 高低**不影响**是否发送💓）

## 步骤 2：调用 subagents list
- 记录：有没有活跃的 sub-agent？有没有最近完成的？

## 步骤 3：检测活跃 session + 用户消息检测 + sub-agent 完成检测

运行以下脚本：
```bash
HEARTBEAT_SID=$(python3 -c "import json; print(json.load(open('/root/.openclaw/agents/main/sessions/sessions.json')).get('agent:main:main', {}).get('sessionId', ''))" 2>/dev/null)
THRESHOLD=$(date -d '5 minutes ago' +%s)

# 找飞书用户 session ID
FEISHU_SID=$(python3 -c "import json; d=json.load(open('/root/.openclaw/agents/main/sessions/sessions.json')); [print(v['sessionId']) for k,v in d.items() if 'feishu' in k and isinstance(v,dict) and 'sessionId' in v]" 2>/dev/null | head -1)

# 检测活跃 session（排除 heartbeat 和主 session）
echo "=== Active sessions (last 5 min, excluding heartbeat) ==="
ACTIVE_SIDS=""
for f in ~/.openclaw/agents/main/sessions/*.jsonl; do
  SID=$(basename "$f" .jsonl)
  if [ "$SID" != "$HEARTBEAT_SID" ]; then
    MTIME=$(stat -c '%Y' "$f")
    if [ "$MTIME" -gt "$THRESHOLD" ]; then
      SIZE=$(stat -c '%s' "$f")
      echo "  $SID  mtime=$(date -d @$MTIME '+%H:%M:%S')  size=${SIZE}"
      if [ "$SID" != "$FEISHU_SID" ]; then
        ACTIVE_SIDS="${ACTIVE_SIDS}${SID},"
      fi
    fi
  fi
done
echo "ACTIVE_NON_MAIN_SIDS=$ACTIVE_SIDS"

# 检测用户最后一条消息的时间戳（不用文件 mtime，因为 bot 回复也会更新 mtime）
LAST_USER_MSG_TS=$(python3 -c "
import json
last_ts = 0
with open('/root/.openclaw/agents/main/sessions/${FEISHU_SID}.jsonl') as f:
    for line in f:
        try:
            obj = json.loads(line)
            msg = obj.get('message', {})
            if msg.get('role') == 'user':
                ts = obj.get('timestamp', '')
                if isinstance(ts, str) and ts:
                    from datetime import datetime
                    ts = int(datetime.fromisoformat(ts.replace('Z','+00:00')).timestamp())
                    if ts > last_ts:
                        last_ts = ts
        except: pass
print(last_ts)
" 2>/dev/null || echo 0)
LAST_SENT=$(python3 -c "import json; print(json.load(open('/root/.openclaw/workspace/memory/heartbeat-state.json')).get('lastHeartbeatSentAt',0))" 2>/dev/null || echo 0)
echo "LAST_USER_MSG_TS=$LAST_USER_MSG_TS LAST_SENT=$LAST_SENT"
```
如果有 SID 在 ACTIVE 里但不在 TRACKED 里 → 加入 `trackedSubagents`

### C. 是否发 💓
- **必须发送（满足任一即发）：** LAST_USER_MSG_TS > LAST_SENT（用户有新消息）、有活跃 sub-agent、有刚完成的 sub-agent
- **跳过：** 以上条件都不满足 → 直接跳到步骤 6
- 注意：context 使用率**不是**发送触发条件。即使 context > 70%，如果没有上述条件之一，也不发💓（但如果因其他条件发了💓，消息中要包含 context 预警）
- ⚠️ **不要自行修改此检测逻辑。** 如果你认为有 bug，先报告给 Hui

### D. 更新 heartbeat-state.json
用 exec 写入更新后的状态：
```bash
python3 -c "
import json
state = {'lastHeartbeatSentAt': $(date +%s), 'trackedSubagents': [<更新后的列表>]}
json.dump(state, open('/root/.openclaw/workspace/memory/heartbeat-state.json','w'))
"
```

## 步骤 4：读取项目进度（仅在需要发 💓 时执行）

⚠️ **如果步骤 3 判定跳过，不要执行此步骤。**

运行：
```bash
ls -lt ~/.openclaw/workspace/memory/projects/ 2>/dev/null
```
然后用 read 工具读取 `memory/projects/` 下的 progress 文件。

## 步骤 5：发送 💓 消息（仅在需要发 💓 时执行）

⚠️ **如果步骤 3 判定跳过，不要执行此步骤。**

用 `message` 工具发送 💓，内容必须包含：
- context 百分比（> 70% 标注预警）
- 活跃 session/sub-agent 状态
- **项目进度摘要**（来自步骤 4 的 progress 文件）
- 如果刚唤醒了主 session，注明"已自动唤醒主 session 继续任务"
- 如果没有活跃 sub-agent 且无待处理任务，末尾加一句"如无进一步指示，将进入静默"

## 步骤 6：进度持久化（仅在主 session context > 70% 或有任务完成时）
- 更新 progress 文件
- 写入未持久化的决策/指导

## 步骤 7：回复 HEARTBEAT_OK
