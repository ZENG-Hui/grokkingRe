# HEARTBEAT.md

**⛔ 你必须按顺序执行以下每一步。不允许跳过任何步骤。不允许在执行完所有步骤之前回复 HEARTBEAT_OK。**

## 步骤 1：调用 session_status
- 检查 context 使用率
- 记录是否 > 70%

## 步骤 2：调用 subagents list
- 记录：有没有活跃的 sub-agent？有没有最近完成的？

## 步骤 3：检测活跃 session + 时间戳比较 + sub-agent 完成检测

运行以下脚本：
```bash
HEARTBEAT_SID="5b0fdcce-940a-4cd4-8a1a-f8c63ae6a116"
THRESHOLD=$(date -d '5 minutes ago' +%s)

# 找飞书用户 session ID
FEISHU_SID=$(python3 -c "import json; d=json.load(open('/root/.openclaw/agents/main/sessions/sessions.json')); [print(v['sessionId']) for k,v in d.items() if 'feishu' in k and isinstance(v,dict) and 'sessionId' in v]" 2>/dev/null | head -1)

# 检测活跃 session
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

# 时间戳比较
SESSION_MTIME=$(stat -c '%Y' ~/.openclaw/agents/main/sessions/${FEISHU_SID}.jsonl 2>/dev/null || echo 0)
LAST_SENT=$(python3 -c "import json; print(json.load(open('/root/.openclaw/workspace/memory/heartbeat-state.json')).get('lastHeartbeatSentAt',0))" 2>/dev/null || echo 0)
echo "SESSION_MTIME=$SESSION_MTIME LAST_SENT=$LAST_SENT"

# 读取已追踪的 sub-agent 列表
TRACKED=$(python3 -c "import json; print(','.join(json.load(open('/root/.openclaw/workspace/memory/heartbeat-state.json')).get('trackedSubagents',[])))" 2>/dev/null || echo "")
echo "TRACKED_SUBAGENTS=$TRACKED"
```

## 判断逻辑（按优先级）

### A. Sub-agent 完成检测 → 唤醒主 session
对比 `TRACKED_SUBAGENTS` 和 `ACTIVE_NON_MAIN_SIDS`：
- 如果有 SID 在 TRACKED 里但不在 ACTIVE 里 → **该 sub-agent 刚完成**
- 执行唤醒：
  ```bash
  nohup openclaw agent --session-id <FEISHU_SID> --message "[auto-heartbeat] Sub-agent completed. Run 'subagents list' to check results, then continue with next steps." --deliver --channel feishu > /dev/null 2>&1 &
  ```
- 从 trackedSubagents 中移除已完成的 SID

### B. 新 sub-agent 检测 → 加入追踪
如果有 SID 在 ACTIVE 里但不在 TRACKED 里 → 加入 `trackedSubagents`

### C. 是否发 💓
- **必须发送：** SESSION_MTIME > LAST_SENT（有新交互）、context > 70%、有活跃 sub-agent、有刚完成的 sub-agent
- **跳过：** 以上条件都不满足 → 直接跳到步骤 6

### D. 更新 heartbeat-state.json
用 exec 写入更新后的状态（包含新的 lastHeartbeatSentAt 和 trackedSubagents）：
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

## 步骤 6：进度持久化（仅在 context > 70% 或有任务完成时）
- 更新 progress 文件
- 写入未持久化的决策/指导

## 步骤 7：回复 HEARTBEAT_OK
