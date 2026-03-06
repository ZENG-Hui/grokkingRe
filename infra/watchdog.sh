#!/bin/bash
# Teneral 看门狗 — 检测 feishu session 卡住并通过飞书 API 直接告警
# 独立于 OpenClaw 运行，即使 gateway 死掉也能告警
#
# 检测逻辑（双条件，降低误报）：
#   1. feishu session 的 run 持续 active 超过 STUCK_THRESHOLD 分钟
#   2. 最近 LOG_SILENT_THRESHOLD 分钟内日志无 tool 活动
#
# 安装: crontab -e → */5 * * * * bash /home/ubuntu/zenghui/openclaw/watchdog.sh

set -euo pipefail

# === 配置 ===
STUCK_THRESHOLD=15           # run 持续时间超过此值(分钟)才可能告警
LOG_SILENT_THRESHOLD=5       # 日志无 tool 活动超过此值(分钟)才确认卡住
CONTAINER="claw"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STATE_FILE="${SCRIPT_DIR}/watchdog-state"
CONFIG_BACKUP="${SCRIPT_DIR}/openclaw.json.backup"
# 冷却：告警后 30 分钟内不重复告警
COOLDOWN_MIN=30

# 从 openclaw.json.backup 动态读取飞书凭据（不硬编码密钥）
if [ ! -f "$CONFIG_BACKUP" ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ERROR: $CONFIG_BACKUP not found, cannot send alerts"
    exit 1
fi
FEISHU_APP_ID=$(python3 -c "import json; print(json.load(open('$CONFIG_BACKUP'))['channels']['feishu']['appId'])")
FEISHU_APP_SECRET=$(python3 -c "import json; print(json.load(open('$CONFIG_BACKUP'))['channels']['feishu']['appSecret'])")
FEISHU_USER_ID=$(python3 -c "
import json
c = json.load(open('$CONFIG_BACKUP'))
pairs = c.get('channels',{}).get('feishu',{}).get('pairing',{}).get('approved',{})
# 取第一个已批准用户
for uid in pairs:
    print(uid)
    break
")

# === 函数 ===
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

send_feishu_alert() {
    local msg="$1"
    # 获取 tenant_access_token
    local token_resp
    token_resp=$(curl -s --max-time 10 -X POST "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal" \
        -H "Content-Type: application/json" \
        -d "{\"app_id\":\"${FEISHU_APP_ID}\",\"app_secret\":\"${FEISHU_APP_SECRET}\"}")
    local token
    token=$(echo "$token_resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('tenant_access_token',''))" 2>/dev/null)
    if [ -z "$token" ]; then
        log "ERROR: Failed to get feishu token"
        return 1
    fi
    # 发送消息
    curl -s --max-time 10 -X POST "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id" \
        -H "Authorization: Bearer $token" \
        -H "Content-Type: application/json" \
        -d "{\"receive_id\":\"${FEISHU_USER_ID}\",\"msg_type\":\"text\",\"content\":\"{\\\"text\\\":\\\"${msg}\\\"}\"}" >/dev/null 2>&1
    log "Alert sent to Feishu"
}

# === 冷却检查 ===
if [ -f "$STATE_FILE" ]; then
    last_alert=$(cat "$STATE_FILE" 2>/dev/null || echo 0)
    now=$(date +%s)
    elapsed=$(( (now - last_alert) / 60 ))
    if [ "$elapsed" -lt "$COOLDOWN_MIN" ]; then
        exit 0
    fi
fi

# === 检查容器是否运行 ===
if ! sudo docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    send_feishu_alert "⚠️ 看门狗告警：容器 ${CONTAINER} 未运行！"
    date +%s > "$STATE_FILE"
    exit 0
fi

# === 检查 gateway 进程 ===
gw_alive=$(sudo docker exec "$CONTAINER" pgrep -c "openclaw" 2>/dev/null || echo 0)
if [ "$gw_alive" -lt 2 ]; then
    send_feishu_alert "⚠️ 看门狗告警：Teneral 的 gateway 进程已死亡！需要从宿主机重启：bash /home/ubuntu/zenghui/openclaw/openclaw-start.sh"
    date +%s > "$STATE_FILE"
    exit 0
fi

# === 检查 session 是否卡住 ===
# 获取 feishu session 最后一次 run registered 和 run cleared 的时间
check_result=$(sudo docker exec "$CONTAINER" python3 -c "
import json, time, os, re

LOG_DIR = '/tmp/openclaw'
now = time.time()

# 找当天日志文件
from datetime import datetime, timezone
today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
log_file = f'{LOG_DIR}/openclaw-{today}.log'

if not os.path.exists(log_file):
    print('NO_LOG')
    exit()

# 从文件尾部读取最后 200KB（足够找最近的 tool 活动）
file_size = os.path.getsize(log_file)
read_size = min(file_size, 200 * 1024)

with open(log_file, 'rb') as f:
    f.seek(max(0, file_size - read_size))
    tail = f.read().decode('utf-8', errors='ignore')

lines = tail.strip().split('\n')

# 找最后一次 feishu session 的 run registered 和 run cleared
last_run_start = None
last_run_clear = None
last_tool_activity = None
feishu_sid = '834e4f8c'

for line in lines:
    try:
        obj = json.loads(line)
        ts_str = obj.get('time', '')
        msg = obj.get('1', '')
        if not isinstance(msg, str):
            continue
        # Parse timestamp
        if ts_str:
            from datetime import datetime as dt
            t = dt.fromisoformat(ts_str.replace('Z', '+00:00')).timestamp()
        else:
            continue

        if feishu_sid in msg or 'totalActive' in msg:
            if 'run registered' in msg and feishu_sid in msg:
                last_run_start = t
            elif 'run cleared' in msg and feishu_sid in msg:
                last_run_clear = t

        if 'tool start' in msg or 'tool end' in msg:
            last_tool_activity = t
    except:
        continue

if last_run_start is None:
    print('NO_RUN')
elif last_run_clear is not None and last_run_clear >= last_run_start:
    print('IDLE')
else:
    run_duration_min = (now - last_run_start) / 60
    if last_tool_activity:
        tool_silent_min = (now - last_tool_activity) / 60
    else:
        tool_silent_min = run_duration_min
    print(f'ACTIVE {run_duration_min:.1f} {tool_silent_min:.1f}')
" 2>/dev/null)

if [ -z "$check_result" ]; then
    exit 0
fi

status=$(echo "$check_result" | awk '{print $1}')

if [ "$status" = "ACTIVE" ]; then
    run_min=$(echo "$check_result" | awk '{print $2}')
    silent_min=$(echo "$check_result" | awk '{print $3}')

    # 转为整数比较
    run_int=${run_min%.*}
    silent_int=${silent_min%.*}

    if [ "$run_int" -ge "$STUCK_THRESHOLD" ] && [ "$silent_int" -ge "$LOG_SILENT_THRESHOLD" ]; then
        send_feishu_alert "⚠️ 看门狗告警：Teneral 疑似卡住！run 已持续 ${run_min} 分钟，最近 ${silent_min} 分钟无 tool 活动。可能需要从宿主机重启：bash /home/ubuntu/zenghui/openclaw/openclaw-start.sh （先 stop 再 start）"
        date +%s > "$STATE_FILE"
    fi
fi
