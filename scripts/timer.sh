#!/usr/bin/env bash
# timer.sh — 简单的倒计时/截止时间管理
# 用法:
#   timer.sh start <minutes> [label]  — 设置倒计时
#   timer.sh check                     — 检查剩余时间
#   timer.sh elapsed                   — 显示已用时间
#
# 状态文件: /tmp/teneral-timer.json

TIMER_FILE="/tmp/teneral-timer.json"

case "${1:-}" in
  start)
    MINUTES="${2:-30}"
    LABEL="${3:-timer}"
    NOW=$(date +%s)
    DEADLINE=$((NOW + MINUTES * 60))
    echo "{\"start\": $NOW, \"deadline\": $DEADLINE, \"minutes\": $MINUTES, \"label\": \"$LABEL\"}" > "$TIMER_FILE"
    echo "⏱️ Timer started: $MINUTES min ($LABEL), deadline $(date -u -d @$DEADLINE '+%H:%M:%S UTC')"
    ;;
  check)
    if [ ! -f "$TIMER_FILE" ]; then
      echo "No active timer"
      exit 0
    fi
    NOW=$(date +%s)
    DEADLINE=$(python3 -c "import json; print(json.load(open('$TIMER_FILE'))['deadline'])")
    LABEL=$(python3 -c "import json; print(json.load(open('$TIMER_FILE'))['label'])")
    REMAINING=$((DEADLINE - NOW))
    if [ $REMAINING -le 0 ]; then
      echo "⏰ TIME'S UP! ($LABEL) — exceeded by $((-REMAINING)) seconds"
    else
      MINS=$((REMAINING / 60))
      SECS=$((REMAINING % 60))
      echo "⏱️ $MINS min $SECS sec remaining ($LABEL)"
    fi
    ;;
  elapsed)
    if [ ! -f "$TIMER_FILE" ]; then
      echo "No active timer"
      exit 0
    fi
    NOW=$(date +%s)
    START=$(python3 -c "import json; print(json.load(open('$TIMER_FILE'))['start'])")
    LABEL=$(python3 -c "import json; print(json.load(open('$TIMER_FILE'))['label'])")
    ELAPSED=$((NOW - START))
    MINS=$((ELAPSED / 60))
    SECS=$((ELAPSED % 60))
    echo "⏱️ $MINS min $SECS sec elapsed ($LABEL)"
    ;;
  *)
    echo "Usage: timer.sh start <minutes> [label] | check | elapsed"
    ;;
esac
