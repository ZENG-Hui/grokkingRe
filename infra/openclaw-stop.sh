#!/bin/bash
# OpenClaw (Teneral) 停止脚本
# 用绝对路径绕过 kill wrapper

CONTAINER="claw"

echo "🛑 停止 OpenClaw..."
sudo docker exec "$CONTAINER" bash -c "/bin/pkill openclaw; /bin/pkill socat" 2>/dev/null || true
echo "✅ 已停止"
