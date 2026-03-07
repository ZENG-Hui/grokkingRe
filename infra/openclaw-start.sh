#!/bin/bash
# OpenClaw (Teneral) 启动脚本
# 包含完整安全配置：kill wrapper + watchdog crontab
set -euo pipefail

CONTAINER="claw"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🦞 启动 OpenClaw..."

# 1. 确保容器运行
sudo docker start "$CONTAINER" 2>/dev/null || true
sleep 2

# 2. 清理旧进程（用绝对路径绕过 kill wrapper）
sudo docker exec "$CONTAINER" bash -c "/bin/pkill openclaw; /bin/pkill socat" 2>/dev/null || true
sleep 1

# 3. 安装 kill wrapper（防止 agent 自杀）
echo "🔒 安装安全防护..."
sudo docker exec "$CONTAINER" bash -c '
# kill wrapper
cat > /usr/local/bin/kill <<'\''WRAPPER'\''
#!/bin/bash
REAL_KILL=/bin/kill
for arg in "$@"; do
    [[ "$arg" == -* ]] && continue
    [[ ! "$arg" =~ ^[0-9]+$ ]] && continue
    PROC_NAME=$(cat /proc/$arg/comm 2>/dev/null)
    if [[ "$PROC_NAME" == openclaw* ]]; then
        echo "BLOCKED: Cannot send signal to openclaw process (PID $arg). Use host start/stop scripts instead." >&2
        exit 1
    fi
done
exec $REAL_KILL "$@"
WRAPPER
chmod +x /usr/local/bin/kill

# killall wrapper
cat > /usr/local/bin/killall <<'\''WRAPPER2'\''
#!/bin/bash
for arg in "$@"; do
    [[ "$arg" == -* ]] && continue
    if [[ "$arg" == *openclaw* || "$arg" == *gateway* ]]; then
        echo "BLOCKED: Cannot killall openclaw processes. Use host start/stop scripts instead." >&2
        exit 1
    fi
done
exec /usr/bin/killall "$@"
WRAPPER2
chmod +x /usr/local/bin/killall

# pkill wrapper
cat > /usr/local/bin/pkill <<'\''WRAPPER3'\''
#!/bin/bash
for arg in "$@"; do
    [[ "$arg" == -* ]] && continue
    if [[ "$arg" == *openclaw* || "$arg" == *gateway* ]]; then
        echo "BLOCKED: Cannot pkill openclaw processes. Use host start/stop scripts instead." >&2
        exit 1
    fi
done
exec /usr/bin/pkill "$@"
WRAPPER3
chmod +x /usr/local/bin/pkill
'

# 4. 安装 hg 透传脚本（飞书 !hg 命令直达 HG opencode）
sudo docker cp "${SCRIPT_DIR}/hg.sh" "${CONTAINER}:/usr/local/bin/hg"
sudo docker exec "$CONTAINER" chmod +x /usr/local/bin/hg

# 5. 同步宿主机脚本到容器内 workspace/infra/（供 git 备份）
echo "📂 同步 infra 文件..."
sudo docker exec "$CONTAINER" mkdir -p /root/.openclaw/workspace/infra
for f in openclaw-start.sh openclaw-stop.sh watchdog.sh hg.sh; do
    sudo docker cp "${SCRIPT_DIR}/${f}" "${CONTAINER}:/root/.openclaw/workspace/infra/"
done
for f in handover-guide.md openclaw-guide.md container-environment.md feishu-commands.md multi-agent-feasibility.md; do
    [ -f "${SCRIPT_DIR}/README/${f}" ] && sudo docker cp "${SCRIPT_DIR}/README/${f}" "${CONTAINER}:/root/.openclaw/workspace/infra/"
done

# 6. 启动 socat + gateway
sudo docker exec -d "$CONTAINER" bash -c "socat TCP-LISTEN:18789,fork,reuseaddr,bind=0.0.0.0 TCP:127.0.0.1:18788 & nohup openclaw gateway run > /tmp/openclaw/openclaw-nohup.log 2>&1 &"
sleep 3

# 7. 确保 watchdog crontab 已安装
CRON_ENTRY="*/5 * * * * bash ${SCRIPT_DIR}/watchdog.sh >> ${SCRIPT_DIR}/watchdog.log 2>&1"
if ! crontab -l 2>/dev/null | grep -qF "watchdog.sh"; then
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
    echo "🐕 看门狗 crontab 已安装"
else
    echo "🐕 看门狗 crontab 已存在"
fi

# 8. 验证
sudo docker exec "$CONTAINER" ps aux | grep -E "openclaw|socat" | grep -v grep
echo ""
echo "✅ 启动完成！关掉电脑也不会停。"
echo "浏览器打开：http://localhost:18789/#token=da1a6138d38986b1983d34d78e8ef165c11f06895088c4ff"
