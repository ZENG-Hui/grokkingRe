#!/bin/bash
# hg - 透传命令到 HG opencode（带本地 session 映射）
# 用法:
#   hg <prompt>                 新 session 执行
#   hg #name <prompt>           用命名 session 执行（自动复用）
#   hg --sessions               列出所有映射的 session
#   hg --rm <name>              删除本地 session 映射

BASE_URL="https://yinghuo-hg.deepseek.com/zenghui/dev-cpu/agent-server"
TOKEN="T1moNCxHPLy4VpVlMFJlDpjPVtUT-tezJCSp0RzUmGo"
PROXY="http://172.29.4.175:22222"
MAP_FILE="/root/.openclaw/workspace/memory/hg-sessions.json"

[ -f "$MAP_FILE" ] || echo '{}' > "$MAP_FILE"

# --sessions
if [ "$1" = "--sessions" ]; then
    python3 -c "
import json
m = json.load(open('$MAP_FILE'))
if not m:
    print('(no sessions)')
else:
    for name, sid in sorted(m.items()):
        print(f'  {name}  ->  {sid}')
"
    exit 0
fi

# --rm
if [ "$1" = "--rm" ] && [ -n "$2" ]; then
    python3 -c "
import json, sys
m = json.load(open('$MAP_FILE'))
name = sys.argv[1]
if name in m:
    del m[name]
    json.dump(m, open('$MAP_FILE', 'w'), indent=2)
    print(f'deleted: {name}')
else:
    print(f'not found: {name}')
" "$2"
    exit 0
fi

SESSION_NAME=""
SESSION_ID=""

# 解析 #session_name
if [[ "$1" == \#* ]]; then
    SESSION_NAME="${1#\#}"
    shift
    SESSION_ID=$(python3 -c "
import json
m = json.load(open('$MAP_FILE'))
print(m.get('$SESSION_NAME', ''))
")
fi
PROMPT="$*"

if [ -z "$PROMPT" ]; then
    echo "hg [#session] <prompt>"
    echo "hg --sessions"
    echo "hg --rm <name>"
    exit 1
fi

# 构建 JSON（用 session 字段传 ses_xxx）
PAYLOAD=$(python3 -c "
import json, sys
d = {'message': ' '.join(sys.argv[1:])}
sid = '$SESSION_ID'
if sid:
    d['session'] = sid
print(json.dumps(d))
" $PROMPT)

# 调用 API
RESULT=$(curl -s --max-time 300 --proxy "$PROXY" \
    -X POST "$BASE_URL/opencode" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" 2>&1)

# 解析结果 + 更新映射
export HG_RESULT="$RESULT"
export HG_SNAME="$SESSION_NAME"
export HG_MAP="$MAP_FILE"

python3 << 'PYEOF'
import json, sys, os

raw = os.environ.get('HG_RESULT', '')
sname = os.environ.get('HG_SNAME', '')
map_file = os.environ.get('HG_MAP', '')

try:
    d = json.loads(raw)
except:
    print(raw)
    sys.exit(0)

new_sid = d.get('session_id', '')
if sname and new_sid and map_file:
    try:
        m = json.load(open(map_file))
        m[sname] = new_sid
        json.dump(m, open(map_file, 'w'), indent=2)
    except:
        pass

if d.get('ok'):
    text = d.get('text', '')
    if text:
        print(text)
    for w in d.get('warnings', []):
        print(f'[WARNING] {w}')
    for e in d.get('events', []):
        if e.get('type') == 'step_finish':
            t = e.get('part', {}).get('tokens', {})
            if t:
                print(f'[tokens: in={t.get("input",0)} out={t.get("output",0)}]')
elif 'detail' in d:
    print(f'ERROR: {d["detail"]}')
else:
    print(json.dumps(d, indent=2, ensure_ascii=False))
PYEOF
