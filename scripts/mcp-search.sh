#!/usr/bin/env bash
# MCP Search — wrapper for the MCP Search Server
# Usage: bash scripts/mcp-search.sh "query string" [query2] [query3] ...
#
# MCP Server: http://10.9.200.200:30130/mcp (Search Server v2.14.3)
# Protocol: JSON-RPC over HTTP with SSE
# Returns: webpage titles, URLs, and full content (markdown)

set -euo pipefail

MCP_URL="http://10.9.200.200:30130/mcp"

if [ $# -eq 0 ]; then
    echo "Usage: mcp-search.sh \"query\" [query2] ..."
    exit 1
fi

# Build queries JSON array
QUERIES="["
for q in "$@"; do
    QUERIES="${QUERIES}\"$(echo "$q" | sed 's/"/\\"/g')\","
done
QUERIES="${QUERIES%,}]"

# Call MCP search
RESPONSE=$(curl -sL --max-time 30 \
    -H "Accept: application/json, text/event-stream" \
    -H "Content-Type: application/json" \
    -X POST "$MCP_URL" \
    -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"search\",\"arguments\":{\"queries\":$QUERIES}}}" 2>/dev/null)

# Extract and output the text content
echo "$RESPONSE" | python3 -c "
import sys, json, re

data = sys.stdin.read()
for line in data.split('\n'):
    if line.startswith('data:'):
        try:
            j = json.loads(line[5:])
            content = j.get('result', {}).get('content', [])
            for item in content:
                text = item.get('text', '')
                # Extract pages
                pages = re.split(r'\[webpage \d+ begin\]', text)
                for page in pages[1:]:  # skip first empty
                    title_m = re.search(r'\[webpage title\](.*?)\n', page)
                    url_m = re.search(r'\[webpage url\](.*?)\n', page)
                    content_m = re.search(r'\[webpage content begin\]\n(.*?)\[webpage content end\]', page, re.DOTALL)
                    
                    title = title_m.group(1).strip() if title_m else '?'
                    url = url_m.group(1).strip() if url_m else '?'
                    body = content_m.group(1).strip()[:500] if content_m else ''
                    
                    print(f'## {title}')
                    print(f'URL: {url}')
                    print(f'{body}')
                    print()
        except Exception as e:
            print(f'Error parsing response: {e}', file=sys.stderr)
" 2>/dev/null
