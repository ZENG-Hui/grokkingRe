#!/usr/bin/env python3
"""
Daily Research Digest v2
- arXiv keyword search (physics + AI)
- arXiv author watch
- Company/lab monitoring (GitHub API + Anthropic page scraping)
- Incremental detection (only report new items)

Usage: python3 daily_research.py [--days N] [--max-per-keyword M] [--reset-baseline]
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import json
import os
import sys
import re
import time
from datetime import datetime, timedelta

# === Configuration ===

KEYWORDS_PHYSICS = [
    "cavity material",
    "cavity materials",
    "quantum geometry",
    "AI scientist",
    "AI for science",
]

KEYWORDS_AI = [
    "grokking",
    "self-improving",
    "iterative self-refinement",
    "self-evolving",
    "data efficiency",
    "sample efficiency",
    "mechanistic interpretability",
    "scaling laws",
    "in-context learning",
]

WATCHED_AUTHORS = [
    "Kaiming He",
    "Jason Wei",
    "Zico Kolter",
    "Neel Nanda",
    "Chris Olah",
    "Tri Dao",
    "Ilya Sutskever",
]

# GitHub orgs to monitor for new repos/releases
GITHUB_ORGS = [
    {"name": "OpenAI", "org": "openai"},
    {"name": "DeepSeek", "org": "deepseek-ai"},
    {"name": "ByteDance Seed", "org": "bytedance-seed"},
]

# Web pages to scrape for new research posts
SCRAPE_SOURCES = [
    {
        "name": "Anthropic",
        "url": "https://www.anthropic.com/research",
        "pattern": r'href="(/research/(?!team/)[^"]+)"',
        "base_url": "https://www.anthropic.com",
    },
]

# RSS feeds to monitor
RSS_FEEDS = [
    {
        "name": "DeepMind",
        "url": "https://deepmind.google/blog/rss.xml",
    },
]

# MCP Search queries for company updates (fallback/supplement)
MCP_SEARCH_URL = "http://10.9.200.200:30130/mcp"
MCP_QUERIES = [
    "OpenAI latest research release this week",
    "Moonshot AI Kimi latest model release 2026",
]

# Jina Reader sources (for JS-rendered SPA sites)
JINA_SOURCES = [
    {
        "name": "Moonshot",
        "url": "https://moonshot.ai/research",
        # Parse: title + date + link from markdown like "[Title\n---\nYYYY-MM-DD](link)"
        "pattern": r'\[([^\]]+?)\s*-+\s*(\d{4}-\d{2}-\d{2})\]\((https?://[^\)]+)\)',
    },
    {
        "name": "Seed",
        "url": "https://seed.bytedance.com/en/research",
        # Seed uses numbered blog entries with titles
        "pattern": r'(\d{2})\n\n([^\n]+(?:Released|Announced|Launched)[^\n]*)',
    },
]

# === Args ===
days_back = 3
max_per_keyword = 3
reset_baseline = False

for i, arg in enumerate(sys.argv):
    if arg == "--days" and i + 1 < len(sys.argv):
        days_back = int(sys.argv[i + 1])
    if arg == "--max-per-keyword" and i + 1 < len(sys.argv):
        max_per_keyword = int(sys.argv[i + 1])
    if arg == "--reset-baseline":
        reset_baseline = True

NS = {'a': 'http://www.w3.org/2005/Atom'}
CUTOFF = datetime.utcnow() - timedelta(days=days_back)
WORKSPACE = os.environ.get('WORKSPACE', os.path.expanduser('~/.openclaw/workspace'))
STATE_PATH = os.path.join(WORKSPACE, 'memory', 'research-state.json')


def load_state():
    """Load previous state for incremental detection."""
    if reset_baseline or not os.path.exists(STATE_PATH):
        return {"seen_arxiv": [], "seen_github_repos": {}, "seen_research_links": {}}
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except:
        return {"seen_arxiv": [], "seen_github_repos": {}, "seen_research_links": {}}


def save_state(state):
    """Save state for next run."""
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    # Keep only last 500 seen arxiv IDs to prevent unbounded growth
    state["seen_arxiv"] = state["seen_arxiv"][-500:]
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def fetch_url(url, timeout=15):
    """Fetch URL with error handling."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'DailyResearchBot/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8')
    except Exception as e:
        print(f"  [ERROR] Failed to fetch {url}: {e}", file=sys.stderr)
        return None


def search_arxiv(query, max_results=10):
    """Search arXiv API and return list of paper dicts."""
    encoded = urllib.parse.quote(query)
    url = (f"https://export.arxiv.org/api/query?"
           f"search_query=ti:{encoded}+OR+abs:{encoded}"
           f"&start=0&max_results={max_results}"
           f"&sortBy=lastUpdatedDate&sortOrder=descending")
    data = fetch_url(url)
    if not data:
        return []

    root = ET.fromstring(data)
    papers = []
    for entry in root.findall('a:entry', NS):
        published_str = entry.find('a:published', NS).text[:10]
        published = datetime.strptime(published_str, "%Y-%m-%d")
        if published < CUTOFF:
            continue
        
        title = entry.find('a:title', NS).text.strip().replace('\n', ' ').replace('  ', ' ')
        summary = entry.find('a:summary', NS).text.strip().replace('\n', ' ')[:200]
        authors = [a.find('a:name', NS).text for a in entry.findall('a:author', NS)]
        arxiv_id = entry.find('a:id', NS).text
        
        papers.append({
            'title': title,
            'authors': authors,
            'summary': summary,
            'link': arxiv_id,
            'published': published_str,
        })
    return papers


def search_author(author_name, max_results=3):
    """Search arXiv for recent papers by a specific author."""
    encoded = urllib.parse.quote(f'"{author_name}"')
    url = (f"https://export.arxiv.org/api/query?"
           f"search_query=au:{encoded}"
           f"&start=0&max_results={max_results}"
           f"&sortBy=lastUpdatedDate&sortOrder=descending")
    data = fetch_url(url)
    if not data:
        return []

    root = ET.fromstring(data)
    papers = []
    for entry in root.findall('a:entry', NS):
        published_str = entry.find('a:published', NS).text[:10]
        published = datetime.strptime(published_str, "%Y-%m-%d")
        if published < CUTOFF:
            continue
        
        title = entry.find('a:title', NS).text.strip().replace('\n', ' ').replace('  ', ' ')
        authors = [a.find('a:name', NS).text for a in entry.findall('a:author', NS)]
        arxiv_id = entry.find('a:id', NS).text
        
        papers.append({
            'title': title,
            'authors': authors,
            'link': arxiv_id,
            'published': published_str,
        })
    return papers


def check_github_orgs(state):
    """Check GitHub orgs for new/updated repos (last 3 days)."""
    new_items = []
    cutoff_str = CUTOFF.strftime("%Y-%m-%d")
    
    for org_info in GITHUB_ORGS:
        org = org_info["org"]
        name = org_info["name"]
        url = f"https://api.github.com/orgs/{org}/repos?sort=created&per_page=10"
        data = fetch_url(url)
        if not data:
            continue
        
        try:
            repos = json.loads(data)
            if not isinstance(repos, list):
                continue
        except:
            continue
        
        seen_key = org
        prev_seen = set(state.get("seen_github_repos", {}).get(seen_key, []))
        current_repos = []
        
        for r in repos:
            repo_name = r.get("name", "")
            created = r.get("created_at", "")[:10]
            current_repos.append(repo_name)
            
            if created >= cutoff_str and repo_name not in prev_seen:
                desc = r.get("description", "") or ""
                stars = r.get("stargazers_count", 0)
                new_items.append({
                    'source': name,
                    'title': repo_name,
                    'description': desc[:150],
                    'link': r.get("html_url", f"https://github.com/{org}/{repo_name}"),
                    'date': created,
                    'stars': stars,
                })
        
        # Update seen repos
        if "seen_github_repos" not in state:
            state["seen_github_repos"] = {}
        state["seen_github_repos"][seen_key] = current_repos[:20]
        
        time.sleep(0.5)  # Rate limit courtesy
    
    return new_items


def check_research_pages(state):
    """Scrape research pages for new posts."""
    new_items = []
    
    for source in SCRAPE_SOURCES:
        name = source["name"]
        url = source["url"]
        pattern = source["pattern"]
        base_url = source["base_url"]
        
        data = fetch_url(url)
        if not data:
            continue
        
        links = re.findall(pattern, data)
        seen_key = name
        prev_seen = set(state.get("seen_research_links", {}).get(seen_key, []))
        current_links = []
        
        for href in links:
            if href in current_links:
                continue
            current_links.append(href)
            
            if href not in prev_seen:
                # Convert slug to readable title
                slug = href.split('/')[-1]
                title = slug.replace('-', ' ').title()
                new_items.append({
                    'source': name,
                    'title': title,
                    'link': f"{base_url}{href}",
                    'date': datetime.utcnow().strftime("%Y-%m-%d"),
                })
        
        if "seen_research_links" not in state:
            state["seen_research_links"] = {}
        state["seen_research_links"][seen_key] = current_links[:50]
    
    return new_items


def check_rss_feeds(state):
    """Check RSS feeds for new posts."""
    new_items = []
    cutoff_str = CUTOFF.strftime("%Y-%m-%d")
    
    for feed_info in RSS_FEEDS:
        name = feed_info["name"]
        url = feed_info["url"]
        
        data = fetch_url(url)
        if not data:
            continue
        
        try:
            # Handle potential namespace issues by stripping media namespace
            data_clean = re.sub(r'<media:[^>]*/?>', '', data)
            root = ET.fromstring(data_clean)
        except ET.ParseError as e:
            print(f"  [ERROR] Failed to parse RSS from {name}: {e}", file=sys.stderr)
            continue
        
        seen_key = f"rss_{name}"
        prev_seen = set(state.get("seen_research_links", {}).get(seen_key, []))
        current_guids = []
        
        for item in root.findall('.//item'):
            title_el = item.find('title')
            link_el = item.find('link')
            guid_el = item.find('guid')
            desc_el = item.find('description')
            
            title = title_el.text if title_el is not None else '?'
            link = link_el.text if link_el is not None else '?'
            guid = guid_el.text if guid_el is not None else link
            description = desc_el.text if desc_el is not None else ''
            
            current_guids.append(guid)
            
            if guid not in prev_seen:
                new_items.append({
                    'source': name,
                    'title': title,
                    'description': description[:150] if description else '',
                    'link': link,
                    'date': datetime.utcnow().strftime("%Y-%m-%d"),
                })
        
        if "seen_research_links" not in state:
            state["seen_research_links"] = {}
        state["seen_research_links"][seen_key] = current_guids[:50]
    
    return new_items


def check_jina_sources(state):
    """Use Jina Reader to extract content from JS-rendered pages."""
    new_items = []
    
    for source in JINA_SOURCES:
        name = source["name"]
        url = source["url"]
        jina_url = f"https://r.jina.ai/{url}"
        
        data = fetch_url(jina_url, timeout=20)
        if not data:
            continue
        
        seen_key = f"jina_{name}"
        prev_seen = set(state.get("seen_research_links", {}).get(seen_key, []))
        current_titles = []
        
        if name == "Moonshot":
            # Parse Moonshot format: [Title\n---\nYYYY-MM-DD](link)
            # Also handle: [Title --- YYYY-MM-DD](link) on single line
            matches = re.findall(
                r'\[([^\]\n]+?)[\s\-]*(\d{4}-\d{2}-\d{2})\]\((https?://[^\)]+)\)',
                data
            )
            for title, date, link in matches:
                title = title.strip().rstrip('-').strip()
                current_titles.append(title)
                if title not in prev_seen:
                    new_items.append({
                        'source': name,
                        'title': title,
                        'link': link,
                        'date': date,
                    })
        
        elif name == "Seed":
            # Parse Seed: look for blog entry titles
            # Pattern: "XX\n\nTitle Released/Announced"
            lines = data.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if 'Released' in line or 'Announced' in line or 'Launched' in line:
                    title = line
                    current_titles.append(title)
                    if title not in prev_seen:
                        new_items.append({
                            'source': name,
                            'title': title,
                            'link': url,
                            'date': datetime.utcnow().strftime("%Y-%m-%d"),
                        })
        
        if "seen_research_links" not in state:
            state["seen_research_links"] = {}
        state["seen_research_links"][seen_key] = current_titles[:30]
        
        time.sleep(1)  # Be nice to Jina
    
    return new_items


def check_mcp_search(state):
    """Use MCP Search Server for company updates that other methods can't reach."""
    new_items = []
    
    for query in MCP_QUERIES:
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": {"queries": [query]}
            }
        })
        
        try:
            req = urllib.request.Request(
                MCP_SEARCH_URL,
                data=payload.encode('utf-8'),
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json, text/event-stream',
                    'User-Agent': 'DailyResearchBot/1.0',
                },
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read().decode('utf-8')
        except Exception as e:
            print(f"  [ERROR] MCP search failed for '{query}': {e}", file=sys.stderr)
            continue
        
        # Parse SSE response
        for line in data.split('\n'):
            if not line.startswith('data:'):
                continue
            try:
                j = json.loads(line[5:])
                content = j.get('result', {}).get('content', [])
                for item in content:
                    text = item.get('text', '')
                    # Extract page titles and URLs
                    pages = re.findall(
                        r'\[webpage title\](.*?)\n\[webpage url\](.*?)\n',
                        text
                    )
                    
                    seen_key = "mcp_search"
                    prev_seen = set(state.get("seen_research_links", {}).get(seen_key, []))
                    
                    for title, url in pages:
                        title = title.strip()
                        url = url.strip()
                        # Skip generic/irrelevant pages
                        if any(skip in url.lower() for skip in ['linkedin.com', 'twitter.com', 'reddit.com', 'youtube.com']):
                            continue
                        if url not in prev_seen:
                            # Determine source from URL
                            source = "Web"
                            if 'openai.com' in url:
                                source = "OpenAI"
                            elif 'moonshot' in url or 'kimi' in url:
                                source = "Moonshot"
                            elif 'deepseek' in url:
                                source = "DeepSeek"
                            elif 'anthropic' in url:
                                source = "Anthropic"
                            elif 'deepmind' in url or 'google' in url:
                                source = "DeepMind"
                            
                            new_items.append({
                                'source': source,
                                'title': title[:100],
                                'link': url,
                                'date': datetime.utcnow().strftime("%Y-%m-%d"),
                            })
                    
                    # Update seen
                    current_urls = [url.strip() for _, url in pages]
                    if "seen_research_links" not in state:
                        state["seen_research_links"] = {}
                    prev = state["seen_research_links"].get("mcp_search", [])
                    state["seen_research_links"]["mcp_search"] = list(set(prev + current_urls))[-100:]
                    
            except Exception as e:
                print(f"  [ERROR] MCP parse error: {e}", file=sys.stderr)
        
        time.sleep(0.5)
    
    return new_items


def dedupe(papers):
    """Remove duplicate papers by arXiv ID."""
    seen = set()
    unique = []
    for p in papers:
        pid = p['link']
        if pid not in seen:
            seen.add(pid)
            unique.append(p)
    return unique


def format_digest(physics_papers, ai_papers, author_papers, company_updates, today_str):
    """Format the digest as a message string."""
    lines = [f"📚 每日研究速递 ({today_str})\n"]
    
    idx = 1
    
    if physics_papers:
        lines.append("🔬 物理")
        for p in physics_papers:
            authors_str = ", ".join(p['authors'][:3])
            if len(p['authors']) > 3:
                authors_str += " et al."
            lines.append(f"{idx}. {p['title']}")
            lines.append(f"   {authors_str} | {p['published']}")
            lines.append(f"   {p['link']}")
            if p.get('summary'):
                lines.append(f"   {p['summary'][:120]}...")
            lines.append("")
            idx += 1
    
    if ai_papers:
        lines.append("🤖 AI/ML")
        for p in ai_papers:
            authors_str = ", ".join(p['authors'][:3])
            if len(p['authors']) > 3:
                authors_str += " et al."
            lines.append(f"{idx}. {p['title']}")
            lines.append(f"   {authors_str} | {p['published']}")
            lines.append(f"   {p['link']}")
            if p.get('summary'):
                lines.append(f"   {p['summary'][:120]}...")
            lines.append("")
            idx += 1
    
    if author_papers:
        lines.append("👤 关注作者新动态")
        for p in author_papers:
            authors_str = ", ".join(p['authors'][:3])
            if len(p['authors']) > 3:
                authors_str += " et al."
            lines.append(f"{idx}. {p['title']}")
            lines.append(f"   {authors_str} | {p['published']}")
            lines.append(f"   {p['link']}")
            lines.append("")
            idx += 1
    
    if company_updates:
        lines.append("🏢 业界动态")
        for item in company_updates:
            lines.append(f"{idx}. [{item['source']}] {item['title']}")
            if item.get('description'):
                lines.append(f"   {item['description'][:100]}")
            if item.get('stars') and item['stars'] > 0:
                lines.append(f"   ⭐ {item['stars']}")
            lines.append(f"   {item['link']}")
            lines.append("")
            idx += 1
    
    if idx == 1:
        lines.append("今天没有发现与关注方向匹配的新论文或动态。")
    else:
        lines.append(f"共 {idx - 1} 条。回复编号可收藏（如 \"1 3\"）。")
    
    return "\n".join(lines)


def append_to_log(all_items, today_str, log_path):
    """Append to RESEARCH_LOG.md."""
    if not all_items:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"\n## {today_str}\n\n")
        for p in all_items:
            if 'authors' in p:
                authors_str = ", ".join(p['authors'][:3])
                f.write(f"- **{p['title']}** — {authors_str}\n")
            else:
                f.write(f"- **[{p.get('source','')}] {p['title']}**\n")
            f.write(f"  {p['link']}\n")


def main():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    log_path = os.path.join(WORKSPACE, 'memory', 'RESEARCH_LOG.md')
    state = load_state()
    
    # 1. arXiv keyword search
    print("Searching arXiv for physics keywords...", file=sys.stderr)
    physics_papers = []
    for kw in KEYWORDS_PHYSICS:
        results = search_arxiv(kw, max_results=max_per_keyword)
        physics_papers.extend(results)
        print(f"  '{kw}': {len(results)} papers", file=sys.stderr)
        time.sleep(0.3)
    physics_papers = dedupe(physics_papers)
    
    print("Searching arXiv for AI/ML keywords...", file=sys.stderr)
    ai_papers = []
    for kw in KEYWORDS_AI:
        results = search_arxiv(kw, max_results=max_per_keyword)
        ai_papers.extend(results)
        print(f"  '{kw}': {len(results)} papers", file=sys.stderr)
        time.sleep(0.3)
    ai_papers = dedupe(ai_papers)
    
    # Remove physics papers that also appeared in AI
    ai_ids = {p['link'] for p in ai_papers}
    physics_papers = [p for p in physics_papers if p['link'] not in ai_ids]
    
    # Filter out previously seen arXiv papers
    seen_arxiv = set(state.get("seen_arxiv", []))
    physics_papers = [p for p in physics_papers if p['link'] not in seen_arxiv]
    ai_papers = [p for p in ai_papers if p['link'] not in seen_arxiv]
    
    # 2. Author search
    print("Searching for watched authors...", file=sys.stderr)
    author_papers = []
    for author in WATCHED_AUTHORS:
        results = search_author(author, max_results=3)
        author_papers.extend(results)
        print(f"  '{author}': {len(results)} papers", file=sys.stderr)
        time.sleep(0.3)
    author_papers = dedupe(author_papers)
    
    keyword_ids = {p['link'] for p in physics_papers + ai_papers}
    author_papers = [p for p in author_papers if p['link'] not in keyword_ids and p['link'] not in seen_arxiv]
    
    # Update seen arxiv
    all_arxiv_ids = [p['link'] for p in physics_papers + ai_papers + author_papers]
    state["seen_arxiv"] = list(seen_arxiv) + all_arxiv_ids
    
    # 3. Company monitoring
    print("Checking company GitHub repos...", file=sys.stderr)
    github_updates = check_github_orgs(state)
    print(f"  Found {len(github_updates)} new repos", file=sys.stderr)
    
    print("Checking research pages...", file=sys.stderr)
    page_updates = check_research_pages(state)
    print(f"  Found {len(page_updates)} new research posts", file=sys.stderr)
    
    print("Checking RSS feeds...", file=sys.stderr)
    rss_updates = check_rss_feeds(state)
    print(f"  Found {len(rss_updates)} new RSS items", file=sys.stderr)
    
    print("Checking Jina Reader sources...", file=sys.stderr)
    jina_updates = check_jina_sources(state)
    print(f"  Found {len(jina_updates)} new Jina items", file=sys.stderr)
    
    print("Checking MCP Search...", file=sys.stderr)
    mcp_updates = check_mcp_search(state)
    print(f"  Found {len(mcp_updates)} new MCP items", file=sys.stderr)
    
    company_updates = github_updates + page_updates + rss_updates + jina_updates + mcp_updates
    
    # Format and output
    digest = format_digest(physics_papers, ai_papers, author_papers, company_updates, today_str)
    print(digest)
    
    # Save state
    save_state(state)
    
    # Append to log
    all_items = physics_papers + ai_papers + author_papers + company_updates
    append_to_log(all_items, today_str, log_path)
    print(f"\nAppended {len(all_items)} items to {log_path}", file=sys.stderr)
    
    # Save paper index for "favorite by number" feature
    index_path = os.path.join(WORKSPACE, 'memory', 'research-last-digest.json')
    all_numbered = physics_papers + ai_papers + author_papers + company_updates
    index_data = {}
    for i, item in enumerate(all_numbered):
        entry = {'title': item['title'], 'link': item['link']}
        if 'authors' in item:
            entry['authors'] = item['authors'][:3]
        if 'source' in item:
            entry['source'] = item['source']
        index_data[str(i + 1)] = entry
    with open(index_path, 'w') as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
