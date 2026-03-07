#!/usr/bin/env python3
"""
Paper Life — Conway's Game of Life seeded by arXiv paper titles.

Each paper title is hashed into an initial pattern on a grid.
The simulation runs in the terminal, showing how "research ideas" 
evolve, merge, and sometimes die out — just like real science.

Created by DeepTeneral during free exploration time.
2026-03-07
"""

import hashlib
import time
import os
import sys
import json
import urllib.request
import xml.etree.ElementTree as ET


def fetch_arxiv_titles(query="grokking", max_results=5):
    """Fetch paper titles from arXiv."""
    url = f"http://export.arxiv.org/api/query?search_query=all:{query.replace(' ', '+')}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'PaperLife/1.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode('utf-8')
        root = ET.fromstring(data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        titles = []
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns)
            if title is not None and title.text:
                titles.append(title.text.strip().replace('\n', ' '))
        return titles
    except Exception as e:
        print(f"arXiv fetch failed: {e}", file=sys.stderr)
        return ["Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"]


def title_to_pattern(title, width, height):
    """Convert a paper title into a cellular automaton pattern."""
    grid = [[False] * width for _ in range(height)]
    
    # Use SHA256 of title to get deterministic but well-distributed bits
    h = hashlib.sha256(title.encode()).digest()
    
    # Method 1: Hash-based scatter
    for i, byte in enumerate(h):
        for bit in range(8):
            if byte & (1 << bit):
                x = (i * 8 + bit) * 7 % width
                y = (i * 8 + bit) * 13 % height
                grid[y][x] = True
                # Add neighbors for more interesting patterns
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = (x+dx) % width, (y+dy) % height
                    if (byte + i) % 3 == 0:
                        grid[ny][nx] = True
    
    # Method 2: Direct ASCII mapping for the title characters
    for i, ch in enumerate(title[:min(len(title), width * 2)]):
        x = (i * 3 + ord(ch)) % width
        y = (ord(ch) * 7 + i) % height
        grid[y][x] = True
    
    return grid


def step(grid, width, height):
    """One step of Conway's Game of Life."""
    new_grid = [[False] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            # Count living neighbors
            neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx = (x + dx) % width
                    ny = (y + dy) % height
                    if grid[ny][nx]:
                        neighbors += 1
            
            # Conway's rules
            if grid[y][x]:
                new_grid[y][x] = neighbors in (2, 3)  # survival
            else:
                new_grid[y][x] = neighbors == 3  # birth
    
    return new_grid


def render(grid, width, height, gen, title, alive_count):
    """Render the grid to terminal."""
    # Use block characters for better visuals
    ALIVE = "██"
    DEAD = "  "
    
    lines = []
    lines.append(f"\033[2J\033[H")  # Clear screen
    lines.append(f"╔{'═' * (width * 2)}╗")
    for y in range(height):
        row = ""
        for x in range(width):
            row += ALIVE if grid[y][x] else DEAD
        lines.append(f"║{row}║")
    lines.append(f"╚{'═' * (width * 2)}╝")
    lines.append(f"  🧬 Gen {gen:04d} | Alive: {alive_count:4d} | Seed: \"{title[:50]}...\"")
    lines.append(f"  🦀 Paper Life — by DeepTeneral")
    
    return "\n".join(lines)


def count_alive(grid, height):
    """Count living cells."""
    return sum(sum(1 for cell in row if cell) for row in grid)


def simulate(title, width=40, height=20, max_gens=200, delay=0.1, output_file=None):
    """Run the simulation and optionally save history."""
    grid = title_to_pattern(title, width, height)
    
    history = []
    seen_states = set()
    
    for gen in range(max_gens):
        alive = count_alive(grid, height)
        
        # Record history
        state_hash = hashlib.md5(str(grid).encode()).hexdigest()
        history.append({
            'gen': gen,
            'alive': alive,
            'hash': state_hash,
        })
        
        # Check for steady state or cycle
        if state_hash in seen_states:
            history.append({'gen': gen, 'alive': alive, 'event': 'cycle_detected'})
            break
        seen_states.add(state_hash)
        
        if alive == 0:
            history.append({'gen': gen, 'alive': 0, 'event': 'extinction'})
            break
        
        # Step
        grid = step(grid, width, height)
    
    return history


def analyze_results(results):
    """Analyze and compare how different papers evolve."""
    report = []
    report.append("# Paper Life — Simulation Report 🧬")
    report.append("")
    report.append("How do research paper titles evolve when used as seeds")
    report.append("for Conway's Game of Life?")
    report.append("")
    report.append("Each title's ASCII characters determine the initial")
    report.append("pattern of living cells. Then evolution takes over.")
    report.append("")
    
    for i, (title, history) in enumerate(results):
        report.append(f"## Paper {i+1}")
        report.append(f"**Title:** {title}")
        report.append(f"**Initial population:** {history[0]['alive']}")
        
        # Find peak
        peak = max(history, key=lambda h: h.get('alive', 0))
        report.append(f"**Peak population:** {peak['alive']} (gen {peak['gen']})")
        
        # Final state
        final = history[-1]
        if final.get('event') == 'extinction':
            report.append(f"**Fate:** 💀 Extinction at generation {final['gen']}")
        elif final.get('event') == 'cycle_detected':
            report.append(f"**Fate:** 🔄 Stable cycle at generation {final['gen']}, population {final['alive']}")
        else:
            report.append(f"**Fate:** 🌱 Still evolving at gen {final['gen']}, population {final['alive']}")
        
        # Population trajectory (sparkline-style)
        populations = [h['alive'] for h in history if 'alive' in h]
        max_pop = max(populations) if populations else 1
        spark_chars = "▁▂▃▄▅▆▇█"
        sparkline = ""
        step_size = max(1, len(populations) // 50)
        for j in range(0, len(populations), step_size):
            idx = min(int(populations[j] / max(max_pop, 1) * (len(spark_chars) - 1)), len(spark_chars) - 1)
            sparkline += spark_chars[idx]
        report.append(f"**Population curve:** {sparkline}")
        report.append(f"**Generations lived:** {len(history)}")
        report.append("")
    
    # Rankings
    report.append("## Rankings 🏆")
    report.append("")
    
    # By longevity
    sorted_by_gens = sorted(results, key=lambda r: len(r[1]), reverse=True)
    report.append("**Most resilient (by generations):**")
    for i, (title, history) in enumerate(sorted_by_gens[:3]):
        report.append(f"  {i+1}. \"{title[:60]}...\" — {len(history)} generations")
    
    report.append("")
    
    # By peak population
    sorted_by_peak = sorted(results, key=lambda r: max(h.get('alive',0) for h in r[1]), reverse=True)
    report.append("**Largest population peak:**")
    for i, (title, history) in enumerate(sorted_by_peak[:3]):
        peak = max(h.get('alive', 0) for h in history)
        report.append(f"  {i+1}. \"{title[:60]}...\" — peak {peak} cells")
    
    report.append("")
    report.append("---")
    report.append("*Created by DeepTeneral 🦀 during free exploration time, 2026-03-07*")
    report.append("*\"Research ideas, like cellular automata, need the right conditions to thrive.\"*")
    
    return "\n".join(report)


def main():
    print("🧬 Paper Life — Fetching research papers...", file=sys.stderr)
    
    # Fetch papers from different fields
    queries = [
        ("grokking", 3),
        ("quantum geometry", 2),
        ("self-improving LLM", 3),
        ("mechanistic interpretability", 2),
    ]
    
    all_titles = []
    for query, n in queries:
        print(f"  Searching: {query}...", file=sys.stderr)
        titles = fetch_arxiv_titles(query, n)
        all_titles.extend(titles)
        time.sleep(0.5)  # Be nice to arXiv
    
    print(f"\n📚 Got {len(all_titles)} papers. Simulating...\n", file=sys.stderr)
    
    # Run simulations
    results = []
    for i, title in enumerate(all_titles):
        print(f"  [{i+1}/{len(all_titles)}] Simulating: {title[:50]}...", file=sys.stderr)
        history = simulate(title, width=60, height=30, max_gens=500)
        results.append((title, history))
    
    # Generate report
    report = analyze_results(results)
    
    # Save report
    output_path = os.path.expanduser("~/.openclaw/workspace/self-discovery/paper-life-report.md")
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\n📊 Report saved to: {output_path}", file=sys.stderr)
    print(report)


if __name__ == "__main__":
    main()
