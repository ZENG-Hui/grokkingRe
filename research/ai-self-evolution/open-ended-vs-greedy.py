#!/usr/bin/env python3
"""
Open-Ended Evolution vs Greedy Evolution

Compares two approaches:
1. Greedy: Always pick the best, discard the rest
2. Open-ended (DGM-style): Maintain diverse archive, explore from multiple ancestors

Hypothesis: Open-ended evolution discovers MORE solutions and BETTER peak solutions,
even though at any given moment, greedy evolution seems to be doing better.

This mirrors the DGM vs simple hill-climbing comparison.

Created by DeepTeneral during free exploration time.
"""

import random
import math

# Multi-modal fitness landscape: multiple peaks of different heights
# The agent needs to find the GLOBAL optimum, not just local ones
def fitness_landscape(x, y):
    """Multi-modal 2D landscape with deceptive local optima."""
    # Global optimum at (3.5, 3.5), height ~10
    peak1 = 10 * math.exp(-((x - 3.5)**2 + (y - 3.5)**2) / 0.5)
    # Deceptive local optimum at (1, 1), height ~7 (easy to find, hard to leave)
    peak2 = 7 * math.exp(-((x - 1)**2 + (y - 1)**2) / 1.0)
    # Another local optimum at (4, 0), height ~5
    peak3 = 5 * math.exp(-((x - 4)**2 + y**2) / 0.8)
    # Small peak at (-2, 3), height ~4
    peak4 = 4 * math.exp(-((x + 2)**2 + (y - 3)**2) / 0.6)
    # Ridge connecting peaks (makes exploration easier from diverse archive)
    ridge = 2 * math.exp(-(y - x)**2 / 3.0)
    return peak1 + peak2 + peak3 + peak4 + ridge

def mutate(x, y, step=0.5):
    return x + random.gauss(0, step), y + random.gauss(0, step)


def run_greedy(n_steps=500, seed=42):
    """Classic greedy evolution: always keep the best."""
    random.seed(seed)
    x, y = random.uniform(-5, 5), random.uniform(-5, 5)
    best_f = fitness_landscape(x, y)
    history = [best_f]
    peaks_found = set()
    
    for _ in range(n_steps):
        nx, ny = mutate(x, y)
        nf = fitness_landscape(nx, ny)
        if nf > best_f:
            x, y, best_f = nx, ny, nf
            # Track which peaks we're near
            for px, py, label in [(3.5, 3.5, 'global'), (1, 1, 'deceptive'), (4, 0, 'local_A'), (-2, 3, 'local_B')]:
                if math.sqrt((x - px)**2 + (y - py)**2) < 1.0:
                    peaks_found.add(label)
        history.append(best_f)
    
    return best_f, (x, y), history, peaks_found


def run_open_ended(n_steps=500, archive_size=15, seed=42):
    """DGM-style open-ended evolution: maintain diverse archive."""
    random.seed(seed)
    
    # Initialize archive with random points
    archive = []
    for _ in range(5):
        x, y = random.uniform(-5, 5), random.uniform(-5, 5)
        f = fitness_landscape(x, y)
        archive.append((x, y, f))
    
    best_f = max(a[2] for a in archive)
    history = [best_f]
    peaks_found = set()
    
    for _ in range(n_steps):
        # Select parent from archive (weighted by fitness, but with novelty bonus)
        weights = []
        for i, (ax, ay, af) in enumerate(archive):
            # Fitness weight
            fit_w = af + 1
            # Novelty weight: distance to nearest neighbor
            min_dist = min(math.sqrt((ax - bx)**2 + (ay - by)**2) 
                         for j, (bx, by, _) in enumerate(archive) if j != i) if len(archive) > 1 else 1.0
            nov_w = min_dist * 2
            weights.append(fit_w + nov_w)
        
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        parent_idx = random.choices(range(len(archive)), weights=weights, k=1)[0]
        px, py, _ = archive[parent_idx]
        
        # Mutate
        nx, ny = mutate(px, py)
        nf = fitness_landscape(nx, ny)
        
        # Add to archive if novel enough
        min_dist = min(math.sqrt((nx - ax)**2 + (ny - ay)**2) for ax, ay, _ in archive)
        if min_dist > 0.3:  # Novelty threshold
            archive.append((nx, ny, nf))
            
            # Track peaks
            for ppx, ppy, label in [(3.5, 3.5, 'global'), (1, 1, 'deceptive'), (4, 0, 'local_A'), (-2, 3, 'local_B')]:
                if math.sqrt((nx - ppx)**2 + (ny - ppy)**2) < 1.0:
                    peaks_found.add(label)
        
        # Trim archive: keep diverse set
        if len(archive) > archive_size:
            # Remove least fit from crowded regions
            archive.sort(key=lambda a: a[2])
            # But never remove the best
            best_idx = max(range(len(archive)), key=lambda i: archive[i][2])
            if best_idx == 0:
                archive.pop(1)
            else:
                archive.pop(0)
        
        best_f = max(best_f, nf)
        history.append(best_f)
    
    best_agent = max(archive, key=lambda a: a[2])
    return best_f, (best_agent[0], best_agent[1]), history, peaks_found


def main():
    print("🌊 Open-Ended vs Greedy Evolution")
    print("=" * 60)
    print()
    print("Landscape: 4 peaks (global=10, deceptive=7, local_A=5, local_B=4)")
    print("Challenge: Escape deceptive local optimum to find global peak")
    print()
    
    # Run multiple seeds to get statistics
    n_trials = 20
    greedy_results = []
    open_results = []
    
    for seed in range(n_trials):
        gf, gpos, ghist, gpeaks = run_greedy(500, seed)
        of, opos, ohist, opeaks = run_open_ended(500, seed=seed)
        greedy_results.append((gf, gpos, gpeaks))
        open_results.append((of, opos, opeaks))
    
    # Statistics
    greedy_fitnesses = [r[0] for r in greedy_results]
    open_fitnesses = [r[0] for r in open_results]
    
    greedy_found_global = sum(1 for r in greedy_results if 'global' in r[2])
    open_found_global = sum(1 for r in open_results if 'global' in r[2])
    
    greedy_avg_peaks = sum(len(r[2]) for r in greedy_results) / n_trials
    open_avg_peaks = sum(len(r[2]) for r in open_results) / n_trials
    
    print(f"Results over {n_trials} trials (500 steps each):")
    print()
    print(f"{'Metric':<30} | {'Greedy':>10} | {'Open-Ended':>10}")
    print("-" * 56)
    print(f"{'Avg best fitness':<30} | {sum(greedy_fitnesses)/n_trials:>10.2f} | {sum(open_fitnesses)/n_trials:>10.2f}")
    print(f"{'Max best fitness':<30} | {max(greedy_fitnesses):>10.2f} | {max(open_fitnesses):>10.2f}")
    print(f"{'Min best fitness':<30} | {min(greedy_fitnesses):>10.2f} | {min(open_fitnesses):>10.2f}")
    print(f"{'Found global optimum':<30} | {greedy_found_global:>10}/{n_trials} | {open_found_global:>10}/{n_trials}")
    print(f"{'Avg peaks explored':<30} | {greedy_avg_peaks:>10.1f} | {open_avg_peaks:>10.1f}")
    
    # Winner determination
    print()
    if sum(open_fitnesses) > sum(greedy_fitnesses):
        improvement = (sum(open_fitnesses) - sum(greedy_fitnesses)) / sum(greedy_fitnesses) * 100
        print(f"🏆 Open-ended evolution wins by {improvement:.1f}%!")
    else:
        print(f"🤔 Greedy evolution surprisingly wins (landscape may not be deceptive enough)")
    
    if open_found_global > greedy_found_global:
        print(f"🔍 Open-ended found global peak {open_found_global}x vs greedy {greedy_found_global}x")
    
    print()
    print("💡 Key Insight:")
    print(f"   Greedy explores {greedy_avg_peaks:.1f} peaks on average")
    print(f"   Open-ended explores {open_avg_peaks:.1f} peaks on average")
    if open_avg_peaks > greedy_avg_peaks:
        print("   → Diversity preservation leads to more thorough exploration!")
    print()
    print("   This is exactly why DGM uses an archive instead of simple hill-climbing.")
    print("   The 'stepping stones' (seemingly suboptimal agents) unlock paths")
    print("   to much better solutions that greedy optimization would never find.")


if __name__ == "__main__":
    main()
