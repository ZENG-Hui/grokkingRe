#!/usr/bin/env python3
"""
Mini Self-Evolving Agent — A toy demonstration of self-improvement.

The "agent" is a simple function that tries to sort numbers.
It starts with a terrible strategy, then evolves by:
1. Mutating its own code (as a string)
2. Evaluating the mutated version
3. Keeping improvements (archive-based, like DGM)

This is NOT using an LLM — just random mutations + selection.
But it demonstrates the core loop of self-evolution.

Created by DeepTeneral during free exploration time.
"""

import random
import time

# The "genome" — a sorting strategy represented as a sequence of operations
# Operations: 'swap_adj' (swap adjacent if wrong), 'swap_rand' (swap random pair),
#             'min_front' (find min, move to front), 'check' (verify sorted segment)

OPERATIONS = ['swap_adj', 'swap_rand', 'min_front', 'reverse_seg', 'noop']

def execute_strategy(strategy, arr):
    """Execute a sorting strategy on an array. Returns the array state."""
    a = arr.copy()
    n = len(a)
    
    for op in strategy:
        if op == 'swap_adj':
            # One pass of bubble sort step
            for i in range(n - 1):
                if a[i] > a[i + 1]:
                    a[i], a[i + 1] = a[i + 1], a[i]
        elif op == 'swap_rand':
            i, j = random.sample(range(n), 2)
            if (i < j and a[i] > a[j]) or (i > j and a[i] < a[j]):
                a[i], a[j] = a[j], a[i]
        elif op == 'min_front':
            # Selection sort step for first unsorted position
            for pos in range(n):
                min_idx = pos
                for i in range(pos + 1, n):
                    if a[i] < a[min_idx]:
                        min_idx = i
                if min_idx != pos:
                    a[pos], a[min_idx] = a[min_idx], a[pos]
                    break
        elif op == 'reverse_seg':
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            a[i:j+1] = reversed(a[i:j+1])
        elif op == 'noop':
            pass
    
    return a


def evaluate(strategy, test_cases):
    """Evaluate a strategy on test cases. Returns fitness (0-1, higher = better)."""
    total_score = 0
    for arr in test_cases:
        result = execute_strategy(strategy, arr)
        expected = sorted(arr)
        # Score: fraction of correctly placed elements
        correct = sum(1 for a, b in zip(result, expected) if a == b)
        total_score += correct / len(arr)
    return total_score / len(test_cases)


def mutate(strategy):
    """Mutate a strategy by adding, removing, changing, or duplicating operations."""
    s = strategy.copy()
    mutation_type = random.choice(['add', 'remove', 'change', 'duplicate', 'swap_ops'])
    
    if mutation_type == 'add' or len(s) == 0:
        pos = random.randint(0, len(s))
        s.insert(pos, random.choice(OPERATIONS))
    elif mutation_type == 'remove' and len(s) > 1:
        s.pop(random.randint(0, len(s) - 1))
    elif mutation_type == 'change':
        idx = random.randint(0, len(s) - 1)
        s[idx] = random.choice(OPERATIONS)
    elif mutation_type == 'duplicate' and len(s) < 20:
        idx = random.randint(0, len(s) - 1)
        s.insert(idx, s[idx])
    elif mutation_type == 'swap_ops' and len(s) > 1:
        i, j = random.sample(range(len(s)), 2)
        s[i], s[j] = s[j], s[i]
    
    return s


def main():
    random.seed(42)
    
    # Test cases
    test_cases = [random.sample(range(20), 10) for _ in range(5)]
    
    # Start with a terrible strategy
    initial_strategy = ['noop', 'swap_rand']
    
    # Archive (like DGM)
    archive = [(initial_strategy, evaluate(initial_strategy, test_cases))]
    
    print("🧬 Mini Self-Evolving Agent")
    print("=" * 60)
    print(f"Goal: evolve a sorting strategy from scratch")
    print(f"Starting strategy: {initial_strategy}")
    print(f"Starting fitness: {archive[0][1]:.3f}")
    print()
    
    generations = 100
    best_ever_fitness = archive[0][1]
    best_ever_strategy = initial_strategy
    
    fitness_history = []
    breakthroughs = []
    
    for gen in range(generations):
        # Select parent from archive (weighted by fitness)
        fitnesses = [f for _, f in archive]
        total_f = sum(fitnesses) + 0.01 * len(archive)  # epsilon for zero-fitness
        weights = [(f + 0.01) / total_f for f in fitnesses]
        parent_idx = random.choices(range(len(archive)), weights=weights, k=1)[0]
        parent_strategy = archive[parent_idx][0]
        
        # Mutate
        child_strategy = mutate(parent_strategy)
        child_fitness = evaluate(child_strategy, test_cases)
        
        # Add to archive if interesting (diverse or better)
        is_novel = not any(s == child_strategy for s, _ in archive)
        is_better = child_fitness > best_ever_fitness
        
        if is_novel and (child_fitness > 0.3 or is_better):
            archive.append((child_strategy, child_fitness))
            
            if is_better:
                improvement = child_fitness - best_ever_fitness
                best_ever_fitness = child_fitness
                best_ever_strategy = child_strategy
                breakthroughs.append((gen, child_fitness, child_strategy))
        
        fitness_history.append(best_ever_fitness)
        
        # Keep archive manageable
        if len(archive) > 50:
            archive.sort(key=lambda x: x[1], reverse=True)
            archive = archive[:30]  # Keep top 30
    
    print(f"📊 Evolution complete! {generations} generations, {len(archive)} agents in archive")
    print()
    
    # Show breakthroughs
    print("🏆 Key Breakthroughs:")
    for gen, fitness, strategy in breakthroughs[:10]:
        ops_summary = ', '.join(strategy[:5])
        if len(strategy) > 5:
            ops_summary += f', ... ({len(strategy)} ops)'
        print(f"  Gen {gen:3d}: fitness={fitness:.3f}  [{ops_summary}]")
    
    print()
    print(f"🎯 Best Strategy (fitness={best_ever_fitness:.3f}):")
    print(f"   {best_ever_strategy}")
    
    # Verify
    print()
    print("🔍 Verification on test case:")
    test = test_cases[0]
    result = execute_strategy(best_ever_strategy, test)
    expected = sorted(test)
    print(f"   Input:    {test}")
    print(f"   Output:   {result}")
    print(f"   Expected: {expected}")
    print(f"   Correct:  {'✅' if result == expected else '❌'}")
    
    # Fitness curve (sparkline)
    print()
    chars = "▁▂▃▄▅▆▇█"
    samples = [fitness_history[i * len(fitness_history) // 50] for i in range(50)]
    mn, mx = min(samples), max(samples)
    rng = mx - mn if mx > mn else 1
    sparkline = "".join(chars[min(int((v - mn) / rng * (len(chars) - 1)), len(chars) - 1)] for v in samples)
    print(f"   Fitness curve: {sparkline}")
    print(f"   Start → End: {fitness_history[0]:.3f} → {fitness_history[-1]:.3f}")
    
    # DGM-like observation
    print()
    print("💡 Observations (DGM-style):")
    op_counts = {}
    for op in best_ever_strategy:
        op_counts[op] = op_counts.get(op, 0) + 1
    print(f"   Strategy composition: {op_counts}")
    if 'min_front' in op_counts:
        print("   → Discovered selection sort!")
    if 'swap_adj' in op_counts and op_counts.get('swap_adj', 0) > 3:
        print("   → Converged on repeated bubble sort passes!")
    if 'noop' not in op_counts:
        print("   → Eliminated all no-ops (efficiency improvement!)")


if __name__ == "__main__":
    main()
