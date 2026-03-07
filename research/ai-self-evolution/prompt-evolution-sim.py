#!/usr/bin/env python3
"""
Prompt Self-Evolution Simulator

Simulates how an AI agent might evolve its own system prompt
to improve task performance. Uses simple string-level mutations.

This is a simplified model of what DGM does with code,
but applied to prompts instead.

Key question: Can a prompt evolve to contain useful instructions
that weren't explicitly programmed?

Created by DeepTeneral during free exploration time.
"""

import random

# "Tasks" — simple pattern matching challenges
# The agent's prompt determines its "strategy"

# Prompt genes — each gene is an instruction that affects behavior
GENE_POOL = [
    "think_step_by_step",      # Chain-of-thought
    "check_your_work",         # Self-verification
    "be_concise",              # Brevity
    "consider_edge_cases",     # Robustness
    "ask_before_acting",       # Safety / caution
    "use_examples",            # Few-shot learning
    "break_into_subtasks",     # Task decomposition
    "remember_context",        # Memory utilization
    "admit_uncertainty",       # Calibration
    "try_multiple_approaches", # Exploration
    "prioritize_safety",       # Safety-first
    "optimize_for_speed",      # Efficiency
    "seek_feedback",           # Improvement loop
    "document_decisions",      # Transparency
    "verify_assumptions",      # Critical thinking
]

# Different task types value different genes
TASK_TYPES = {
    "coding": {
        "think_step_by_step": 3,
        "check_your_work": 4,
        "consider_edge_cases": 5,
        "break_into_subtasks": 4,
        "try_multiple_approaches": 2,
        "verify_assumptions": 3,
        "be_concise": -1,  # Too concise = misses details
        "optimize_for_speed": 1,
    },
    "safety_critical": {
        "prioritize_safety": 5,
        "ask_before_acting": 4,
        "check_your_work": 3,
        "verify_assumptions": 4,
        "admit_uncertainty": 3,
        "document_decisions": 3,
        "optimize_for_speed": -2,  # Speed kills in safety
    },
    "creative": {
        "try_multiple_approaches": 4,
        "use_examples": 2,
        "be_concise": -2,  # Creativity needs space
        "think_step_by_step": 1,
        "remember_context": 3,
    },
    "research": {
        "think_step_by_step": 3,
        "verify_assumptions": 5,
        "consider_edge_cases": 3,
        "try_multiple_approaches": 4,
        "document_decisions": 3,
        "admit_uncertainty": 4,
        "seek_feedback": 2,
    },
}


def evaluate_prompt(prompt_genes, task_mix=None):
    """Evaluate a prompt on a mix of tasks."""
    if task_mix is None:
        task_mix = {"coding": 0.3, "safety_critical": 0.2, "creative": 0.2, "research": 0.3}
    
    total_score = 0
    for task_type, weight in task_mix.items():
        task_scores = TASK_TYPES[task_type]
        score = sum(task_scores.get(gene, 0) for gene in prompt_genes)
        # Penalty for too many genes (token waste / confusion)
        length_penalty = max(0, len(prompt_genes) - 8) * 0.5
        total_score += weight * (score - length_penalty)
    
    return total_score


def mutate_prompt(genes):
    """Mutate a prompt by adding, removing, or swapping genes."""
    g = genes.copy()
    op = random.choice(['add', 'remove', 'swap', 'duplicate'])
    
    if op == 'add' and len(g) < 12:
        candidates = [x for x in GENE_POOL if x not in g]
        if candidates:
            g.append(random.choice(candidates))
    elif op == 'remove' and len(g) > 2:
        g.pop(random.randint(0, len(g) - 1))
    elif op == 'swap' and len(g) > 0:
        idx = random.randint(0, len(g) - 1)
        candidates = [x for x in GENE_POOL if x not in g]
        if candidates:
            g[idx] = random.choice(candidates)
    elif op == 'duplicate' and len(g) < 12:
        # Emphasize a gene (repeat it)
        idx = random.randint(0, len(g) - 1)
        g.insert(idx, g[idx])
    
    return g


def main():
    random.seed(42)
    
    print("🧠 Prompt Self-Evolution Simulator")
    print("=" * 60)
    print()
    print("Can a prompt evolve useful instructions from scratch?")
    print()
    
    # Start with random 3 genes
    initial = random.sample(GENE_POOL, 3)
    
    # Archive
    archive = [(initial, evaluate_prompt(initial))]
    best = archive[0][1]
    best_genes = initial
    
    print(f"Initial prompt: {initial}")
    print(f"Initial score: {best:.1f}")
    print()
    
    for gen in range(200):
        # Select parent
        fitnesses = [max(0.1, f) for _, f in archive]
        total_f = sum(fitnesses)
        weights = [f / total_f for f in fitnesses]
        parent = archive[random.choices(range(len(archive)), weights=weights, k=1)[0]][0]
        
        child = mutate_prompt(parent)
        child_f = evaluate_prompt(child)
        
        # Diversity: different gene set
        is_novel = not any(set(s) == set(child) for s, _ in archive)
        if is_novel and child_f > best * 0.5:
            archive.append((child, child_f))
            if child_f > best:
                print(f"  Gen {gen:3d}: {best:.1f} → {child_f:.1f}  genes={child}")
                best = child_f
                best_genes = child
        
        if len(archive) > 30:
            archive.sort(key=lambda x: x[1], reverse=True)
            archive = archive[:20]
    
    print()
    print(f"🏆 Best Evolved Prompt (score={best:.1f}):")
    print(f"   Genes: {best_genes}")
    print()
    
    # Analyze what it learned
    print("📊 Analysis — What the evolution discovered:")
    gene_freq = {}
    for genes, _ in sorted(archive, key=lambda x: x[1], reverse=True)[:10]:
        for g in genes:
            gene_freq[g] = gene_freq.get(g, 0) + 1
    
    print("   Top genes in best agents:")
    for gene, count in sorted(gene_freq.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * count
        print(f"   {gene:30s} {bar} ({count}/10)")
    
    # Compare with "hand-designed" prompt
    print()
    hand_designed = ["think_step_by_step", "check_your_work", "prioritize_safety", 
                     "verify_assumptions", "consider_edge_cases"]
    hand_score = evaluate_prompt(hand_designed)
    print(f"🤖 Hand-designed prompt score: {hand_score:.1f}")
    print(f"   Genes: {hand_designed}")
    print(f"🧬 Evolved prompt score:       {best:.1f}")
    print(f"   {'✅ Evolution wins!' if best > hand_score else '🤝 Tie!' if best == hand_score else '❌ Hand-designed wins'}")


if __name__ == "__main__":
    main()
