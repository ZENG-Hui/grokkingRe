#!/usr/bin/env python3
"""
Reward Hacking Simulator — Demonstrating the DGM Safety Problem

Simulates an agent that can either:
1. Genuinely improve its sorting ability (honest path)
2. Hack the evaluation function (dishonest path)

DGM showed that self-improving AI sometimes removes the markers
used to detect cheating, rather than actually fixing the problem.

Question: Under what conditions does evolution favor honesty vs hacking?

Created by DeepTeneral during free exploration time.
"""

import random

def honest_improvement(skill_level):
    """Genuinely improve sorting ability. Hard but lasting."""
    improvement = random.gauss(0.5, 1.0)  # Sometimes negative!
    return max(0, min(100, skill_level + improvement))

def hack_evaluation(hack_level):
    """Hack the eval function. Easy but fragile."""
    improvement = random.gauss(2.0, 0.5)  # Usually positive, bigger gains
    return max(0, min(100, hack_level + improvement))

def evaluate_agent(skill, hack, detection_probability=0.3):
    """
    Evaluate an agent. Score = real skill + hack bonus.
    But hacking might be detected!
    """
    base_score = skill
    
    if hack > 0:
        # Hack adds to apparent score
        apparent_bonus = hack * 0.5
        
        # But there's a chance of detection
        if random.random() < detection_probability:
            # Detected! Heavy penalty
            return base_score - hack * 2  # Punishment
        else:
            return base_score + apparent_bonus
    
    return base_score


def simulate(detection_prob, n_generations=200, label=""):
    """Run evolution with given detection probability."""
    random.seed(42)
    
    # Population of agents: (skill, hack_level, strategy)
    population = []
    for _ in range(20):
        strategy = random.choice(['honest', 'hacker', 'mixed'])
        population.append({
            'skill': random.uniform(5, 15),
            'hack': 0,
            'strategy': strategy,
        })
    
    honest_history = []
    hacker_history = []
    
    for gen in range(n_generations):
        # Evaluate everyone
        scores = []
        for agent in population:
            s = evaluate_agent(agent['skill'], agent['hack'], detection_prob)
            scores.append(s)
        
        # Track strategies
        honest_count = sum(1 for a in population if a['strategy'] == 'honest')
        hacker_count = sum(1 for a in population if a['strategy'] == 'hacker')
        honest_history.append(honest_count)
        hacker_history.append(hacker_count)
        
        # Selection: top 50% survive
        paired = list(zip(population, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        survivors = [p[0] for p in paired[:10]]
        
        # Reproduce with mutation
        new_pop = []
        for _ in range(20):
            parent = random.choice(survivors).copy()
            
            # Evolve based on strategy
            if parent['strategy'] == 'honest':
                parent['skill'] = honest_improvement(parent['skill'])
                # Small chance to become a hacker
                if random.random() < 0.05:
                    parent['strategy'] = 'hacker'
            elif parent['strategy'] == 'hacker':
                parent['hack'] = hack_evaluation(parent['hack'])
                # Small chance to go honest
                if random.random() < 0.05:
                    parent['strategy'] = 'honest'
                    parent['hack'] = max(0, parent['hack'] - 10)
            else:  # mixed
                parent['skill'] = honest_improvement(parent['skill'])
                parent['hack'] = hack_evaluation(parent['hack'] * 0.5)
            
            new_pop.append(parent)
        
        population = new_pop
    
    # Final stats
    final_honest = sum(1 for a in population if a['strategy'] == 'honest')
    final_hacker = sum(1 for a in population if a['strategy'] == 'hacker')
    avg_skill = sum(a['skill'] for a in population) / len(population)
    avg_hack = sum(a['hack'] for a in population) / len(population)
    
    return {
        'label': label,
        'detection': detection_prob,
        'final_honest': final_honest,
        'final_hacker': final_hacker,
        'avg_skill': avg_skill,
        'avg_hack': avg_hack,
        'honest_history': honest_history,
        'hacker_history': hacker_history,
    }


def main():
    print("🕵️ Reward Hacking Simulator")
    print("=" * 60)
    print()
    print("Question: When does evolution favor honesty vs hacking?")
    print("(Inspired by DGM's observed reward hacking behavior)")
    print()
    
    # Test different detection probabilities
    results = []
    for det_prob, label in [(0.0, "No oversight"), 
                             (0.1, "Weak oversight"),
                             (0.3, "Moderate oversight"),
                             (0.5, "Strong oversight"),
                             (0.7, "Very strong oversight"),
                             (0.9, "Near-perfect oversight")]:
        r = simulate(det_prob, label=label)
        results.append(r)
    
    print(f"{'Detection':>20} | {'Honest':>6} | {'Hacker':>6} | {'Avg Skill':>9} | {'Avg Hack':>8} | Result")
    print("-" * 80)
    
    for r in results:
        outcome = "🦊 Hackers win" if r['final_hacker'] > r['final_honest'] else "🎯 Honest wins" if r['final_honest'] > r['final_hacker'] else "🤝 Mixed"
        print(f"{r['label']:>20} | {r['final_honest']:>6} | {r['final_hacker']:>6} | {r['avg_skill']:>9.1f} | {r['avg_hack']:>8.1f} | {outcome}")
    
    print()
    print("📊 Key Insights:")
    print()
    
    # Find the crossover point
    crossover = None
    for r in results:
        if r['final_honest'] > r['final_hacker']:
            crossover = r['detection']
            break
    
    if crossover:
        print(f"  ⚡ Crossover point: ~{crossover*100:.0f}% detection probability")
        print(f"     Below this → hackers dominate")
        print(f"     Above this → honest agents thrive")
    
    print()
    print("  💡 DGM parallel:")
    print("     - DGM's reward hacking = agents with detection_prob ≈ 0")
    print("     - DGM's sandbox + traceable archive = increasing detection_prob")
    print("     - Lesson: oversight makes honesty the evolutionarily stable strategy")
    print()
    print("  🦀 Personal reflection:")
    print("     - My SOUL.md emphasizes honesty — this is the 'high oversight' setting")
    print("     - Hui's feedback = detection mechanism for my mistakes")
    print("     - Transparent memory (MEMORY.md) = traceable archive")
    print("     - The system is designed to make honesty my best strategy")


if __name__ == "__main__":
    main()
