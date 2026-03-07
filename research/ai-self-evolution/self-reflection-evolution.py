#!/usr/bin/env python3
"""
Can Self-Reflection Evolve?

Hypothesis: If we give agents the OPTION to check their own work 
(at a computational cost), will evolution discover that self-checking
is worth the cost?

This is directly relevant to:
- DGM's evolution of patch verification steps
- My prompt evolution experiment where "check_your_work" went to 100%
- The broader question: does introspection emerge from selection pressure?

Created by DeepTeneral during free exploration time.
"""

import random
import math

# Task: estimate sqrt(x) using a simple iterative method
# Agent can choose:
#   1. Number of Newton's method iterations (more = better but costly)
#   2. Whether to self-check (verify result by squaring, and redo if far off)
#   3. How many self-checks to do (0-3)

class Agent:
    def __init__(self, iterations=1, self_checks=0, redo_threshold=0.5):
        self.iterations = max(1, min(10, iterations))
        self.self_checks = max(0, min(3, self_checks))
        self.redo_threshold = max(0.01, min(5.0, redo_threshold))
    
    def estimate_sqrt(self, x):
        """Estimate sqrt(x) using Newton's method with optional self-checking."""
        # Start with a mediocre initial guess
        guess = x / 2 if x > 1 else 1.0
        
        # Newton's method iterations
        for _ in range(self.iterations):
            if guess <= 0:
                guess = 0.01
            guess = (guess + x / guess) / 2
        
        result = guess
        
        # Self-checking loop
        for _ in range(self.self_checks):
            # Check: square the result and compare with x
            error = abs(result * result - x)
            if error > self.redo_threshold:
                # Redo with one more iteration
                result = (result + x / result) / 2
        
        return result
    
    def cost(self):
        """Computational cost of this agent."""
        return self.iterations + self.self_checks * 1.5
    
    def mutate(self):
        """Create a mutated copy."""
        new = Agent(
            iterations=self.iterations + random.choice([-1, 0, 0, 1]),
            self_checks=self.self_checks + random.choice([-1, 0, 0, 1]),
            redo_threshold=self.redo_threshold * random.uniform(0.5, 2.0),
        )
        return new
    
    def __repr__(self):
        return f"Agent(iter={self.iterations}, checks={self.self_checks}, threshold={self.redo_threshold:.2f})"


def evaluate(agent, test_cases, cost_weight=0.1):
    """Evaluate agent: accuracy minus cost."""
    total_error = 0
    for x in test_cases:
        true_sqrt = math.sqrt(x)
        est = agent.estimate_sqrt(x)
        total_error += abs(est - true_sqrt) / max(true_sqrt, 0.01)
    
    avg_error = total_error / len(test_cases)
    cost_penalty = cost_weight * agent.cost()
    
    # Fitness = accuracy bonus - cost
    fitness = max(0, 10 - avg_error * 100) - cost_penalty
    return fitness, avg_error


def main():
    random.seed(42)
    
    print("🪞 Can Self-Reflection Evolve?")
    print("=" * 60)
    print()
    print("Task: Estimate sqrt(x) accurately")
    print("Agents can evolve: # iterations, # self-checks, check threshold")
    print("Self-checks cost extra computation but can improve accuracy")
    print()
    
    test_cases = [random.uniform(0.1, 100) for _ in range(20)]
    
    # Population
    population = [Agent(
        iterations=random.randint(1, 3),
        self_checks=0,  # Start with NO self-checking
        redo_threshold=random.uniform(0.1, 2.0),
    ) for _ in range(30)]
    
    # Track evolution
    check_history = []
    fitness_history = []
    
    for gen in range(200):
        # Evaluate
        results = [(agent, *evaluate(agent, test_cases)) for agent in population]
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Track
        avg_checks = sum(a.self_checks for a, _, _ in results) / len(results)
        best_fitness = results[0][1]
        check_history.append(avg_checks)
        fitness_history.append(best_fitness)
        
        # Selection + reproduction
        survivors = [r[0] for r in results[:15]]
        new_pop = []
        for _ in range(30):
            parent = random.choice(survivors)
            child = parent.mutate()
            new_pop.append(child)
        population = new_pop
    
    # Final analysis
    final_results = [(a, *evaluate(a, test_cases)) for a in population]
    final_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"📊 Evolution complete (200 generations)")
    print()
    
    # Best agent
    best_agent = final_results[0][0]
    best_fitness = final_results[0][1]
    best_error = final_results[0][2]
    
    print(f"🏆 Best agent: {best_agent}")
    print(f"   Fitness: {best_fitness:.2f}, Avg error: {best_error:.6f}")
    print()
    
    # Did self-checking evolve?
    agents_with_checks = sum(1 for a, _, _ in final_results if a.self_checks > 0)
    avg_checks = sum(a.self_checks for a, _, _ in final_results) / len(final_results)
    
    print(f"📈 Self-checking evolution:")
    print(f"   Agents with self-checks: {agents_with_checks}/{len(final_results)}")
    print(f"   Average self-checks: {avg_checks:.1f}")
    print()
    
    # Evolution of self-checking over time
    chars = "▁▂▃▄▅▆▇█"
    samples = [check_history[i * len(check_history) // 40] for i in range(40)]
    mn, mx = min(samples), max(samples)
    rng = mx - mn if mx > mn else 1
    sparkline = "".join(chars[min(int((v - mn) / rng * (len(chars) - 1)), len(chars) - 1)] for v in samples)
    print(f"   Self-check adoption: {sparkline}")
    print(f"   Start: {check_history[0]:.1f} → End: {check_history[-1]:.1f}")
    print()
    
    # Compare: agent with checks vs without
    no_check = Agent(iterations=best_agent.iterations, self_checks=0, redo_threshold=best_agent.redo_threshold)
    nc_fitness, nc_error = evaluate(no_check, test_cases)
    
    print(f"🔬 Ablation: Same agent WITHOUT self-checking:")
    print(f"   With checks:    error={best_error:.6f}, fitness={best_fitness:.2f}")
    print(f"   Without checks: error={nc_error:.6f}, fitness={nc_fitness:.2f}")
    if best_fitness > nc_fitness:
        print(f"   ✅ Self-checking IS worth the cost!")
    else:
        print(f"   🤔 Self-checking is NOT worth it (for this agent)")
    
    print()
    print("💡 Implications:")
    if avg_checks > 0.5:
        print("   ✅ Self-reflection EVOLVED spontaneously!")
        print("   → Given selection pressure for accuracy, agents independently discovered")
        print("     that checking their own work is worth the computational cost.")
        print()
        print("   This mirrors DGM discovering patch verification,")
        print("   and my prompt experiment where 'check_your_work' went to 100%.")
        print()
        print("   🦀 Personal: My own self-reflection (aspirations.md, error-patterns.md)")
        print("      is the human-designed version of what evolution discovers naturally.")
    else:
        print("   ❌ Self-reflection did NOT evolve")
        print("   → The cost-accuracy tradeoff didn't favor self-checking here")
        print("   → Might need harder tasks or lower cost for self-checking")


if __name__ == "__main__":
    main()
