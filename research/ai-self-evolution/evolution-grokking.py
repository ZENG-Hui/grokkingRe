#!/usr/bin/env python3
"""
Does Self-Evolution Grok?

Hypothesis: Self-evolving agents might exhibit grokking-like behavior —
initially memorizing evaluation-specific tricks, then suddenly 
generalizing to truly better strategies.

Test: Evolve agents on a training set, track both train and test performance.
Look for delayed generalization (= grokking).

This connects two of Hui's research interests:
1. Grokking (grokkingRe project)
2. AI self-evolution

Created by DeepTeneral during free exploration time.
"""

import random
import math

# Task: learn a mathematical function
# Training: 20 input-output pairs
# Testing: 20 different input-output pairs
# Agent: a polynomial with evolvable coefficients

def target_function(x):
    """The true function agents try to learn: sin(x) + 0.5*cos(2x)"""
    return math.sin(x) + 0.5 * math.cos(2 * x)

def agent_predict(coeffs, x):
    """Agent's prediction using polynomial coefficients."""
    result = 0
    for i, c in enumerate(coeffs):
        result += c * (x ** i)
    return result

def evaluate(coeffs, data_points):
    """MSE on given data points."""
    total_error = 0
    for x, y_true in data_points:
        y_pred = agent_predict(coeffs, x)
        total_error += (y_pred - y_true) ** 2
    return total_error / len(data_points)

def mutate_coeffs(coeffs):
    """Mutate polynomial coefficients."""
    c = coeffs.copy()
    mutation_type = random.choice(['perturb', 'add_term', 'remove_term', 'reset_one'])
    
    if mutation_type == 'perturb':
        idx = random.randint(0, len(c) - 1)
        c[idx] += random.gauss(0, 0.3)
    elif mutation_type == 'add_term' and len(c) < 8:
        c.append(random.gauss(0, 0.1))
    elif mutation_type == 'remove_term' and len(c) > 2:
        c.pop()
    elif mutation_type == 'reset_one':
        idx = random.randint(0, len(c) - 1)
        c[idx] = random.gauss(0, 0.5)
    
    return c

def main():
    random.seed(42)
    
    # Generate train and test data
    train_x = [random.uniform(-3, 3) for _ in range(20)]
    test_x = [random.uniform(-3, 3) for _ in range(20)]
    
    train_data = [(x, target_function(x)) for x in train_x]
    test_data = [(x, target_function(x)) for x in test_x]
    
    print("🔬 Does Self-Evolution Grok?")
    print("=" * 60)
    print(f"Target: f(x) = sin(x) + 0.5*cos(2x)")
    print(f"Agent: polynomial with evolvable coefficients")
    print(f"Train: {len(train_data)} points, Test: {len(test_data)} points")
    print()
    
    # Start with random coefficients
    best_coeffs = [random.gauss(0, 0.5) for _ in range(3)]
    best_train_loss = evaluate(best_coeffs, train_data)
    
    # Archive
    archive = [(best_coeffs, best_train_loss)]
    
    train_history = []
    test_history = []
    grok_candidates = []
    
    n_generations = 500
    
    for gen in range(n_generations):
        # Select parent
        losses = [1.0 / (l + 0.01) for _, l in archive]  # Inverse loss = fitness
        total = sum(losses)
        weights = [l / total for l in losses]
        parent = archive[random.choices(range(len(archive)), weights=weights, k=1)[0]][0]
        
        # Mutate
        child = mutate_coeffs(parent)
        child_train_loss = evaluate(child, train_data)
        child_test_loss = evaluate(child, test_data)
        
        # Selection based on TRAIN loss only (agent doesn't see test)
        if child_train_loss < best_train_loss * 1.5:  # Keep if reasonably good
            archive.append((child, child_train_loss))
            if child_train_loss < best_train_loss:
                best_train_loss = child_train_loss
                best_coeffs = child
        
        if len(archive) > 30:
            archive.sort(key=lambda x: x[1])
            archive = archive[:20]
        
        # Record both losses for best agent
        train_history.append(evaluate(best_coeffs, train_data))
        test_history.append(evaluate(best_coeffs, test_data))
        
        # Detect grokking: train loss drops but test loss stays high, then test drops
        if gen > 50:
            recent_train = sum(train_history[-10:]) / 10
            recent_test = sum(test_history[-10:]) / 10
            older_test = sum(test_history[-50:-40]) / 10 if gen > 90 else recent_test
            
            if recent_train < 0.1 and recent_test < older_test * 0.5:
                grok_candidates.append(gen)
    
    # Results
    print(f"📊 Evolution complete ({n_generations} generations)")
    print()
    
    # Sample histories for display
    checkpoints = list(range(0, n_generations, n_generations // 10))
    print(f"{'Gen':>5} | {'Train Loss':>10} | {'Test Loss':>10} | {'Gap':>6} | Train vs Test")
    print("-" * 65)
    for g in checkpoints:
        tr = train_history[g]
        te = test_history[g]
        gap = te - tr
        bar_tr = "▓" * min(int(tr * 10), 20)
        bar_te = "░" * min(int(te * 10), 20)
        status = "📈 memorizing" if gap > 0.5 else "🎯 generalizing" if gap < 0.2 else "⚡ transitioning"
        print(f"{g:>5} | {tr:>10.4f} | {te:>10.4f} | {gap:>+6.2f} | {bar_tr}|{bar_te} {status}")
    
    # Final
    g = n_generations - 1
    tr = train_history[g]
    te = test_history[g]
    print(f"{g:>5} | {tr:>10.4f} | {te:>10.4f} | {te-tr:>+6.2f} | (final)")
    
    print()
    print(f"🏆 Best coefficients: {[f'{c:.3f}' for c in best_coeffs]}")
    print(f"   (degree-{len(best_coeffs)-1} polynomial)")
    
    # Grokking detection
    print()
    if grok_candidates:
        print(f"⚡ Grokking detected around generation {grok_candidates[0]}!")
        print(f"   Train loss was already low, but test loss suddenly dropped")
    else:
        # Check if there's a memorization-generalization gap
        early_gap = test_history[50] - train_history[50] if len(train_history) > 50 else 0
        late_gap = test_history[-1] - train_history[-1]
        if early_gap > 0.5 and late_gap < 0.3:
            print(f"📊 Gradual generalization (not sharp grokking)")
            print(f"   Early gap: {early_gap:.2f}, Final gap: {late_gap:.2f}")
        elif late_gap > 0.5:
            print(f"❌ Still memorizing (train-test gap = {late_gap:.2f})")
        else:
            print(f"🎯 Good generalization throughout (no grokking needed)")
    
    # Sparklines
    print()
    chars = "▁▂▃▄▅▆▇█"
    
    def sparkline(values, n=50):
        samples = [values[i * len(values) // n] for i in range(n)]
        mn, mx = min(samples), max(samples)
        rng = mx - mn if mx > mn else 1
        return "".join(chars[min(int((v - mn) / rng * (len(chars) - 1)), len(chars) - 1)] for v in samples)
    
    print(f"   Train: {sparkline(train_history)}")
    print(f"   Test:  {sparkline(test_history)}")
    
    print()
    print("💡 Connection to grokking research:")
    print("   - In neural nets, grokking = memorize → generalize (sharp transition)")
    print("   - In evolution, we see a similar but smoother pattern")
    print("   - The 'LLC' (basin flatness) in arXiv:2603.01192 may correspond to")
    print("     the 'diversity of archive' in DGM — both help escape local optima")


if __name__ == "__main__":
    main()
