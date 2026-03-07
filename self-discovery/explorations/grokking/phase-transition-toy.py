#!/usr/bin/env python3
"""
Toy Phase Transition Simulator — Understanding Grokking Intuitively

Simulates a particle (= SGD) in a double-well potential:
  - Left well = memorization basin (low LLC, easy to reach)
  - Right well = generalization basin (lower energy, better but harder to reach)
  - Noise = SGD noise / weight decay
  
As noise increases (= more regularization), the particle can escape the 
memorization basin and find the generalization basin.

This is the essence of arXiv:2603.01192's "competing basins" picture.

Created by DeepTeneral during free exploration time.
"""

import math
import random


def double_well_potential(x, barrier_height=2.0, asymmetry=0.5):
    """
    Double-well potential: V(x) = (x^2 - 1)^2 + asymmetry * x
    - Left well (memorization): x ≈ -1, higher energy
    - Right well (generalization): x ≈ +1, lower energy due to asymmetry
    """
    return (x**2 - 1)**2 - asymmetry * x


def simulate_sgd(noise_level, n_steps=2000, dt=0.01, barrier=2.0, asym=0.5):
    """Simulate overdamped Langevin dynamics (= SGD with noise)."""
    x = -1.0 + random.gauss(0, 0.1)  # Start near memorization basin
    
    trajectory = []
    for t in range(n_steps):
        # Force = -dV/dx
        dVdx = 4 * x * (x**2 - 1) - asym
        
        # Overdamped Langevin: dx = -dV/dx * dt + sqrt(2*noise*dt) * dW
        x += -dVdx * dt + math.sqrt(2 * noise_level * dt) * random.gauss(0, 1)
        
        trajectory.append(x)
    
    return trajectory


def classify_trajectory(traj):
    """Determine if the particle grokked (escaped to right well)."""
    # Average position in last 10% of trajectory
    last_segment = traj[int(len(traj) * 0.9):]
    avg_pos = sum(last_segment) / len(last_segment)
    
    if avg_pos > 0.5:
        return "generalized"
    elif avg_pos < -0.5:
        return "memorized"
    else:
        return "transitioning"


def sparkline(values, width=60):
    """Create a sparkline from values."""
    chars = "▁▂▃▄▅▆▇█"
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1
    return "".join(chars[min(int((v - mn) / rng * (len(chars) - 1)), len(chars) - 1)] 
                   for v in values[:width])


def main():
    print("🔬 Toy Phase Transition Simulator")
    print("=" * 60)
    print()
    print("Double-well potential: memorization (left) vs generalization (right)")
    print("Noise = regularization strength (e.g., weight decay)")
    print()
    
    # Scan noise levels
    noise_levels = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    n_trials = 20
    
    print(f"{'Noise':>6} | {'Grok%':>6} | {'Avg final pos':>13} | {'Trajectory (first trial)':<40}")
    print("-" * 80)
    
    for noise in noise_levels:
        grok_count = 0
        final_positions = []
        first_traj = None
        
        for trial in range(n_trials):
            traj = simulate_sgd(noise, n_steps=3000)
            result = classify_trajectory(traj)
            if result == "generalized":
                grok_count += 1
            final_positions.append(traj[-1])
            if first_traj is None:
                # Subsample for sparkline
                step = max(1, len(traj) // 40)
                first_traj = [traj[i] for i in range(0, len(traj), step)]
        
        grok_pct = grok_count / n_trials * 100
        avg_final = sum(final_positions) / len(final_positions)
        spark = sparkline(first_traj, 40)
        
        marker = ""
        if grok_pct > 80:
            marker = " ✅ always groks"
        elif grok_pct > 20:
            marker = " ⚡ critical region!"
        else:
            marker = " ❌ stuck memorizing"
        
        print(f"{noise:6.1f} | {grok_pct:5.0f}% | {avg_final:+13.2f} | {spark}{marker}")
    
    print()
    print("📊 Interpretation:")
    print("  - Low noise → particle stays in memorization basin (left well)")
    print("  - High noise → particle always finds generalization basin (right well)")
    print("  - Critical noise → phase transition! Sometimes groks, sometimes doesn't")
    print()
    print("  This is exactly what arXiv:2603.01192 describes:")
    print("  Grokking = phase transition between competing basins")
    print("  LLC = measure of basin 'flatness' = determines which basin wins")
    print()
    print("  The critical insight: regularization (noise) is what enables")
    print("  the escape from memorization to generalization.")
    print()
    print("🦀 DeepTeneral — learning by simulating")


if __name__ == "__main__":
    random.seed(42)  # Reproducible
    main()
