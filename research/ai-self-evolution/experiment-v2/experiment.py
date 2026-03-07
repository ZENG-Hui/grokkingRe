#!/usr/bin/env python3
"""
Model Collapse Phase Transition Experiment (Pure Python)

No numpy/scipy — everything from scratch.

Question: Does model collapse exhibit a sharp phase transition?
"""

import random
import math
import json
import os

# ============================================================
# Ground truth: 3-component Gaussian mixture
# ============================================================

TRUE_COMPONENTS = [
    (0.3, -3.0, 0.8),   # (weight, mean, std)
    (0.5,  1.0, 1.2),
    (0.2,  5.0, 0.5),
]

def gauss_pdf(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))

def sample_true(n, rng):
    """Sample from true GMM."""
    samples = []
    for _ in range(n):
        r = rng.random()
        cum = 0
        for w, mu, sigma in TRUE_COMPONENTS:
            cum += w
            if r < cum:
                samples.append(rng.gauss(mu, sigma))
                break
    return samples

def true_pdf(x):
    """True mixture density at x."""
    return sum(w * gauss_pdf(x, mu, sigma) for w, mu, sigma in TRUE_COMPONENTS)

def true_ll(test_data):
    """Average log-likelihood under true distribution."""
    return sum(math.log(max(true_pdf(x), 1e-300)) for x in test_data) / len(test_data)


# ============================================================
# GMM with EM (pure Python)
# ============================================================

class GMM:
    def __init__(self, k=3):
        self.k = k
        self.weights = [1.0 / k] * k
        self.means = [0.0] * k
        self.stds = [1.0] * k
    
    def fit(self, data, n_iter=20):
        n = len(data)
        if n < self.k:
            return
        
        # Initialize: sort and split
        sd = sorted(data)
        for i in range(self.k):
            s = int(i * n / self.k)
            e = int((i + 1) * n / self.k)
            chunk = sd[s:e]
            self.means[i] = sum(chunk) / len(chunk)
            var = sum((x - self.means[i])**2 for x in chunk) / len(chunk)
            self.stds[i] = max(math.sqrt(var), 0.1)
        
        for _ in range(n_iter):
            # E-step
            resp = []
            for x in data:
                r = [self.weights[j] * gauss_pdf(x, self.means[j], self.stds[j]) 
                     for j in range(self.k)]
                s = max(sum(r), 1e-300)
                resp.append([rj / s for rj in r])
            
            # M-step
            for j in range(self.k):
                nk = max(sum(resp[i][j] for i in range(n)), 1e-10)
                self.weights[j] = nk / n
                self.means[j] = sum(resp[i][j] * data[i] for i in range(n)) / nk
                var = sum(resp[i][j] * (data[i] - self.means[j])**2 for i in range(n)) / nk
                self.stds[j] = max(math.sqrt(var), 0.05)
    
    def sample(self, n, rng):
        samples = []
        for _ in range(n):
            r = rng.random()
            cum = 0
            for j in range(self.k):
                cum += self.weights[j]
                if r < cum:
                    samples.append(rng.gauss(self.means[j], self.stds[j]))
                    break
        return samples
    
    def log_likelihood(self, data):
        total = 0
        for x in data:
            p = sum(self.weights[j] * gauss_pdf(x, self.means[j], self.stds[j]) 
                    for j in range(self.k))
            total += math.log(max(p, 1e-300))
        return total / len(data)


# ============================================================
# Experiment
# ============================================================

def run_experiment(alpha, T, n_samples=300, mode='replace', seed=0):
    rng = random.Random(seed)
    test_rng = random.Random(seed + 10000)
    test_data = sample_true(500, test_rng)
    
    # Gen 0: pure real data
    real_data = sample_true(n_samples, rng)
    model = GMM(k=3)
    model.fit(real_data)
    
    lls = [model.log_likelihood(test_data)]
    data_pool = real_data[:]  # for accumulate mode
    
    for t in range(T):
        n_syn = int(n_samples * alpha)
        n_real = n_samples - n_syn
        
        syn_data = model.sample(n_syn, rng) if n_syn > 0 else []
        fresh_real = sample_true(n_real, rng) if n_real > 0 else []
        
        if mode == 'replace':
            training = fresh_real + syn_data
        else:  # accumulate
            data_pool = data_pool + fresh_real + syn_data
            # Cap pool size to avoid slowdown
            if len(data_pool) > 2000:
                rng_sample = random.Random(seed + t)
                data_pool = rng_sample.sample(data_pool, 2000)
            training = data_pool
        
        model = GMM(k=3)
        model.fit(training)
        lls.append(model.log_likelihood(test_data))
    
    return lls


def main():
    print("=" * 65)
    print("  MODEL COLLAPSE PHASE TRANSITION EXPERIMENT")
    print("=" * 65)
    
    # Smaller sweep for pure Python (slower)
    alphas = [i * 0.1 for i in range(11)]  # 0.0 to 1.0, step 0.1
    T_values = [1, 5, 10, 20]
    n_seeds = 8
    
    test_rng = random.Random(99999)
    test_data = sample_true(500, test_rng)
    baseline = true_ll(test_data)
    print(f"\nTrue distribution LL: {baseline:.4f}")
    
    results = {}
    
    for mode in ['replace', 'accumulate']:
        print(f"\n{'='*55}")
        print(f"  MODE: {mode.upper()}")
        print(f"{'='*55}")
        results[mode] = {}
        
        for T in T_values:
            results[mode][T] = {}
            ll_baseline = None
            
            row_data = []
            for alpha in alphas:
                lls_final = []
                for seed in range(n_seeds):
                    lls = run_experiment(alpha, T, 300, mode, seed)
                    lls_final.append(lls[-1])
                
                mean_ll = sum(lls_final) / len(lls_final)
                std_ll = (sum((x - mean_ll)**2 for x in lls_final) / len(lls_final)) ** 0.5
                results[mode][T][f"{alpha:.1f}"] = {'mean': mean_ll, 'std': std_ll}
                row_data.append((alpha, mean_ll, std_ll))
                
                if alpha == 0:
                    ll_baseline = mean_ll
            
            # Print
            print(f"\n  T={T}:")
            print(f"  {'α':>4} | {'LL':>8} | {'Δ':>7} | quality")
            print(f"  ----+----------+---------+--------------------")
            for alpha, mean_ll, std_ll in row_data:
                delta = mean_ll - ll_baseline
                # Bar: normalize between -4 and baseline
                frac = max(0, min(1, (mean_ll - (-4)) / (baseline - (-4))))
                bar_len = int(frac * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"  {alpha:4.1f} | {mean_ll:8.4f} | {delta:+7.4f} | {bar}")
    
    # ============================================================
    # Phase transition detection
    # ============================================================
    print(f"\n{'='*65}")
    print("  PHASE TRANSITION DETECTION")
    print(f"{'='*65}")
    print()
    print("  Sharpness = max|dLL/dα| / mean|dLL/dα|")
    print("  >2 suggests phase transition, ≈1 suggests smooth degradation")
    print()
    
    for mode in ['replace', 'accumulate']:
        print(f"  --- {mode.upper()} ---")
        for T in T_values:
            lls = [results[mode][T][f"{a:.1f}"]['mean'] for a in alphas]
            derivs = [(lls[i+1] - lls[i]) / 0.1 for i in range(len(lls)-1)]
            abs_derivs = [abs(d) for d in derivs]
            
            if max(abs_derivs) == 0:
                print(f"  T={T:3d}: no change")
                continue
            
            max_idx = abs_derivs.index(max(abs_derivs))
            peak_alpha = alphas[max_idx] + 0.05
            avg_abs = sum(abs_derivs) / len(abs_derivs)
            sharpness = max(abs_derivs) / avg_abs if avg_abs > 0 else 0
            total_drop = lls[0] - lls[-1]
            
            print(f"  T={T:3d}: peak at α≈{peak_alpha:.2f}, "
                  f"sharpness={sharpness:.2f}, drop={total_drop:.4f}")
        print()
    
    # ============================================================
    # Key question: does sharpness increase with T?
    # ============================================================
    print(f"  --- SHARPNESS vs T (replace mode) ---")
    print(f"  Does the transition get sharper with more iterations?")
    print(f"  T   | sharpness | interpretation")
    print(f"  ----+-----------+--------------")
    
    for T in T_values:
        lls = [results['replace'][T][f"{a:.1f}"]['mean'] for a in alphas]
        derivs = [(lls[i+1] - lls[i]) / 0.1 for i in range(len(lls)-1)]
        abs_derivs = [abs(d) for d in derivs]
        avg_abs = sum(abs_derivs) / len(abs_derivs)
        sharpness = max(abs_derivs) / avg_abs if avg_abs > 0 else 0
        
        if sharpness > 2:
            interp = "SHARP transition"
        elif sharpness > 1.5:
            interp = "moderate"  
        else:
            interp = "smooth"
        print(f"  {T:3d} | {sharpness:9.2f} | {interp}")
    
    # Save
    output = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_results.json')
    with open(output, 'w') as f:
        json_r = {mode: {str(T): results[mode][T] for T in results[mode]} for mode in results}
        json.dump(json_r, f, indent=2)
    print(f"\n  Results saved to raw_results.json")


if __name__ == "__main__":
    main()
