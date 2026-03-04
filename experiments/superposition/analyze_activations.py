"""
Activation analysis: how does the model internally represent identity and inverse?

Feed actual input sequences through the model, record hidden activations at each layer,
and compare activation patterns for:
  1. Identity: [x, +, 0, =] vs [x, +, y, =] (does y=0 produce a "shortcut"?)
  2. Inverse: [x, +, (97-x), =] vs [x, +, y, =] (does the inverse sum produce a special pattern?)
  3. General: how does the hidden state at "=" position encode the answer?

Usage:
    python experiments/superposition/analyze_activations.py
"""

import sys
import os
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import Transformer

CKPT_DIR = SCRIPT_DIR / "results" / "checkpoints"
OUT_DIR = SCRIPT_DIR / "results" / "geometry" / "activations"
P = 97
EQ_TOKEN = P       # 97 (eq_token = prime)
OP_TOKEN = P + 1   # 98 (op_token = prime + 1)


def load_model(ckpt_name):
    ckpt_path = CKPT_DIR / f"{ckpt_name}.pt"
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    config = ckpt.get('config', {})

    model = Transformer(
        num_layers=config.get('num_layers', 2),
        dim_model=config.get('dim_model', 128),
        num_heads=config.get('num_heads', 4),
        num_tokens=P + 2,
        seq_len=5,
        norm_type=config.get('norm_type', 'layernorm'),
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()
    return model


def make_input(x, y):
    """Create input tensor [x, op, y, eq] with shape [1, 4]."""
    return torch.tensor([[x, OP_TOKEN, y, EQ_TOKEN]])


def collect_activations(model, inputs):
    """
    Run forward pass and collect activations at key points.
    Returns dict: layer_name -> activation at "=" position (last token, index 3).
    
    Points collected:
      - after_embed: embedding + position_embedding [4, 1, 128] -> take pos 3
      - after_attn_block{i}: after each DecoderBlock
      - after_norm: after final norm
      - logits: final output
    """
    activations = {}
    hooks = []

    # Hook into embedding output
    def embed_hook(act_dict):
        def hook_fn(module, input, output):
            # output shape: [seq_len, batch, dim] after rearrange
            act_dict['after_embed'] = output[:, 0, :].detach().clone()  # [4, 128]
        return hook_fn

    # We need to manually trace through the model since nn.Sequential doesn't expose intermediates
    # Instead, do a manual forward pass

    with torch.no_grad():
        batch_size, context_len = inputs.shape
        token_emb = model.token_embeddings(inputs)
        from einops import repeat
        positions = repeat(torch.arange(context_len, device=inputs.device), "p -> b p", b=batch_size)
        pos_emb = model.position_embeddings(positions)
        embedding = token_emb + pos_emb  # [1, 4, 128]

        from einops import rearrange
        x = rearrange(embedding, 'b s d -> s b d')  # [4, 1, 128]
        activations['after_embed'] = x[:, 0, :].clone()  # [4, 128]

        # Walk through nn.Sequential manually
        for i, layer in enumerate(model.model):
            if hasattr(layer, 'self_attn'):
                # DecoderBlock
                x = layer(x)
                activations[f'after_block{i}'] = x[:, 0, :].clone()  # [4, 128]
            elif isinstance(layer, (torch.nn.LayerNorm,)):
                x = layer(x)
                activations['after_final_norm'] = x[:, 0, :].clone()
            elif hasattr(layer, 'weight') and hasattr(layer, 'bias'):
                # Final Linear
                x = layer(x)
                activations['logits'] = x[:, 0, :].clone()  # [4, 99]
            else:
                # RMSNorm or other
                x = layer(x)
                activations['after_final_norm'] = x[:, 0, :].clone()

    return activations


def analyze_identity(model, label):
    """
    Compare activations for [x, +, 0, =] vs [x, +, y, =] for various x, y.
    Question: does the model take a "shortcut" when y=0?
    """
    print(f"\n{'='*70}")
    print(f"IDENTITY ANALYSIS (y=0): {label}")
    print(f"{'='*70}")

    # Collect activations for all x with y=0
    identity_acts = {}  # x -> activations dict
    for x in range(P):
        acts = collect_activations(model, make_input(x, 0))
        identity_acts[x] = acts

    # Collect activations for random (x, y) pairs as baseline
    np.random.seed(42)
    baseline_acts = {}
    for _ in range(200):
        x = np.random.randint(0, P)
        y = np.random.randint(1, P)  # y != 0
        acts = collect_activations(model, make_input(x, y))
        baseline_acts[(x, y)] = acts

    # For each layer, compare the "=" position (index 3) activations
    layers = list(identity_acts[0].keys())

    print(f"\n  Analysis at '=' position (index 3):")
    print(f"  {'Layer':<25} {'y=0 norm':>10} {'y!=0 norm':>10} {'ratio':>7} {'cos_sim(y0,y!=0)':>17}")
    print("  " + "-" * 72)

    results = {}
    for layer in layers:
        # Get "=" position activation (index 3, or last for logits)
        pos = 3 if identity_acts[0][layer].shape[0] == 4 else -1

        # y=0 activations
        y0_vecs = torch.stack([identity_acts[x][layer][pos] for x in range(P)])  # [97, d]
        y0_norms = y0_vecs.norm(dim=1)

        # baseline activations (y!=0) at "=" position
        bl_vecs = torch.stack([baseline_acts[k][layer][pos] for k in baseline_acts])  # [200, d]
        bl_norms = bl_vecs.norm(dim=1)

        # Compare norms
        norm_ratio = y0_norms.mean() / (bl_norms.mean() + 1e-8)

        # For each x, find cos_sim between [x,+,0,=] and mean of [x,+,y,=] for y!=0
        cos_sims = []
        for x in range(min(20, P)):  # sample 20 x values
            y0_vec = identity_acts[x][layer][pos]
            # Get baseline with same x
            same_x_vecs = [baseline_acts[k][layer][pos] for k in baseline_acts if k[0] == x]
            if same_x_vecs:
                mean_bl = torch.stack(same_x_vecs).mean(dim=0)
                cs = F.cosine_similarity(y0_vec.unsqueeze(0), mean_bl.unsqueeze(0)).item()
                cos_sims.append(cs)

        mean_cos = np.mean(cos_sims) if cos_sims else 0

        print(f"  {layer:<25} {y0_norms.mean():>10.4f} {bl_norms.mean():>10.4f} "
              f"{norm_ratio:>7.3f} {mean_cos:>17.4f}")

        results[layer] = {
            'y0_norm_mean': float(y0_norms.mean()),
            'baseline_norm_mean': float(bl_norms.mean()),
            'norm_ratio': float(norm_ratio),
            'cos_sim_y0_vs_baseline': float(mean_cos),
        }

    # Key test: does [x,+,0,=] produce logits that peak at x?
    print(f"\n  Identity accuracy test: does argmax(logits[x,+,0,=]) == x?")
    correct = 0
    for x in range(P):
        logits = identity_acts[x]['logits'][3] if identity_acts[x]['logits'].shape[0] == 4 else identity_acts[x]['logits'][-1]
        pred = logits[:P].argmax().item()
        if pred == x:
            correct += 1
    print(f"  {correct}/{P} correct ({correct/P:.1%})")

    return results, identity_acts, baseline_acts


def analyze_inverse(model, label):
    """
    Compare activations for [x, +, (97-x), =] (result=0) vs other [x, +, y, =].
    Question: does x + (97-x) = 0 produce a special activation pattern?
    """
    print(f"\n{'='*70}")
    print(f"INVERSE ANALYSIS (x+y≡0): {label}")
    print(f"{'='*70}")

    # Collect inverse pair activations: [x, +, 97-x, =]
    inverse_acts = {}
    for x in range(1, P):  # skip x=0 since 97-0=97 is out of range... actually 97%97=0
        y = (P - x) % P
        acts = collect_activations(model, make_input(x, y))
        inverse_acts[x] = acts

    # Baseline: random pairs where answer != 0
    np.random.seed(123)
    baseline_acts = {}
    count = 0
    while count < 200:
        x = np.random.randint(0, P)
        y = np.random.randint(0, P)
        if (x + y) % P != 0:
            acts = collect_activations(model, make_input(x, y))
            baseline_acts[(x, y)] = acts
            count += 1

    layers = list(inverse_acts[1].keys())

    print(f"\n  Analysis at '=' position (index 3):")
    print(f"  {'Layer':<25} {'inv norm':>10} {'other norm':>10} {'ratio':>7} {'inv_var':>9} {'other_var':>10}")
    print("  " + "-" * 75)

    for layer in layers:
        pos = 3 if inverse_acts[1][layer].shape[0] == 4 else -1

        inv_vecs = torch.stack([inverse_acts[x][layer][pos] for x in range(1, P)])
        bl_vecs = torch.stack([baseline_acts[k][layer][pos] for k in baseline_acts])

        inv_norms = inv_vecs.norm(dim=1)
        bl_norms = bl_vecs.norm(dim=1)
        norm_ratio = inv_norms.mean() / (bl_norms.mean() + 1e-8)

        # Variance of activations (are inverse-pair activations more "concentrated"?)
        inv_var = inv_vecs.var(dim=0).mean()
        bl_var = bl_vecs.var(dim=0).mean()

        print(f"  {layer:<25} {inv_norms.mean():>10.4f} {bl_norms.mean():>10.4f} "
              f"{norm_ratio:>7.3f} {inv_var:>9.4f} {bl_var:>10.4f}")

    # Are inverse activations more similar to each other?
    # (all produce answer=0, so their "=" hidden states should converge)
    print(f"\n  Convergence test: do all [x, +, 97-x, =] produce similar '=' activations?")
    for layer in ['after_block0', 'after_block1', 'after_final_norm']:
        if layer not in inverse_acts[1]:
            continue
        pos = 3 if inverse_acts[1][layer].shape[0] == 4 else -1
        inv_vecs = torch.stack([inverse_acts[x][layer][pos] for x in range(1, P)])
        inv_vecs_norm = F.normalize(inv_vecs, dim=1)
        # Mean pairwise cosine similarity
        G = inv_vecs_norm @ inv_vecs_norm.T
        mask = ~torch.eye(len(G), dtype=torch.bool)
        mean_cos = G[mask].mean().item()

        # Same for baseline (sample 96 from baseline for fair comparison)
        bl_keys = list(baseline_acts.keys())[:96]
        bl_vecs = torch.stack([baseline_acts[k][layer][pos] for k in bl_keys])
        bl_vecs_norm = F.normalize(bl_vecs, dim=1)
        G_bl = bl_vecs_norm @ bl_vecs_norm.T
        mask_bl = ~torch.eye(len(G_bl), dtype=torch.bool)
        mean_cos_bl = G_bl[mask_bl].mean().item()

        print(f"  {layer:<25} inverse mean_cos={mean_cos:.4f}  baseline mean_cos={mean_cos_bl:.4f}  "
              f"{'CONVERGE' if mean_cos > mean_cos_bl + 0.05 else 'no signal'}")

    # Accuracy test
    print(f"\n  Inverse accuracy: does argmax(logits[x,+,97-x,=]) == 0?")
    correct = 0
    for x in range(1, P):
        logits = inverse_acts[x]['logits']
        pred_logit = logits[3] if logits.shape[0] == 4 else logits[-1]
        pred = pred_logit[:P].argmax().item()
        if pred == 0:
            correct += 1
    print(f"  {correct}/{P-1} correct ({correct/(P-1):.1%})")

    return inverse_acts, baseline_acts


def analyze_answer_encoding(model, label):
    """
    For all 97 possible answers, collect the "=" position hidden state
    and check if same-answer inputs cluster together.
    """
    print(f"\n{'='*70}")
    print(f"ANSWER ENCODING ANALYSIS: {label}")
    print(f"{'='*70}")

    # Group inputs by answer
    answer_acts = defaultdict(list)  # answer -> list of hidden states

    # Sample: for each answer a, collect a few (x, y) pairs where (x+y)%97 = a
    np.random.seed(456)
    for a in range(P):
        count = 0
        for x in range(P):
            y = (a - x) % P
            if count >= 5:
                break
            acts = collect_activations(model, make_input(x, y))
            pos = 3 if acts['after_final_norm'].shape[0] == 4 else -1
            answer_acts[a].append(acts['after_final_norm'][pos])
            count += 1

    # For each answer, compute mean hidden state
    answer_means = {}
    for a in range(P):
        vecs = torch.stack(answer_acts[a])
        answer_means[a] = vecs.mean(dim=0)

    # Compute pairwise similarity between answer-mean vectors
    mean_matrix = torch.stack([answer_means[a] for a in range(P)])  # [97, 128]
    mean_norm = F.normalize(mean_matrix, dim=1)
    G = (mean_norm @ mean_norm.T).numpy()
    np.fill_diagonal(G, 0)

    # Check neighborhood structure (same as before but on answer-conditioned hidden states)
    print(f"\n  Neighborhood in answer-conditioned hidden states (after_final_norm):")
    for dist in [1, 2, 5, 10, 48]:
        pairs = [(a, (a + dist) % P) for a in range(P)]
        sims = [G[a, b] for a, b in pairs if a != b]
        print(f"    Distance {dist:>2}: mean_cos={np.mean(sims):.4f}")

    # Within-answer consistency: are hidden states for same answer similar?
    print(f"\n  Within-answer consistency (5 inputs per answer):")
    within_sims = []
    for a in range(P):
        vecs = torch.stack(answer_acts[a])
        vn = F.normalize(vecs, dim=1)
        G_w = vn @ vn.T
        mask = ~torch.eye(len(G_w), dtype=torch.bool)
        if mask.sum() > 0:
            within_sims.append(G_w[mask].mean().item())
    print(f"    Mean within-answer cos_sim: {np.mean(within_sims):.4f}")
    print(f"    Std:                        {np.std(within_sims):.4f}")

    # Cross-answer similarity (random pairs of different answers)
    cross_sims = []
    for _ in range(500):
        a1 = np.random.randint(0, P)
        a2 = np.random.randint(0, P)
        if a1 != a2:
            v1 = answer_acts[a1][0]
            v2 = answer_acts[a2][0]
            cs = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
            cross_sims.append(cs)
    print(f"    Mean cross-answer cos_sim:  {np.mean(cross_sims):.4f}")
    print(f"    Within/Cross ratio:         {np.mean(within_sims)/np.mean(cross_sims):.2f}x")

    return answer_means


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    models = [
        ("dense_baseline", "Dense_wd1"),
    ]

    for ckpt_name, label in models:
        model = load_model(ckpt_name)

        id_results, id_acts, id_baseline = analyze_identity(model, label)
        inv_acts, inv_baseline = analyze_inverse(model, label)
        answer_means = analyze_answer_encoding(model, label)

    print("\nDone.")


if __name__ == '__main__':
    main()
