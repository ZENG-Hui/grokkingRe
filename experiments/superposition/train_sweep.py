"""
Train a sweep of models (dense + multiple sparse L0 levels) and save checkpoints
+ full training history (loss, accuracy, geometry metrics every N steps).

Usage:
    python experiments/superposition/train_sweep.py

Saves:
    experiments/superposition/results/checkpoints/{label}.pt
    experiments/superposition/results/checkpoints/{label}_history.json
"""

import sys
import os
import json
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import torch
from tqdm import tqdm

# ============================================================
# Sweep configuration — matching previous successful experiments
# ============================================================

SWEEP = [
    # (label, config_overrides)
    ("dense_baseline", {
        "regularization_type": "l2",
        "final_L0": 1.0,
        "initial_L0": 1.0,
        "anneal_end_ratio": 0.0,
        "use_L0_lr_scaling": False,
    }),
    ("sparse_L0_0.80", {
        "regularization_type": "l2-topk",
        "final_L0": 0.80,
        "anneal_end_ratio": 0.3,
        "use_L0_lr_scaling": True,
    }),
    ("sparse_L0_0.50", {
        "regularization_type": "l2-topk",
        "final_L0": 0.50,
        "anneal_end_ratio": 0.3,
        "use_L0_lr_scaling": True,
    }),
    ("sparse_L0_0.20", {
        "regularization_type": "l2-topk",
        "final_L0": 0.20,
        "anneal_end_ratio": 0.3,
        "use_L0_lr_scaling": True,
    }),
    ("sparse_L0_0.10", {
        "regularization_type": "l2-topk",
        "final_L0": 0.10,
        "anneal_end_ratio": 0.3,
        "use_L0_lr_scaling": True,
    }),
]

# Shared config — matching previous successful experiments (wd=1)
SHARED = {
    "operation": "x+y",
    "prime": 97,
    "training_fraction": 0.5,
    "num_layers": 2,
    "dim_model": 128,
    "num_heads": 4,
    "batch_size": 512,
    "num_steps": 4000,
    "learning_rate": 1e-3,
    "weight_decay": 1.0,        # wd=1, matching previous grokking experiments
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "adam_eps": 1e-8,
    "warmup_ratio": 0.01,
    "use_cosine_decay": True,
    "min_lr_ratio": 0.0,
    "grad_clip_rms": 1.0,
    "device": "cpu",
    "seed": 42,
    "norm_type": "layernorm",
    # Logging
    "eval_every_steps": 50,     # evaluate val + geometry every N steps
}

CKPT_DIR = ROOT / "experiments" / "superposition" / "results" / "checkpoints"

# Geometry layers to track during training
GEOMETRY_LAYERS = ["output_proj", "ffn_up_block1"]


def eval_val(model, val_loader, criterion, device):
    """Evaluate on full validation set. Returns (val_loss, val_acc)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = tuple(t.to(device) for t in batch)
            output = model(inputs)[-1, :, :]
            loss = criterion(output, labels)
            total_loss += loss.item() * len(labels)
            total_correct += (torch.argmax(output, dim=1) == labels).sum().item()
            total_samples += len(labels)
    model.train()
    return total_loss / total_samples, total_correct / total_samples


def compute_geometry_snapshot(model):
    """Compute geometry metrics for tracked layers. Returns dict."""
    from superposition_utils import compute_overlap_stats, compute_norm_stats, extract_weight_matrices

    matrices = extract_weight_matrices(model, model_type="dense")
    snapshot = {}

    for layer_name in GEOMETRY_LAYERS:
        if layer_name not in matrices:
            continue
        W = matrices[layer_name]
        ovs = compute_overlap_stats(W)
        ns = compute_norm_stats(W)
        one_m = ovs['one_over_m']
        snapshot[layer_name] = {
            "overlap_ratio": ovs['mean_sq_overlap'] / one_m if one_m > 0 else 0,
            "mean_sq_overlap": ovs['mean_sq_overlap'],
            "max_abs_overlap": ovs['max_abs_overlap'],
            "var_ratio": ovs['var_sq_overlap'] / ovs['random_var'] if ovs['random_var'] > 0 else 0,
            "mean_norm": ns['mean_norm'],
            "std_norm": ns['std_norm'],
        }
    return snapshot


def train_and_save(label: str, overrides: dict):
    """Train one model with full history tracking."""
    from core.data import get_data
    from core.model import Transformer
    from core.lr_scheduler import get_unified_lr_scheduler, clip_grad_rms, calculate_current_L0
    from core.sparse_utils import enforce_weight_sparsity

    config = {**SHARED, **overrides}

    torch.manual_seed(config['seed'])
    device = torch.device(config['device'])

    train_loader, val_loader = get_data(
        config['operation'], config['prime'],
        config['training_fraction'], config['batch_size']
    )

    model = Transformer(
        num_layers=config['num_layers'], dim_model=config['dim_model'],
        num_heads=config['num_heads'], num_tokens=config['prime'] + 2,
        seq_len=5, norm_type=config['norm_type']
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['learning_rate'],
        betas=(config['adam_beta1'], config['adam_beta2']),
        eps=config['adam_eps'], weight_decay=config['weight_decay']
    )

    scheduler = get_unified_lr_scheduler(optimizer, config, config['num_steps'])

    reg_type = config.get('regularization_type', 'l2')
    use_L0 = reg_type in ['l2-topk', 'l0']

    num_epochs = ceil(config['num_steps'] / len(train_loader))
    criterion = torch.nn.CrossEntropyLoss()
    eval_every = config['eval_every_steps']

    # History tracking
    history = {
        "label": label,
        "config": {k: v for k, v in config.items()},
        "steps": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "geometry": {layer: {
            "overlap_ratio": [], "mean_sq_overlap": [], "max_abs_overlap": [],
            "var_ratio": [], "mean_norm": [], "std_norm": [],
        } for layer in GEOMETRY_LAYERS},
    }

    print(f"\n{'='*60}")
    print(f"TRAINING: {label}")
    print(f"  Type: {'sparse (L2+TopK)' if use_L0 else 'dense (L2 only)'}")
    print(f"  L0: {config.get('final_L0', 1.0)}, WD: {config['weight_decay']}")
    print(f"  Steps: {config['num_steps']}, eval every {eval_every} steps")
    print(f"{'='*60}")

    global_step = 0
    # Accumulate train metrics between eval points
    batch_losses = []
    batch_accs = []

    for epoch in tqdm(range(num_epochs), desc=label):
        model.train()
        for batch in train_loader:
            if global_step >= config['num_steps']:
                break
            inputs, labels = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()
            output = model(inputs)[-1, :, :]
            loss = criterion(output, labels)
            acc = (torch.argmax(output, dim=1) == labels).float().mean()
            loss.backward()

            if config.get('grad_clip_rms'):
                clip_grad_rms(model.parameters(), config['grad_clip_rms'])

            optimizer.step()

            if use_L0 and config.get('final_L0', 1.0) < 1.0:
                current_L0 = calculate_current_L0(global_step, config, config['num_steps'])
                enforce_weight_sparsity(model, current_L0, config.get('min_connections', 1))

            scheduler.step()
            global_step += 1

            batch_losses.append(loss.item())
            batch_accs.append(acc.item())

            # Periodic evaluation + geometry snapshot
            if global_step % eval_every == 0 or global_step == config['num_steps']:
                # Train metrics: average over last eval_every batches
                avg_train_loss = sum(batch_losses) / len(batch_losses)
                avg_train_acc = sum(batch_accs) / len(batch_accs)
                batch_losses.clear()
                batch_accs.clear()

                # Val metrics
                val_loss, val_acc = eval_val(model, val_loader, criterion, device)

                # Geometry snapshot
                geo = compute_geometry_snapshot(model)

                # Record
                history["steps"].append(global_step)
                history["train_loss"].append(avg_train_loss)
                history["train_acc"].append(avg_train_acc)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                for layer_name in GEOMETRY_LAYERS:
                    if layer_name in geo:
                        for key in history["geometry"][layer_name]:
                            history["geometry"][layer_name][key].append(
                                geo[layer_name].get(key, 0)
                            )

                # Print progress
                geo_str = ""
                if "output_proj" in geo:
                    geo_str = f"  out_ratio={geo['output_proj']['overlap_ratio']:.2f}"
                print(f"  Step {global_step:5d}: "
                      f"train_loss={avg_train_loss:.4f} train_acc={avg_train_acc:.4f} "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}{geo_str}")

        if global_step >= config['num_steps']:
            break

    # Save checkpoint
    ckpt_path = CKPT_DIR / f"{label}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'val_acc': history["val_acc"][-1] if history["val_acc"] else 0,
        'label': label,
    }, ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    # Save history
    hist_path = CKPT_DIR / f"{label}_history.json"
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Saved history:    {hist_path}")

    return history


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("=" * 60)
    print("SUPERPOSITION GEOMETRY SWEEP")
    print(f"Training {len(SWEEP)} models, saving to {CKPT_DIR}")
    print(f"Config: wd={SHARED['weight_decay']}, steps={SHARED['num_steps']}, "
          f"eval_every={SHARED['eval_every_steps']}")
    print("=" * 60)

    summaries = {}
    for label, overrides in SWEEP:
        ckpt = CKPT_DIR / f"{label}.pt"
        hist = CKPT_DIR / f"{label}_history.json"
        # Skip if both checkpoint and history exist
        if ckpt.exists() and hist.exists():
            print(f"\nSKIP: {label} (already exists)")
            summaries[label] = "skipped"
            continue

        history = train_and_save(label, overrides)
        final_val = history["val_acc"][-1] if history["val_acc"] else 0
        summaries[label] = final_val

    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"{'='*60}")
    for label, val in summaries.items():
        if val == "skipped":
            print(f"  {label:<25} [skipped]")
        else:
            print(f"  {label:<25} val_acc={val:.4f}")


if __name__ == '__main__':
    main()
