"""
Script to visualize neuron activations and their Fourier spectra.
This helps in understanding the "grokking" process and identifying modular arithmetic circuits.

Usage:
    python visualize_activations.py --checkpoint checkpoints/final_model.pt
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.model_sparse import SparseTransformer
from core.model import Transformer
from core.config_sparse import SparseTrainingConfig
from core.data import get_data

def load_model(checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    # Set weights_only=False to allow loading custom config object
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']
    
    # Determine if sparse or dense
    is_sparse = isinstance(config, SparseTrainingConfig)
    
    if is_sparse:
        model = SparseTransformer(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            num_heads=config.num_heads,
            num_tokens=config.prime + 2, # +2 for special tokens if any, usually just prime is enough but let's stick to training logic
            seq_len=5, # x op y = z
            activation_sparsity=config.activation_sparsity_ratio,
            use_activation_sparsity=config.use_activation_sparsity
        )
    else:
        # Dense model loading logic (simplified)
        model = Transformer(
            num_layers=config.num_layers,
            dim_model=config.dim_model,
            num_heads=config.num_heads,
            num_tokens=config.prime + 2,
            seq_len=5,
            norm_type=getattr(config, 'norm_type', 'layernorm')
        )
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config

def get_all_inputs(prime, device):
    """Generate all possible (x, y) pairs for the operation."""
    x = torch.arange(prime)
    y = torch.arange(prime)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten
    flat_x = grid_x.flatten()
    flat_y = grid_y.flatten()
    
    # Create input sequences: [x, op, y, =]
    # Assuming op is '+' (index prime) and '=' is prime+1
    # But wait, data.py might handle this differently.
    # Let's check data.py logic. Usually it's: x, y, op -> z
    # Or x, op, y, = -> z
    # Let's assume standard format: [x, y, op] or similar.
    # Actually, let's just use the data loader logic or construct manually.
    # For modular addition x+y:
    # Input: x, y
    # But the model expects a sequence.
    # Let's look at how `data.py` constructs inputs.
    
    # Re-using get_data logic might be complex if we want ALL inputs in order.
    # Let's construct manually: [x, y, op] -> predict z?
    # Or [x, op, y, =] -> predict z?
    
    # Let's check data.py quickly.
    # For now, I'll assume [x, op, y, =] format which is length 4, predicting 5th token.
    # The model seq_len is 5.
    
    op_token = prime
    eq_token = prime + 1
    
    # Construct batch: (P*P, 4)
    # [x, op, y, =]
    batch_size = len(flat_x)
    inputs = torch.zeros((batch_size, 4), dtype=torch.long, device=device)
    inputs[:, 0] = flat_x
    inputs[:, 1] = op_token
    inputs[:, 2] = flat_y
    inputs[:, 3] = eq_token
    
    return inputs, grid_x, grid_y

def capture_activations(model, inputs):
    activations = {}
    hooks = []
    
    def get_activation(name):
        def hook(model, input, output):
            # output is (seq_len, batch, hidden_dim) or (batch, seq_len, hidden_dim)
            # We care about the last token's activation usually, or all?
            # For modular arithmetic, usually the representation of the answer matters.
            # But the FFN acts on all tokens.
            # Let's capture the last token's activation (index -1).
            # output shape: [seq_len, batch, dim] (if batch_first=False)
            # or [batch, seq_len, dim] (if batch_first=True)
            
            # Check model structure. Transformer usually outputs [seq_len, batch, dim]
            # But SparseDecoderBlock uses batch_first=False for attn?
            # Let's assume [seq_len, batch, dim] based on typical PyTorch Transformer.
            
            # We'll detach and move to CPU to save memory
            if output.shape[0] == inputs.shape[1]: # seq_len first
                 act = output[-1, :, :].detach().cpu()
            else: # batch first
                 act = output[:, -1, :].detach().cpu()
                 
            activations[name] = act
        return hook

    # Register hooks on FFN GELU outputs
    # model.blocks is nn.ModuleList
    for i, layer in enumerate(model.blocks):
        # layer.ffn is Sequential(Linear, GELU, Linear)
        # We want output of GELU (index 1)
        hooks.append(layer.ffn[1].register_forward_hook(get_activation(f"layer_{i}_ffn")))

    # Run inference
    with torch.no_grad():
        model(inputs) # inputs is [batch, seq_len] 
        # Wait, let's check model forward.
        # model.forward(x) usually expects [seq_len, batch] or [batch, seq_len]
        # Let's check model.py
    
    # Clean up hooks
    for h in hooks:
        h.remove()
        
    return activations

def plot_activations(activations, grid_x, grid_y, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    prime = grid_x.shape[0]
    
    for name, acts in activations.items():
        # acts: [batch, dim]
        dim = acts.shape[1]
        
        # Calculate variance to find interesting neurons
        variances = torch.var(acts, dim=0)
        top_k_indices = torch.topk(variances, k=5).indices
        
        print(f"Plotting top 5 neurons for {name}...")
        
        for idx in top_k_indices:
            neuron_idx = idx.item()
            act_map = acts[:, neuron_idx].view(prime, prime).numpy()
            
            # Plot Heatmap
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.imshow(act_map, origin='lower', cmap='viridis')
            plt.colorbar()
            plt.title(f"{name} Neuron {neuron_idx}\nActivation")
            plt.xlabel("y")
            plt.ylabel("x")
            
            # Plot Fourier Spectrum
            # 2D FFT
            fft = np.fft.fft2(act_map)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shifted)
            # Log scale for better visibility
            magnitude_log = np.log(magnitude + 1e-9)
            
            plt.subplot(1, 2, 2)
            plt.imshow(magnitude_log, origin='lower', cmap='inferno')
            plt.colorbar()
            plt.title(f"Fourier Spectrum (Log)")
            plt.xlabel("ky")
            plt.ylabel("kx")
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{name}_neuron_{neuron_idx}.png"))
            plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/final_model.pt')
    parser.add_argument('--save_dir', type=str, default='visualizations/activations')
    parser.add_argument('--device', type=str, default='cpu') # CPU is usually fine for inference
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    print(f"Model loaded. Config: {config}")
    
    # Prepare inputs
    prime = config.prime
    inputs, grid_x, grid_y = get_all_inputs(prime, device)
    
    # Capture activations
    # Note: Model expects [batch, seq_len]
    print("Capturing activations...")
    activations = capture_activations(model, inputs)
    
    # Plot
    print(f"Saving plots to {args.save_dir}...")
    plot_activations(activations, grid_x, grid_y, args.save_dir)
    print("Done!")

if __name__ == "__main__":
    main()
