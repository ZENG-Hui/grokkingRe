"""
Script to scan final_L0 parameter for sparse training.
Runs experiments with final_L0 ranging from 0.1 to 1.0 and saves the results.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from copy import deepcopy

# Add current directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

# Import configurations and training function
from run_dense_vs_sparse import SHARED_CONFIG, SPARSE_SPECIFIC, WANDB_CONFIG
from config_sparse import SparseTrainingConfig
import training_sparse

class WandbProxy:
    """
    Proxy for wandb to capture log data while still allowing 
    normal wandb functionality if enabled.
    """
    def __init__(self, real_wandb):
        self.real_wandb = real_wandb
        self.run_history = []
        self.current_run_config = {}
        
    def init(self, *args, **kwargs):
        self.current_run_config = kwargs.get('config', {})
        if self.real_wandb:
            return self.real_wandb.init(*args, **kwargs)
        return None

    def log(self, data, *args, **kwargs):
        # Store data with a timestamp or step if available
        self.run_history.append(data)
        
        # Print progress every 100 steps or if it's a validation metric
        step = data.get('step', 0)
        if 'training/loss' in data and step % 100 == 0:
            print(f"\rStep {step}: Loss={data['training/loss']:.4f}", end="", flush=True)
        elif 'validation/loss' in data:
            print(f"\nValidation: Loss={data['validation/loss']:.4f}, Acc={data['validation/accuracy']:.4f}")
            
        if self.real_wandb:
            self.real_wandb.log(data, *args, **kwargs)

    def finish(self, *args, **kwargs):
        if self.real_wandb:
            self.real_wandb.finish(*args, **kwargs)
            
    def login(self, *args, **kwargs):
        if self.real_wandb:
            self.real_wandb.login(*args, **kwargs)
            
    def define_metric(self, *args, **kwargs):
        if self.real_wandb:
            self.real_wandb.define_metric(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.real_wandb, name)

def run_scan():
    # Range of final_L0 to scan: 0.1 to 1.0 with step 0.1
    l0_values = [round(x, 2) for x in np.arange(0.15, 0.25, 0.06)]
    
    all_results = {}
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting scan for final_L0 values: {l0_values}")
    print(f"Scan ID: {timestamp}")
    
    # Save original wandb to restore later if needed
    original_wandb = training_sparse.wandb
    
    try:
        for l0 in l0_values:
            print(f"\n" + "="*60)
            print(f"🧪 Running Experiment: final_L0 = {l0}")
            print("="*60)
            
            # Setup proxy to capture data
            proxy = WandbProxy(original_wandb)
            training_sparse.wandb = proxy
            
            # Prepare config
            current_sparse_config = deepcopy(SPARSE_SPECIFIC)
            current_sparse_config['final_L0'] = l0
            
            # Create config object
            config = SparseTrainingConfig(
                **SHARED_CONFIG, 
                **current_sparse_config,
                use_wandb=True,
                wandb_mode=WANDB_CONFIG['mode'],
                wandb_project=WANDB_CONFIG['project'],
            )
            
            # Setup wandb init to include scan info
            orig_init = proxy.init
            def custom_init(*a, **k):
                k.update({
                    'name': f'scan-{timestamp}-L0_{l0}',
                    'tags': ['scan', timestamp, f'L0_{l0}'],
                    'group': f'scan-{timestamp}'
                })
                proxy.login(key=WANDB_CONFIG['api_key'])
                # Call the bound init method of the proxy
                if proxy.real_wandb:
                    return proxy.real_wandb.init(*a, **k)
                return None
            
            # We need to patch the proxy's init method on the instance, 
            # but since training_sparse calls wandb.init, and training_sparse.wandb is our proxy,
            # we can just override the init method of our proxy instance.
            # However, a cleaner way is to just let the proxy handle it, 
            # but we need to inject the name/tags.
            # Let's monkeypatch the init on the proxy instance.
            proxy.init = custom_init

            try:
                # Run training
                training_sparse.main(config)
                
                # Store results
                all_results[str(l0)] = {
                    'config': config.to_dict(),
                    'history': proxy.run_history
                }
                print(f"✅ Completed L0={l0}")
                
            except Exception as e:
                print(f"❌ Failed L0={l0}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Ensure wandb run is closed
                proxy.finish()
                
    finally:
        # Restore original wandb
        training_sparse.wandb = original_wandb

    # Save all results to file
    scan_dir = "scan"
    os.makedirs(scan_dir, exist_ok=True)
    output_file = os.path.join(scan_dir, f"scan_results_{timestamp}.json")
    print(f"\n💾 Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    run_scan()
