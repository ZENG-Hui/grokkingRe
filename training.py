from math import ceil
import torch
from tqdm import tqdm
import wandb

from data import get_data
from model import Transformer
from lr_scheduler import get_unified_lr_scheduler, clip_grad_rms, calculate_current_L0
from sparse_utils import enforce_weight_sparsity

def main(args: dict):
    # logging
    # wandb.login(key="2695076dd0f87f2dba081c03fdc9cfa84acef643")

    # wandb.init(project="grokking_jd", config=args)
    wandb.init(mode="offline", config=args) 
    config = wandb.config
    device = torch.device(config.device)

    # Define time scales
    wandb.define_metric("step")
    wandb.define_metric("epoch")

    # Define metrics
    wandb.define_metric("training/accuracy", step_metric='step')
    wandb.define_metric("training/loss", step_metric='step')
    wandb.define_metric("validation/accuracy", step_metric='epoch')
    wandb.define_metric("validation/loss", step_metric='epoch')

    train_loader, val_loader = get_data(
        config.operation,
        config.prime,
        config.training_fraction,
        config.batch_size
        )
    # Create model with norm_type support
    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=config.prime + 2,
        seq_len=5,
        norm_type=getattr(config, 'norm_type', 'layernorm')  # Default to layernorm
    ).to(device)
    
    # Create optimizer with configurable params
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(getattr(config, 'adam_beta1', 0.9), getattr(config, 'adam_beta2', 0.98)),
        eps=getattr(config, 'adam_eps', 1e-8),
        weight_decay=config.weight_decay
    )
    
    # Use unified LR scheduler
    scheduler = get_unified_lr_scheduler(optimizer, config, config.num_steps)

    num_epochs = ceil(config.num_steps / len(train_loader))

    # Training loop with step tracking
    global_step = 0
    for epoch in tqdm(range(num_epochs)):
        global_step = train(model, train_loader, optimizer, scheduler, device, config, global_step)
        
        # Evaluate every N epochs
        eval_every = getattr(config, 'eval_every', 1)
        if (epoch + 1) % eval_every == 0:
            evaluate(model, val_loader, device, epoch)
        
        if global_step >= config.num_steps:
            break

def train(model, train_loader, optimizer, scheduler, device, config, global_step):
    """Train for one epoch with unified configuration support."""
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    
    # Check regularization type
    reg_type = getattr(config, 'regularization_type', 'l2')
    use_L0 = (reg_type in ['l2-topk', 'l0'])  # 支持新名称和旧名称

    for batch in train_loader:
        # Copy data to device
        batch = tuple(t.to(device) for t in batch)
        inputs, labels = batch

        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(inputs)[-1,:,:]
        loss = criterion(output, labels)
        acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_rms = 0.0
        if hasattr(config,  'grad_clip_rms'):
            grad_rms = clip_grad_rms(model.parameters(), config.grad_clip_rms)

        # Update weights
        optimizer.step()
        
        # Apply L0 sparsification if enabled
        if use_L0 and hasattr(config, 'final_L0'):
            current_L0 = calculate_current_L0(global_step, config, config.num_steps)
            min_conn = getattr(config, 'min_connections', 1)
            enforce_weight_sparsity(model, current_L0, min_conn)
        
        # Step scheduler
        scheduler.step()
        global_step += 1

        # Log metrics
        metrics = {
            "training/accuracy": acc,
            "training/loss": loss,
            "training/learning_rate": optimizer.param_groups[0]['lr'],
            "step": global_step
        }
        if use_L0:
            metrics["sparsity/current_L0"] = current_L0 if 'current_L0' in locals() else 1.0
        if grad_rms > 0:
            metrics["training/grad_rms"] = grad_rms
        
        wandb.log(metrics)

        # Finish training at maximum steps
        if global_step >= config.num_steps:
            return global_step
    
    return global_step

def evaluate(model, val_loader, device, epoch):
    # Set model to evaluation mode
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    correct = 0
    loss = 0.

    # Loop over each batch from the validation set
    for batch in val_loader:
        
        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch
        
        # Forward pass
        with torch.no_grad():
            output = model(inputs)[-1,:,:]
            correct += (torch.argmax(output, dim=1) == labels).sum()
            loss += criterion(output, labels) * len(labels)
    
    acc = correct / len(val_loader.dataset)
    loss = loss / len(val_loader.dataset)

    metrics = {
        "validation/accuracy": acc,
        "validation/loss": loss,
        "epoch": epoch
    }
    wandb.log(metrics, commit=False)
