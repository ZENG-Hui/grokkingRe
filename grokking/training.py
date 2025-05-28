from math import ceil
import torch
from tqdm import tqdm # 进度条
import wandb # Weights and Biases for experiment tracking

# 用于保存模型
import os
from pathlib import Path

from data import get_data
from model import Transformer

def main(args: dict):
    wandb.init(project="grokking", config=args) 
    config = wandb.config 
    device = torch.device(config.device)

    # 添加模型保存目录
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    best_val_acc = 0.0 # 用于保存最佳验证准确率，准确率提高了就保存

    # Define time scales
    wandb.define_metric("step")
    wandb.define_metric("epoch")

    # Define metrics
    # 也就是我们在网页上看到的指标
    wandb.define_metric("training/accuracy", step_metric='step')
    wandb.define_metric("training/loss", step_metric='step')
    wandb.define_metric("validation/accuracy", step_metric='epoch')
    wandb.define_metric("validation/loss", step_metric='epoch')

    # 生成训练和验证数据加载器
    train_loader, val_loader = get_data(
        config.operation,
        config.prime,
        config.training_fraction,
        config.batch_size
        )
    # 调用模型，将模型转移到指定设备
    model = Transformer(
        num_layers=config.num_layers,
        dim_model=config.dim_model,
        num_heads=config.num_heads,
        num_tokens=config.prime + 2,
        seq_len=5
        ).to(device)
    # 定义优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=config.weight_decay
        )
    
    # 原来用的是线性学习率调度器，为了避免后期震荡曾晖改成余弦退火了
    # scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer, start_factor = 0.1, total_iters=9
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_steps)

    # 进行epoch轮训练，每轮包括num_steps次梯度更新和一次验证
    # 计算总的epoch数为总训练步数除以每个epoch的训练步数(batch_size)
    num_epochs = ceil(config.num_steps / len(train_loader)) 
    
    for epoch in tqdm(range(num_epochs)):
        train(model, train_loader, optimizer, scheduler, device, config.num_steps)
        val_acc = evaluate(model, val_loader, device, epoch)

        # # 如果验证准确率提高了，就保存模型
        # if val_acc > best_val_acc:
        #     best_val_acc = val_acc
        #     model_path = save_dir / f"model_epoch_{epoch}_acc_{val_acc:.4f}.pt"
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_accuracy': val_acc,
        #         'config': dict(config),
        #     }, model_path)
        #     wandb.log({"best_model_saved": True, "best_val_accuracy": val_acc})

        # # 每10个epoch保存一次检查点
        # if epoch % 10 == 0:
        #     checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}_{wandb.run.name}.pt"
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_accuracy': val_acc,
        #         'config': dict(config),
        #     }, checkpoint_path)
        
    # 保存最终模型，名称末尾为wandb任务ID
    final_model_path = save_dir / f"final_model_{wandb.run.id}.pt"
    torch.save({
        'epoch': num_epochs-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_acc,
        'config': dict(config),
    }, final_model_path)
    wandb.log({"final_model_saved": True})
            

# 训练模式
def train(model, train_loader, optimizer, scheduler, device, num_steps):
    # Set model to training mode
    model.train()
    criterion = torch.nn.CrossEntropyLoss() # 使用交叉熵损失函数

    # Loop over each batch from the training set
    for batch in train_loader:

        # Copy data to device if needed
        batch = tuple(t.to(device) for t in batch)

        # Unpack the batch from the loader
        inputs, labels = batch

        # Zero gradient buffers
        optimizer.zero_grad()
        
        # Forward pass
        output = model(inputs)[-1,:,:] #前向传播：[-1,:,:]提取序列最后位置（等号后）的预测
        loss = criterion(output, labels)
        acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels) # 计算准确率
        
        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()
        scheduler.step()

        metrics = {
            "training/accuracy": acc,
            "training/loss": loss,
            "step": wandb.run.step
        }
        wandb.log(metrics)

        # Finish training at maximum gradient updates
        if wandb.run.step == num_steps:
            return

# 评估模式
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
        with torch.no_grad(): # 禁用梯度计算以节省内存和计算资源
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

    # 返回验证准确率，用于保存最佳模型
    return acc
