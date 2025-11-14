"""
Utility Functions
Device handling, random seed setting, model parameter counting, memory tracking, etc.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import psutil
import gc
from typing import Dict, Any


def set_seed(seed=42):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


def get_device(prefer_cuda=True):
    """
    Get the best available device

    Args:
        prefer_cuda: Whether to prefer CUDA if available

    Returns:
        device: torch.device object
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")

    return device


def count_parameters(model, trainable_only=True):
    """
    Count model parameters

    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters

    Returns:
        num_params: Number of parameters
    """
    if trainable_only:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_params = sum(p.numel() for p in model.parameters())

    return num_params


def print_model_summary(model):
    """
    Print detailed model summary

    Args:
        model: PyTorch model
    """
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # Parameter breakdown by layer type
    print("\nParameter breakdown:")
    layer_params = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 0:
                layer_type = module.__class__.__name__
                if layer_type not in layer_params:
                    layer_params[layer_type] = 0
                layer_params[layer_type] += num_params

    for layer_type, num_params in sorted(layer_params.items(), key=lambda x: x[1], reverse=True):
        print(f"  {layer_type}: {num_params:,} ({num_params/total_params*100:.2f}%)")

    print("=" * 60)


def get_memory_usage():
    """
    Get current memory usage

    Returns:
        memory_info: Dictionary with memory statistics
    """
    memory_info = {}

    # CPU memory
    process = psutil.Process(os.getpid())
    memory_info['cpu_mb'] = process.memory_info().rss / 1024 / 1024

    # GPU memory
    if torch.cuda.is_available():
        memory_info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        memory_info['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
        memory_info['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024

    return memory_info


def print_memory_usage():
    """Print current memory usage"""
    memory_info = get_memory_usage()

    print("Memory Usage:")
    print(f"  CPU: {memory_info['cpu_mb']:.2f} MB")

    if 'gpu_allocated_mb' in memory_info:
        print(f"  GPU Allocated: {memory_info['gpu_allocated_mb']:.2f} MB")
        print(f"  GPU Reserved: {memory_info['gpu_reserved_mb']:.2f} MB")
        print(f"  GPU Max Allocated: {memory_info['gpu_max_allocated_mb']:.2f} MB")


def clear_memory():
    """Clear GPU and CPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory cleared")


def save_checkpoint(model, optimizer, epoch, loss, save_path, **kwargs):
    """
    Save model checkpoint

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint
        **kwargs: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None, device='cuda'):
    """
    Load model checkpoint

    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint
        optimizer: Optimizer (optional)
        device: Device to load model to

    Returns:
        checkpoint: Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Loss: {checkpoint.get('loss', 'N/A'):.4f}")

    return checkpoint


def get_lr(optimizer):
    """
    Get current learning rate from optimizer

    Args:
        optimizer: PyTorch optimizer

    Returns:
        lr: Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def clip_gradients(model, max_norm=1.0):
    """
    Clip gradients by norm

    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm

    Returns:
        total_norm: Total gradient norm before clipping
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm.item()


def freeze_layers(model, layer_names):
    """
    Freeze specific layers

    Args:
        model: PyTorch model
        layer_names: List of layer names to freeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = False
                print(f"Frozen layer: {name}")


def unfreeze_layers(model, layer_names):
    """
    Unfreeze specific layers

    Args:
        model: PyTorch model
        layer_names: List of layer names to unfreeze
    """
    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name:
                param.requires_grad = True
                print(f"Unfrozen layer: {name}")


def get_model_size_mb(model):
    """
    Calculate model size in MB

    Args:
        model: PyTorch model

    Returns:
        size_mb: Model size in megabytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024

    return size_mb


def format_time(seconds):
    """
    Format seconds into human-readable time

    Args:
        seconds: Time in seconds

    Returns:
        time_str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def calculate_perplexity(loss):
    """
    Calculate perplexity from cross-entropy loss

    Args:
        loss: Cross-entropy loss value

    Returns:
        perplexity: Perplexity value
    """
    return torch.exp(torch.tensor(loss)).item()


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """
    def __init__(self, patience=5, min_delta=0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' (for loss or accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        """
        Check if training should stop

        Args:
            score: Current validation metric

        Returns:
            should_stop: Boolean indicating whether to stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter += 1
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            print(f"Early stopping triggered after {self.counter} epochs without improvement")
            return True

        return False


def create_learning_rate_scheduler(optimizer, scheduler_type='cosine', **kwargs):
    """
    Create learning rate scheduler

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('cosine', 'step', 'plateau')
        **kwargs: Additional arguments for scheduler

    Returns:
        scheduler: Learning rate scheduler
    """
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 10),
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 10)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


if __name__ == '__main__':
    print("Utility functions loaded ")
    print("\nAvailable utilities:")
    print("  - set_seed(): Set random seed for reproducibility")
    print("  - get_device(): Get best available device")
    print("  - count_parameters(): Count model parameters")
    print("  - print_model_summary(): Print detailed model summary")
    print("  - get_memory_usage(): Get memory statistics")
    print("  - save_checkpoint(): Save model checkpoint")
    print("  - load_checkpoint(): Load model checkpoint")
    print("  - AverageMeter: Track running averages")
    print("  - EarlyStopping: Early stopping callback")
