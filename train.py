"""
Training Module
Handles model training with support for gradient accumulation (Part 2.3) and gradient checkpointing (Part 2.4)
"""

import torch
import torch.nn as nn
import os
from tqdm import tqdm
import time
from utils import load_checkpoint
from config import Config
from utils import (
    save_checkpoint, get_lr, clip_gradients, calculate_perplexity,
    AverageMeter, format_time
)


def train_epoch(model, train_loader, optimizer, device, clip_grad_norm=1.0, accum_steps=1):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    total_tokens = 0
    optimizer.zero_grad()

    progress_bar = tqdm(train_loader, desc="Training")

    for batch_idx, (input_seq, target_seq) in enumerate(progress_bar):
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        logits, loss = model(input_seq, targets=target_seq)
        loss = loss / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        total_tokens += (target_seq != model.pad_idx).sum().item()

        progress_bar.set_postfix({
            'loss': f'{loss.item() * accum_steps:.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })

    if (batch_idx + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, device):
    """
    Validate the model

    Args:
        model: Transformer model
        val_loader: Validation dataloader
        device: Device

    Returns:
        avg_loss: Average validation loss
        perplexity: Perplexity metric
    """
    model.eval()

    total_loss = 0

    progress_bar = tqdm(val_loader, desc="Validation")

    with torch.no_grad():
        for input_seq, target_seq in progress_bar:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            logits, loss = model(input_seq, targets=target_seq)

            # Track loss
            total_loss += loss.item()

            progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(val_loader)
    perplexity = torch.exp(torch.tensor(avg_loss))

    return avg_loss, perplexity.item()


def train_transformer(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=3e-4,
    device='cuda',
    save_dir='checkpoints',
    log_interval=100,
    accum_steps=1,
    use_scheduler=True,
    early_stopping_patience=None
):
    """
    Complete training loop with validation and checkpointing

    Args:
        model: Transformer model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs: Number of epochs to train
        learning_rate: Initial learning rate
        device: Device to use
        save_dir: Directory to save checkpoints
        log_interval: How often to log progress
        accum_steps: Gradient accumulation steps (Part 2.3)
        use_scheduler: Whether to use learning rate scheduler
        early_stopping_patience: Patience for early stopping (None to disable)

    Returns:
        train_losses: List of training losses
        val_losses: List of validation losses
        perplexities: List of perplexities
    """
    os.makedirs(save_dir, exist_ok=True)

    # Move model to device
    model = model.to(device)
    print(f"Model moved to {device}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Gradient accumulation steps: {accum_steps}")
    print(f"Effective batch size: {train_loader.batch_size * accum_steps}")

    # Optimizer (Adam is standard for transformers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler (optional but recommended)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )

    # Early stopping
    early_stopping = None
    if early_stopping_patience is not None:
        from utils import EarlyStopping
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')

    # Tracking metrics
    train_losses = []
    val_losses = []
    perplexities = []
    best_val_loss = float('inf')

    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)

    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            clip_grad_norm=Config.CLIP_GRAD_NORM,
            accum_steps=accum_steps
        )
        train_losses.append(train_loss)

        # Validate
        val_loss, perplexity = validate(model, val_loader, device)
        val_losses.append(val_loss)
        perplexities.append(perplexity)

        # Update learning rate
        if scheduler is not None:
            scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start_time

        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Perplexity: {perplexity:.2f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Epoch Time: {format_time(epoch_time)}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, 'best_model.pt')
            save_checkpoint(
                model, optimizer, epoch, val_loss, checkpoint_path,
                train_loss=train_loss,
                perplexity=perplexity
            )
            print(f"   Saved best model (val_loss: {val_loss:.4f})")

            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        save_checkpoint(
            model, optimizer, epoch, val_loss, checkpoint_path,
            train_loss=train_loss,
            perplexity=perplexity
        )

        # Early stopping check
        if early_stopping is not None:
            if early_stopping(val_loss):
                print("\nEarly stopping triggered!")
                break

    total_time = time.time() - start_time

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Total Training Time: {format_time(total_time)}")

    return train_losses, val_losses, perplexities


def train_with_config(model, train_loader, val_loader, config=Config):
    """
    Train model using configuration

    Args:
        model: Transformer model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Configuration object

    Returns:
        train_losses: List of training losses
        val_losses: List of validation losses
        perplexities: List of perplexities
    """
    return train_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        device=config.DEVICE,
        save_dir=config.CHECKPOINT_DIR,
        accum_steps=config.GRADIENT_ACCUMULATION_STEPS,
        use_scheduler=config.USE_LR_SCHEDULER
    )


def resume_training(
    model,
    train_loader,
    val_loader,
    checkpoint_path,
    num_additional_epochs=5,
    device='cuda'
):



    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    checkpoint = load_checkpoint(model, checkpoint_path, optimizer, device)

    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"\nResuming from epoch {start_epoch}")

    # Train
    return train_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_additional_epochs,
        learning_rate=Config.LEARNING_RATE,
        device=device,
        save_dir=Config.CHECKPOINT_DIR
    )



