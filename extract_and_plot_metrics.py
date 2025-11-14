"""
Extract training metrics from checkpoints and create plots
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Paths
checkpoint_dir = '/home/silversage22/Desktop/sem7/AIL861/assignment/submission/code/checkpoints'
results_dir = '/home/silversage22/Desktop/sem7/AIL861/assignment/submission/code/results'

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

print("=" * 60)
print("EXTRACTING TRAINING METRICS FROM CHECKPOINTS")
print("=" * 60)

# Extract metrics from each checkpoint
epochs = []
train_losses = []
val_losses = []
perplexities = []

checkpoint_files = [
    'checkpoint_epoch_1.pt',
    'checkpoint_epoch_2.pt',
    'checkpoint_epoch_3.pt',
    'checkpoint_epoch_4.pt'
]

print("\nLoading checkpoints...")
for i, checkpoint_file in enumerate(checkpoint_files, start=1):
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)

    if os.path.exists(checkpoint_path):
        print(f"\nEpoch {i}:")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        epoch = checkpoint.get('epoch', i-1) + 1
        train_loss = checkpoint.get('train_loss', None)
        val_loss = checkpoint.get('val_loss', checkpoint.get('loss', None))
        perplexity = checkpoint.get('perplexity', None)

        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        perplexities.append(perplexity)

        print(f"  Epoch: {epoch}")
        print(f"  Train Loss: {train_loss:.4f}" if train_loss else "  Train Loss: N/A")
        print(f"  Val Loss: {val_loss:.4f}" if val_loss else "  Val Loss: N/A")
        print(f"  Perplexity: {perplexity:.2f}" if perplexity else "  Perplexity: N/A")
    else:
        print(f"\nWarning: {checkpoint_file} not found!")

print("\n" + "=" * 60)
print("METRICS EXTRACTED")
print("=" * 60)
print(f"Epochs: {epochs}")
print(f"Train Losses: {[f'{l:.4f}' if l else 'N/A' for l in train_losses]}")
print(f"Val Losses: {[f'{l:.4f}' if l else 'N/A' for l in val_losses]}")
print(f"Perplexities: {[f'{p:.2f}' if p else 'N/A' for p in perplexities]}")

# Filter out None values for plotting
valid_train = [(e, l) for e, l in zip(epochs, train_losses) if l is not None]
valid_val = [(e, l) for e, l in zip(epochs, val_losses) if l is not None]
valid_perp = [(e, p) for e, p in zip(epochs, perplexities) if p is not None]

if valid_train:
    train_epochs, train_vals = zip(*valid_train)
else:
    train_epochs, train_vals = [], []

if valid_val:
    val_epochs, val_vals = zip(*valid_val)
else:
    val_epochs, val_vals = [], []

if valid_perp:
    perp_epochs, perp_vals = zip(*valid_perp)
else:
    perp_epochs, perp_vals = [], []

# ============================================
# PLOT 1: Training Loss Curve
# ============================================
print("\n" + "=" * 60)
print("CREATING PLOTS")
print("=" * 60)

if train_vals:
    print("\n1. Creating Training Loss Curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_vals, 'b-', marker='o', linewidth=2, markersize=8, label='Training Loss')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Training Loss Curve', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(train_epochs)

    # Add value annotations
    for e, l in zip(train_epochs, train_vals):
        plt.annotate(f'{l:.3f}', (e, l), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'training_loss_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Saved to: {save_path}")
    plt.close()

# ============================================
# PLOT 2: Validation Loss Curve
# ============================================
if val_vals:
    print("\n2. Creating Validation Loss Curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(val_epochs, val_vals, 'r-', marker='s', linewidth=2, markersize=8, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Validation Loss Curve', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(val_epochs)

    # Add value annotations
    for e, l in zip(val_epochs, val_vals):
        plt.annotate(f'{l:.3f}', (e, l), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'validation_loss_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Saved to: {save_path}")
    plt.close()

# ============================================
# PLOT 3: Perplexity Curve
# ============================================
if perp_vals:
    print("\n3. Creating Perplexity Curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(perp_epochs, perp_vals, 'g-', marker='^', linewidth=2, markersize=8, label='Perplexity')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Perplexity', fontsize=14, fontweight='bold')
    plt.title('Perplexity Curve', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(perp_epochs)

    # Add value annotations
    for e, p in zip(perp_epochs, perp_vals):
        plt.annotate(f'{p:.2f}', (e, p), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'perplexity_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Saved to: {save_path}")
    plt.close()

# ============================================
# PLOT 4: Combined Training & Validation Loss
# ============================================
if train_vals or val_vals:
    print("\n4. Creating Combined Loss Plot...")
    plt.figure(figsize=(12, 7))

    if train_vals:
        plt.plot(train_epochs, train_vals, 'b-', marker='o', linewidth=2.5,
                markersize=10, label='Training Loss', alpha=0.8)

    if val_vals:
        plt.plot(val_epochs, val_vals, 'r-', marker='s', linewidth=2.5,
                markersize=10, label='Validation Loss', alpha=0.8)

    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=13, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)

    # Use all unique epochs for x-ticks
    all_epochs = sorted(set(list(train_epochs) + list(val_epochs)))
    plt.xticks(all_epochs)

    plt.tight_layout()
    save_path = os.path.join(results_dir, 'combined_loss_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   Saved to: {save_path}")
    plt.close()

# ============================================
# PLOT 5: All Metrics in One Figure
# ============================================
print("\n5. Creating Comprehensive Training Curves...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Training Loss
if train_vals:
    axes[0].plot(train_epochs, train_vals, 'b-', marker='o', linewidth=2, markersize=8)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(train_epochs)

# Validation Loss
if val_vals:
    axes[1].plot(val_epochs, val_vals, 'r-', marker='s', linewidth=2, markersize=8)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(val_epochs)

# Perplexity
if perp_vals:
    axes[2].plot(perp_epochs, perp_vals, 'g-', marker='^', linewidth=2, markersize=8)
    axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Perplexity', fontsize=12, fontweight='bold')
    axes[2].set_title('Perplexity', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(perp_epochs)

plt.suptitle('Training Progress - All Metrics', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
save_path = os.path.join(results_dir, 'training_curves_all.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"   Saved to: {save_path}")
plt.close()

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nTotal Epochs Trained: {len(epochs)}")
if train_vals:
    print(f"Final Training Loss: {train_vals[-1]:.4f}")
if val_vals:
    print(f"Final Validation Loss: {val_vals[-1]:.4f}")
if perp_vals:
    print(f"Final Perplexity: {perp_vals[-1]:.2f}")

print(f"\nAll plots saved to: {results_dir}")
print("\nPlots created:")
print("  1. training_loss_curve.png")
print("  2. validation_loss_curve.png")
print("  3. perplexity_curve.png")
print("  4. combined_loss_plot.png")
print("  5. training_curves_all.png")

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)
