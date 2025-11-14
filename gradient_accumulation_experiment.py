"""
Gradient Accumulation Experiments (Part 2.3)
Compare training with different gradient accumulation steps
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import pickle
from tqdm import tqdm
import pandas as pd

from config import Config
from model import DecoderOnlyTransformer
from data_utils import create_dataloaders, load_tinystories_dataset
from train import train_epoch, validate
from utils import set_seed

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

print("=" * 70)
print("GRADIENT ACCUMULATION EXPERIMENTS")
print("=" * 70)

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_BATCH_SIZE = 16  # Fixed mini-batch size
ACCUMULATION_STEPS = [1, 2, 4, 8]  # Effective batch sizes: 16, 32, 64, 128
NUM_EPOCHS = 2  # Number of epochs for each experiment
NUM_TRAIN_SAMPLES = 50000  # Subset for faster experiments
NUM_VAL_SAMPLES = 5000

print(f"\nDevice: {DEVICE}")
print(f"Base Batch Size: {BASE_BATCH_SIZE}")
print(f"Accumulation Steps to test: {ACCUMULATION_STEPS}")
print(f"Effective Batch Sizes: {[BASE_BATCH_SIZE * acc for acc in ACCUMULATION_STEPS]}")
print(f"Epochs per experiment: {NUM_EPOCHS}")
print(f"Training samples: {NUM_TRAIN_SAMPLES:,}")

# Set seed for reproducibility
set_seed(Config.SEED)

# Create results directory
results_dir = 'results/gradient_accumulation'
os.makedirs(results_dir, exist_ok=True)

# ============================================
# LOAD DATA
# ============================================
print("\n" + "=" * 70)
print("LOADING DATA")
print("=" * 70)

# Load vocabulary
vocab_path = Config.VOCAB_PATH
with open(vocab_path, 'rb') as f:
    vocab_data = pickle.load(f)
    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']

print(f"Vocabulary size: {len(word2idx)}")

# Load dataset
print("\nLoading TinyStories dataset...")
train_texts, val_texts = load_tinystories_dataset(
    num_train_samples=NUM_TRAIN_SAMPLES,
    num_val_samples=NUM_VAL_SAMPLES
)

# Load embedding matrix
embedding_matrix = None
if Config.USE_PRETRAINED_EMBEDDINGS and os.path.exists(Config.EMBEDDING_MATRIX_PATH):
    embedding_matrix = np.load(Config.EMBEDDING_MATRIX_PATH)
    print(f"Loaded embedding matrix: {embedding_matrix.shape}")

# ============================================
# EXPERIMENT LOOP
# ============================================
results = {
    'accumulation_steps': [],
    'effective_batch_size': [],
    'train_losses': [],
    'val_losses': [],
    'perplexities': [],
    'epoch_times': [],
    'total_time': []
}

print("\n" + "=" * 70)
print("RUNNING EXPERIMENTS")
print("=" * 70)

for accum_steps in ACCUMULATION_STEPS:
    effective_batch_size = BASE_BATCH_SIZE * accum_steps

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: Accumulation Steps = {accum_steps}")
    print(f"Effective Batch Size = {effective_batch_size}")
    print(f"{'=' * 70}")

    train_loader, val_loader = create_dataloaders(
        train_texts, val_texts, word2idx,
        batch_size=BASE_BATCH_SIZE,
        max_seq_len=Config.MAX_SEQ_LEN,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    model = DecoderOnlyTransformer(
        vocab_size=len(word2idx),
        d_model=Config.D_MODEL,
        num_heads=Config.NUM_HEADS,
        num_layers=Config.NUM_LAYERS,
        d_ff=Config.D_FF,
        max_seq_len=Config.MAX_SEQ_LEN,
        dropout=Config.DROPOUT,
        pad_idx=Config.PAD_IDX,
        pretrained_embeddings=embedding_matrix,
        use_checkpoint=False
    )
    model = model.to(DEVICE)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Training
    train_losses = []
    val_losses = []
    perplexities = []
    epoch_times = []

    experiment_start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, DEVICE,
            clip_grad_norm=Config.CLIP_GRAD_NORM,
            accum_steps=accum_steps
        )

        # Validate
        val_loss, perplexity = validate(model, val_loader, DEVICE)

        epoch_time = time.time() - epoch_start_time

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        perplexities.append(perplexity)
        epoch_times.append(epoch_time)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Epoch Time: {epoch_time:.2f}s")

    total_time = time.time() - experiment_start_time
    avg_epoch_time = np.mean(epoch_times)

    print(f"\nTotal Time: {total_time:.2f}s")
    print(f"Average Time per Epoch: {avg_epoch_time:.2f}s")

    # Store results
    results['accumulation_steps'].append(accum_steps)
    results['effective_batch_size'].append(effective_batch_size)
    results['train_losses'].append(train_losses)
    results['val_losses'].append(val_losses)
    results['perplexities'].append(perplexities)
    results['epoch_times'].append(avg_epoch_time)
    results['total_time'].append(total_time)

    # Clean up
    del model, optimizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ============================================
# SAVE RESULTS
# ============================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Save raw results
results_file = os.path.join(results_dir, 'experiment_results.pkl')
with open(results_file, 'wb') as f:
    pickle.dump(results, f)
print(f"Saved results to: {results_file}")

# ============================================
# CREATE PLOTS
# ============================================
print("\n" + "=" * 70)
print("CREATING PLOTS")
print("=" * 70)

# Plot 1: Training Loss Curves Comparison
print("\n1. Training Loss Curves Comparison...")
plt.figure(figsize=(12, 7))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

for idx, (accum_steps, train_losses) in enumerate(zip(results['accumulation_steps'], results['train_losses'])):
    effective_bs = results['effective_batch_size'][idx]
    epochs = range(1, len(train_losses) + 1)

    label = f'Accum={accum_steps}, Eff.BS={effective_bs}'
    plt.plot(epochs, train_losses,
             marker=markers[idx],
             linewidth=2.5,
             markersize=10,
             label=label,
             color=colors[idx],
             alpha=0.8)

plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Training Loss', fontsize=14, fontweight='bold')
plt.title('Training Loss: Gradient Accumulation Comparison', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='best', framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

save_path = os.path.join(results_dir, 'training_loss_comparison.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"   Saved to: {save_path}")
plt.close()

# Plot 2: Validation Loss Curves Comparison
print("\n2. Validation Loss Curves Comparison...")
plt.figure(figsize=(12, 7))

for idx, (accum_steps, val_losses) in enumerate(zip(results['accumulation_steps'], results['val_losses'])):
    effective_bs = results['effective_batch_size'][idx]
    epochs = range(1, len(val_losses) + 1)

    label = f'Accum={accum_steps}, Eff.BS={effective_bs}'
    plt.plot(epochs, val_losses,
             marker=markers[idx],
             linewidth=2.5,
             markersize=10,
             label=label,
             color=colors[idx],
             alpha=0.8)

plt.xlabel('Epoch', fontsize=14, fontweight='bold')
plt.ylabel('Validation Loss', fontsize=14, fontweight='bold')
plt.title('Validation Loss: Gradient Accumulation Comparison', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='best', framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

save_path = os.path.join(results_dir, 'validation_loss_comparison.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"   Saved to: {save_path}")
plt.close()

# Plot 3: Runtime Comparison
print("\n3. Runtime per Epoch Comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar plot
bars = ax1.bar(range(len(results['accumulation_steps'])),
               results['epoch_times'],
               color=colors,
               edgecolor='black',
               linewidth=1.5,
               alpha=0.8)

ax1.set_xlabel('Configuration', fontsize=13, fontweight='bold')
ax1.set_ylabel('Time per Epoch (seconds)', fontsize=13, fontweight='bold')
ax1.set_title('Runtime per Epoch', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(results['accumulation_steps'])))
ax1.set_xticklabels([f'Accum={a}\nBS={b}' for a, b in
                      zip(results['accumulation_steps'], results['effective_batch_size'])])
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar, time_val in zip(bars, results['epoch_times']):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{time_val:.1f}s',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Line plot
ax2.plot(results['effective_batch_size'],
         results['epoch_times'],
         marker='o',
         linewidth=2.5,
         markersize=12,
         color='steelblue',
         alpha=0.8)

ax2.set_xlabel('Effective Batch Size', fontsize=13, fontweight='bold')
ax2.set_ylabel('Time per Epoch (seconds)', fontsize=13, fontweight='bold')
ax2.set_title('Runtime vs Effective Batch Size', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Annotate points
for bs, time_val in zip(results['effective_batch_size'], results['epoch_times']):
    ax2.annotate(f'{time_val:.1f}s', (bs, time_val),
                textcoords="offset points", xytext=(0,10),
                ha='center', fontsize=10)

plt.tight_layout()
save_path = os.path.join(results_dir, 'runtime_comparison.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"   Saved to: {save_path}")
plt.close()

# Plot 4: Comprehensive Comparison
print("\n4. Comprehensive Comparison (All Metrics)...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Training Loss
for idx, (accum_steps, train_losses) in enumerate(zip(results['accumulation_steps'], results['train_losses'])):
    epochs = range(1, len(train_losses) + 1)
    axes[0, 0].plot(epochs, train_losses, marker=markers[idx], linewidth=2,
                    markersize=8, label=f'Accum={accum_steps}', color=colors[idx])
axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Training Loss', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Training Loss', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Validation Loss
for idx, (accum_steps, val_losses) in enumerate(zip(results['accumulation_steps'], results['val_losses'])):
    epochs = range(1, len(val_losses) + 1)
    axes[0, 1].plot(epochs, val_losses, marker=markers[idx], linewidth=2,
                    markersize=8, label=f'Accum={accum_steps}', color=colors[idx])
axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Validation Loss', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Perplexity
for idx, (accum_steps, perps) in enumerate(zip(results['accumulation_steps'], results['perplexities'])):
    epochs = range(1, len(perps) + 1)
    axes[1, 0].plot(epochs, perps, marker=markers[idx], linewidth=2,
                    markersize=8, label=f'Accum={accum_steps}', color=colors[idx])
axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Perplexity', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Perplexity', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Runtime
axes[1, 1].bar(range(len(results['accumulation_steps'])), results['epoch_times'],
               color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
axes[1, 1].set_xlabel('Accumulation Steps', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Time per Epoch (s)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Runtime per Epoch', fontsize=13, fontweight='bold')
axes[1, 1].set_xticks(range(len(results['accumulation_steps'])))
axes[1, 1].set_xticklabels([f'{a}' for a in results['accumulation_steps']])
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('Gradient Accumulation: Comprehensive Comparison',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
save_path = os.path.join(results_dir, 'comprehensive_comparison.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"   Saved to: {save_path}")
plt.close()

# ============================================
# CREATE TABLES
# ============================================
print("\n" + "=" * 70)
print("CREATING TABLES")
print("=" * 70)

# Table 1: Runtime Comparison
print("\n1. Runtime Comparison Table...")
runtime_data = []
for accum, eff_bs, epoch_time, total_time in zip(
    results['accumulation_steps'],
    results['effective_batch_size'],
    results['epoch_times'],
    results['total_time']
):
    runtime_data.append({
        'Accumulation Steps': accum,
        'Effective Batch Size': eff_bs,
        'Avg Time per Epoch (s)': f'{epoch_time:.2f}',
        'Total Time (s)': f'{total_time:.2f}',
        'Speedup vs Baseline': f'{results["epoch_times"][0] / epoch_time:.2f}x'
    })

runtime_df = pd.DataFrame(runtime_data)
print("\n" + "=" * 70)
print("RUNTIME COMPARISON TABLE")
print("=" * 70)
print(runtime_df.to_string(index=False))

runtime_table_path = os.path.join(results_dir, 'runtime_table.csv')
runtime_df.to_csv(runtime_table_path, index=False)
print(f"\nSaved to: {runtime_table_path}")

# Table 2: Final Metrics Comparison
print("\n2. Final Metrics Comparison Table...")
metrics_data = []
for idx, accum in enumerate(results['accumulation_steps']):
    metrics_data.append({
        'Accumulation Steps': accum,
        'Effective Batch Size': results['effective_batch_size'][idx],
        'Final Train Loss': f'{results["train_losses"][idx][-1]:.4f}',
        'Final Val Loss': f'{results["val_losses"][idx][-1]:.4f}',
        'Final Perplexity': f'{results["perplexities"][idx][-1]:.2f}',
        'Avg Time/Epoch (s)': f'{results["epoch_times"][idx]:.2f}'
    })

metrics_df = pd.DataFrame(metrics_data)
print("\n" + "=" * 70)
print("FINAL METRICS COMPARISON TABLE")
print("=" * 70)
print(metrics_df.to_string(index=False))

metrics_table_path = os.path.join(results_dir, 'metrics_table.csv')
metrics_df.to_csv(metrics_table_path, index=False)
print(f"\nSaved to: {metrics_table_path}")

# ============================================
# CREATE SUMMARY REPORT
# ============================================
print("\n" + "=" * 70)
print("CREATING SUMMARY REPORT")
print("=" * 70)

summary_report = f"""{'=' * 70}
GRADIENT ACCUMULATION EXPERIMENTS - SUMMARY REPORT
{'=' * 70}

CONFIGURATION
{'=' * 70}
Base Batch Size: {BASE_BATCH_SIZE}
Accumulation Steps Tested: {ACCUMULATION_STEPS}
Effective Batch Sizes: {[BASE_BATCH_SIZE * acc for acc in ACCUMULATION_STEPS]}
Epochs per Experiment: {NUM_EPOCHS}
Training Samples: {NUM_TRAIN_SAMPLES:,}
Validation Samples: {NUM_VAL_SAMPLES:,}
Device: {DEVICE}

RUNTIME COMPARISON
{'=' * 70}
{runtime_df.to_string(index=False)}

FINAL METRICS
{'=' * 70}
{metrics_df.to_string(index=False)}

KEY FINDINGS
{'=' * 70}
1. Baseline (Accum=1, BS=16):
   - Final Train Loss: {results['train_losses'][0][-1]:.4f}
   - Final Val Loss: {results['val_losses'][0][-1]:.4f}
   - Time per Epoch: {results['epoch_times'][0]:.2f}s

2. Best Configuration (based on validation loss):
"""

# Find best configuration
best_idx = np.argmin([val_losses[-1] for val_losses in results['val_losses']])
summary_report += f"""   - Accumulation Steps: {results['accumulation_steps'][best_idx]}
   - Effective Batch Size: {results['effective_batch_size'][best_idx]}
   - Final Val Loss: {results['val_losses'][best_idx][-1]:.4f}
   - Time per Epoch: {results['epoch_times'][best_idx]:.2f}s

3. Runtime Analysis:
   - Fastest: Accum={results['accumulation_steps'][0]} ({results['epoch_times'][0]:.2f}s/epoch)
   - Slowest: Accum={results['accumulation_steps'][-1]} ({results['epoch_times'][-1]:.2f}s/epoch)
   - Time increase from baseline: {(results['epoch_times'][-1] / results['epoch_times'][0] - 1) * 100:.1f}%

PLOTS GENERATED
{'=' * 70}
1. training_loss_comparison.png    - Training loss curves
2. validation_loss_comparison.png  - Validation loss curves
3. runtime_comparison.png          - Runtime analysis
4. comprehensive_comparison.png    - All metrics together

TABLES GENERATED
{'=' * 70}
1. runtime_table.csv              - Runtime comparison
2. metrics_table.csv              - Final metrics comparison

{'=' * 70}
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 70}
"""

summary_path = os.path.join(results_dir, 'summary_report.txt')
with open(summary_path, 'w') as f:
    f.write(summary_report)

print(summary_report)
print(f"\nSummary saved to: {summary_path}")

print("\n" + "=" * 70)
print("EXPERIMENTS COMPLETE!")
print("=" * 70)
print(f"\nAll results saved to: {results_dir}")
