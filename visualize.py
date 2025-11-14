"""
Visualization Module
Attention heatmaps, training curves, and comparison plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd

from config import Config


def visualize_attention(attn_weights, tokens=None, heads_to_show=4, save_path=None):

    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.cpu().detach().numpy()

    num_heads = min(heads_to_show, attn_weights.shape[0])
    seq_len = attn_weights.shape[1]

    fig, axes = plt.subplots(1, num_heads, figsize=(5*num_heads, 5))

    if num_heads == 1:
        axes = [axes]

    for head_idx in range(num_heads):
        ax = axes[head_idx]

        # Plot heatmap
        sns.heatmap(
            attn_weights[head_idx],
            cmap='viridis',
            ax=ax,
            cbar=True,
            square=True,
            vmin=0,
            vmax=1
        )

        ax.set_title(f'Head {head_idx + 1}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Key Position', fontsize=12)
        ax.set_ylabel('Query Position', fontsize=12)

        if tokens is not None:
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens, rotation=0)

    plt.suptitle('Multi-Head Attention Weights', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")

    plt.show()


def plot_training_curves(train_losses, val_losses, perplexities, save_path=None):
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(train_losses) + 1)

    # Plot 1: Training Loss
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2, marker='o')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)

    # Plot 2: Validation Loss
    axes[1].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)

    # Plot 3: Perplexity
    axes[2].plot(epochs, perplexities, 'g-', label='Perplexity', linewidth=2, marker='^')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Perplexity', fontsize=12)
    axes[2].set_title('Perplexity', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=11)

    plt.suptitle('Training Progress', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    plt.show()


def plot_combined_loss(train_losses, val_losses, save_path=None):

    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined loss plot saved to {save_path}")

    plt.show()


def plot_beam_search_comparison(beam_widths, generation_times, save_path=None):
 
    plt.figure(figsize=(10, 6))

    plt.plot(beam_widths, generation_times, 'o-', linewidth=2, markersize=10, color='steelblue')

    plt.xlabel('Beam Width', fontsize=14)
    plt.ylabel('Average Generation Time (s)', fontsize=14)
    plt.title('Beam Search: Generation Time vs Beam Width', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)

    for bw, time in zip(beam_widths, generation_times):
        plt.annotate(f'{time:.3f}s', (bw, time), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Beam search comparison saved to {save_path}")

    plt.show()


def plot_kv_cache_comparison(time_without, time_with, speedup, save_path=None):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot for times
    methods = ['Without Cache', 'With Cache']
    times = [time_without, time_with]
    colors = ['coral', 'lightgreen']

    bars = ax1.bar(methods, times, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Generation Time (s)', fontsize=14)
    ax1.set_title('KV Cache: Generation Time Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Speedup visualization
    ax2.bar(['Speedup'], [speedup], color='gold', edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Speedup Factor', fontsize=14)
    ax2.set_title('KV Cache Speedup', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.text(0, speedup, f'{speedup:.2f}x', ha='center', va='bottom',
            fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"KV cache comparison saved to {save_path}")

    plt.show()


def plot_gradient_accumulation_comparison(accum_steps, times, memory_usage, save_path=None):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time plot
    ax1.plot(accum_steps, times, 'o-', linewidth=2, markersize=10, color='steelblue')
    ax1.set_xlabel('Accumulation Steps', fontsize=14)
    ax1.set_ylabel('Time (s)', fontsize=14)
    ax1.set_title('Gradient Accumulation: Execution Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(accum_steps)

    # Memory plot
    ax2.plot(accum_steps, memory_usage, 'o-', linewidth=2, markersize=10, color='coral')
    ax2.set_xlabel('Accumulation Steps', fontsize=14)
    ax2.set_ylabel('Peak GPU Memory (MB)', fontsize=14)
    ax2.set_title('Gradient Accumulation: Memory Usage', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(accum_steps)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gradient accumulation comparison saved to {save_path}")

    plt.show()


def plot_gradient_checkpointing_comparison(results, save_path=None):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Time comparison
    methods = ['Without CP', 'With CP']
    times = [
        results['without_checkpointing']['time'],
        results['with_checkpointing']['time']
    ]
    colors = ['coral', 'lightblue']

    bars1 = ax1.bar(methods, times, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Time (s)', fontsize=14)
    ax1.set_title('Gradient Checkpointing: Time Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for bar, time in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.3f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Memory comparison
    memory = [
        results['without_checkpointing']['memory'],
        results['with_checkpointing']['memory']
    ]

    bars2 = ax2.bar(methods, memory, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Peak GPU Memory (MB)', fontsize=14)
    ax2.set_title('Gradient Checkpointing: Memory Comparison', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    for bar, mem in zip(bars2, memory):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f} MB', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gradient checkpointing comparison saved to {save_path}")

    plt.show()


def plot_learning_rate_schedule(optimizer_history, save_path=None):


    plt.figure(figsize=(10, 6))

    epochs = range(1, len(optimizer_history) + 1)
    plt.plot(epochs, optimizer_history, 'o-', linewidth=2, markersize=8, color='purple')

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate schedule saved to {save_path}")

    plt.show()


def create_comparison_table(results, save_path=None):


    df = pd.DataFrame(results)

    print("\nExperiment Results:")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Results table saved to {save_path}")

    return df



