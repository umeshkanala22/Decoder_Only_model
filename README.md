# Decoder-Only Transformer for TinyStories

Implementation of a decoder-only transformer language model from scratch for the TinyStories dataset, as part of the AIL861 assignment.

## 📋 Overview

This project implements a complete transformer-based language model with:
- Custom implementation of all core components (LayerNorm, Multi-head Attention, Positional Encoding)
- Training on the TinyStories dataset
- FastText embeddings integration
- Advanced inference techniques (Beam Search, KV Caching)
- Gradient accumulation for memory-efficient training

## 🏗️ Model Architecture

- **Type**: Decoder-Only Transformer
- **Parameters**: 17.4 Million
- **Layers**: 5
- **Attention Heads**: 6 per layer
- **Embedding Dimension**: 300 (FastText)
- **Feed-forward Dimension**: 1200
- **Vocabulary Size**: 10,004 words (10,000 + 4 special tokens)
- **Max Sequence Length**: 64 tokens
- **Context Window**: 64 tokens

### Special Tokens
- `<pad>` - Padding token (index: 0)
- `<sos>` - Start of sequence (index: 1)
- `<eos>` - End of sequence (index: 2)
- `<unk>` - Unknown word (index: 3)

## 📊 Training Results (4 Epochs)

| Epoch | Training Loss | Validation Loss | Perplexity |
|-------|--------------|-----------------|------------|
| 1     | 3.2612       | 2.5748         | 13.13      |
| 2     | 2.5221       | 2.2983         | 9.96       |
| 3     | 2.3232       | 2.1758         | 8.81       |
| 4     | 2.2134       | 2.1014         | 8.18       |

**Final Metrics:**
- Training Loss: 2.21 (32% improvement)
- Validation Loss: 2.10 (18% improvement)
- Perplexity: 8.18 (38% improvement)

## 🚀 Quick Start

### Requirements
```bash
pip install torch numpy pandas matplotlib seaborn tqdm datasets gensim evaluate
```

### Training
```bash
# Train with default configuration
python main.py --mode train

# Train with custom config
python main.py --mode train --config full
```

### Inference
```bash
# Generate text from a prompt
python main.py --mode generate --checkpoint checkpoints/best_model.pt --prompt "once upon a time"

# Evaluate on validation set
python main.py --mode eval --checkpoint checkpoints/best_model.pt
```

### Experiments
```bash
# Run gradient accumulation experiments
python gradient_accumulation_experiment.py

# Extract and plot training metrics
python extract_and_plot_metrics.py
```

## 📁 Project Structure

```
.
├── config.py                              # Configuration and hyperparameters
├── model.py                               # Transformer model implementation
├── train.py                               # Training loop and utilities
├── inference.py                           # Text generation (greedy, beam search, KV cache)
├── evaluation.py                          # Evaluation metrics (perplexity, BLEU)
├── visualize.py                           # Plotting and visualization
├── data_utils.py                          # Data loading and preprocessing
├── utils.py                               # Helper functions
├── main.py                                # Main entry point
├── extract_and_plot_metrics.py           # Extract metrics from checkpoints
├── gradient_accumulation_experiment.py    # Gradient accumulation experiments
│
├── checkpoints/
│   └── best_model.pt                      # Best trained model (epoch 3)
│
├── data/
│   ├── vocab.pkl                          # Vocabulary mappings
│   └── embedding_matrix.npy               # FastText embeddings (not in repo)
│
└── results/
    ├── training_loss_curve.png            # Training loss plot
    ├── validation_loss_curve.png          # Validation loss plot
    ├── perplexity_curve.png               # Perplexity plot
    ├── combined_loss_plot.png             # Combined train/val loss
    ├── training_curves_all.png            # All metrics
    ├── training_summary.txt               # Training summary
    └── metrics_table.md                   # Metrics table
```

## 🔧 Implementation Details

### Core Components (All Implemented from Scratch)

1. **LayerNorm** (`model.py:50-63`)
   - Custom normalization layer
   - Learnable scale (gamma) and shift (beta) parameters

2. **Multi-Head Self-Attention** (`model.py:66-150`)
   - Masked causal attention
   - Parallelized head computation
   - Optional KV caching for faster inference

3. **Sinusoidal Positional Encoding** (`model.py:31-41`)
   - As per Vaswani et al. (2017)
   - Fixed positional embeddings

4. **Feed-Forward Network** (`model.py:152-167`)
   - Two-layer MLP with GELU activation
   - Dropout for regularization

### Training Features

- **Teacher Forcing**: Standard autoregressive training
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: ReduceLROnPlateau scheduler
- **Gradient Accumulation**: Simulate larger batch sizes
- **Early Stopping**: Prevent overfitting
- **Checkpointing**: Save best model and epoch checkpoints

### Inference Techniques

1. **Greedy Decoding**: Fast, deterministic generation
2. **Sampling with Temperature**: Controlled randomness
3. **Top-k Sampling**: Sample from top k tokens
4. **Top-p (Nucleus) Sampling**: Sample from cumulative probability mass
5. **Beam Search**: Explore multiple hypotheses
6. **KV Caching**: Reuse past key-value pairs for faster generation

## 📈 Visualization

All training metrics are automatically saved as high-resolution plots:

- Training and validation loss curves
- Perplexity progression
- Gradient accumulation comparison
- Attention weight visualizations (when applicable)

## ⚙️ Configuration

Edit `config.py` to modify:
- Model architecture (layers, heads, dimensions)
- Training hyperparameters (learning rate, batch size, epochs)
- Data configuration (vocab size, sequence length)
- Inference settings (temperature, beam width, top-k/p)

### Pre-configured Settings

- **Default**: Balanced configuration for training
- **QuickTest**: Fast testing with smaller data
- **FullScale**: Best performance with full dataset

## 🎯 Assignment Requirements

### Part 1: Implementation (35 points)

✅ **1.1 Pre-Training Dataset**
- TinyStories dataset from HuggingFace
- 2.1M training samples, 21,990 validation samples

✅ **1.2 Model Architecture**
- All components implemented from scratch
- FastText embeddings (300-dim)
- Sinusoidal positional encoding
- Special token handling

✅ **1.3 Training (20 points)**
- 4 epochs completed
- Training/validation loss curves
- Perplexity tracking
- Efficient parallel training

✅ **1.4 Inference (15 points)**
- Auto-regressive generation
- Stochastic sampling
- Sample generations with metrics

### Part 2: Enhancements

✅ **2.1 Beam Search Decoding**
- Multiple beam widths (1, 5, 10)
- Length penalty
- Batch processing

✅ **2.2 KV Caching**
- Faster inference
- Reuses computed key-value pairs
- Significant speedup

✅ **2.3 Gradient Accumulation**
- Simulate larger batch sizes
- Memory-efficient training
- Experiments with steps: 1, 2, 4, 8

✅ **2.4 Gradient Checkpointing**
- Memory-efficient backpropagation
- Trade compute for memory

## 📝 Citation

Dataset:
```bibtex
@article{eldan2023tinystories,
  title={TinyStories: How Small Can Language Models Be and Still Speak Coherent English?},
  author={Eldan, Ronen and Li, Yuanzhi},
  journal={arXiv preprint arXiv:2305.07759},
  year={2023}
}
```

Transformer Architecture:
```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 📄 License

This project is for educational purposes as part of AIL861 coursework.

---

**Note**: Large files (checkpoints, embeddings) are excluded from the repository. See `.gitignore` for details.
