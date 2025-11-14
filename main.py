"""
Main Entry Point
Orchestrates training, evaluation, and inference
"""

import argparse
import torch
import numpy as np

from config import Config, QuickTestConfig, FullScaleConfig
from utils import set_seed, get_device, print_model_summary, print_memory_usage
from data_utils import (
    load_tinystories_dataset, build_vocab, load_fasttext_embeddings,
    create_embedding_matrix, create_dataloaders,
    save_vocabulary, save_embedding_matrix
)
from model import DecoderOnlyTransformer
from train import train_transformer
from evaluation import (
    evaluate_perplexity, benchmark_beam_search, benchmark_kv_cache,
    benchmark_gradient_accumulation
)
from visualize import plot_training_curves, plot_combined_loss
from inference import generate_samples


def setup_environment(config):
    """
    Setup environment: seed, device, etc.

    Args:
        config: Configuration object

    Returns:
        device: Device to use
    """
    print("=" * 60)
    print("ENVIRONMENT SETUP")
    print("=" * 60)


    set_seed(config.SEED)


    device = get_device(prefer_cuda=(config.DEVICE == 'cuda'))

    return device


def prepare_data(config):
    """
    Prepare data: load dataset, build vocabulary, create embeddings

    Args:
        config: Configuration object

    Returns:
        word2idx: Word to index mapping
        idx2word: Index to word mapping
        embedding_matrix: Pretrained embedding matrix
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)

    # Load TinyStories dataset
    print("\n1. Loading TinyStories dataset...")
    train_texts, val_texts = load_tinystories_dataset(
        num_train_samples=config.NUM_TRAIN_SAMPLES,
        num_val_samples=config.NUM_VAL_SAMPLES
    )

    # Check if vocabulary and embeddings are already cached
    import os
    vocab_exists = os.path.exists(config.VOCAB_PATH)
    embeddings_exist = os.path.exists(config.EMBEDDING_MATRIX_PATH)

    if vocab_exists and embeddings_exist:
        print("\n2. Loading cached vocabulary and embeddings...")
        from data_utils import load_vocabulary, load_embedding_matrix
        word2idx, idx2word = load_vocabulary(config.VOCAB_PATH)
        embedding_matrix = load_embedding_matrix(config.EMBEDDING_MATRIX_PATH) if config.USE_PRETRAINED_EMBEDDINGS else None
        print(f" Loaded from cache: {len(word2idx)} vocab size")
    else:
        # Build vocabulary
        print("\n2. Building vocabulary...")
        word2idx, idx2word, word_counter = build_vocab(
            train_texts,
            min_freq=config.MIN_FREQ,
            max_vocab_size=config.VOCAB_SIZE
        )

        # Save vocabulary
        save_vocabulary(word2idx, idx2word, config.VOCAB_PATH)

        # Load FastText embeddings (if using pretrained)
        embedding_matrix = None
        if config.USE_PRETRAINED_EMBEDDINGS:
            print("\n3. Loading FastText embeddings...")
            try:
                embeddings_dict, fasttext_model = load_fasttext_embeddings(
                    word2idx,
                    embedding_dim=config.EMBEDDING_DIM,
                    kaggle_mode=config.KAGGLE_MODE
                )


                print("\n4. Creating embedding matrix...")
                embedding_matrix = create_embedding_matrix(
                    word2idx, embeddings_dict, config.EMBEDDING_DIM
                )

                # Save embedding matrix
                save_embedding_matrix(embedding_matrix, config.EMBEDDING_MATRIX_PATH)

                # Clean up to save memory
                import gc
                del embeddings_dict, fasttext_model
                gc.collect()

            except Exception as e:
                print(f"Warning: Failed to load FastText embeddings: {e}")
                print("Continuing with random embeddings...")
                embedding_matrix = None


    print("\n3. Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_texts, val_texts, word2idx,
        batch_size=config.BATCH_SIZE,
        max_seq_len=config.MAX_SEQ_LEN,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    return word2idx, idx2word, embedding_matrix, train_loader, val_loader


def create_model(config, vocab_size, embedding_matrix=None):
    """
    Create the transformer model

    Args:
        config: Configuration object
        vocab_size: Vocabulary size
        embedding_matrix: Pretrained embedding matrix (optional)

    Returns:
        model: Decoder-only transformer model
    """
    print("\n" + "=" * 60)
    print("MODEL CREATION")
    print("=" * 60)

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        d_ff=config.D_FF,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout=config.DROPOUT,
        pad_idx=config.PAD_IDX,
        pretrained_embeddings=embedding_matrix,
        use_checkpoint=config.USE_GRADIENT_CHECKPOINTING
    )

    # Print model summary
    print_model_summary(model)

    return model


def run_training(model, train_loader, val_loader, config, device):
    """
    Run training

    Args:
        model: Transformer model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        config: Configuration object
        device: Device to use

    Returns:
        train_losses: List of training losses
        val_losses: List of validation losses
        perplexities: List of perplexities
    """
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    train_losses, val_losses, perplexities = train_transformer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        device=device,
        save_dir=config.CHECKPOINT_DIR,
        accum_steps=config.GRADIENT_ACCUMULATION_STEPS,
        use_scheduler=config.USE_LR_SCHEDULER
    )

    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, perplexities,
        save_path=config.TRAINING_CURVES_PATH
    )
    plot_combined_loss(
        train_losses, val_losses,
        save_path=config.COMBINED_LOSS_PATH
    )

    return train_losses, val_losses, perplexities


def run_evaluation(model, val_loader, word2idx, idx2word, config, device):
    """
    Run evaluation and benchmarks

    Args:
        model: Transformer model
        val_loader: Validation dataloader
        word2idx: Word to index mapping
        idx2word: Index to word mapping
        config: Configuration object
        device: Device to use
    """
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    # Evaluate perplexity
    print("\n1. Evaluating perplexity...")
    perplexity, avg_loss = evaluate_perplexity(model, val_loader, device)
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Average Loss: {avg_loss:.4f}")

    # Generate sample texts
    print("\n2. Generating sample texts...")
    prompts = [
        "once upon a time",
        "there was a little",
        "the cat and the"
    ]

    generated_texts = generate_samples(
        model, prompts, word2idx, idx2word,
        method='greedy',
        max_new_tokens=50,
        device=device
    )

    print("\nGenerated Samples:")
    for prompt, text in zip(prompts, generated_texts):
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {text}")


def run_part2_experiments(model, train_loader, val_loader, word2idx, idx2word, config, device):
    """
    Run Part 2 enhancement experiments

    Args:
        model: Transformer model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        word2idx: Word to index mapping
        idx2word: Index to word mapping
        config: Configuration object
        device: Device to use
    """
    print("\n" + "=" * 60)
    print("PART 2: ENHANCEMENT EXPERIMENTS")
    print("=" * 60)

    # Part 2.1: Beam Search
    print("\n" + "-" * 60)
    print("Part 2.1: Beam Search Experiments")
    print("-" * 60)

    prompts = ["once upon a time"] * config.NUM_SAMPLES_BEAM
    beam_results = benchmark_beam_search(
        model, prompts, word2idx, idx2word,
        beam_widths=config.BEAM_WIDTHS,
        max_new_tokens=50,
        device=device
    )

    # Part 2.2: KV Caching
    print("\n" + "-" * 60)
    print("Part 2.2: KV Caching Benchmark")
    print("-" * 60)

    prompts_kv = ["there was a little"] * config.NUM_SAMPLES_KV
    kv_results = benchmark_kv_cache(
        model, prompts_kv, word2idx, idx2word,
        num_samples=config.NUM_SAMPLES_KV,
        max_new_tokens=50,
        device=device
    )

    # Part 2.3: Gradient Accumulation
    print("\n" + "-" * 60)
    print("Part 2.3: Gradient Accumulation Benchmark")
    print("-" * 60)

    accum_results = benchmark_gradient_accumulation(
        model, train_loader,
        accum_steps_list=config.ACCUM_STEPS_LIST,
        num_batches=10,
        device=device
    )

    print("\n" + "=" * 60)
    print("PART 2 EXPERIMENTS COMPLETE")
    print("=" * 60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Decoder-Only Transformer for TinyStories')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'generate', 'experiments', 'full'],
                       help='Mode: train, eval, generate, experiments, or full')
    parser.add_argument('--config', type=str, default='default',
                       choices=['default', 'quick', 'full'],
                       help='Configuration: default, quick (test), or full (best)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation/generation')
    parser.add_argument('--prompt', type=str, default="once upon a time",
                       help='Prompt for text generation')

    args = parser.parse_args()

    # Select configuration
    if args.config == 'quick':
        config = QuickTestConfig()
        print("Using Quick Test Configuration")
    elif args.config == 'full':
        config = FullScaleConfig()
        print("Using Full Scale Configuration")
    else:
        config = Config()
        print("Using Default Configuration")

    # Display configuration
    config.display()

    # Setup environment
    device = setup_environment(config)

    # Prepare data
    word2idx, idx2word, embedding_matrix, train_loader, val_loader = prepare_data(config)

    # Create model
    model = create_model(config, len(word2idx), embedding_matrix)
    model = model.to(device)

    # Run based on mode
    if args.mode in ['train', 'full']:
        # Training
        train_losses, val_losses, perplexities = run_training(
            model, train_loader, val_loader, config, device
        )

    if args.mode in ['eval', 'full']:
        # Evaluation
        if args.checkpoint:
            from utils import load_checkpoint
            load_checkpoint(model, args.checkpoint, device=device)

        run_evaluation(model, val_loader, word2idx, idx2word, config, device)

    if args.mode == 'generate':
        # Text generation
        if args.checkpoint:
            from utils import load_checkpoint
            load_checkpoint(model, args.checkpoint, device=device)

        generated_texts = generate_samples(
            model, [args.prompt], word2idx, idx2word,
            method='beam_search',
            beam_width=5,
            max_new_tokens=100,
            device=device
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {generated_texts[0]}")

    if args.mode in ['experiments', 'full']:
        # Part 2 experiments
        if args.checkpoint:
            from utils import load_checkpoint
            load_checkpoint(model, args.checkpoint, device=device)

        run_part2_experiments(model, train_loader, val_loader, word2idx, idx2word, config, device)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == '__main__':
    main()
