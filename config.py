"""
Configuration file for Decoder-Only Transformer
Centralizes all hyperparameters and settings
"""

import torch
import os


class Config:
    """Configuration class with all hyperparameters"""

    # ============================================
    # DEVICE CONFIGURATION
    # ============================================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ============================================
    # DATA CONFIGURATION
    # ============================================
    # Choose your scale: Quick test: 10k, Medium: 100k, Full: entire dataset
    # Using balanced 90/10 split with max available validation data
    NUM_TRAIN_SAMPLES = 197910  # Number of training samples to use (90% of balanced split)
    NUM_VAL_SAMPLES = 21990     # Number of validation samples to use (all available - 10% of balanced split)

    # ============================================
    # VOCABULARY CONFIGURATION
    # ============================================
    VOCAB_SIZE = 10000          # Maximum vocabulary size (optimized for fastest training)
    MIN_FREQ = 1                # Minimum word frequency to include in vocab

    # Special tokens
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'

    PAD_IDX = 0
    SOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    # ============================================
    # MODEL ARCHITECTURE
    # ============================================
    D_MODEL = 300               # Embedding dimension (matches FastText)
    NUM_HEADS = 6               # Number of attention heads (must divide D_MODEL)
    NUM_LAYERS = 5              # Number of transformer blocks
    D_FF = 1200                 # Feed-forward dimension (typically 4 * D_MODEL)
    MAX_SEQ_LEN = 64            # Maximum sequence length
    DROPOUT = 0.1               # Dropout probability

    # Embedding configuration
    EMBEDDING_DIM = 300         # FastText embedding dimension
    USE_PRETRAINED_EMBEDDINGS = True  # Whether to use FastText embeddings
    FREEZE_EMBEDDINGS = False   # Whether to freeze embedding weights

    # ============================================
    # TRAINING CONFIGURATION
    # ============================================
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 5
    WEIGHT_DECAY = 0.01

    # Gradient clipping
    CLIP_GRAD_NORM = 1.0

    # Learning rate scheduler
    USE_LR_SCHEDULER = True
    LR_SCHEDULER_TYPE = 'cosine'  # 'cosine', 'step', or 'plateau'
    WARMUP_STEPS = 1000

    # ============================================
    # OPTIMIZATION ENHANCEMENTS (Part 2)
    # ============================================
    # Gradient accumulation (Part 2.3)
    GRADIENT_ACCUMULATION_STEPS = 1  # Effective batch size = BATCH_SIZE * ACCUM_STEPS

    # Gradient checkpointing (Part 2.4)
    USE_GRADIENT_CHECKPOINTING = False  # Memory-efficient training

    # ============================================
    # INFERENCE CONFIGURATION
    # ============================================
    # Basic generation
    MAX_GEN_LENGTH = 100        # Maximum tokens to generate
    TEMPERATURE = 1.0           # Sampling temperature
    TOP_K = 50                  # Top-k sampling
    TOP_P = 0.9                 # Nucleus sampling (top-p)

    # Beam search (Part 2.1)
    BEAM_WIDTH = 5              # Number of beams for beam search
    LENGTH_PENALTY = 1.0        # Length penalty for beam search

    # KV caching (Part 2.2)
    USE_KV_CACHE = True         # Enable KV caching for faster inference

    # ============================================
    # EVALUATION CONFIGURATION
    # ============================================
    # Number of samples for different experiments
    NUM_SAMPLES_BEAM = 5        # Samples for beam search evaluation
    NUM_SAMPLES_KV = 20         # Samples for KV cache benchmark

    # Beam search experiments
    BEAM_WIDTHS = [1, 5, 10]    # Different beam widths to compare

    # Gradient accumulation experiments
    ACCUM_STEPS_LIST = [1, 2, 4, 8]  # Different accumulation steps to compare

    # ============================================
    # FILE PATHS
    # ============================================
    # Base directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')

    # Create directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Checkpoint paths
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pt')
    LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'last_model.pt')

    # Vocabulary paths
    VOCAB_PATH = os.path.join(DATA_DIR, 'vocab.pkl')
    EMBEDDING_MATRIX_PATH = os.path.join(DATA_DIR, 'embedding_matrix.npy')

    # Results paths
    TRAINING_CURVES_PATH = os.path.join(RESULTS_DIR, 'training_curves.png')
    COMBINED_LOSS_PATH = os.path.join(RESULTS_DIR, 'combined_loss.png')
    ATTENTION_VIZ_PATH = os.path.join(RESULTS_DIR, 'attention_visualization.png')

    # ============================================
    # LOGGING CONFIGURATION
    # ============================================
    LOG_INTERVAL = 100          # Log every N batches
    SAVE_INTERVAL = 1           # Save checkpoint every N epochs
    EVAL_INTERVAL = 1           # Evaluate every N epochs

    # ============================================
    # REPRODUCIBILITY
    # ============================================
    SEED = 42                   # Random seed for reproducibility

    # ============================================
    # DATA LOADING
    # ============================================
    NUM_WORKERS = 2             # Number of workers for data loading
    PIN_MEMORY = True if DEVICE == 'cuda' else False

    # ============================================
    # FASTTEXT CONFIGURATION
    # ============================================
    FASTTEXT_MODEL_NAME = 'fasttext-wiki-news-subwords-300'
    KAGGLE_MODE = True          # Use gensim API (works on Kaggle)

    # ============================================
    # EXPERIMENT TRACKING
    # ============================================
    EXPERIMENT_NAME = 'decoder_transformer'
    TRACK_EXPERIMENTS = False   # Enable experiment tracking (e.g., wandb)

    @classmethod
    def display(cls):
        """Display current configuration"""
        print("=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"Device: {cls.DEVICE}")
        print(f"\nData:")

        # Handle None values for training samples
        train_samples_str = "ALL (2.1M+)" if cls.NUM_TRAIN_SAMPLES is None else f"{cls.NUM_TRAIN_SAMPLES:,}"
        val_samples_str = "ALL (22K)" if cls.NUM_VAL_SAMPLES is None else f"{cls.NUM_VAL_SAMPLES:,}"

        print(f"  Training samples: {train_samples_str}")
        print(f"  Validation samples: {val_samples_str}")
        print(f"  Vocabulary size: {cls.VOCAB_SIZE:,}")
        print(f"\nModel Architecture:")
        print(f"  d_model: {cls.D_MODEL}")
        print(f"  num_heads: {cls.NUM_HEADS}")
        print(f"  num_layers: {cls.NUM_LAYERS}")
        print(f"  d_ff: {cls.D_FF}")
        print(f"  max_seq_len: {cls.MAX_SEQ_LEN}")
        print(f"  dropout: {cls.DROPOUT}")
        print(f"\nTraining:")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Learning rate: {cls.LEARNING_RATE}")
        print(f"  Gradient accumulation: {cls.GRADIENT_ACCUMULATION_STEPS}")
        print(f"  Gradient checkpointing: {cls.USE_GRADIENT_CHECKPOINTING}")
        print(f"\nInference:")
        print(f"  Beam width: {cls.BEAM_WIDTH}")
        print(f"  KV caching: {cls.USE_KV_CACHE}")
        print(f"  Temperature: {cls.TEMPERATURE}")
        print("=" * 60)

    @classmethod
    def estimate_training_time(cls):
        """Estimate training time"""
        if cls.NUM_TRAIN_SAMPLES is None:
            # Use approximate full dataset size (2.1M samples)
            num_samples = 2119719
            print(f"Using full dataset (~{num_samples:,} samples)")
        else:
            num_samples = cls.NUM_TRAIN_SAMPLES

        batches_per_epoch = num_samples // cls.BATCH_SIZE
        estimated_time_min = (batches_per_epoch * 0.025 * cls.NUM_EPOCHS) / 60
        estimated_time_hr = estimated_time_min / 60

        print(f"Estimated training time: {estimated_time_min:.1f} minutes ({estimated_time_hr:.1f} hours)")
        print(f"(Not including evaluation time)")
        return estimated_time_min


# Create a default config instance
config = Config()


# Alternative configurations for different scenarios
class QuickTestConfig(Config):
    """Quick test configuration with smaller parameters"""
    NUM_TRAIN_SAMPLES = 10000
    NUM_VAL_SAMPLES = 1000
    VOCAB_SIZE = 10000
    NUM_EPOCHS = 1
    NUM_LAYERS = 3


class FullScaleConfig(Config):
    """Full-scale configuration for best performance"""
    NUM_TRAIN_SAMPLES = None  # Use entire dataset
    NUM_VAL_SAMPLES = None    # Use entire validation set
    VOCAB_SIZE = 200000
    NUM_EPOCHS = 10
    NUM_LAYERS = 6
    BATCH_SIZE = 64


if __name__ == '__main__':
    # Display configuration when run directly
    Config.display()
    print("\n")
    Config.estimate_training_time()
