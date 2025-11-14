

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from tqdm import tqdm
from datasets import load_dataset
import pickle
import gensim.downloader as api
GENSIM_AVAILABLE = True

from config import Config


def build_vocab(train_texts, min_freq=2, max_vocab_size=160000):
    word_counter = Counter()
    for text in tqdm(train_texts, desc="Counting words"):
        words = text.lower().strip().split()
        word_counter.update(words)

    print(f"Total unique words found: {len(word_counter)}")
    print(f"Total word occurrences: {sum(word_counter.values())}")

    # Filter by minimum frequency
    filtered_words = {word: freq for word, freq in word_counter.items()
                     if freq >= min_freq}
    print(f"Words after min_freq={min_freq} filter: {len(filtered_words)}")


    most_common = word_counter.most_common(max_vocab_size)


    # Reserve indices for special tokens
    word2idx = {
        Config.PAD_TOKEN: Config.PAD_IDX,
        Config.SOS_TOKEN: Config.SOS_IDX,
        Config.EOS_TOKEN: Config.EOS_IDX,
        Config.UNK_TOKEN: Config.UNK_IDX
    }

    idx2word = {
        Config.PAD_IDX: Config.PAD_TOKEN,
        Config.SOS_IDX: Config.SOS_TOKEN,
        Config.EOS_IDX: Config.EOS_TOKEN,
        Config.UNK_IDX: Config.UNK_TOKEN
    }

    # Add words from most common
    current_idx = 4
    for word, freq in most_common:
        if freq >= min_freq:
            word2idx[word] = current_idx
            idx2word[current_idx] = word
            current_idx += 1

    vocab_size = len(word2idx)
    print(f"Final vocabulary size: {vocab_size}")
    print(f"Sample words: {list(word2idx.keys())[4:14]}")

    return word2idx, idx2word, word_counter


def load_fasttext_embeddings(vocab, embedding_dim=300, kaggle_mode=True):
    """
    Load FastText embeddings for your vocabulary

    Args:
        vocab: word2idx dictionary
        embedding_dim: Dimension of embeddings (300 for FastText)
        kaggle_mode: If True, uses gensim API (works on Kaggle)
                     If False, loads from local file

    Returns:
        embeddings_dict: Dictionary mapping words to their embedding vectors
        fasttext_model: The loaded FastText model (for reference)
    """
    if not GENSIM_AVAILABLE:
        raise ImportError(
            "gensim is required for FastText embeddings. "
            "Install it with: pip install gensim\n"
            "Or set USE_PRETRAINED_EMBEDDINGS=False in config.py to use random embeddings."
        )

    print(f"Loading FastText embeddings (dim={embedding_dim})...")

    if kaggle_mode:
        # Check if already downloaded, otherwise download
        import os
        cache_dir = os.path.expanduser('~/.cache/gensim/data')
        model_path = os.path.join(cache_dir, Config.FASTTEXT_MODEL_NAME)

        if os.path.exists(model_path):
            print("Loading FastText model from cache...")
        else:
            print("Downloading FastText model via gensim API (this will take a few minutes)...")

        fasttext_model = api.load(Config.FASTTEXT_MODEL_NAME)
        print("FastText model loaded ")

    else:
        # Load from local file if you've downloaded it
        print("Loading from local file...")
        import gensim
        fasttext_model = gensim.models.KeyedVectors.load_word2vec_format(
            'path/to/wiki-news-300d-1M.vec',
            binary=False
        )

    # Extract embeddings for words in your vocabulary
    embeddings_dict = {}
    found_count = 0

    print("Extracting embeddings for vocabulary words...")
    for word in tqdm(vocab.keys()):
        if word in [Config.PAD_TOKEN, Config.SOS_TOKEN, Config.EOS_TOKEN, Config.UNK_TOKEN]:
            # Will handle special tokens separately
            continue

        try:

            embeddings_dict[word] = fasttext_model[word]
            found_count += 1
        except KeyError:
            # Word not in FastText vocabulary
            embeddings_dict[word] = None

    coverage = (found_count / (len(vocab) - 4)) * 100
    print(f"Embedding coverage: {found_count}/{len(vocab)-4} ({coverage:.2f}%)")

    return embeddings_dict, fasttext_model


def create_embedding_matrix(vocab, embeddings_dict, embedding_dim=300):
    """
    Create embedding matrix to load into PyTorch nn.Embedding

    Args:
        vocab: word2idx dictionary
        embeddings_dict: Dictionary with word -> embedding vector
        embedding_dim: Dimension of embeddings

    Returns:
        embedding_matrix: numpy array of shape (vocab_size, embedding_dim)
    """
    print("Creating embedding matrix...")

    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)

    # Track statistics
    found = 0
    not_found = 0

    for word, idx in tqdm(vocab.items(), desc="Building matrix"):
        if word in [Config.PAD_TOKEN, Config.SOS_TOKEN, Config.EOS_TOKEN, Config.UNK_TOKEN]:

            # <pad> can be zeros or random (you choose)
            if word == Config.PAD_TOKEN:
                embedding_matrix[idx] = np.zeros(embedding_dim)  # or random
            else:
                embedding_matrix[idx] = np.random.normal(
                    loc=0.0,
                    scale=0.1,
                    size=embedding_dim
                )
        else:

            embedding_vector = embeddings_dict.get(word)

            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
                found += 1
            else:
                # Word not found in FastText - initialize randomly
                embedding_matrix[idx] = np.random.normal(
                    loc=0.0,
                    scale=0.1,
                    size=embedding_dim
                )
                not_found += 1

    print(f"Embeddings found: {found}, not found (random init): {not_found}")
    print(f"Embedding matrix shape: {embedding_matrix.shape}")

    return embedding_matrix


class TinyStoriesDataset(Dataset):
    """
    PyTorch Dataset for TinyStories
    Tokenizes text and converts to tensor indices
    """
    def __init__(self, texts, word2idx, max_seq_len=64):
        """
        Args:
            texts: List of text strings
            word2idx: Vocabulary mapping
            max_seq_len: Maximum sequence length (for padding/truncation)
        """
        self.texts = texts
        self.word2idx = word2idx
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def tokenize(self, text):
        """
        Tokenize text into word indices

        Args:
            text: Input text string

        Returns:
            tokens: List of token indices
        """
        # Lowercase and split
        words = text.lower().strip().split()

        # Convert to indices
        tokens = [self.word2idx.get(word, self.word2idx[Config.UNK_TOKEN])
                 for word in words]

        # Add SOS and EOS tokens
        tokens = [self.word2idx[Config.SOS_TOKEN]] + tokens + [self.word2idx[Config.EOS_TOKEN]]

        # Truncate if too long
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        # Pad if too short
        elif len(tokens) < self.max_seq_len:
            tokens = tokens + [self.word2idx[Config.PAD_TOKEN]] * (self.max_seq_len - len(tokens))

        return tokens

    def __getitem__(self, idx):
        """
        Get a single item

        Returns:
            input_ids: Token indices for input (all tokens except last)
            target_ids: Token indices for target (all tokens except first)
        """
        text = self.texts[idx]
        tokens = self.tokenize(text)

        # Convert to tensor
        tokens_tensor = torch.LongTensor(tokens)


        # Input: all tokens except the last one
        # Target: all tokens except the first one (shifted by 1)
        input_ids = tokens_tensor[:-1]
        target_ids = tokens_tensor[1:]

        return input_ids, target_ids


def create_dataloaders(train_texts, val_texts, word2idx, batch_size=32,
                       max_seq_len=64, num_workers=2, pin_memory=True):
    """
    Create PyTorch DataLoaders for training and validation

    Args:
        train_texts: List of training text strings
        val_texts: List of validation text strings
        word2idx: Vocabulary mapping
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory (faster GPU transfer)

    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    print("Creating datasets...")


    train_dataset = TinyStoriesDataset(train_texts, word2idx, max_seq_len)
    val_dataset = TinyStoriesDataset(val_texts, word2idx, max_seq_len)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    return train_loader, val_loader


def load_tinystories_dataset(num_train_samples=None, num_val_samples=None):
    """
    Load TinyStories dataset from HuggingFace

    Args:
        num_train_samples: Number of training samples (None for all)
        num_val_samples: Number of validation samples (None for all)

    Returns:
        train_texts: List of training texts
        val_texts: List of validation texts
    """
    print("Loading TinyStories dataset from HuggingFace...")

    dataset = load_dataset("roneneldan/TinyStories")

    # Extract texts
    if num_train_samples is not None:
        train_dataset = dataset['train'].select(range(num_train_samples))
    else:
        train_dataset = dataset['train']

    if num_val_samples is not None:
        val_dataset = dataset['validation'].select(range(num_val_samples))
    else:
        val_dataset = dataset['validation']

    train_texts = train_dataset['text']
    val_texts = val_dataset['text']

    print(f"Loaded {len(train_texts)} training texts")
    print(f"Loaded {len(val_texts)} validation texts")

    return train_texts, val_texts


def save_vocabulary(word2idx, idx2word, save_path):
    """Save vocabulary dictionaries to file"""
    vocab_data = {
        'word2idx': word2idx,
        'idx2word': idx2word
    }
    with open(save_path, 'wb') as f:
        pickle.dump(vocab_data, f)
    print(f"Vocabulary saved to {save_path}")


def load_vocabulary(load_path):
    """Load vocabulary dictionaries from file"""
    with open(load_path, 'rb') as f:
        vocab_data = pickle.load(f)
    print(f"Vocabulary loaded from {load_path}")
    return vocab_data['word2idx'], vocab_data['idx2word']


def save_embedding_matrix(embedding_matrix, save_path):
    """Save embedding matrix to file"""
    np.save(save_path, embedding_matrix)
    print(f"Embedding matrix saved to {save_path}")


def load_embedding_matrix(load_path):
    """Load embedding matrix from file"""
    embedding_matrix = np.load(load_path)
    print(f"Embedding matrix loaded from {load_path}")
    return embedding_matrix


if __name__ == '__main__':
    # Test data utilities
    print("Testing data utilities...")

    # Load small subset for testing
    train_texts, val_texts = load_tinystories_dataset(
        num_train_samples=1000,
        num_val_samples=100
    )

    # Build vocabulary
    word2idx, idx2word, word_counter = build_vocab(
        train_texts,
        min_freq=2,
        max_vocab_size=5000
    )


    train_loader, val_loader = create_dataloaders(
        train_texts, val_texts, word2idx,
        batch_size=8,
        max_seq_len=32
    )

    # Test a batch
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Inputs shape: {inputs.shape}")
        print(f"  Targets shape: {targets.shape}")
        print(f"  Sample input: {inputs[0][:10]}")
        print(f"  Sample target: {targets[0][:10]}")
        break

    print("\nData utilities test complete!")
