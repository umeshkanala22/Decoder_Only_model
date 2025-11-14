

## Download Vocabulary File

The vocabulary file (`vocab.pkl` - 197 KB) is hosted on Hugging Face:

**🔗 Download from**: [https://huggingface.co/silvesage22/Decoder_only_model](https://huggingface.co/silvesage22/Decoder_only_model)

### Quick Download

```bash
# Using Hugging Face CLI
pip install huggingface_hub
huggingface-cli download silvesage22/Decoder_only_model vocab.pkl --local-dir ./data

# Or using Python
from huggingface_hub import hf_hub_download
vocab_path = hf_hub_download(
    repo_id="silvesage22/Decoder_only_model",
    filename="vocab.pkl",
    local_dir="./data"
)
```

## Expected Data Structure

After downloading the vocabulary and preparing your dataset, this directory should contain:

```
data/
├── vocab.pkl           # Vocabulary file (download from HF)
├── train.txt           # Training data (your own data)
├── val.txt             # Validation data (your own data)
└── test.txt            # Test data (your own data)
```

## Vocabulary File Contents

The `vocab.pkl` file contains:
```python
{
    'word2idx': {...},      # Word to index mapping
    'idx2word': {...},      # Index to word mapping
    'vocab_size': N,        # Total vocabulary size
    'special_tokens': {
        'PAD': 0,           # Padding token
        'UNK': 1,           # Unknown token
        'SOS': 2,           # Start of sequence
        'EOS': 3            # End of sequence
    }
}
```

## Dataset Format

Your training data should be in plain text format:
- One example per line
- Source and target separated by delimiter (e.g., tab or special token)
- UTF-8 encoding

Example:
```
source sequence 1 <SEP> target sequence 1
source sequence 2 <SEP> target sequence 2
source sequence 3 <SEP> target sequence 3
```

## Building Your Own Vocabulary

If you want to create a new vocabulary from your data:

```python
from data_utils import build_vocabulary

# Build vocabulary from your data
vocab = build_vocabulary(
    train_file='train.txt',
    min_freq=2,
    max_vocab_size=50000
)

# Save vocabulary
import pickle
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
```
