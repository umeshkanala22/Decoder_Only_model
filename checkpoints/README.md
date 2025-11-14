# Model Checkpoints

The trained model checkpoint (`best_model.pt` - 131 MB) is hosted on Hugging Face:

**🔗 Download from**: [https://huggingface.co/silvesage22/Decoder_only_model](https://huggingface.co/silvesage22/Decoder_only_model)

### Quick Download

```bash
# Using Hugging Face CLI
pip install huggingface_hub
huggingface-cli download silvesage22/Decoder_only_model best_model.pt --local-dir ./checkpoints

# Or using Python
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="silvesage22/Decoder_only_model",
    filename="best_model.pt",
    local_dir="./checkpoints"
)
```

After downloading, place the file in this directory:
```
checkpoints/
└── best_model.pt
```

## Checkpoint Contents

The checkpoint file contains:
- Model state dictionary
- Optimizer state dictionary
- Training configuration
- Epoch number
- Loss history
- Vocabulary size

## File Structure

```python
checkpoint = {
    'epoch': 4,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'loss': ...,
    'config': {...}
}
```
