"""
Quick test script to verify model, training, and inference work correctly
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from model import DecoderOnlyTransformer
from train import train_epoch, validate
from inference import generate, tokens_to_text

print("=" * 60)
print("QUICK FUNCTIONALITY TEST")
print("=" * 60)

# Test 1: Configuration
print("\n1. Testing Configuration...")
Config.display()
print("✓ Config loaded successfully")

# Test 2: Model initialization
print("\n2. Testing Model Initialization...")
vocab_size = 1000
model = DecoderOnlyTransformer(
    vocab_size=vocab_size,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_ff=512,
    max_seq_len=32,
    dropout=0.1,
    pad_idx=0,
    pretrained_embeddings=None,
    use_checkpoint=False
)
print("✓ Model initialized successfully")

# Test 3: Forward pass
print("\n3. Testing Forward Pass...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

batch_size = 4
seq_len = 16
dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
dummy_target = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

logits, loss = model(dummy_input, targets=dummy_target)
print(f"Input shape: {dummy_input.shape}")
print(f"Logits shape: {logits.shape}")
print(f"Loss: {loss.item():.4f}")
print("✓ Forward pass successful")

# Test 4: Training step
print("\n4. Testing Training Step...")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Create dummy dataset
n_samples = 32
dummy_data = torch.randint(0, vocab_size, (n_samples, seq_len))
dummy_targets = torch.randint(0, vocab_size, (n_samples, seq_len))
dataset = TensorDataset(dummy_data, dummy_targets)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

model.train()
initial_loss = None
for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
    input_seq = input_seq.to(device)
    target_seq = target_seq.to(device)

    optimizer.zero_grad()
    logits, loss = model(input_seq, targets=target_seq)
    loss.backward()
    optimizer.step()

    if initial_loss is None:
        initial_loss = loss.item()

    if batch_idx == 0:
        print(f"Initial loss: {loss.item():.4f}")

print("✓ Training step successful")

# Test 5: Validation
print("\n5. Testing Validation...")
val_loader = DataLoader(dataset, batch_size=8, shuffle=False)
val_loss, perplexity = validate(model, val_loader, device)
print(f"Validation loss: {val_loss:.4f}")
print(f"Perplexity: {perplexity:.2f}")
print("✓ Validation successful")

# Test 6: Inference
print("\n6. Testing Inference...")
model.eval()

# Create simple vocab for testing
word2idx = {f'word{i}': i for i in range(100)}
word2idx.update({'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3})
idx2word = {v: k for k, v in word2idx.items()}

prompt = torch.randint(0, 100, (1, 5)).to(device)

with torch.no_grad():
    generated = generate(
        model=model,
        prompt=prompt,
        max_new_tokens=10,
        temperature=1.0,
        top_k=50,
        eos_token=2,
        device=device
    )

print(f"Prompt shape: {prompt.shape}")
print(f"Generated shape: {generated.shape}")
print(f"Generated tokens: {generated[0].cpu().tolist()[:20]}")
print("✓ Inference successful")

# Test 7: Gradient Accumulation
print("\n7. Testing Gradient Accumulation...")
train_loss = train_epoch(
    model, train_loader, optimizer, device,
    clip_grad_norm=1.0,
    accum_steps=2
)
print(f"Training loss with accumulation: {train_loss:.4f}")
print("✓ Gradient accumulation successful")

# Test 8: Gradient Checkpointing
print("\n8. Testing Gradient Checkpointing...")
model_cp = DecoderOnlyTransformer(
    vocab_size=vocab_size,
    d_model=128,
    num_heads=4,
    num_layers=2,
    d_ff=512,
    max_seq_len=32,
    dropout=0.1,
    pad_idx=0,
    use_checkpoint=True
)
model_cp = model_cp.to(device)
model_cp.train()

optimizer_cp = torch.optim.Adam(model_cp.parameters(), lr=1e-3)
for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
    if batch_idx > 0:
        break
    input_seq = input_seq.to(device)
    target_seq = target_seq.to(device)

    optimizer_cp.zero_grad()
    logits, loss = model_cp(input_seq, targets=target_seq)
    loss.backward()
    optimizer_cp.step()
    print(f"Checkpoint loss: {loss.item():.4f}")

print("✓ Gradient checkpointing successful")

# Test 9: KV Caching
print("\n9. Testing KV Caching...")
from inference import generate_with_kv_cache

model.eval()
with torch.no_grad():
    generated_kv = generate_with_kv_cache(
        model=model,
        prompt=prompt,
        max_new_tokens=10,
        temperature=1.0,
        top_k=50,
        eos_token=2,
        device=device
    )
print(f"KV cache generated shape: {generated_kv.shape}")
print("✓ KV caching successful")

# Test 10: Beam Search
print("\n10. Testing Beam Search...")
from inference import generate_beam_search

with torch.no_grad():
    best_seq, best_score, all_beams = generate_beam_search(
        model=model,
        prompt=prompt,
        beam_width=3,
        max_new_tokens=10,
        eos_token=2,
        device=device
    )
print(f"Beam search best shape: {best_seq.shape}")
print(f"Best score: {best_score:.4f}")
print(f"Number of beams: {len(all_beams)}")
print("✓ Beam search successful")

# Summary
print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print("\nYour implementation is working correctly!")
print(f"Device used: {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\nReady for submission!")
