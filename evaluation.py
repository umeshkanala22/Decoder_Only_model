"""
Evaluation Module
Perplexity calculation, BLEU score, and experiment benchmarks for Part 2
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
from collections import defaultdict
from inference import text_to_tokens, generate, generate_with_kv_cache
from evaluate import load
from config import Config
from utils import calculate_perplexity, format_time
from inference import text_to_tokens, tokens_to_text, generate_beam_search, generate

def evaluate_perplexity(model, dataloader, device='cuda'):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for input_seq, target_seq in tqdm(dataloader, desc="Evaluating"):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            logits, loss = model(input_seq, targets=target_seq)

            # Count non-padding tokens
            num_tokens = (target_seq != model.pad_idx).sum().item()

            total_loss += loss.item() * len(input_seq)
            total_tokens += num_tokens

    avg_loss = total_loss / len(dataloader.dataset)
    perplexity = calculate_perplexity(avg_loss)

    return perplexity, avg_loss


def calculate_bleu_score(references, hypotheses):

    
    bleu = load("bleu")


    results = bleu.compute(predictions=hypotheses, references=references)
    return results['bleu']



def benchmark_beam_search(model, prompts, word2idx, idx2word, beam_widths=[1, 5, 10],
                          max_new_tokens=50, device='cuda'):


    results = {
        'beam_widths': beam_widths,
        'generation_times': [],
        'generated_texts': {},
        'scores': {}
    }

    print("\n" + "="*60)
    print("BEAM SEARCH BENCHMARK")
    print("="*60)

    for beam_width in beam_widths:
        print(f"\nBeam Width: {beam_width}")
        print("-" * 40)

        texts = []
        scores_list = []
        total_time = 0

        for prompt_text in tqdm(prompts, desc=f"Beam={beam_width}"):
            prompt_tokens = text_to_tokens(prompt_text, word2idx, model.max_seq_len).to(device)

            start_time = time.time()

            if beam_width == 1:
                # Greedy decoding (beam_width=1)
                generated = generate(
                    model, prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    eos_token=word2idx.get(Config.EOS_TOKEN),
                    device=device
                )
                score = 0.0
            else:
                # Beam search
                generated, score, _ = generate_beam_search(
                    model, prompt_tokens,
                    beam_width=beam_width,
                    max_new_tokens=max_new_tokens,
                    eos_token=word2idx.get(Config.EOS_TOKEN),
                    device=device
                )

            gen_time = time.time() - start_time
            total_time += gen_time

            generated_text = tokens_to_text(generated[0], idx2word)
            texts.append(generated_text)
            scores_list.append(score)

        avg_time = total_time / len(prompts)
        results['generation_times'].append(avg_time)
        results['generated_texts'][beam_width] = texts
        results['scores'][beam_width] = scores_list

        print(f"Average generation time: {avg_time:.4f}s")
        print(f"Sample output: {texts[0][:100]}...")


    print("BEAM SEARCH COMPARISON")
    print("="*60)
    for beam_width, avg_time in zip(beam_widths, results['generation_times']):
        print(f"Beam Width {beam_width}: {avg_time:.4f}s per sample")

    return results


def benchmark_kv_cache(model, prompts, word2idx, idx2word, num_samples=20,
                      max_new_tokens=50, device='cuda'):

    results = {
        'without_cache': {'time': 0, 'texts': []},
        'with_cache': {'time': 0, 'texts': []},
        'speedup': 0
    }

    print("\n" + "="*60)
    print("KV CACHE BENCHMARK")
    print("="*60)

    # Test WITHOUT KV cache
    print("\nWithout KV Cache:")
    print("-" * 40)
    start_time = time.time()

    for prompt_text in tqdm(prompts[:num_samples], desc="No Cache"):
        prompt_tokens = text_to_tokens(prompt_text, word2idx, model.max_seq_len).to(device)

        generated = generate(
            model, prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            eos_token=word2idx.get(Config.EOS_TOKEN),
            device=device
        )

        from inference import tokens_to_text
        generated_text = tokens_to_text(generated[0], idx2word)
        results['without_cache']['texts'].append(generated_text)

    time_without_cache = time.time() - start_time
    results['without_cache']['time'] = time_without_cache

    print(f"Total time: {format_time(time_without_cache)}")
    print(f"Average time per sample: {time_without_cache/num_samples:.4f}s")

    # Test WITH KV cache
    print("\nWith KV Cache:")
    print("-" * 40)
    start_time = time.time()

    for prompt_text in tqdm(prompts[:num_samples], desc="With Cache"):
        prompt_tokens = text_to_tokens(prompt_text, word2idx, model.max_seq_len).to(device)

        generated = generate_with_kv_cache(
            model, prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            eos_token=word2idx.get(Config.EOS_TOKEN),
            device=device
        )

        from inference import tokens_to_text
        generated_text = tokens_to_text(generated[0], idx2word)
        results['with_cache']['texts'].append(generated_text)

    time_with_cache = time.time() - start_time
    results['with_cache']['time'] = time_with_cache

    print(f"Total time: {format_time(time_with_cache)}")
    print(f"Average time per sample: {time_with_cache/num_samples:.4f}s")


    speedup = time_without_cache / time_with_cache
    results['speedup'] = speedup

    return results


def benchmark_gradient_accumulation(model, train_loader, accum_steps_list=[1, 2, 4, 8],
                                    num_batches=10, device='cuda'):

    from train import train_epoch

    results = {
        'accum_steps': accum_steps_list,
        'times': [],
        'memory_usage': []
    }

    print("\n" + "="*60)
    print("GRADIENT ACCUMULATION BENCHMARK")
    print("="*60)

    for accum_steps in accum_steps_list:
        print(f"\nAccumulation Steps: {accum_steps}")
        print(f"Effective batch size: {train_loader.batch_size * accum_steps}")
        print("-" * 40)


        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

        # Measure time and memory
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

        start_time = time.time()

        # Train for a few batches
        model.train()
        optimizer.zero_grad()

        for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break

            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            logits, loss = model(input_seq, targets=target_seq)
            loss = loss / accum_steps
            loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        elapsed_time = time.time() - start_time


        peak_memory = 0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        results['times'].append(elapsed_time)
        results['memory_usage'].append(peak_memory)

        print(f"Time: {elapsed_time:.4f}s")
        if torch.cuda.is_available():
            print(f"Peak GPU memory: {peak_memory:.2f} MB")

    # Print comparison
    print("\n" + "="*60)
    print("GRADIENT ACCUMULATION COMPARISON")
    print("="*60)
    for accum, time_val, mem in zip(accum_steps_list, results['times'], results['memory_usage']):
        print(f"Accum Steps {accum}: {time_val:.4f}s, Memory: {mem:.2f} MB")

    return results


def benchmark_gradient_checkpointing(model_with_cp, model_without_cp, train_loader,
                                    num_batches=10, device='cuda'):

    results = {
        'without_checkpointing': {'time': 0, 'memory': 0},
        'with_checkpointing': {'time': 0, 'memory': 0},
        'memory_savings': 0,
        'time_overhead': 0
    }

    print("\n" + "="*60)
    print("GRADIENT CHECKPOINTING BENCHMARK")
    print("="*60)

    # Test WITHOUT checkpointing
    print("\nWithout Checkpointing:")
    print("-" * 40)

    model_without_cp.to(device)
    optimizer = torch.optim.Adam(model_without_cp.parameters(), lr=Config.LEARNING_RATE)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start_time = time.time()

    model_without_cp.train()
    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        optimizer.zero_grad()
        logits, loss = model_without_cp(input_seq, targets=target_seq)
        loss.backward()
        optimizer.step()

    time_without = time.time() - start_time
    memory_without = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    results['without_checkpointing']['time'] = time_without
    results['without_checkpointing']['memory'] = memory_without

    print(f"Time: {time_without:.4f}s")
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {memory_without:.2f} MB")

    # Test WITH checkpointing
    print("\nWith Checkpointing:")
    print("-" * 40)

    model_with_cp.to(device)
    optimizer = torch.optim.Adam(model_with_cp.parameters(), lr=Config.LEARNING_RATE)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start_time = time.time()

    model_with_cp.train()
    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)

        optimizer.zero_grad()
        logits, loss = model_with_cp(input_seq, targets=target_seq)
        loss.backward()
        optimizer.step()

    time_with = time.time() - start_time
    memory_with = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    results['with_checkpointing']['time'] = time_with
    results['with_checkpointing']['memory'] = memory_with

    print(f"Time: {time_with:.4f}s")
    if torch.cuda.is_available():
        print(f"Peak GPU memory: {memory_with:.2f} MB")


    memory_savings = memory_without - memory_with
    time_overhead = time_with - time_without

    results['memory_savings'] = memory_savings
    results['time_overhead'] = time_overhead

    print("\n" + "="*60)
    print(f"Memory Savings: {memory_savings:.2f} MB ({memory_savings/memory_without*100:.1f}%)")
    print(f"Time Overhead: {time_overhead:.4f}s ({time_overhead/time_without*100:.1f}%)")
    print("="*60)

    return results



