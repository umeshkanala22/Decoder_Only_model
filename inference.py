"""
Inference and Text Generation Module
Supports: basic generation, beam search (Part 2.1), and KV caching (Part 2.2)
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

from config import Config


def generate(
    model,
    prompt,
    max_new_tokens=50,
    temperature=1.0,
    top_k=None,
    top_p=None,
    eos_token=None,
    device='cuda'
):

    model.eval()
    prompt = prompt.to(device)

    generated = prompt.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):

            # Truncate to max_seq_len if too long
            context = generated if generated.size(1) <= model.max_seq_len else generated[:, -model.max_seq_len:]

            logits, _ = model.forward(context)


            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS token generated (for all sequences in batch)
            if eos_token is not None and (next_token == eos_token).all():
                break

    return generated


def generate_beam_search(
    model,
    prompt,
    beam_width=5,
    max_new_tokens=50,
    temperature=1.0,
    length_penalty=1.0,
    eos_token=None,
    device='cuda'
):

    model.eval()
    prompt = prompt.to(device)

    # Beam search works on single input
    assert prompt.size(0) == 1, "Beam search only supports batch_size=1"


    beams = [(prompt, 0.0)]
    completed_beams = []

    with torch.no_grad():
        for step in range(max_new_tokens):
            all_candidates = []

            # Expand each current beam
            for seq, score in beams:
                if eos_token is not None and seq[0, -1].item() == eos_token:
                    completed_beams.append((seq, score))
                    continue

                context = seq if seq.size(1) <= model.max_seq_len else seq[:, -model.max_seq_len:]

                logits, _ = model.forward(context)

                next_token_logits = logits[0, -1, :] / temperature

                log_probs = F.log_softmax(next_token_logits, dim=-1)


                top_log_probs, top_indices = torch.topk(log_probs, beam_width)


                for log_prob, token_idx in zip(top_log_probs, top_indices):
                    new_seq = torch.cat([seq, token_idx.unsqueeze(0).unsqueeze(0)], dim=1)

                    new_score = score + log_prob.item()

                    seq_len = new_seq.size(1)
                    normalized_score = new_score / (seq_len ** length_penalty)

                    all_candidates.append((new_seq, new_score, normalized_score))

            # If no candidates (all beams finished), break
            if not all_candidates:
                break

            # Sort all candidates by normalized score and keep top beam_width
            all_candidates.sort(key=lambda x: x[2], reverse=True)
            beams = [(seq, score) for seq, score, _ in all_candidates[:beam_width]]

            if len(beams) == 0:
                break

        # Add remaining beams to completed beams after loop finishes
        completed_beams.extend(beams)

        # Sort completed beams by normalized score
        if completed_beams:
            completed_beams.sort(
                key=lambda x: x[1] / (x[0].size(1) ** length_penalty),
                reverse=True
            )
            best_sequence = completed_beams[0][0]
            best_score = completed_beams[0][1]
        else:
            # Fallback if no beams completed
            best_sequence = beams[0][0] if beams else prompt
            best_score = beams[0][1] if beams else 0.0

    return best_sequence, best_score, completed_beams


def generate_beam_search_batch(
    model,
    prompts,
    beam_width=5,
    max_new_tokens=50,
    temperature=1.0,
    length_penalty=1.0,
    eos_token=None,
    device='cuda'
):


    batch_size = prompts.size(0)
    generated_sequences = []
    scores = []

    for i in range(batch_size):
        prompt = prompts[i:i+1]  # Single prompt
        best_seq, best_score, _ = generate_beam_search(
            model=model,
            prompt=prompt,
            beam_width=beam_width,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            length_penalty=length_penalty,
            eos_token=eos_token,
            device=device
        )
        generated_sequences.append(best_seq)
        scores.append(best_score)

    return generated_sequences, scores


def generate_with_kv_cache(
    model,
    prompt,
    max_new_tokens=50,
    temperature=1.0,
    top_k=None,
    eos_token=None,
    device='cuda'
):

    model.eval()
    prompt = prompt.to(device)
    batch_size = prompt.size(0)


    kv_caches = [None] * model.num_layers

    generated = prompt.clone()

    with torch.no_grad():
        for step in range(max_new_tokens):
            # For first step, process full prompt
            # For subsequent steps, only process last token
            if step == 0:
                input_tokens = generated
            else:
                input_tokens = generated[:, -1:]  # Only last token

            # === EMBEDDING LAYER ===
            x = model.embedding(input_tokens)
            x = model.dropout(x)

            # === TRANSFORMER BLOCKS WITH KV CACHE ===
            new_kv_caches = []

            for layer_idx, block in enumerate(model.blocks):

                layer_cache = kv_caches[layer_idx]

                # Forward through block with cache
                x, updated_cache = block.forward_with_cache(
                    x,
                    mask=None,
                    kv_cache=layer_cache,
                    use_cache=True
                )

                new_kv_caches.append(updated_cache)

            # Update caches for next iteration
            kv_caches = new_kv_caches

            # === FINAL LAYERS ===
            x = model.final_norm(x)
            logits = model.output_projection(x)


            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token is not None and (next_token == eos_token).all():
                break

    return generated


def tokens_to_text(tokens, idx2word):

    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy().tolist()

    if isinstance(tokens[0], list):
        tokens = tokens[0]

    words = []
    for token in tokens:
        word = idx2word.get(token, Config.UNK_TOKEN)
        # Skip special tokens in output
        if word not in [Config.PAD_TOKEN, Config.SOS_TOKEN, Config.EOS_TOKEN]:
            words.append(word)
        # Stop at EOS
        elif word == Config.EOS_TOKEN:
            break

    return ' '.join(words)


def text_to_tokens(text, word2idx, max_seq_len=None):

    # Tokenize
    words = text.lower().strip().split()

    tokens = [word2idx.get(word, word2idx[Config.UNK_TOKEN]) for word in words]

    tokens = [word2idx[Config.SOS_TOKEN]] + tokens

    # Truncate if needed
    if max_seq_len is not None and len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]

    tokens_tensor = torch.LongTensor(tokens).unsqueeze(0)

    return tokens_tensor


def generate_samples(
    model,
    prompts,
    word2idx,
    idx2word,
    method='greedy',
    max_new_tokens=50,
    temperature=1.0,
    beam_width=5,
    top_k=None,
    top_p=None,
    device='cuda',
    use_kv_cache=False
):

    model.eval()
    generated_texts = []

    for prompt_text in prompts:
        prompt_tokens = text_to_tokens(prompt_text, word2idx, model.max_seq_len)

        # Generate based on method
        if method == 'beam_search':
            generated_tokens, _, _ = generate_beam_search(
                model, prompt_tokens,
                beam_width=beam_width,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                eos_token=word2idx.get(Config.EOS_TOKEN),
                device=device
            )
        elif use_kv_cache:
            generated_tokens = generate_with_kv_cache(
                model, prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token=word2idx.get(Config.EOS_TOKEN),
                device=device
            )
        else:
            generated_tokens = generate(
                model, prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                eos_token=word2idx.get(Config.EOS_TOKEN),
                device=device
            )

        generated_text = tokens_to_text(generated_tokens[0], idx2word)
        generated_texts.append(generated_text)

    return generated_texts



