#!/usr/bin/env python3
"""NES Step 1 (GPU): Extract Gemma-3-27B last-token hidden states for each domain's L1 prompts.

Loads unsloth/gemma-3-27b-it-bnb-4bit (base, no LoRA), extracts the hidden state
at the last prompt token from layer 31 (middle of 62 layers) for N_RECORDS sampled
records from each domain's L1 dataset. Averages across records to get a stable
domain embedding.

Secondary: also extract from the final layer (layer 61) for sensitivity check.

Output: results/spinout/idea_nes/domain_embeddings.npz
  Keys: {domain}_mid, {domain}_final  (shape: [hidden_size])
  Also saves individual record embeddings: {domain}_mid_records  (shape: [N, hidden_size])
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

DOMAINS = ["dti", "ct", "ppi", "ge"]

L1_PATHS = {
    "dti": "exports/llm_benchmarks/l1_mcq.jsonl",
    "ct": "exports/ct_llm/ct_l1_dataset.jsonl",
    "ppi": "exports/ppi_llm/ppi_l1_dataset.jsonl",
    "ge": "exports/ge_llm/ge_l1_dataset.jsonl",
}

SYSTEM_PROMPTS = {
    "dti": None,  # pulled from prompt module
    "ct": None,
    "ppi": None,
    "ge": None,
}


def load_prompt_fn(domain: str):
    """Return the L1 prompt formatting function for a domain."""
    if domain == "dti":
        from negbiodb.llm_prompts import format_l1_prompt
        return format_l1_prompt
    elif domain == "ct":
        from negbiodb_ct.llm_prompts import format_ct_l1_prompt
        return format_ct_l1_prompt
    elif domain == "ppi":
        from negbiodb_ppi.llm_prompts import format_ppi_l1_prompt
        return format_ppi_l1_prompt
    elif domain == "ge":
        from negbiodb_depmap.llm_prompts import format_ge_l1_prompt
        return format_ge_l1_prompt
    raise ValueError(f"Unknown domain: {domain}")


def load_records(path: str, n: int, seed: int) -> list[dict]:
    """Load N records from a JSONL file."""
    rng = random.Random(seed)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if len(records) > n:
        records = rng.sample(records, n)
    return records


def format_chat(system: str | None, user: str, tokenizer) -> str:
    """Apply chat template to get the full formatted string."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_last_token_hidden(
    text: str,
    model,
    tokenizer,
    layers: list[int],
    device: str,
    max_length: int = 2048,
) -> dict[int, np.ndarray]:
    """Extract hidden state at last token for each specified layer."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    result = {}
    for layer in layers:
        # hidden_states[0] = embedding layer, [1..] = transformer layers
        h = outputs.hidden_states[layer + 1]  # +1 because index 0 is embeddings
        last = h[0, -1, :].float().cpu().numpy()  # last token, float32
        result[layer] = last

    return result


def main():
    ap = argparse.ArgumentParser(description="NES: Extract Gemma-3-27B embeddings per domain")
    ap.add_argument("--model", default="unsloth/gemma-3-27b-it-bnb-4bit")
    ap.add_argument("--n-records", type=int, default=20,
                    help="Records to sample per domain for averaging (default 20)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mid-layer", type=int, default=31,
                    help="Middle layer index (default 31 of 62)")
    ap.add_argument("--out", default="results/spinout/idea_nes/domain_embeddings.npz")
    ap.add_argument("--max-length", type=int, default=2048)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map="auto",
        )
    except Exception as e:
        print(f"ERROR loading model: {e}", file=sys.stderr)
        sys.exit(1)

    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}")

    n_layers = model.config.num_hidden_layers
    print(f"Model layers: {n_layers}")
    final_layer = n_layers - 1
    layers_to_extract = sorted(set([args.mid_layer, final_layer]))
    print(f"Extracting layers: {layers_to_extract}")

    # ── Extract per domain ────────────────────────────────────────────────────
    all_embeddings = {}
    for domain in DOMAINS:
        path = L1_PATHS[domain]
        if not Path(path).exists():
            print(f"WARN: L1 file not found: {path}, skipping {domain}")
            continue

        print(f"\n--- {domain.upper()} ---")
        records = load_records(path, args.n_records, args.seed)
        print(f"  Loaded {len(records)} records")

        prompt_fn = load_prompt_fn(domain)
        layer_hiddens: dict[int, list[np.ndarray]] = {l: [] for l in layers_to_extract}

        for i, rec in enumerate(records):
            system, user = prompt_fn(rec, config="zero-shot")
            text = format_chat(system, user, tokenizer)
            hiddens = extract_last_token_hidden(
                text, model, tokenizer, layers_to_extract, str(device), args.max_length
            )
            for layer, h in hiddens.items():
                layer_hiddens[layer].append(h)
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{len(records)} records", flush=True)

        for layer in layers_to_extract:
            hs = np.stack(layer_hiddens[layer])  # [N, hidden_size]
            mean_h = hs.mean(axis=0)             # [hidden_size]
            tag_mid = "mid" if layer == args.mid_layer else "final"
            all_embeddings[f"{domain}_{tag_mid}"] = mean_h
            all_embeddings[f"{domain}_{tag_mid}_records"] = hs
            print(f"  Layer {layer} ({tag_mid}): shape={hs.shape}, norm={np.linalg.norm(mean_h):.3f}")

    np.savez(out_path, **all_embeddings)
    print(f"\nSaved: {out_path}")
    print(f"Keys: {list(all_embeddings.keys())}")


if __name__ == "__main__":
    main()
