#!/usr/bin/env python3
"""
Correlation Analysis: Output Distance vs KL Divergence

Tests whether minimizing L2 distance between full and sparse attention outputs
correlates with minimizing KL(p_sparse || p_full) between their distributions.

Usage:
    python experiments/correlation_analysis.py [--model meta-llama/Llama-3.2-1B] [--num_sequences 100]
"""

import os
import sys
import math
import argparse
import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def _import_parallel_nsa():
    """Import parallel_nsa, working around fla / transformers version conflicts."""
    # Fix 1: fla registers model configs (e.g. 'bitnet') that already exist in
    # transformers>=4.54, causing ValueError on re-registration.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    _Cls = CONFIG_MAPPING.__class__
    _orig_register = _Cls.register
    _Cls.register = lambda self, key, value, exist_ok=False: _orig_register(self, key, value, exist_ok=True)

    try:
        # Fix 2: fla v0.2.0 lacks some utility functions that the patched
        # native_sparse_attention/ops/parallel.py imports at module level.
        # They are only called when cu_seqlens is not None (variable-length),
        # which we never use, so None stubs are safe.
        import fla.ops.utils as _fla_utils
        for _name in ("prepare_chunk_indices", "prepare_chunk_offsets",
                       "prepare_lens", "prepare_token_indices"):
            if not hasattr(_fla_utils, _name):
                setattr(_fla_utils, _name, None)

        from native_sparse_attention.ops.parallel import parallel_nsa as _fn
        return _fn
    finally:
        _Cls.register = _orig_register


parallel_nsa = _import_parallel_nsa()

# ── Constants ─────────────────────────────────────────────────────────────────
BLOCK_SIZE = 16
BLOCK_COUNTS = 16
SEQ_LEN = 1024
NUM_SEQUENCES = 100
NUM_SAMPLED = 64
MIN_POS = BLOCK_SIZE * BLOCK_COUNTS  # 256: ensure enough blocks for full selection


# ── RoPE helpers (self-contained, no dependency on transformers internals) ────
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """q, k: [B, H, T, D].  cos, sin: [B, T, D]."""
    cos = cos.unsqueeze(1)  # [B, 1, T, D]
    sin = sin.unsqueeze(1)
    return (
        (q * cos) + (rotate_half(q) * sin),
        (k * cos) + (rotate_half(k) * sin),
    )


# ── Data loading ──────────────────────────────────────────────────────────────
def load_sequences(tokenizer, num_sequences):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in dataset["text"] if t.strip())
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    n = len(tokens) // SEQ_LEN
    tokens = tokens[: n * SEQ_LEN].reshape(n, SEQ_LEN)
    if tokens.shape[0] < num_sequences:
        print(f"Warning: only {tokens.shape[0]} sequences available")
    return tokens[: min(num_sequences, tokens.shape[0])]


# ── Per-layer computation ─────────────────────────────────────────────────────
def process_layer(hidden_states, layer, cos, sin, sampled_pos, device,
                  num_heads, num_kv_heads, head_dim, causal_mask, block_offsets):
    """
    Run one decoder layer:
      - compute Q, K, V + RoPE
      - full attention (manual eager)
      - sparse attention (parallel_nsa)
      - metrics at sampled positions
      - propagate hidden_states for next layer

    Returns:
        hidden_states: [B, T, D] for next layer
        d_output: [N, H] output L2 distances
        d_kl:     [N, H] KL(p_sparse || p_full)
    """
    attn = layer.self_attn
    B, T, D = hidden_states.shape
    groups = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    N = len(sampled_pos)
    S = BLOCK_COUNTS * BLOCK_SIZE  # selected tokens per query

    # ── 1. Pre-norm + project Q, K, V ──
    normed = layer.input_layernorm(hidden_states)
    q = attn.q_proj(normed).view(B, T, num_heads, head_dim).transpose(1, 2)      # [B, H, T, D]
    k = attn.k_proj(normed).view(B, T, num_kv_heads, head_dim).transpose(1, 2)   # [B, Hkv, T, D]
    v = attn.v_proj(normed).view(B, T, num_kv_heads, head_dim)                    # [B, T, Hkv, D]

    # ── 2. Apply RoPE ──
    q, k = apply_rotary_pos_emb(q, k, cos, sin)  # still [B, H, T, D] / [B, Hkv, T, D]

    # ── 3. Full attention ──
    k_exp = k.repeat_interleave(groups, dim=1)                      # [B, H, T, D]
    v_exp = v.transpose(1, 2).repeat_interleave(groups, dim=1)      # [B, H, T, D]

    scores_full = (q @ k_exp.transpose(-2, -1)) * scale             # [B, H, T, T]
    scores_full = scores_full + causal_mask
    attn_w = torch.softmax(scores_full, dim=-1, dtype=torch.float32)  # [B, H, T, T]
    out_full = (attn_w @ v_exp.float()).to(hidden_states.dtype)       # [B, H, T, D]

    # ── 4. Sparse attention ──
    q_nsa = q.transpose(1, 2)   # [B, T, H, D]
    k_nsa = k.transpose(1, 2)   # [B, T, Hkv, D]

    g_slc = torch.ones(B, T, num_heads, device=device, dtype=hidden_states.dtype)
    out_sparse, block_idx = parallel_nsa(
        q=q_nsa, k=k_nsa, v=v,
        g_cmp=0, g_slc=g_slc, g_swa=0,
        block_size=BLOCK_SIZE, block_counts=BLOCK_COUNTS,
        window_size=0, head_first=False,
    )
    # out_sparse: [B, T, H, D],  block_idx: [B, T, Hkv, BLOCK_COUNTS]

    # ── 5. Metrics at sampled positions ──
    # Output distance
    of = out_full[0, :, sampled_pos, :].transpose(0, 1).float()  # [N, H, D]
    sp = out_sparse[0, sampled_pos, :, :].float()                 # [N, H, D]
    d_output = torch.norm(of - sp, dim=-1)                        # [N, H]

    # Reconstruct sparse attention distribution
    blocks = block_idx[0, sampled_pos, :, :]                      # [N, Hkv, BC]
    # Expand blocks to token indices
    tok_idx = (blocks.unsqueeze(-1) * BLOCK_SIZE + block_offsets)  # [N, Hkv, BC, BS]
    tok_idx = tok_idx.reshape(N, num_kv_heads, S)                 # [N, Hkv, S]
    tok_gqa = tok_idx.repeat_interleave(groups, dim=1)            # [N, H, S]
    tok_clamped = tok_gqa.clamp(0, T - 1)

    # Mask: causal + out-of-bounds + invalid (-1 from unfilled topk slots)
    qp = sampled_pos.view(-1, 1, 1)  # [N, 1, 1]
    invalid = (tok_gqa >= qp) | (tok_gqa >= T) | (tok_gqa < 0)

    # Gather K at selected positions
    k_all = k_nsa[0].repeat_interleave(groups, dim=1)            # [T, H, D]
    h_idx = torch.arange(num_heads, device=device).view(1, -1, 1).expand(N, -1, S)
    k_sel = k_all[tok_clamped, h_idx, :]                         # [N, H, S, D]

    # Sparse scores -> distribution
    q_sam = q_nsa[0, sampled_pos, :, :].float()                  # [N, H, D]
    sp_scores = torch.einsum("nhd,nhsd->nhs", q_sam, k_sel.float()) * scale
    sp_scores.masked_fill_(invalid, float("-inf"))
    p_sparse = torch.softmax(sp_scores, dim=-1)                  # [N, H, S]

    # Gather full attention weights at the same positions
    aw_sam = attn_w[0, :, sampled_pos, :].transpose(0, 1)       # [N, H, T]
    p_full_sel = torch.gather(aw_sam, 2, tok_clamped)            # [N, H, S]
    p_full_sel.masked_fill_(invalid, 0.0)

    # KL(p_sparse || p_full)
    valid = p_sparse > 0
    kl_terms = torch.where(
        valid,
        p_sparse * torch.log(p_sparse / (p_full_sel + 1e-10)),
        torch.zeros_like(p_sparse),
    )
    d_kl = kl_terms.sum(dim=-1)  # [N, H]

    # ── 6. Continue forward pass (full attention path) ──
    attn_out = out_full.transpose(1, 2).contiguous().reshape(B, T, -1)
    attn_out = attn.o_proj(attn_out)
    hidden_states = hidden_states + attn_out
    hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))

    return hidden_states, d_output.cpu().numpy(), d_kl.cpu().numpy()


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_per_layer(corr_per_layer, head_corrs, num_layers, output_dir):
    """Line plot: per-layer Spearman rho with error bars over heads."""
    means = np.array([corr_per_layer[l]["spearman"] for l in range(num_layers)])
    stds = np.array([np.std(head_corrs[l]) for l in range(num_layers)])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(range(num_layers), means, yerr=stds, marker="o", capsize=4, linewidth=2)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel(r"Spearman $\rho$ (output dist. vs KL div.)")
    ax.set_title("Per-Layer Correlation: Output Distance vs KL Divergence")
    ax.set_xticks(range(num_layers))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_layer_correlation.pdf"))
    plt.close(fig)


def plot_scatter(d_output_all, d_kl_all, corr_per_layer, num_layers, output_dir):
    """Scatter plots for 4 selected layers."""
    rhos = [corr_per_layer[l]["spearman"] for l in range(num_layers)]
    selected = sorted(set([
        0, num_layers // 2, num_layers - 1, int(np.argmax(rhos))
    ]))[:4]

    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4.5))
    if len(selected) == 1:
        axes = [axes]

    for ax, li in zip(axes, selected):
        x = d_output_all[li].flatten()
        y = d_kl_all[li].flatten()
        if len(x) > 2000:
            idx = np.random.RandomState(42).choice(len(x), 2000, replace=False)
            x, y = x[idx], y[idx]
        ax.scatter(x, y, alpha=0.15, s=3)
        ax.set_xlabel("Output Distance (L2)")
        ax.set_ylabel(r"$\mathrm{KL}(p_\mathrm{sparse}\|p_\mathrm{full})$")
        ax.set_title(f"Layer {li} ($\\rho$={rhos[li]:.3f})")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scatter_plots.pdf"))
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Output-distance vs KL-divergence correlation")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--num_sequences", type=int, default=NUM_SEQUENCES)
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "output"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda")

    # ── Load model ──
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // num_heads)
    print(f"  {num_layers} layers, {num_heads} Q heads, {num_kv_heads} KV heads, d={head_dim}")

    # ── Load data ──
    print("Loading WikiText-2...")
    sequences = load_sequences(tokenizer, args.num_sequences)
    num_seq = sequences.shape[0]
    print(f"  {num_seq} sequences of length {SEQ_LEN}")

    # ── Pre-allocate results ──
    d_output_all = np.zeros((num_layers, num_seq, NUM_SAMPLED, num_heads), dtype=np.float32)
    d_kl_all = np.zeros((num_layers, num_seq, NUM_SAMPLED, num_heads), dtype=np.float32)

    # ── Pre-compute reusable tensors ──
    causal_mask = torch.triu(
        torch.full((SEQ_LEN, SEQ_LEN), float("-inf"), device=device), diagonal=1
    )
    block_offsets = torch.arange(BLOCK_SIZE, device=device)

    # ── Main loop ──
    for seq_idx in tqdm(range(num_seq), desc="Sequences"):
        input_ids = sequences[seq_idx].unsqueeze(0).to(device)

        sampled_pos = torch.from_numpy(
            np.sort(np.random.choice(range(MIN_POS, SEQ_LEN), NUM_SAMPLED, replace=False))
        ).to(device)

        with torch.no_grad():
            hidden_states = model.model.embed_tokens(input_ids)
            position_ids = torch.arange(SEQ_LEN, device=device).unsqueeze(0)
            cos, sin = model.model.rotary_emb(hidden_states, position_ids)

            for li in range(num_layers):
                hidden_states, d_out, d_kl = process_layer(
                    hidden_states, model.model.layers[li],
                    cos, sin, sampled_pos, device,
                    num_heads, num_kv_heads, head_dim,
                    causal_mask, block_offsets,
                )
                d_output_all[li, seq_idx] = d_out
                d_kl_all[li, seq_idx] = d_kl

        # periodic checkpoint
        if (seq_idx + 1) % 10 == 0:
            np.savez(
                os.path.join(args.output_dir, "partial_results.npz"),
                d_output=d_output_all, d_kl=d_kl_all, completed=seq_idx + 1,
            )

    # ── Correlations ──
    print("\n" + "=" * 70)
    print(f"{'Layer':>5} {'Spearman':>10} {'Pearson':>10} {'Head std':>10}")
    print("-" * 70)

    corr_per_layer = {}
    head_corrs = {}

    for li in range(num_layers):
        x = d_output_all[li].flatten()
        y = d_kl_all[li].flatten()
        rho, _ = spearmanr(x, y)
        r, _ = pearsonr(x, y)

        # per-head
        hr = []
        for h in range(num_heads):
            xh = d_output_all[li, :, :, h].flatten()
            yh = d_kl_all[li, :, :, h].flatten()
            rh, _ = spearmanr(xh, yh)
            hr.append(rh)

        corr_per_layer[li] = {"spearman": rho, "pearson": r}
        head_corrs[li] = np.array(hr)
        print(f"{li:5d} {rho:10.4f} {r:10.4f} {np.std(hr):10.4f}")

    # ── Save CSVs ──
    import pandas as pd

    # Correlation summary (per layer-head)
    rows = []
    for li in range(num_layers):
        for h in range(num_heads):
            xh = d_output_all[li, :, :, h].flatten()
            yh = d_kl_all[li, :, :, h].flatten()
            rho_h, _ = spearmanr(xh, yh)
            r_h, _ = pearsonr(xh, yh)
            rows.append({"layer": li, "head": h, "spearman_rho": rho_h, "pearson_r": r_h})
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "correlation_summary.csv"), index=False)

    # Full raw data — built vectorized from numpy arrays
    layer_idx = np.repeat(np.arange(num_layers), num_seq * NUM_SAMPLED * num_heads)
    seq_idx = np.tile(np.repeat(np.arange(num_seq), NUM_SAMPLED * num_heads), num_layers)
    sample_idx = np.tile(np.repeat(np.arange(NUM_SAMPLED), num_heads), num_layers * num_seq)
    head_idx = np.tile(np.arange(num_heads), num_layers * num_seq * NUM_SAMPLED)
    pd.DataFrame({
        "layer": layer_idx, "head": head_idx, "seq_idx": seq_idx, "sample_idx": sample_idx,
        "d_output": d_output_all.flatten(), "d_kl": d_kl_all.flatten(),
    }).to_csv(os.path.join(args.output_dir, "raw_data.csv"), index=False)

    # Also save numpy for fast reloading
    np.savez(
        os.path.join(args.output_dir, "results.npz"),
        d_output=d_output_all, d_kl=d_kl_all,
    )

    # ── Plots ──
    plot_per_layer(corr_per_layer, head_corrs, num_layers, args.output_dir)
    plot_scatter(d_output_all, d_kl_all, corr_per_layer, num_layers, args.output_dir)

    # ── Summary ──
    rhos = [corr_per_layer[l]["spearman"] for l in range(num_layers)]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"  Mean Spearman rho: {np.mean(rhos):.4f} +/- {np.std(rhos):.4f}")
    print(f"  Min:  layer {np.argmin(rhos)} = {np.min(rhos):.4f}")
    print(f"  Max:  layer {np.argmax(rhos)} = {np.max(rhos):.4f}")
    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
