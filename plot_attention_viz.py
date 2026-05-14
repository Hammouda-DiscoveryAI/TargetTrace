"""
Attention weight visualisation: for a handful of well-characterised drug-target pairs
in PDBbind, plot cross-attention weights over residue positions and overlay known
binding-site residue indices where available.
Produces: results/supplementary/attention_viz.png
"""
import gc, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent))
from trainer   import load_latest, device, USE_AMP
from embedders import (_seq_key, _load, _PROT_POOLED, load_residues_subset,
                       unload_models, ESM_DIM, embed_proteins)
from features  import load_fp_subset, precompute_fps

OUT = Path(__file__).parent / "results" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

BLUE  = "#4C72B0"
GREEN = "#55A868"
RED   = "#C44E52"

# Well-characterised pairs from PDBbind, with approximate binding-site residue
# indices (0-based in the full-length training sequence) for annotation.
# These are approximate and drawn from literature/UniProt active-site annotations.
PAIRS_OF_INTEREST = [
    {
        "pdb_id":  "6sbh",
        "label":   "CA-II / sulfonamide\n(6SBH)",
        "protein": "Carbonic anhydrase 2",
        "site_hint": "Active site: H94, H96, H119, T199",
    },
    {
        "pdb_id":  "5kj2",
        "label":   "EP300 / inhibitor\n(5KJ2)",
        "protein": "Histone acetyltransferase p300",
        "site_hint": "Lys-acetyl site: C1438, W1436",
    },
]


def get_attention_weights(model, fp_t, esm_p_t, res_t, mask_t):
    """Run a single forward pass and return per-head attention from both layers."""
    from model import _MolProtCrossAttn

    # Monkey-patch to capture attn weights
    attn_out = {}

    orig_fwd1 = model.cross_attn.forward
    orig_fwd2 = model.cross_attn_2.forward

    def fwd1(mol, residues, pad_mask=None, return_attn=False):
        out, attn = orig_fwd1(mol, residues, pad_mask, return_attn=True)
        attn_out["layer1"] = attn.detach().cpu()  # (B, H, L)
        return out

    def fwd2(mol, residues, pad_mask=None, return_attn=False):
        out, attn = orig_fwd2(mol, residues, pad_mask, return_attn=True)
        attn_out["layer2"] = attn.detach().cpu()
        return out

    model.cross_attn.forward   = fwd1
    model.cross_attn_2.forward = fwd2

    with torch.no_grad():
        with autocast(device_type="cuda", enabled=USE_AMP):
            model(fp_t, esm_p_t, res_t, mask_t)

    # Restore
    model.cross_attn.forward   = orig_fwd1
    model.cross_attn_2.forward = orig_fwd2

    return attn_out.get("layer1"), attn_out.get("layer2")


def run_attention_viz(scores_csv: str = "results/pdbbind_scores.csv",
                      csv_path: str   = "pdbbind_eval.csv") -> None:
    df_scores = pd.read_csv(scores_csv)
    df_full   = pd.read_csv(csv_path)

    print("Loading model…")
    model, _, _, _, norm_stats = load_latest()
    assert model is not None
    p_mean = norm_stats.get("pic50_mean", 7.0)
    p_std  = norm_stats.get("pic50_std",  1.5)
    model.eval()

    # Pick representative pairs: highest-scoring active per target in the interest list
    pairs = []
    for poi in PAIRS_OF_INTEREST:
        sub = df_full[(df_full["pdb_id"] == poi["pdb_id"]) & (df_full["active"] == 1)]
        if len(sub) == 0:
            sub = df_full[(df_full["target_name"].str.contains(poi["protein"],
                           case=False, na=False)) & (df_full["active"] == 1)].head(1)
        if len(sub) == 0:
            print(f"  Skipping {poi['label']} — not found in PDBbind CSV")
            continue
        row = sub.iloc[0]
        pairs.append({**poi, "smiles": row["smiles"],
                      "protein_sequence": row["protein_sequence"],
                      "pic50": row["pic50"]})

    if not pairs:
        print("No matching pairs found; skipping attention visualisation.")
        return

    # Also add the top 2 highest-scoring actives from the full dataset
    top_rows = (df_scores[df_scores["active"] == 1]
                .sort_values("score", ascending=False)
                .drop_duplicates("uniprot")
                .head(2))
    for _, r in top_rows.iterrows():
        if any(r["pdb_id"] == p.get("pdb_id", "") for p in pairs):
            continue
        full_row = df_full[df_full["pdb_id"] == r["pdb_id"]]
        if len(full_row) == 0:
            continue
        pairs.append({
            "label": f"{r['target_name'][:25]}\n({r['pdb_id'].upper()})",
            "site_hint": "", "smiles": full_row.iloc[0]["smiles"],
            "protein_sequence": full_row.iloc[0]["protein_sequence"],
            "pic50": r.get("pic50", np.nan),
        })

    # Embed proteins
    seqs = [p["protein_sequence"] for p in pairs]
    smis = [p["smiles"] for p in pairs]
    precompute_fps(smis)
    embed_proteins(seqs); unload_models()

    fp_cache  = load_fp_subset(set(smis))
    prot_pool = _load(_PROT_POOLED)
    ukeys     = [_seq_key(s) for s in seqs]
    res_cache = load_residues_subset(set(ukeys))

    max_L = max(r.shape[0] for r in res_cache.values())
    N     = len(seqs)
    res_t  = torch.zeros(N, max_L, ESM_DIM, device=device, dtype=torch.float32)
    mask_t = torch.ones(N, max_L, device=device, dtype=torch.bool)
    for i, k in enumerate(ukeys):
        r = res_cache.get(k)
        if r is not None:
            L = r.shape[0]; res_t[i, :L] = torch.tensor(r); mask_t[i, :L] = False
    del res_cache; gc.collect()

    # Build batch tensors for all pairs
    from features import FP_DIM
    fp_batch   = torch.zeros(N, FP_DIM, device=device)
    esm_batch  = torch.zeros(N, ESM_DIM, device=device)
    for i, (smi, seq) in enumerate(zip(smis, seqs)):
        fp_batch[i]  = torch.tensor(fp_cache[smi], device=device)
        k = _seq_key(seq)
        if k in prot_pool:
            esm_batch[i] = torch.tensor(prot_pool[k], device=device)

    attn1, attn2 = get_attention_weights(model, fp_batch, esm_batch, res_t, mask_t)
    del fp_cache, prot_pool; gc.collect()

    # ── Figure ────────────────────────────────────────────────────────────
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(15, 4 * n_pairs))
    if n_pairs == 1:
        axes = axes[None, :]
    fig.suptitle("Cross-Attention Weights over Residue Positions",
                 fontsize=13, fontweight="bold")

    for i, (pair, ax_row) in enumerate(zip(pairs, axes)):
        seq_len = int((~mask_t[i]).sum().item())  # non-padding residues
        for j, (ax, attn, layer_name) in enumerate(
                zip(ax_row,
                    [attn1[i].mean(0)[:seq_len].numpy(),   # mean over heads, layer 1
                     attn2[i].mean(0)[:seq_len].numpy()],  # layer 2
                    ["Layer 1", "Layer 2"])):
            positions = np.arange(seq_len)
            # Smoothed for readability
            window = max(1, seq_len // 80)
            smoothed = np.convolve(attn, np.ones(window) / window, mode="same")
            ax.fill_between(positions, smoothed, alpha=0.4, color=BLUE)
            ax.plot(positions, smoothed, color=BLUE, lw=1.0)
            ax.set_xlim(0, seq_len)
            ax.set_ylabel("Attention weight\n(mean over heads)", fontsize=8)
            ax.set_xlabel("Residue position", fontsize=8)
            title = pair["label"] if j == 0 else ""
            site = pair.get("site_hint", "")
            ax.set_title(f"{title}  [{layer_name}]\n{site}", fontsize=8.5,
                         fontweight="bold" if j == 0 else "normal")
            ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)
            # Mark top-5 attended positions
            top5 = np.argsort(smoothed)[-5:]
            ax.scatter(top5, smoothed[top5], color=RED, s=30, zorder=4,
                       label="Top-5 positions")
            if j == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / "attention_viz.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUT / 'attention_viz.png'}")


if __name__ == "__main__":
    run_attention_viz()
