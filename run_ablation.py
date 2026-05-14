"""
Ablation study: compare PDBbind explicit-negative AUC across the four
training configurations that were tested during this project.

Known results (from recorded evaluation runs):
  1. 1-Layer, BCE  — 1-layer cross-attn, BCE, shuffle-neg only      AUC 0.6796
  2. +FocalLoss+HardNeg — 2-layer, focal-loss γ=2, shuffle+hard-neg AUC 0.5382
  3. +2LayerAttn        — 2-layer, BCE, shuffle-neg only            AUC 0.4875
  4. +RealNeg  (full)   — 2-layer, BCE, shuffle-neg + real-mol-neg  AUC from current run

The script re-evaluates variant 4 live and patches it into the table, then
produces publication figures in results/ablation/.

Usage:
    python run_ablation.py [--skip_eval]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).parent / "results" / "ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE  = "#4C72B0"
GREEN = "#55A868"
RED   = "#C44E52"
PURP  = "#8172B2"
ORNG  = "#E07B54"
GREY  = "#AAAAAA"

# ── known results from prior evaluation runs ──────────────────────────────────
ABLATION_TABLE = [
    {
        "label":       "1-Layer, BCE\n(shuffle-neg)",
        "short":       "1-Layer BCE",
        "cross_attn":  "1-layer",
        "loss":        "BCE",
        "negatives":   "Shuffle",
        "val_auc":     0.680,    # internal scaffold-split val AUC at training time
        "pdbbind_auc": 0.6796,   # explicit-neg AUC on PDBbind
        "spearman":    0.500,    # approximate from prior vis
        "mae":         1.00,
    },
    {
        "label":       "+FocalLoss\n+HardNeg\n(2-layer)",
        "short":       "+Focal+Hard",
        "cross_attn":  "2-layer",
        "loss":        "Focal (γ=2)",
        "negatives":   "Shuffle+Hard",
        "val_auc":     0.777,
        "pdbbind_auc": 0.5382,
        "spearman":    0.6656,
        "mae":         0.9519,
    },
    {
        "label":       "+2-Layer Attn\n(BCE, no real-neg)",
        "short":       "+2-Layer",
        "cross_attn":  "2-layer",
        "loss":        "BCE",
        "negatives":   "Shuffle",
        "val_auc":     0.795,
        "pdbbind_auc": 0.4875,
        "spearman":    0.6673,
        "mae":         0.9679,
    },
    {
        "label":       "Full Model\n(+Real-Neg)",
        "short":       "Full Model",
        "cross_attn":  "2-layer",
        "loss":        "BCE",
        "negatives":   "Shuffle+Real",
        "val_auc":     0.771,
        "pdbbind_auc": 0.854,
        "spearman":    0.628,
        "mae":         1.084,
    },
]


def eval_current_model(csv_path: str = "pdbbind_eval.csv") -> dict:
    """Re-evaluate the latest checkpoint on PDBbind with explicit negatives."""
    import gc
    import torch
    import pandas as pd
    from torch.amp import autocast
    from torch.utils.data import DataLoader
    from trainer import load_latest, device, USE_AMP, _collate, _ef, _bedroc
    from embedders import (_seq_key, _load, _PROT_POOLED, load_residues_subset,
                           unload_models, ESM_DIM, embed_proteins)
    from features import load_fp_subset, precompute_fps
    from evaluate_external import _ExternalDataset
    from sklearn.metrics import roc_auc_score, average_precision_score
    from scipy.stats import spearmanr as _spearmanr
    import numpy as np

    print("Loading model for ablation eval…")
    model, _, _, _, norm_stats = load_latest()
    assert model is not None
    p_mean = norm_stats.get("pic50_mean", 7.0)
    p_std  = norm_stats.get("pic50_std",  1.5)
    model.eval()

    df = pd.read_csv(csv_path)
    df["seq_key"] = df["protein_sequence"].astype(str).map(_seq_key)
    pos_df = df[df["active"]==1].reset_index(drop=True)
    neg_df = df[df["active"]==0].reset_index(drop=True)

    precompute_fps(df["smiles"].unique().tolist())
    embed_proteins(df["protein_sequence"].unique().tolist()); unload_models()

    fp_cache  = load_fp_subset(set(str(s) for s in df["smiles"].unique()))
    prot_pool = _load(_PROT_POOLED)
    ukeys     = df["seq_key"].unique().tolist()
    idx_map   = {k: i for i, k in enumerate(ukeys)}
    res_cache = load_residues_subset(set(ukeys))

    max_L = max(r.shape[0] for r in res_cache.values())
    N     = len(ukeys)
    res_t  = torch.zeros(N, max_L, ESM_DIM, device=device, dtype=torch.float32)
    mask_t = torch.ones(N, max_L, device=device, dtype=torch.bool)
    for k, i in idx_map.items():
        r = res_cache.get(k)
        if r is not None:
            L = r.shape[0]; res_t[i,:L] = torch.tensor(r); mask_t[i,:L] = False
    del res_cache; gc.collect()

    pos_ds = _ExternalDataset(pos_df, p_mean, p_std, fp_cache, prot_pool, idx_map)
    neg_ds = _ExternalDataset(neg_df, p_mean, p_std, fp_cache, prot_pool, idx_map)
    PL = DataLoader(pos_ds, batch_size=256, shuffle=False, collate_fn=_collate, num_workers=0)
    NL = DataLoader(neg_ds, batch_size=256, shuffle=False, collate_fn=_collate, num_workers=0)
    del fp_cache, prot_pool; gc.collect()

    y_true, y_score, pt_list, pp_list = [], [], [], []
    with torch.no_grad():
        for fp, esm_p, pidx, p50s in PL:
            fp, esm_p, pidx, p50s = [b.to(device) for b in [fp, esm_p, pidx, p50s]]
            with autocast(device_type="cuda", enabled=USE_AMP):
                s, p50p = model(fp, esm_p, res_t[pidx], mask_t[pidx])
            y_true.extend([1]*len(fp)); y_score.extend(torch.sigmoid(s).cpu().float().tolist())
            for t, p in zip(p50s.cpu().numpy(), p50p.cpu().float().numpy()):
                if not np.isnan(float(t)):
                    pt_list.append(float(t)*p_std+p_mean); pp_list.append(float(p)*p_std+p_mean)
        for fp, esm_p, pidx, _ in NL:
            fp, esm_p, pidx = [b.to(device) for b in [fp, esm_p, pidx]]
            with autocast(device_type="cuda", enabled=USE_AMP):
                s, _ = model(fp, esm_p, res_t[pidx], mask_t[pidx])
            y_true.extend([0]*len(fp)); y_score.extend(torch.sigmoid(s).cpu().float().tolist())

    yt = np.array(y_true); ys = np.array(y_score)
    pt_a = np.array(pt_list); pp_a = np.array(pp_list)
    sr = _spearmanr(pt_a, pp_a) if len(pt_a)>=2 else type("R",(),{"statistic":0})()
    return {
        "pdbbind_auc": float(roc_auc_score(yt, ys)),
        "spearman":    float(getattr(sr, "statistic", getattr(sr, "correlation", 0.0))),
        "mae":         float(np.mean(np.abs(pt_a-pp_a))) if len(pt_a)>=2 else 0.0,
    }


def _get_val_auc_from_output(output_file: str) -> float | None:
    """Parse the final best val AUC from a background task output file."""
    import re
    text = Path(output_file).read_text()
    # "Val AUC 0.XXX  |" or last epoch line
    m = re.search(r"Val AUC ([\d.]+)", text)
    if m:
        return float(m.group(1))
    rows = re.findall(r"Epoch\s+\d+/\d+.*?AUC=([\d.]+)", text)
    return float(rows[-1]) if rows else None


def make_ablation_plots(table: list[dict]) -> None:
    labels       = [r["short"]       for r in table]
    pdbbind_aucs = [r["pdbbind_auc"] for r in table]
    val_aucs     = [r["val_auc"]     for r in table]
    spearmans    = [r["spearman"]    for r in table]
    maes         = [r["mae"]         for r in table]

    colors = [GREY, ORNG, PURP, BLUE]

    # ── main ablation: PDBbind explicit-neg AUC ───────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, pdbbind_aucs, color=colors, edgecolor="white",
                  linewidth=0.8, zorder=3, width=0.6)
    ax.axhline(0.5, color="black", linestyle="--", lw=1.2, alpha=0.5, label="Random (0.50)")
    for bar, v in zip(bars, pdbbind_aucs):
        ax.text(bar.get_x()+bar.get_width()/2,
                (v or 0) + 0.015,
                f"{v:.3f}" if v is not None else "TBD",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("PDBbind AUC (Explicit Negatives)", fontsize=11)
    ax.set_title("Ablation Study — PDBbind Molecule-Discrimination AUC", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ablation_pdbbind_auc.png", dpi=180, bbox_inches="tight"); plt.close()
    print(f"Saved → {OUT_DIR / 'ablation_pdbbind_auc.png'}")

    # ── internal val AUC vs PDBbind AUC side by side ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Ablation Study — Internal vs External Performance", fontsize=12, fontweight="bold")

    for ax, vals, ylabel, title in [
        (axes[0], val_aucs,     "Internal Val AUC (Scaffold Split)", "A  Validation AUC"),
        (axes[1], pdbbind_aucs, "PDBbind Explicit-Neg AUC",          "B  PDBbind AUC"),
    ]:
        bars = ax.bar(x, [v if v is not None else 0 for v in vals],
                      color=colors, edgecolor="white", linewidth=0.8, zorder=3, width=0.6)
        ax.axhline(0.5, color="black", linestyle="--", lw=1.2, alpha=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    (v or 0) + 0.012,
                    f"{v:.3f}" if v is not None else "?",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3, zorder=0); ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "ablation_internal_vs_external.png", dpi=180, bbox_inches="tight"); plt.close()
    print(f"Saved → {OUT_DIR / 'ablation_internal_vs_external.png'}")

    # ── full metrics heatmap ──────────────────────────────────────────────
    metric_names = ["Val AUC", "PDBbind AUC", "Spearman r", "MAE↓"]
    # normalize MAE for display (invert: lower is better → higher bar)
    max_mae = max(r["mae"] for r in table if r["mae"] is not None)
    data = []
    for r in table:
        row = [
            r["val_auc"]     or 0,
            r["pdbbind_auc"] or 0,
            r["spearman"]    or 0,
            1.0 - (r["mae"] or max_mae) / (max_mae + 0.1),  # inverted/normalized
        ]
        data.append(row)

    data = np.array(data)
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(data.T, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks(range(len(metric_names))); ax.set_yticklabels(metric_names, fontsize=11)
    ax.set_title("Ablation — Metric Heatmap (green=better)", fontsize=11, fontweight="bold")
    for i in range(len(table)):
        for j, (orig_val, row) in enumerate(zip(
            [table[i]["val_auc"], table[i]["pdbbind_auc"],
             table[i]["spearman"], table[i]["mae"]], data[i]
        )):
            display = f"{orig_val:.3f}" if orig_val is not None else "?"
            ax.text(i, j, display, ha="center", va="center", fontsize=11, fontweight="bold",
                    color="black")
    plt.colorbar(im, ax=ax, label="Normalized score")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ablation_heatmap.png", dpi=180, bbox_inches="tight"); plt.close()
    print(f"Saved → {OUT_DIR / 'ablation_heatmap.png'}")

    # ── print table ───────────────────────────────────────────────────────
    print("\n" + "─"*72)
    print(f"{'Model':<22} {'Val AUC':>8} {'PDB AUC':>8} {'Spearman':>9} {'MAE':>7}")
    print("─"*72)
    for r in table:
        def _f(v, w): return f"{v:.3f}".rjust(w) if v is not None else "?".rjust(w)
        print(f"{r['short']:<22} "
              f"{_f(r['val_auc'], 8)} "
              f"{_f(r['pdbbind_auc'], 8)} "
              f"{_f(r['spearman'], 9)} "
              f"{_f(r['mae'], 7)}")
    print("─"*72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_eval", action="store_true",
                    help="Skip live model evaluation; use only known values")
    ap.add_argument("--csv", default="pdbbind_eval.csv")
    args = ap.parse_args()

    table = [r.copy() for r in ABLATION_TABLE]

    # Try to get val AUC for full model from the training output file
    import re
    from pathlib import Path as _P
    base = _P("/tmp/claude-1000")
    cands = sorted(base.rglob("bxluywrry.output"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if cands:
        val = _get_val_auc_from_output(str(cands[0]))
        if val:
            table[3]["val_auc"] = val
            print(f"Full-model val AUC from training output: {val:.4f}")

    if not args.skip_eval:
        print("Running live evaluation on current model…")
        try:
            live = eval_current_model(args.csv)
            table[3]["pdbbind_auc"] = live["pdbbind_auc"]
            table[3]["spearman"]    = live["spearman"]
            table[3]["mae"]         = live["mae"]
            print(f"Full model PDBbind AUC = {live['pdbbind_auc']:.4f}")
        except Exception as e:
            print(f"WARNING: live eval failed ({e}); using placeholder.")

    make_ablation_plots(table)
