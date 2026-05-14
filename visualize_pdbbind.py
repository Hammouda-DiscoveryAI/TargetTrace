"""
Visualize PDBbind external evaluation — cold-cold model vs latest model.

Panels
------
  A  ROC curve (both models)
  B  Precision-Recall curve (both models)
  C  pIC50 scatter — cold-cold
  D  pIC50 scatter — latest
  E  Metric bar chart comparison

Run:
  python visualize_pdbbind.py
  python visualize_pdbbind.py --out my_figure.png
"""

import argparse
import gc
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

from embedders import ESM_DIM, _PROT_POOLED, _load, _seq_key, embed_proteins, load_residues_subset, unload_models
from evaluate_external import _ExternalDataset
from features import FP_DIM, precompute_fps, load_fp_subset
from trainer import USE_AMP, _collate, device, load_latest

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
)
from scipy.stats import spearmanr, pearsonr

MODELS_DIR = Path("models")

# ── Colour palette ─────────────────────────────────────────────────────────────
_C_COLD   = "#E07B54"   # warm orange  — cold-cold
_C_LATEST = "#4C72B0"   # steel blue   — latest
_C_REF    = "#AAAAAA"   # grey         — random / diagonal


# ── Model loader ───────────────────────────────────────────────────────────────

def _load_model(ckpt_path: Path):
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)
    m    = data["model"].to(device)
    norm = data.get("norm_stats", {"pic50_mean": 7.0, "pic50_std": 1.5})
    m.eval()
    return m, norm


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(model, norm_stats, fp_cache, prot_pool, prot_key_to_idx,
                  prot_res_t, prot_mask_t,
                  pos_df, neg_df, batch_size=256):
    """
    Returns (y_true, y_score, pic50_true, pic50_pred).
    """
    p_mean = norm_stats.get("pic50_mean", 7.0)
    p_std  = norm_stats.get("pic50_std",  1.5)

    pos_ds = _ExternalDataset(pos_df, p_mean, p_std, fp_cache,
                               prot_pool.copy(), prot_key_to_idx)
    pos_loader = DataLoader(pos_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=_collate, num_workers=0,
                            pin_memory=(device.type == "cuda"))

    neg_ds = _ExternalDataset(neg_df, p_mean, p_std, fp_cache,
                               prot_pool.copy(), prot_key_to_idx)
    neg_loader = DataLoader(neg_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=_collate, num_workers=0,
                            pin_memory=(device.type == "cuda"))

    y_true, y_score, pt, pp = [], [], [], []

    model.eval()
    with torch.no_grad():
        for fp, esm_p, prot_idx, pic50s in pos_loader:
            fp, esm_p, prot_idx, pic50s = [b.to(device, non_blocking=True)
                                            for b in [fp, esm_p, prot_idx, pic50s]]
            res  = prot_res_t[prot_idx]
            mask = prot_mask_t[prot_idx]
            with autocast(device_type="cuda", enabled=USE_AMP):
                logit, p50 = model(fp, esm_p, res, mask)
            B = len(fp)
            y_true.extend([1] * B)
            y_score.extend(torch.sigmoid(logit).cpu().float().tolist())
            for t, p in zip(pic50s.cpu().numpy(), p50.cpu().float().numpy()):
                if not np.isnan(float(t)):
                    pt.append(float(t) * p_std + p_mean)
                    pp.append(float(p) * p_std + p_mean)

        for fp, esm_p, prot_idx, _ in neg_loader:
            fp, esm_p, prot_idx = [b.to(device, non_blocking=True)
                                    for b in [fp, esm_p, prot_idx]]
            res  = prot_res_t[prot_idx]
            mask = prot_mask_t[prot_idx]
            with autocast(device_type="cuda", enabled=USE_AMP):
                logit, _ = model(fp, esm_p, res, mask)
            B = len(fp)
            y_true.extend([0] * B)
            y_score.extend(torch.sigmoid(logit).cpu().float().tolist())

    return (np.array(y_true), np.array(y_score),
            np.array(pt, dtype=np.float32), np.array(pp, dtype=np.float32))


# ── Main ───────────────────────────────────────────────────────────────────────

def main(csv_path: str = "pdbbind_eval.csv",
         out_path: str = "pdbbind_evaluation.png"):

    csv_path = Path(csv_path)
    out_path = Path(out_path)

    ckpts     = sorted(MODELS_DIR.glob("*.pkl"))
    ckpt_cold = ckpts[0]
    ckpt_new  = ckpts[-1]
    print(f"Cold-cold : {ckpt_cold.name}")
    print(f"Latest    : {ckpt_new.name}")

    # ── Load & prepare data once ───────────────────────────────────────────────
    print("Loading CSV…")
    df = pd.read_csv(csv_path)
    df = df[df["smiles"].notna() & df["protein_sequence"].notna()].copy()
    df["active"]  = pd.to_numeric(df["active"],  errors="coerce").fillna(0).astype(int)
    df["pic50"]   = pd.to_numeric(df["pic50"],   errors="coerce")
    df["seq_key"] = df["protein_sequence"].astype(str).map(_seq_key)

    pos_df = df[df["active"] == 1].reset_index(drop=True)
    neg_df = df[df["active"] == 0].reset_index(drop=True)

    all_smiles   = df["smiles"].unique().tolist()
    all_seqs     = df["protein_sequence"].unique().tolist()
    unique_keys  = df["seq_key"].unique().tolist()
    prot_key_to_idx = {k: i for i, k in enumerate(unique_keys)}

    print("Computing fingerprints…")
    precompute_fps(all_smiles)

    print("Computing ESM-2 embeddings…")
    embed_proteins(all_seqs)
    unload_models()

    print("Loading caches…")
    fp_cache  = load_fp_subset(set(str(s) for s in all_smiles))
    prot_pool = _load(_PROT_POOLED)
    res_cache = load_residues_subset(set(unique_keys))

    max_L       = max((r.shape[0] for r in res_cache.values()), default=1)
    N_prot      = len(unique_keys)
    prot_res_t  = torch.zeros(N_prot, max_L, ESM_DIM, device=device, dtype=torch.float32)
    prot_mask_t = torch.ones( N_prot, max_L,           device=device, dtype=torch.bool)
    for k, i in prot_key_to_idx.items():
        r = res_cache.get(k)
        if r is not None:
            L = r.shape[0]
            prot_res_t[i, :L]  = torch.tensor(r, dtype=torch.float32)
            prot_mask_t[i, :L] = False
    del res_cache
    gc.collect()

    # ── Run inference for both models ──────────────────────────────────────────
    results = {}
    for label, ckpt, colour in [
        ("Cold-cold", ckpt_cold, _C_COLD),
        ("Latest",    ckpt_new,  _C_LATEST),
    ]:
        print(f"Running inference — {label} ({ckpt.name})…")
        model, norm = _load_model(ckpt)
        y_true, y_score, pt, pp = run_inference(
            model, norm, fp_cache, prot_pool, prot_key_to_idx,
            prot_res_t, prot_mask_t, pos_df, neg_df,
        )
        del model
        torch.cuda.empty_cache()
        gc.collect()

        auc   = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        fpr, tpr, _   = roc_curve(y_true, y_score)
        prec, rec, _  = precision_recall_curve(y_true, y_score)

        spear = float(spearmanr(pt, pp).statistic) if len(pt) >= 2 else 0.0
        pear  = float(pearsonr(pt, pp)[0])          if len(pt) >= 2 else 0.0
        mae   = float(np.mean(np.abs(pt - pp)))     if len(pt) >= 2 else 0.0
        rmse  = float(np.sqrt(np.mean((pt - pp)**2))) if len(pt) >= 2 else 0.0

        results[label] = {
            "colour": colour,
            "y_true": y_true, "y_score": y_score,
            "pt": pt, "pp": pp,
            "fpr": fpr, "tpr": tpr,
            "prec": prec, "rec": rec,
            "AUC": auc, "AUPRC": auprc,
            "Spearman": spear, "Pearson": pear,
            "MAE": mae, "RMSE": rmse,
        }
        print(f"  AUC={auc:.4f}  AUPRC={auprc:.4f}  "
              f"Spearman={spear:.4f}  MAE={mae:.4f}")

    # ── Figure layout ──────────────────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.42, wspace=0.38,
        left=0.07, right=0.97, top=0.91, bottom=0.09,
    )

    ax_roc     = fig.add_subplot(gs[0, 0])
    ax_pr      = fig.add_subplot(gs[0, 1])
    ax_bar     = fig.add_subplot(gs[0, 2])
    ax_scatter_cold   = fig.add_subplot(gs[1, 0])
    ax_scatter_latest = fig.add_subplot(gs[1, 1])
    ax_radar   = fig.add_subplot(gs[1, 2])

    # ── A: ROC curve ──────────────────────────────────────────────────────────
    ax_roc.plot([0, 1], [0, 1], "--", color=_C_REF, lw=1.2, label="Random (0.50)")
    for label, res in results.items():
        # Downsample curve for clean rendering
        step = max(1, len(res["fpr"]) // 300)
        ax_roc.plot(res["fpr"][::step], res["tpr"][::step],
                    color=res["colour"], lw=2.2,
                    label=f"{label}  (AUC = {res['AUC']:.3f})")
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate",  fontsize=11)
    ax_roc.set_title("A  ROC Curve", fontsize=12, fontweight="bold", loc="left")
    ax_roc.legend(fontsize=9.5)
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02)

    # ── B: Precision-Recall curve ──────────────────────────────────────────────
    baseline = results["Latest"]["y_true"].mean()
    ax_pr.axhline(baseline, linestyle="--", color=_C_REF, lw=1.2,
                  label=f"Baseline ({baseline:.2f})")
    for label, res in results.items():
        step = max(1, len(res["rec"]) // 300)
        ax_pr.plot(res["rec"][::step], res["prec"][::step],
                   color=res["colour"], lw=2.2,
                   label=f"{label}  (AUPRC = {res['AUPRC']:.3f})")
    ax_pr.set_xlabel("Recall",    fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title("B  Precision-Recall Curve", fontsize=12, fontweight="bold", loc="left")
    ax_pr.legend(fontsize=9.5)
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)

    # ── C & D: pIC50 scatter ───────────────────────────────────────────────────
    for ax, label, panel in [
        (ax_scatter_cold,   "Cold-cold", "C"),
        (ax_scatter_latest, "Latest",    "D"),
    ]:
        res = results[label]
        pt, pp = res["pt"], res["pp"]
        if len(pt) == 0:
            ax.text(0.5, 0.5, "No pIC50 data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        # 2-D density via hexbin
        hb = ax.hexbin(pt, pp, gridsize=30, cmap="YlOrRd",
                       mincnt=1, linewidths=0.2)
        plt.colorbar(hb, ax=ax, label="Count", pad=0.02)

        lo = min(pt.min(), pp.min()) - 0.3
        hi = max(pt.max(), pp.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], "--", color="#555555", lw=1.4, label="y = x")

        ax.set_xlabel("Experimental pIC50", fontsize=11)
        ax.set_ylabel("Predicted pIC50",    fontsize=11)
        ax.set_title(
            f"{panel}  pIC50 — {label}\n"
            f"Spearman={res['Spearman']:.3f}  "
            f"Pearson={res['Pearson']:.3f}  "
            f"MAE={res['MAE']:.3f}",
            fontsize=10.5, fontweight="bold", loc="left",
        )
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.legend(fontsize=9)

    # ── E: Metric bar chart ────────────────────────────────────────────────────
    metric_keys   = ["AUC", "AUPRC", "Spearman", "Pearson"]
    metric_labels = ["AUC", "AUPRC", "Spearman ρ", "Pearson r"]
    x = np.arange(len(metric_keys))
    w = 0.32

    for i, (label, res) in enumerate(results.items()):
        vals = [res[k] for k in metric_keys]
        bars = ax_bar.bar(x + (i - 0.5) * w, vals, w,
                          color=res["colour"], alpha=0.88,
                          label=label, edgecolor="white", linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels, fontsize=10.5)
    ax_bar.set_ylim(0, 1.12)
    ax_bar.set_ylabel("Score", fontsize=11)
    ax_bar.set_title("E  Classification & Regression Metrics",
                     fontsize=12, fontweight="bold", loc="left")
    ax_bar.legend(fontsize=10)
    ax_bar.axhline(0.5, linestyle=":", color=_C_REF, lw=1.1)

    # ── F: MAE / RMSE grouped bar ──────────────────────────────────────────────
    err_keys   = ["MAE", "RMSE"]
    err_labels = ["MAE (pIC50)", "RMSE (pIC50)"]
    x2 = np.arange(len(err_keys))

    for i, (label, res) in enumerate(results.items()):
        vals = [res[k] for k in err_keys]
        bars = ax_radar.bar(x2 + (i - 0.5) * w, vals, w,
                            color=res["colour"], alpha=0.88,
                            label=label, edgecolor="white", linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax_radar.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + 0.012,
                          f"{v:.3f}", ha="center", va="bottom",
                          fontsize=8.5, fontweight="bold")

    ax_radar.set_xticks(x2)
    ax_radar.set_xticklabels(err_labels, fontsize=10.5)
    ax_radar.set_ylabel("pIC50 units", fontsize=11)
    ax_radar.set_title("F  Regression Error",
                       fontsize=12, fontweight="bold", loc="left")
    ax_radar.legend(fontsize=10)
    ax_radar.set_ylim(0, max(results["Cold-cold"]["RMSE"],
                              results["Latest"]["RMSE"]) * 1.35)

    # ── Title ──────────────────────────────────────────────────────────────────
    fig.suptitle(
        "TargetTrace — External Evaluation on PDBbind\n"
        f"Cold-cold ({ckpt_cold.stem})  vs  Latest ({ckpt_new.stem})",
        fontsize=13, fontweight="bold", y=0.975,
    )

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="pdbbind_eval.csv")
    ap.add_argument("--out", default="pdbbind_evaluation.png")
    args = ap.parse_args()
    main(args.csv, args.out)
