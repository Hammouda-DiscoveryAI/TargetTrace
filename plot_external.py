"""
Run the latest model on the PDBbind test set using EXPLICIT negatives
(measured inactive compounds) and produce ROC, PR, scatter and metric
figures in results/external/.

This tests molecule-level discrimination: active vs measured-inactive
compounds against their actual target proteins.

Usage:
    python plot_external.py [pdbbind_eval.csv]
"""
import gc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score, precision_recall_curve,
    roc_auc_score, roc_curve,
)
from scipy.stats import pearsonr as _pearsonr, spearmanr as _spearmanr
from torch.amp import autocast
from torch.utils.data import DataLoader

OUT_DIR = Path(__file__).parent / "results" / "external"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE  = "#4C72B0"
GREEN = "#55A868"
RED   = "#C44E52"
PURP  = "#8172B2"
ORNG  = "#E07B54"


def run_explicit_eval(csv_path: str = "pdbbind_eval.csv") -> dict:
    from trainer import load_latest, device, USE_AMP, _collate, _ef, _bedroc
    from embedders import _seq_key, _load, _PROT_POOLED, load_residues_subset, unload_models, ESM_DIM
    from features import load_fp_subset, precompute_fps
    from evaluate_external import _ExternalDataset
    from embedders import embed_proteins

    print("Loading model…")
    model, _, _, _, norm_stats = load_latest()
    assert model is not None, "No trained model found."
    p_mean = norm_stats.get("pic50_mean", 7.0)
    p_std  = norm_stats.get("pic50_std",  1.5)
    model.eval()

    print(f"Reading {csv_path}…")
    df = pd.read_csv(csv_path)
    df["seq_key"] = df["protein_sequence"].astype(str).map(_seq_key)
    df["active"]  = df["active"].astype(int)
    if "pic50" not in df.columns:
        df["pic50"] = np.nan

    pos_df = df[df["active"] == 1].reset_index(drop=True)
    neg_df = df[df["active"] == 0].reset_index(drop=True)
    print(f"  {len(pos_df)} positives | {len(neg_df)} explicit negatives")

    all_smiles = df["smiles"].unique().tolist()
    all_seqs   = df["protein_sequence"].unique().tolist()
    precompute_fps(all_smiles)
    embed_proteins(all_seqs); unload_models()

    fp_cache  = load_fp_subset(set(str(s) for s in all_smiles))
    prot_pool = _load(_PROT_POOLED)
    unique_keys     = df["seq_key"].unique().tolist()
    prot_key_to_idx = {k: i for i, k in enumerate(unique_keys)}
    residue_cache   = load_residues_subset(set(unique_keys))

    max_L  = max(r.shape[0] for r in residue_cache.values())
    N      = len(unique_keys)
    res_t  = torch.zeros(N, max_L, ESM_DIM, device=device, dtype=torch.float32)
    mask_t = torch.ones(N, max_L, device=device, dtype=torch.bool)
    for k, i in prot_key_to_idx.items():
        r = residue_cache.get(k)
        if r is not None:
            L = r.shape[0]; res_t[i,:L] = torch.tensor(r); mask_t[i,:L] = False
    del residue_cache; gc.collect()

    pos_ds = _ExternalDataset(pos_df, p_mean, p_std, fp_cache, prot_pool, prot_key_to_idx)
    neg_ds = _ExternalDataset(neg_df, p_mean, p_std, fp_cache, prot_pool, prot_key_to_idx)
    pos_loader = DataLoader(pos_ds, batch_size=256, shuffle=False, collate_fn=_collate, num_workers=0)
    neg_loader = DataLoader(neg_ds, batch_size=256, shuffle=False, collate_fn=_collate, num_workers=0)
    del fp_cache, prot_pool; gc.collect()

    y_true, y_score, pt_list, pp_list = [], [], [], []
    with torch.no_grad():
        for fp, esm_p, prot_idx, pic50s in pos_loader:
            fp, esm_p, prot_idx, pic50s = [b.to(device) for b in [fp, esm_p, prot_idx, pic50s]]
            r = res_t[prot_idx]; m = mask_t[prot_idx]
            with autocast(device_type="cuda", enabled=USE_AMP):
                s, p50 = model(fp, esm_p, r, m)
            y_true.extend([1]*len(fp))
            y_score.extend(torch.sigmoid(s).cpu().float().tolist())
            for t, p in zip(pic50s.cpu().numpy(), p50.cpu().float().numpy()):
                if not np.isnan(float(t)):
                    pt_list.append(float(t)*p_std + p_mean)
                    pp_list.append(float(p)*p_std + p_mean)

        for fp, esm_p, prot_idx, _ in neg_loader:
            fp, esm_p, prot_idx = [b.to(device) for b in [fp, esm_p, prot_idx]]
            r = res_t[prot_idx]; m = mask_t[prot_idx]
            with autocast(device_type="cuda", enabled=USE_AMP):
                s, _ = model(fp, esm_p, r, m)
            y_true.extend([0]*len(fp))
            y_score.extend(torch.sigmoid(s).cpu().float().tolist())

    y_true_a  = np.array(y_true)
    y_score_a = np.array(y_score)
    auc   = roc_auc_score(y_true_a, y_score_a)
    auprc = average_precision_score(y_true_a, y_score_a)
    ef1   = _ef(y_true_a, y_score_a, 0.01)
    ef5   = _ef(y_true_a, y_score_a, 0.05)
    bedr  = _bedroc(y_true_a, y_score_a)
    fpr, tpr, _ = roc_curve(y_true_a, y_score_a)
    prec, rec, _ = precision_recall_curve(y_true_a, y_score_a)

    pt_a = np.array(pt_list, dtype=np.float32)
    pp_a = np.array(pp_list, dtype=np.float32)
    mae  = float(np.mean(np.abs(pt_a - pp_a))) if len(pt_a)>=2 else 0.0
    rmse = float(np.sqrt(np.mean((pt_a-pp_a)**2))) if len(pt_a)>=2 else 0.0
    sr   = _spearmanr(pt_a, pp_a) if len(pt_a)>=2 else type("R",(),{"statistic":0})()
    pr_  = _pearsonr(pt_a, pp_a)  if len(pt_a)>=2 else (0.0,0.0)
    spear = float(getattr(sr, "statistic", getattr(sr, "correlation", 0.0)))
    pear  = float(getattr(pr_, "statistic", pr_[0] if isinstance(pr_, tuple) else 0.0))

    return dict(
        y_true=y_true_a, y_score=y_score_a,
        fpr=fpr, tpr=tpr, prec=prec, rec=rec,
        pt=pt_a, pp=pp_a,
        auc=auc, auprc=auprc, ef1=ef1, ef5=ef5, bedroc=bedr,
        mae=mae, rmse=rmse, spearman=spear, pearson=pear,
        n_pos=int(y_true_a.sum()), n_neg=int((1-y_true_a).sum()),
    )


def make_plots(res: dict) -> None:
    # ── combined 2×3 panel ────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"TargetTrace — External Evaluation on PDBbind  "
        f"(n_pos={res['n_pos']}, n_neg={res['n_neg']})",
        fontsize=13, fontweight="bold"
    )

    # A: ROC
    ax = axes[0,0]
    ax.plot(res["fpr"], res["tpr"], color=BLUE, lw=2, label=f"AUC = {res['auc']:.3f}")
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
    ax.fill_between(res["fpr"], res["tpr"], alpha=0.1, color=BLUE)
    ax.set_title("A  ROC Curve", fontweight="bold"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(); ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)

    # B: PR
    ax = axes[0,1]
    baseline = res["n_pos"] / (res["n_pos"] + res["n_neg"])
    ax.plot(res["rec"], res["prec"], color=GREEN, lw=2, label=f"AUPRC = {res['auprc']:.3f}")
    ax.axhline(baseline, color="grey", linestyle="--", lw=1.2, alpha=0.6,
               label=f"No-skill ({baseline:.2f})")
    ax.set_title("B  Precision-Recall Curve", fontweight="bold")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.legend()
    ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)

    # C: score distributions
    ax = axes[0,2]
    pos_scores = res["y_score"][res["y_true"] == 1]
    neg_scores = res["y_score"][res["y_true"] == 0]
    bins = np.linspace(0, 1, 30)
    ax.hist(pos_scores, bins=bins, alpha=0.65, color=BLUE,  label=f"Active (n={len(pos_scores)})")
    ax.hist(neg_scores, bins=bins, alpha=0.65, color=ORNG, label=f"Inactive (n={len(neg_scores)})")
    ax.set_title("C  Score Distribution", fontweight="bold")
    ax.set_xlabel("P(active)"); ax.set_ylabel("Count"); ax.legend()
    ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)

    # D: pIC50 scatter
    ax = axes[1,0]
    if len(res["pt"]) >= 2:
        lo = min(res["pt"].min(), res["pp"].min()) - 0.3
        hi = max(res["pt"].max(), res["pp"].max()) + 0.3
        hb = ax.hexbin(res["pt"], res["pp"], gridsize=35, cmap="Blues", mincnt=1)
        plt.colorbar(hb, ax=ax, label="Count")
        ax.plot([lo,hi],[lo,hi],"r--",lw=1.5,alpha=0.7,label="Ideal")
        ax.set_title(f"D  pIC50  (Spear={res['spearman']:.3f})", fontweight="bold")
        ax.set_xlabel("Measured pIC50"); ax.set_ylabel("Predicted pIC50"); ax.legend()
        ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)

    # E: classification metrics
    ax = axes[1,1]
    clf_m = {"AUC": res["auc"], "AUPRC": res["auprc"], "BEDROC": res["bedroc"]}
    cols = [BLUE, GREEN, PURP]
    bars = ax.bar(clf_m.keys(), clf_m.values(), color=cols, edgecolor="white")
    for bar, v in zip(bars, clf_m.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.15); ax.axhline(0.5, color="grey", linestyle="--", lw=1, alpha=0.5)
    ax.set_title("E  Classification Metrics", fontweight="bold"); ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3); ax.spines[["top","right"]].set_visible(False)

    # F: regression metrics
    ax = axes[1,2]
    reg_m = {"MAE": res["mae"], "RMSE": res["rmse"]}
    ax2 = ax.twinx()
    rank_m = {"Spearman": res["spearman"], "Pearson": res["pearson"]}
    b1 = ax.bar(["MAE","RMSE"], [res["mae"],res["rmse"]], color=[RED,ORNG], edgecolor="white", width=0.35,
                align="center", label="Error (left)")
    b2 = ax2.bar(["Spearman","Pearson"], [res["spearman"],res["pearson"]],
                 color=[PURP,BLUE], edgecolor="white", width=0.35,
                 align="center", label="Correlation (right)")
    ax.set_ylabel("Error (MAE / RMSE)", color=RED)
    ax2.set_ylabel("Correlation", color=BLUE)
    ax2.set_ylim(0, 1.15)
    ax.set_title("F  Regression Metrics", fontweight="bold")
    ax.grid(axis="y", alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in b2:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = OUT_DIR / "pdbbind_evaluation.png"
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
    print(f"Saved → {out}")

    # ── standalone ROC ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(res["fpr"], res["tpr"], color=BLUE, lw=2.5, label=f"AUC = {res['auc']:.3f}")
    ax.plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
    ax.fill_between(res["fpr"], res["tpr"], alpha=0.12, color=BLUE)
    ax.set_xlabel("False Positive Rate", fontsize=12); ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC — PDBbind External Test (Explicit Negatives)", fontsize=11)
    ax.legend(fontsize=11); ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pdbbind_roc.png", dpi=180, bbox_inches="tight"); plt.close()

    print(f"\nExternal evaluation complete:")
    print(f"  AUC={res['auc']:.4f}  AUPRC={res['auprc']:.4f}  BEDROC={res['bedroc']:.4f}")
    print(f"  EF1%={res['ef1']:.2f}  EF5%={res['ef5']:.2f}")
    print(f"  MAE={res['mae']:.4f}  Spearman={res['spearman']:.4f}")


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "pdbbind_eval.csv"
    res = run_explicit_eval(csv)
    make_plots(res)
