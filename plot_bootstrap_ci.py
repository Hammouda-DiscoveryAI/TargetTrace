"""
Bootstrap confidence intervals (95%) for all PDBbind evaluation metrics.
Produces: results/supplementary/bootstrap_ci.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from trainer import _ef, _bedroc

OUT = Path(__file__).parent / "results" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#4C72B0"

def bootstrap_metrics(df: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> dict:
    rng   = np.random.default_rng(seed)
    y     = df["active"].values
    s     = df["score"].values
    pos   = df[df["active"] == 1]
    pt    = pos["pic50_true"].values
    pp    = pos["pic50_pred"].values
    valid = ~np.isnan(pt)
    pt, pp = pt[valid], pp[valid]

    metrics = {
        "AUC-ROC":   [], "AUPRC":     [],
        "BEDROC":    [], "EF@1%":     [],
        "Spearman ρ": [], "Pearson r": [],
        "MAE":       [],
    }

    for _ in range(n_boot):
        idx = rng.integers(0, len(df), len(df))
        yi, si = y[idx], s[idx]
        if yi.sum() == 0 or yi.sum() == len(yi):
            continue
        metrics["AUC-ROC"].append(roc_auc_score(yi, si))
        metrics["AUPRC"].append(average_precision_score(yi, si))
        metrics["BEDROC"].append(_bedroc(yi, si))
        metrics["EF@1%"].append(_ef(yi, si, 0.01))

        ri = rng.integers(0, len(pt), len(pt))
        if len(ri) >= 2:
            metrics["Spearman ρ"].append(float(spearmanr(pt[ri], pp[ri]).statistic))
            metrics["Pearson r"].append(float(pearsonr(pt[ri], pp[ri])[0]))
            metrics["MAE"].append(float(np.mean(np.abs(pt[ri] - pp[ri]))))

    return {k: np.array(v) for k, v in metrics.items()}


def plot_ci(df: pd.DataFrame) -> None:
    boots = bootstrap_metrics(df)
    y     = df["active"].values
    s     = df["score"].values
    pos   = df[df["active"] == 1]
    pt    = pos["pic50_true"].values
    pp    = pos["pic50_pred"].values
    valid = ~np.isnan(pt)

    point_estimates = {
        "AUC-ROC":    roc_auc_score(y, s),
        "AUPRC":      average_precision_score(y, s),
        "BEDROC":     _bedroc(y, s),
        "EF@1%":      _ef(y, s, 0.01),
        "Spearman ρ": float(spearmanr(pt[valid], pp[valid]).statistic),
        "Pearson r":  float(pearsonr(pt[valid], pp[valid])[0]),
        "MAE":        float(np.mean(np.abs(pt[valid] - pp[valid]))),
    }

    names   = list(boots.keys())
    centers = [point_estimates[n] for n in names]
    lows    = [np.percentile(boots[n], 2.5)  for n in names]
    highs   = [np.percentile(boots[n], 97.5) for n in names]
    errs    = [[c - l for c, l in zip(centers, lows)],
               [h - c for c, h in zip(centers, highs)]]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, centers, color=BLUE, alpha=0.8, edgecolor="white",
                  linewidth=0.8, zorder=3, width=0.55)
    ax.errorbar(x, centers, yerr=errs, fmt="none", color="black",
                capsize=5, linewidth=1.5, zorder=4)
    for xi, c, lo, hi in zip(x, centers, lows, highs):
        ax.text(xi, hi + 0.01, f"{c:.3f}\n[{lo:.3f}–{hi:.3f}]",
                ha="center", va="bottom", fontsize=8.5, color="#333333")

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_title("TargetTrace — PDBbind Metrics with 95% Bootstrap CIs (n=2,000)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(highs) * 1.18)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "bootstrap_ci.png", dpi=180, bbox_inches="tight"); plt.close()
    print(f"Saved → {OUT / 'bootstrap_ci.png'}")

    # Print table
    print("\n" + "─" * 62)
    print(f"{'Metric':<14} {'Point Est':>10} {'95% CI Lower':>13} {'95% CI Upper':>13}")
    print("─" * 62)
    for n in names:
        c, lo, hi = point_estimates[n], np.percentile(boots[n], 2.5), np.percentile(boots[n], 97.5)
        print(f"{n:<14} {c:>10.4f} {lo:>13.4f} {hi:>13.4f}")
    print("─" * 62)


if __name__ == "__main__":
    scores_csv = Path(__file__).parent / "results" / "pdbbind_scores.csv"
    df = pd.read_csv(scores_csv)
    plot_ci(df)
