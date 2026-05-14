"""
Model calibration analysis: reliability diagram + Expected Calibration Error (ECE).
Produces: results/supplementary/calibration.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

OUT = Path(__file__).parent / "results" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#4C72B0"
RED  = "#C44E52"


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray,
                       n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    frac_pos, mean_pred, bin_counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        frac_pos.append(y_true[mask].mean())
        mean_pred.append(y_prob[mask].mean())
        bin_counts.append(mask.sum())
    return np.array(mean_pred), np.array(frac_pos), np.array(bin_counts)


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(y_true)
    error = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        error += mask.sum() / total * abs(acc - conf)
    return error


def plot_calibration(df: pd.DataFrame) -> None:
    y = df["active"].values
    s = df["score"].values

    mean_pred, frac_pos, counts = calibration_curve(y, s, n_bins=15)
    ece_val = ece(y, s, n_bins=15)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("TargetTrace — Calibration Analysis", fontsize=13, fontweight="bold")

    # ── Reliability diagram ───────────────────────────────────────────────
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration", alpha=0.6)
    sc = ax.scatter(mean_pred, frac_pos, s=counts / counts.max() * 300 + 20,
                    c=counts, cmap="Blues", edgecolors="#4C72B0",
                    linewidths=0.8, zorder=3, alpha=0.85)
    ax.plot(mean_pred, frac_pos, color=BLUE, lw=1.5, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Bin count")
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives", fontsize=11)
    ax.set_title(f"A  Reliability Diagram\n(ECE = {ece_val:.4f})", fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)

    # ── Histogram of predicted probabilities ─────────────────────────────
    ax = axes[1]
    bins = np.linspace(0, 1, 26)
    ax.hist(s[y == 1], bins=bins, color=BLUE, alpha=0.6,
            label=f"Active (n={int(y.sum())})", density=True)
    ax.hist(s[y == 0], bins=bins, color=RED,  alpha=0.6,
            label=f"Inactive (n={int((1-y).sum())})", density=True)
    ax.axvline(0.5, color="black", ls="--", lw=1.2, alpha=0.7)
    ax.set_xlabel("Predicted binding probability", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("B  Predicted Score Histogram", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "calibration.png", dpi=180, bbox_inches="tight"); plt.close()
    print(f"Saved → {OUT / 'calibration.png'}")
    print(f"ECE = {ece_val:.4f}")


if __name__ == "__main__":
    scores_csv = Path(__file__).parent / "results" / "pdbbind_scores.csv"
    df = pd.read_csv(scores_csv)
    plot_calibration(df)
