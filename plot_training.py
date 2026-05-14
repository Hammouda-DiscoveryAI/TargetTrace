"""
Parse training epoch logs from background-task output files and produce
publication-quality training curves saved to results/training/.

Usage:
    python plot_training.py [output_file]

If no output_file is given, auto-discovers the most recent bxluywrry-style
output in /tmp and falls back to bg-task output paths.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = Path(__file__).parent / "results" / "training"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+loss=([\d.]+)\s+AUC=([\d.]+)\s+AUPRC=([\d.]+)\s+MAE=([\d.]+)"
)


def parse_log(text: str) -> list[dict]:
    rows = []
    for m in _EPOCH_RE.finditer(text):
        rows.append({
            "epoch":  int(m.group(1)),
            "total":  int(m.group(2)),
            "loss":   float(m.group(3)),
            "auc":    float(m.group(4)),
            "auprc":  float(m.group(5)),
            "mae":    float(m.group(6)),
        })
    return rows


def find_latest_output() -> str | None:
    base = Path("/tmp/claude-1000")
    candidates = sorted(base.rglob("bxluywrry.output"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0])
    # fallback: any .output with Epoch lines
    for p in sorted(base.rglob("*.output"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            if "Epoch" in p.read_text():
                return str(p)
        except Exception:
            pass
    return None


def plot(rows: list[dict], title_suffix: str = "") -> None:
    if not rows:
        print("No epoch data found — nothing to plot.")
        return

    epochs = [r["epoch"] for r in rows]
    losses = [r["loss"]  for r in rows]
    aucs   = [r["auc"]   for r in rows]
    auprcs = [r["auprc"] for r in rows]
    maes   = [r["mae"]   for r in rows]
    best_e = epochs[int(np.argmax(aucs))]

    BLUE  = "#4C72B0"
    GREEN = "#55A868"
    RED   = "#C44E52"
    PURP  = "#8172B2"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"TargetTrace — Training Curves{title_suffix}", fontsize=14, fontweight="bold")

    def _ax(ax, xs, ys, color, ylabel, best_epoch=None):
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4)
        if best_epoch:
            bv = ys[xs.index(best_epoch)]
            ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=1, alpha=0.6)
            ax.scatter([best_epoch], [bv], zorder=5, s=80, color=color,
                       edgecolors="white", linewidths=1.5)
            ax.annotate(f" best={bv:.4f}", xy=(best_epoch, bv),
                        fontsize=8, color="grey", va="bottom")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)

    _ax(axes[0,0], epochs, losses, RED,   "Training Loss")
    _ax(axes[0,1], epochs, aucs,   BLUE,  "Val AUC-ROC",  best_epoch=best_e)
    _ax(axes[1,0], epochs, auprcs, GREEN, "Val AUPRC",    best_epoch=best_e)
    _ax(axes[1,1], epochs, maes,   PURP,  "Val pIC50 MAE")

    plt.tight_layout()
    out = OUT_DIR / "training_curves.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")

    # ── individual panels ──────────────────────────────────────────────────
    for ys, color, ylabel, fname in [
        (losses, RED,   "Training Loss",  "loss.png"),
        (aucs,   BLUE,  "Val AUC-ROC",    "val_auc.png"),
        (auprcs, GREEN, "Val AUPRC",      "val_auprc.png"),
        (maes,   PURP,  "Val pIC50 MAE",  "val_mae.png"),
    ]:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        best_e2 = epochs[int(np.argmax(aucs))] if "AUC" in ylabel or "AUPRC" in ylabel else None
        _ax(ax2, epochs, ys, color, ylabel, best_epoch=best_e2)
        ax2.set_title(ylabel, fontsize=12)
        plt.tight_layout()
        out2 = OUT_DIR / fname
        plt.savefig(out2, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Saved → {out2}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else find_latest_output()
    if path is None:
        print("ERROR: no output file found. Pass path as argument.")
        sys.exit(1)
    print(f"Parsing: {path}")
    text = Path(path).read_text()
    rows = parse_log(text)
    print(f"Found {len(rows)} epoch entries")
    plot(rows)
