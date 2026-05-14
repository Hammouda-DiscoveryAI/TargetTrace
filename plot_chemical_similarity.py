"""
Chemical similarity analysis: Tanimoto similarity between PDBbind test compounds
and the nearest training compound in the ChEMBL database.
Tests whether good performance is driven by memorisation vs. genuine generalisation.
Produces: results/supplementary/chemical_similarity.png
"""
import sys, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

OUT = Path(__file__).parent / "results" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

BLUE  = "#4C72B0"
GREEN = "#55A868"
RED   = "#C44E52"
ORNG  = "#E07B54"


def fp2048(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def nearest_tanimoto(test_fps, train_fps_sample, batch=500) -> np.ndarray:
    """For each test FP compute max Tanimoto to any training FP (batched)."""
    n = len(test_fps)
    best = np.zeros(n)
    for i in range(0, len(train_fps_sample), batch):
        chunk = train_fps_sample[i:i + batch]
        sims  = np.array(DataStructs.BulkTanimotoSimilarity.__class__  # type annotation bypass
                         if False else
                         [max(DataStructs.BulkTanimotoSimilarity(fp, chunk))
                          for fp in test_fps])
        best = np.maximum(best, sims)
    return best


def run_analysis(scores_csv: str, db_path: str = "bioactivity.db",
                 n_train_sample: int = 30_000) -> None:
    df_test = pd.read_csv(scores_csv)

    # Compute test fingerprints
    print("Computing test fingerprints…")
    test_fps = []
    valid_idx = []
    for i, smi in enumerate(df_test["smiles"]):
        fp = fp2048(str(smi))
        if fp is not None:
            test_fps.append(fp); valid_idx.append(i)
    df_test = df_test.iloc[valid_idx].reset_index(drop=True)
    print(f"  {len(test_fps)} valid test FPs")

    # Sample training SMILES from DB
    print(f"Sampling {n_train_sample} training compounds from DB…")
    conn = sqlite3.connect(db_path)
    train_df = pd.read_sql(
        f"SELECT DISTINCT smiles FROM bioactivity WHERE standard_value <= 10000 "
        f"ORDER BY RANDOM() LIMIT {n_train_sample}", conn)
    conn.close()

    print("Computing training fingerprints…")
    train_fps = []
    for smi in train_df["smiles"]:
        fp = fp2048(str(smi))
        if fp is not None:
            train_fps.append(fp)
    print(f"  {len(train_fps)} valid training FPs")

    # Compute nearest-neighbour Tanimoto for each test compound
    print("Computing nearest-Tanimoto similarities…")
    sim = nearest_tanimoto(test_fps, train_fps)
    df_test["tanimoto_nn"] = sim
    print(f"  Similarity: mean={sim.mean():.3f}, median={np.median(sim):.3f}, "
          f"min={sim.min():.3f}, max={sim.max():.3f}")

    # Novel scaffold fraction (Tanimoto < 0.4)
    novel_frac = (sim < 0.4).mean()
    print(f"  Novel compounds (Tan < 0.4): {novel_frac:.1%}")

    # AUC by similarity decile
    bins   = np.percentile(sim, np.arange(0, 110, 25))  # quartiles
    labels = ["Q1 (most novel)", "Q2", "Q3", "Q4 (most similar)"]
    auc_by_quartile = []
    for lo, hi, lbl in zip(bins[:-1], bins[1:], labels):
        mask = (sim >= lo) & (sim <= hi)
        sub  = df_test[mask]
        if sub["active"].sum() > 0 and (sub["active"] == 0).sum() > 0:
            a = roc_auc_score(sub["active"], sub["score"])
        else:
            a = np.nan
        auc_by_quartile.append((lbl, lo, hi, mask.sum(), a))
        print(f"  {lbl} [{lo:.2f}–{hi:.2f}]: n={mask.sum()}, AUC={a:.3f}")

    # ── Figures ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Chemical Similarity Analysis: TargetTrace PDBbind vs Training Set",
                 fontsize=13, fontweight="bold")

    # Panel A: Tanimoto histogram (actives and inactives)
    ax = axes[0]
    bins_h = np.linspace(0, 1, 26)
    ax.hist(sim[df_test["active"] == 1], bins=bins_h, color=BLUE, alpha=0.6,
            density=True, label=f"Active (n={int((df_test['active']==1).sum())})")
    ax.hist(sim[df_test["active"] == 0], bins=bins_h, color=RED, alpha=0.6,
            density=True, label=f"Inactive (n={int((df_test['active']==0).sum())})")
    ax.axvline(0.4, color="black", ls="--", lw=1.2, alpha=0.6,
               label="Novel threshold (0.4)")
    ax.set_xlabel("Nearest-Neighbour Tanimoto to training set", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"A  Tanimoto Distribution\n"
                 f"{novel_frac:.1%} novel (Tan < 0.4)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)

    # Panel B: AUC by similarity quartile
    ax = axes[1]
    q_labels = [r[0] for r in auc_by_quartile]
    q_aucs   = [r[4] for r in auc_by_quartile]
    q_colors = [BLUE if a >= 0.7 else (ORNG if a >= 0.5 else RED)
                for a in q_aucs]
    bars = ax.bar(range(len(q_labels)), q_aucs, color=q_colors,
                  edgecolor="white", linewidth=0.8, zorder=3, width=0.6)
    ax.axhline(0.5, color="black", ls="--", lw=1.2, alpha=0.5)
    ax.axhline(roc_auc_score(df_test["active"], df_test["score"]),
               color=GREEN, ls=":", lw=1.5, label=f"Overall AUC", alpha=0.8)
    for b, v in zip(bars, q_aucs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(q_labels))); ax.set_xticklabels(q_labels, fontsize=8.5)
    ax.set_ylabel("AUC-ROC", fontsize=10)
    ax.set_title("B  AUC by Similarity Quartile\n(does performance decay for novel compounds?)",
                 fontweight="bold", fontsize=9.5)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)

    # Panel C: Score vs Tanimoto scatter (sampled)
    ax = axes[2]
    n_show = min(800, len(df_test))
    idx_s  = np.random.default_rng(42).choice(len(df_test), n_show, replace=False)
    sub    = df_test.iloc[idx_s]
    for act, col, lbl in [(1, BLUE, "Active"), (0, RED, "Inactive")]:
        m = sub["active"] == act
        ax.scatter(sub[m]["tanimoto_nn"], sub[m]["score"],
                   color=col, alpha=0.35, s=12, label=lbl)
    ax.axhline(0.5, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(0.4, color="black", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("Nearest-Neighbour Tanimoto", fontsize=10)
    ax.set_ylabel("Predicted binding probability", fontsize=10)
    ax.set_title("C  Score vs Similarity\n(sampled)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "chemical_similarity.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUT / 'chemical_similarity.png'}")


if __name__ == "__main__":
    scores_csv = Path(__file__).parent / "results" / "pdbbind_scores.csv"
    run_analysis(scores_csv)
