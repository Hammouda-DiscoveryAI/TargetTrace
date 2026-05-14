"""
Per-target and per-family AUC breakdown on PDBbind.
Produces: results/supplementary/per_target_auc.png
          results/supplementary/per_family_auc.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

OUT = Path(__file__).parent / "results" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#4C72B0"
RED  = "#C44E52"
GREEN = "#55A868"

# Simple keyword mapping from target_name → protein family
FAMILY_MAP = {
    "Carbonic anhydrase":           "Carbonic anhydrases",
    "Transthyretin":                "Transport proteins",
    "Aldo-keto reductase":          "Aldo-keto reductases",
    "Mcl-1":                        "Anti-apoptotic BCL-2",
    "Epidermal growth factor":      "Receptor tyrosine kinases",
    "Renin":                        "Aspartic proteases",
    "Glycogen phosphorylase":       "Glycogen phosphorylases",
    "CREB-binding protein":         "HATs / bromodomains",
    "YTH domain":                   "RNA-binding proteins",
    "Estrogen receptor":            "Nuclear receptors",
    "Kelch-like":                   "Kelch / E3 ligases",
    "phosphodiesterase":            "Phosphodiesterases",
    "Amine oxidase":                "Amine oxidases (MAO)",
    "nerve growth factor":          "Receptor tyrosine kinases",
    "Leukotriene":                  "Hydrolases / lyases",
    "Androgen receptor":            "Nuclear receptors",
    "Farnesyl pyrophosphate":       "Isoprenoid biosynthesis",
    "p53":                          "Tumor suppressors",
    "adrenergic":                   "GPCRs",
    "kinase":                       "Kinases (other)",
    "Thrombin":                     "Serine proteases",
    "Factor Xa":                    "Serine proteases",
    "protease":                     "Serine proteases",
    "HIV":                          "Viral proteins",
    "Bromodomain":                  "HATs / bromodomains",
    "acetylcholinesterase":         "Esterases",
    "Acetylcholinesterase":         "Esterases",
    "Histone":                      "HATs / bromodomains",
    "dehydrogenase":                "Dehydrogenases",
    "reductase":                    "Reductases (other)",
    "Cathepsin":                    "Cysteine proteases",
}


def assign_family(name: str) -> str:
    for kw, fam in FAMILY_MAP.items():
        if kw.lower() in name.lower():
            return fam
    return "Other"


def plot_per_target(df: pd.DataFrame) -> None:
    # Require ≥3 actives AND ≥1 inactive for AUC
    records = []
    for tgt, grp in df.groupby("target_name"):
        n_pos = (grp["active"] == 1).sum()
        n_neg = (grp["active"] == 0).sum()
        if n_pos < 3 or n_neg < 1:
            continue
        auc = roc_auc_score(grp["active"], grp["score"])
        records.append({"target": tgt, "auc": auc,
                        "n_active": n_pos, "n_inactive": n_neg,
                        "n_total": len(grp)})
    tgt_df = pd.DataFrame(records).sort_values("auc", ascending=True)
    print(f"Targets with ≥3 actives + ≥1 inactive: {len(tgt_df)}")

    # ── Per-target bar chart ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, max(5, len(tgt_df) * 0.3)))
    colors = [RED if a < 0.5 else (BLUE if a >= 0.7 else "#E07B54")
              for a in tgt_df["auc"]]
    y_pos = np.arange(len(tgt_df))
    bars = ax.barh(y_pos, tgt_df["auc"], color=colors, edgecolor="white",
                   linewidth=0.6, height=0.7)
    ax.axvline(0.5, color="black", ls="--", lw=1.2, alpha=0.5)
    ax.axvline(tgt_df["auc"].mean(), color=GREEN, ls=":", lw=1.5,
               alpha=0.8, label=f"Mean AUC = {tgt_df['auc'].mean():.3f}")
    # Truncate long target names
    labels = [t[:40] + ("…" if len(t) > 40 else "") for t in tgt_df["target"]]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("AUC-ROC", fontsize=11)
    ax.set_title("Per-Target AUC on PDBbind\n(targets with ≥3 actives + ≥1 inactive)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1.12)
    for bar, v in zip(bars, tgt_df["auc"]):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=7)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "per_target_auc.png", dpi=180, bbox_inches="tight"); plt.close()
    print(f"Saved → {OUT / 'per_target_auc.png'}")

    # ── Per-family aggregated bar chart ───────────────────────────────────
    tgt_df["family"] = tgt_df["target"].apply(assign_family)
    fam_df = (tgt_df.groupby("family")
              .agg(mean_auc=("auc", "mean"), std_auc=("auc", "std"),
                   n_targets=("auc", "count"))
              .reset_index()
              .sort_values("mean_auc", ascending=True))
    fam_df["std_auc"] = fam_df["std_auc"].fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_f = [RED if a < 0.5 else (BLUE if a >= 0.7 else "#E07B54")
                for a in fam_df["mean_auc"]]
    y_f = np.arange(len(fam_df))
    ax.barh(y_f, fam_df["mean_auc"], xerr=fam_df["std_auc"],
            color=colors_f, edgecolor="white", linewidth=0.6,
            height=0.65, capsize=3, error_kw=dict(lw=1.2, color="grey"))
    ax.axvline(0.5, color="black", ls="--", lw=1.2, alpha=0.5)
    ax.axvline(tgt_df["auc"].mean(), color=GREEN, ls=":", lw=1.5, alpha=0.8,
               label=f"Overall mean = {tgt_df['auc'].mean():.3f}")
    ax.set_yticks(y_f)
    ax.set_yticklabels([f"{r.family} (n={r.n_targets})"
                        for _, r in fam_df.iterrows()], fontsize=9.5)
    ax.set_xlabel("Mean AUC-ROC (± SD across targets)", fontsize=11)
    ax.set_title("Per-Protein-Family AUC on PDBbind",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.15)
    for yi, (_, row) in zip(y_f, fam_df.iterrows()):
        ax.text(row.mean_auc + row.std_auc + 0.02, yi,
                f"{row.mean_auc:.3f}", va="center", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "per_family_auc.png", dpi=180, bbox_inches="tight"); plt.close()
    print(f"Saved → {OUT / 'per_family_auc.png'}")


if __name__ == "__main__":
    scores_csv = Path(__file__).parent / "results" / "pdbbind_scores.csv"
    df = pd.read_csv(scores_csv)
    plot_per_target(df)
