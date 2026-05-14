"""
Baseline comparison: TargetTrace vs. simple ML baselines on PDBbind.

Baselines trained on ChEMBL data:
  1. Tanimoto-NN   — nearest-neighbour in Morgan FP space (no model needed)
  2. FP-only LR    — Morgan FP (2048) → Logistic Regression (no protein)
  3. FP+prot LR    — Morgan FP (2048) + ESM-2 pooled (480) → LR

All evaluated on the same 1,115-pair PDBbind test set.
Produces: results/supplementary/baselines_comparison.png
"""
import sys, gc, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from trainer import _bedroc, _ef

OUT = Path(__file__).parent / "results" / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)

BLUE  = "#4C72B0"
GREEN = "#55A868"
RED   = "#C44E52"
ORNG  = "#E07B54"
PURP  = "#8172B2"
GREY  = "#AAAAAA"

N_TRAIN = 60_000   # ChEMBL pairs to train baselines on (balanced: 30k pos + 30k neg)


def morgan_fp(smi: str, n_bits: int = 2048) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto_nn_score(test_smiles, train_smiles_pos, batch=500) -> np.ndarray:
    """Score each test compound by max Tanimoto to any training active."""
    print("  Building training FP pool for NN baseline…")
    train_fps = []
    for s in train_smiles_pos:
        mol = Chem.MolFromSmiles(str(s))
        if mol:
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))

    print(f"  Scoring {len(test_smiles)} test compounds against {len(train_fps)} training actives…")
    scores = []
    for smi in test_smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            scores.append(0.0); continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        best = 0.0
        for i in range(0, len(train_fps), batch):
            chunk = train_fps[i:i + batch]
            sims  = DataStructs.BulkTanimotoSimilarity(fp, chunk)
            best  = max(best, max(sims))
        scores.append(best)
    return np.array(scores)


def build_training_data(db_path: str, n_pos: int, n_neg: int,
                        use_prot_emb: bool = False):
    conn = sqlite3.connect(db_path)
    pos_df = pd.read_sql(
        f"SELECT smiles, protein_sequence FROM bioactivity "
        f"WHERE standard_value <= 10000 ORDER BY RANDOM() LIMIT {n_pos}", conn)
    neg_df = pd.read_sql(
        f"SELECT smiles, protein_sequence FROM bioactivity "
        f"WHERE standard_value > 10000 ORDER BY RANDOM() LIMIT {n_neg}", conn)
    conn.close()
    pos_df["y"] = 1; neg_df["y"] = 0
    df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1, random_state=42)

    print(f"  Computing FPs for {len(df)} training pairs…")
    fps, valid_idx = [], []
    for i, smi in enumerate(df["smiles"]):
        fp = morgan_fp(smi)
        if fp is not None:
            fps.append(fp); valid_idx.append(i)
    df = df.iloc[valid_idx].reset_index(drop=True)
    X_fp = np.stack(fps)
    y    = df["y"].values

    if use_prot_emb:
        from embedders import _seq_key, _load, _PROT_POOLED, embed_proteins, unload_models
        print("  Loading ESM-2 pooled embeddings for training proteins…")
        seqs = df["protein_sequence"].unique().tolist()
        embed_proteins(seqs); unload_models()
        prot_pool = _load(_PROT_POOLED)
        seq_to_key = {s: _seq_key(s) for s in seqs}
        prot_embs = np.stack([
            prot_pool.get(_seq_key(s), np.zeros(480, dtype=np.float32))
            for s in df["protein_sequence"]
        ])
        del prot_pool; gc.collect()
        X = np.hstack([X_fp, prot_embs])
    else:
        X = X_fp

    return X, y, df


def build_test_features(scores_df: pd.DataFrame, use_prot_emb: bool = False):
    fps, valid_idx = [], []
    for i, smi in enumerate(scores_df["smiles"]):
        fp = morgan_fp(smi)
        if fp is not None:
            fps.append(fp); valid_idx.append(i)
    sub = scores_df.iloc[valid_idx].reset_index(drop=True)
    X_fp = np.stack(fps)

    if use_prot_emb:
        from embedders import _seq_key, _load, _PROT_POOLED, embed_proteins, unload_models
        seqs = sub["protein_sequence"].unique().tolist()
        embed_proteins(seqs); unload_models()
        prot_pool = _load(_PROT_POOLED)
        prot_embs = np.stack([
            prot_pool.get(_seq_key(s), np.zeros(480, dtype=np.float32))
            for s in sub["protein_sequence"]
        ])
        del prot_pool; gc.collect()
        X = np.hstack([X_fp, prot_embs])
    else:
        X = X_fp
    return X, sub


def metrics_dict(y, s) -> dict:
    return {
        "AUC-ROC": roc_auc_score(y, s),
        "AUPRC":   average_precision_score(y, s),
        "BEDROC":  _bedroc(y, s),
        "EF@1%":   _ef(y, s, 0.01),
    }


def run_baselines(scores_csv: str, db_path: str = "bioactivity.db",
                  full_csv: str = "pdbbind_eval.csv") -> None:
    df_scores = pd.read_csv(scores_csv)
    df_full   = pd.read_csv(full_csv)[["smiles", "protein_sequence"]]
    df_test   = df_scores.merge(df_full, on="smiles", how="left")
    tt_metrics = metrics_dict(df_scores["active"].values, df_scores["score"].values)

    results = {"TargetTrace (full model)": tt_metrics}

    # ── 1. Tanimoto-NN baseline ───────────────────────────────────────────
    print("\n[1/3] Tanimoto-NN baseline…")
    conn = sqlite3.connect(db_path)
    train_pos_smiles = pd.read_sql(
        f"SELECT smiles FROM bioactivity WHERE standard_value<=10000 "
        f"ORDER BY RANDOM() LIMIT 30000", conn)["smiles"].tolist()
    conn.close()
    nn_scores = tanimoto_nn_score(df_test["smiles"], train_pos_smiles)
    results["Tanimoto-NN"] = metrics_dict(df_test["active"].values, nn_scores)

    # ── 2. FP-only LR baseline ────────────────────────────────────────────
    print("\n[2/3] FP-only Logistic Regression…")
    X_tr, y_tr, _ = build_training_data(db_path, N_TRAIN // 2, N_TRAIN // 2,
                                         use_prot_emb=False)
    scaler = StandardScaler(with_mean=False)
    X_tr_s = scaler.fit_transform(X_tr)
    lr_fp = LogisticRegression(max_iter=300, C=0.1, solver="saga", n_jobs=-1)
    lr_fp.fit(X_tr_s, y_tr)
    # test features
    X_te_fp, sub_fp = build_test_features(df_test, use_prot_emb=False)
    X_te_fp_s = scaler.transform(X_te_fp)
    s_fp = lr_fp.predict_proba(X_te_fp_s)[:, 1]
    results["FP-only LR"] = metrics_dict(sub_fp["active"].values, s_fp)
    del X_tr, X_tr_s, X_te_fp, X_te_fp_s; gc.collect()

    # ── 3. FP + ESM-2 pooled LR ───────────────────────────────────────────
    print("\n[3/3] FP + ESM-2 pooled LR…")
    X_tr2, y_tr2, _ = build_training_data(db_path, N_TRAIN // 2, N_TRAIN // 2,
                                           use_prot_emb=True)
    scaler2 = StandardScaler(with_mean=False)
    X_tr2_s = scaler2.fit_transform(X_tr2)
    lr_fp_prot = LogisticRegression(max_iter=300, C=0.1, solver="saga", n_jobs=-1)
    lr_fp_prot.fit(X_tr2_s, y_tr2)
    X_te2, sub2 = build_test_features(df_test, use_prot_emb=True)
    X_te2_s = scaler2.transform(X_te2)
    s2 = lr_fp_prot.predict_proba(X_te2_s)[:, 1]
    results["FP + ESM-2 LR"] = metrics_dict(sub2["active"].values, s2)
    del X_tr2, X_tr2_s, X_te2, X_te2_s; gc.collect()

    # ── Print comparison table ────────────────────────────────────────────
    print("\n" + "─" * 65)
    print(f"{'Model':<28} {'AUC-ROC':>8} {'AUPRC':>7} {'BEDROC':>8} {'EF@1%':>7}")
    print("─" * 65)
    for name, m in results.items():
        print(f"{name:<28} {m['AUC-ROC']:>8.4f} {m['AUPRC']:>7.4f} "
              f"{m['BEDROC']:>8.4f} {m['EF@1%']:>7.3f}")
    print("─" * 65)

    # ── Figure ────────────────────────────────────────────────────────────
    model_names = list(results.keys())
    metric_keys = ["AUC-ROC", "AUPRC", "BEDROC", "EF@1%"]
    colors = [BLUE, GREY, ORNG, PURP]

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    fig.suptitle("TargetTrace vs. Baseline Models — PDBbind Performance",
                 fontsize=13, fontweight="bold")

    for ax, mk in zip(axes, metric_keys):
        vals = [results[n][mk] for n in model_names]
        bar_colors = [BLUE if n == "TargetTrace (full model)" else GREY
                      for n in model_names]
        bars = ax.bar(range(len(model_names)), vals, color=bar_colors,
                      edgecolor="white", linewidth=0.8, zorder=3, width=0.6)
        if mk in ("AUC-ROC", "AUPRC", "BEDROC"):
            ax.axhline(0.5, color="black", ls="--", lw=1.0, alpha=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=8)
        ax.set_ylabel(mk, fontsize=10)
        ax.set_title(mk, fontweight="bold")
        ax.set_ylim(0, min(max(vals) * 1.20, 1.05) if mk != "EF@1%" else max(vals) * 1.25)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "baselines_comparison.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUT / 'baselines_comparison.png'}")

    return results


if __name__ == "__main__":
    scores_csv = Path(__file__).parent / "results" / "pdbbind_scores.csv"
    run_baselines(scores_csv)
