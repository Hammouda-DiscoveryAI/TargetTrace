"""
Ablation study + baseline comparison for TargetTrace3.

All variants are evaluated on the SAME held-out scaffold-split val set so the
numbers are directly comparable.  The val set is rebuilt deterministically from
the database using the same seed as training (seed=42).

Variants
--------
  TargetTrace3 (Full)   — cross-attention, multi-radius FP, Hadamard fusion
  No Cross-Attention    — same weights, pooled-ESM path (ablates cross-attn)
  No Hadamard           — same weights, zero out mol⊙prot interaction term
  ECFP4 Only (2048-dim) — same weights, zero-pad FP beyond first ECFP4 block
  Logistic Regression   — sklearn LR on concat(FP, pooled-ESM)
  Random Forest         — sklearn RF on concat(FP, pooled-ESM)
  Random Baseline       — random scores, lower-bound reference

Returns
-------
  metrics_df  : pd.DataFrame  [Variant × metrics]
  curves      : dict[str, dict]  {"roc": [(fpr,tpr)…], "pr": [(rec,prec)…]}
  scatter     : dict[str, list]  [(true_pic50, pred_pic50)…]  (up to 500 pts)
"""
import gc
import time as _time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader

from database import get_training_data
from embedders import ESM_DIM, _PROT_POOLED, _load, _seq_key, load_residues_subset
from features import FP_DIM, load_fp_subset
from model import TargetTrace3
from trainer import (
    TargetTrace3Dataset, _collate, _ef, _bedroc,
    _fast_stratified_sample, _scaffold_split,
    device, USE_AMP,
)

try:
    from sklearn.metrics import (
        average_precision_score, precision_recall_curve,
        roc_auc_score, roc_curve,
    )
    from scipy.stats import pearsonr as _pearsonr, spearmanr as _spearmanr
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


# ── Concordance Index ──────────────────────────────────────────────────────

def _ci(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance index — fraction of pairs correctly ordered."""
    n = len(y_true)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            if (y_true[i] > y_true[j]) == (y_pred[i] > y_pred[j]):
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    return concordant / denom if denom else 0.5


def _ci_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Vectorised CI (O(n²) but numpy — fast enough for n ≤ 2000)."""
    if len(y_true) > 2000:
        idx = np.random.choice(len(y_true), 2000, replace=False)
        y_true, y_pred = y_true[idx], y_pred[idx]
    diff_t = y_true[:, None] - y_true[None, :]   # (n, n)
    diff_p = y_pred[:, None] - y_pred[None, :]
    mask   = diff_t != 0
    concordant  = ((diff_t > 0) == (diff_p > 0)) & mask
    discordant  = ((diff_t > 0) != (diff_p > 0)) & mask
    denom = concordant.sum() + discordant.sum()
    return float(concordant.sum() / denom) if denom else 0.5


# ── Metric computation ─────────────────────────────────────────────────────

def _compute_metrics(y_true, y_score, pt, pp, label: str) -> dict:
    """Return dict of all evaluation metrics for one variant."""
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)

    auc   = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    ef1   = _ef(y_true, y_score, 0.01)
    ef5   = _ef(y_true, y_score, 0.05)
    bedr  = _bedroc(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    step = max(1, len(fpr) // 200)
    roc_pts = list(zip(fpr[::step].tolist(), tpr[::step].tolist()))

    prec_c, rec_c, _ = precision_recall_curve(y_true, y_score)
    step = max(1, len(prec_c) // 200)
    pr_pts = list(zip(rec_c[::step].tolist(), prec_c[::step].tolist()))

    if len(pt) >= 2 and len(pt) == len(pp):
        pt_a = np.array(pt, dtype=np.float32)
        pp_a = np.array(pp, dtype=np.float32)
        mae  = float(np.mean(np.abs(pt_a - pp_a)))
        rmse = float(np.sqrt(np.mean((pt_a - pp_a) ** 2)))
        sr   = _spearmanr(pt_a, pp_a)
        pr_  = _pearsonr(pt_a, pp_a)
        spear = float(getattr(sr,  "statistic", getattr(sr,  "correlation", 0.0)))
        pear  = float(getattr(pr_, "statistic", pr_[0] if isinstance(pr_, tuple) else 0.0))
        ci    = _ci_fast(pt_a, pp_a)
        n_s   = min(len(pt_a), 500)
        idx_s = (np.random.choice(len(pt_a), n_s, replace=False)
                 if len(pt_a) > n_s else np.arange(len(pt_a)))
        scatter = list(zip(pt_a[idx_s].tolist(), pp_a[idx_s].tolist()))
    else:
        mae = rmse = spear = pear = ci = 0.0
        scatter = []

    return {
        "metrics": {
            "Variant": label, "AUC": auc, "AUPRC": auprc,
            "EF1%": ef1, "EF5%": ef5, "BEDROC": bedr,
            "MAE": mae, "RMSE": rmse, "Spearman": spear,
            "Pearson": pear, "CI": ci,
        },
        "roc":     roc_pts,
        "pr":      pr_pts,
        "scatter": scatter,
    }


# ── Custom forward passes for ablation variants ────────────────────────────

@torch.no_grad()
def _infer(
    model: TargetTrace3,
    variant: str,
    fp_t:    torch.Tensor,      # (2B, FP_DIM)
    esm_t:   torch.Tensor,      # (2B, ESM_DIM)
    res_t:   torch.Tensor,      # (2B, L, ESM_DIM)
    mask_t:  torch.Tensor,      # (2B, L)
    p_mean: float,
    p_std:  float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns act_scores (2B,), pic50_pos (B,), pic50_neg (B,) in original scale.
    First B rows = positive pairs, last B rows = negatives.
    """
    B = fp_t.shape[0] // 2
    model.eval()

    with autocast(device_type="cuda", enabled=USE_AMP):
        if variant == "full":
            logit, p50 = model(fp_t, esm_t, res_t, mask_t)

        elif variant == "no_xattn":
            # Force pooled-ESM path by passing residues=None
            logit, p50 = model(fp_t, esm_t, residues=None, pad_mask=None)

        elif variant == "no_hadamard":
            # Same weights but zero out the mol*prot interaction term
            mol  = model.fp_enc(fp_t)
            # Use cross-attention (residues present)
            prot = model.cross_attn(mol, res_t, mask_t)
            zeros = torch.zeros_like(mol)
            fused = model.fusion(torch.cat([mol, prot, zeros], dim=1))
            logit = model.act_head(fused).squeeze(1)
            p50   = model.pic50_head(fused).squeeze(1)

        elif variant == "ecfp4_only":
            # Zero-pad everything outside the ECFP4 block [2048:4096]
            fp_masked = torch.zeros_like(fp_t)
            fp_masked[:, 2048:4096] = fp_t[:, 2048:4096]
            logit, p50 = model(fp_masked, esm_t, res_t, mask_t)

        else:
            raise ValueError(f"Unknown variant: {variant}")

    scores  = torch.sigmoid(logit).cpu().float().numpy()           # (2B,)
    p50_arr = p50.cpu().float().numpy() * p_std + p_mean           # (2B,)
    return scores, p50_arr[:B], p50_arr[B:]                        # pos only for pIC50


# ── GPU baseline helpers ───────────────────────────────────────────────────

def _build_X_fast(
    ds: "TargetTrace3Dataset",
    max_pairs: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorised feature extraction — no Python loop over samples.
    Returns X_pos (n, FP_DIM+ESM_DIM), X_neg, pic50_arr (original scale).
    Positive = real (mol, prot) pair. Negative = same mol, shuffled prot.
    """
    # Build unique FP and ESM matrices once, then index with numpy fancy-indexing
    unique_smis = list(ds._fp.keys())
    fp_matrix   = np.stack([ds._fp[s]  for s in unique_smis], dtype=np.float32)
    smi_to_i    = {s: i for i, s in enumerate(unique_smis)}

    unique_keys = list(ds._prot.keys())
    esm_matrix  = np.stack([ds._prot[k] for k in unique_keys], dtype=np.float32)
    key_to_i    = {k: i for i, k in enumerate(unique_keys)}

    smi_idx  = np.array([smi_to_i.get(s, -1)  for s in ds._smiles_arr],   dtype=np.int32)
    prot_idx = np.array([key_to_i.get(k, -1)  for k in ds._prot_key_arr], dtype=np.int32)
    valid    = (smi_idx >= 0) & (prot_idx >= 0)

    smi_idx  = smi_idx[valid]
    prot_idx = prot_idx[valid]
    pic50s   = ds._pic50_arr[valid]     # normalised; decode later

    rng = np.random.default_rng(seed)
    if max_pairs and len(smi_idx) > max_pairs:
        sel      = rng.choice(len(smi_idx), max_pairs, replace=False)
        smi_idx  = smi_idx[sel]
        prot_idx = prot_idx[sel]
        pic50s   = pic50s[sel]

    fps_arr  = fp_matrix[smi_idx]      # (n, FP_DIM)
    esms_arr = esm_matrix[prot_idx]    # (n, ESM_DIM)

    neg_prot = rng.permutation(len(prot_idx))
    same     = neg_prot == np.arange(len(prot_idx))
    neg_prot[same] = (neg_prot[same] + 1) % len(prot_idx)

    X_pos = np.hstack([fps_arr, esms_arr])
    X_neg = np.hstack([fps_arr, esm_matrix[prot_idx[neg_prot]]])
    return X_pos, X_neg, pic50s


def _gpu_fit_predict(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray,
    hidden:  int = 0,           # 0 → logistic regression, >0 → MLP with that hidden dim
    n_epochs: int = 300,
    lr:       float = 3e-3,
    batch_size: int = 4096,
    weight_decay: float = 1e-4,
) -> np.ndarray:
    """
    GPU-accelerated classifier (Adam + BCE).
    hidden=0  → Logistic Regression   (nn.Linear → sigmoid)
    hidden>0  → 2-layer MLP           (Linear → SiLU → BN → Linear → sigmoid)
    Returns predicted probabilities for X_val.
    """
    # Feature-wise standardise on CPU (fast, no memory pressure)
    mu  = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    sig = (X_train.std( axis=0, keepdims=True) + 1e-7).astype(np.float32)
    Xtr = torch.tensor((X_train - mu) / sig, dtype=torch.float32)
    ytr = torch.tensor(y_train,              dtype=torch.float32)
    Xva = torch.tensor((X_val   - mu) / sig, dtype=torch.float32, device=device)

    d = Xtr.shape[1]
    if hidden == 0:
        net = nn.Linear(d, 1)
    else:
        net = nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(),
            nn.BatchNorm1d(hidden), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    n = len(Xtr)
    net.train()
    for _ in range(n_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb  = Xtr[idx].to(device, non_blocking=True)
            yb  = ytr[idx].to(device, non_blocking=True)
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(xb).squeeze(1), yb
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        scores = torch.sigmoid(net(Xva).squeeze(1)).cpu().numpy()
    del net, Xtr, ytr, Xva
    torch.cuda.empty_cache()
    return scores


def _gpu_fit_predict_reg(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray,
    hidden: int = 256,
    n_epochs: int = 300, lr: float = 3e-3, batch_size: int = 4096,
) -> np.ndarray:
    """GPU MLP regressor for pIC50. Returns predictions for X_val."""
    mu  = X_train.mean(axis=0, keepdims=True).astype(np.float32)
    sig = (X_train.std( axis=0, keepdims=True) + 1e-7).astype(np.float32)
    Xtr = torch.tensor((X_train - mu) / sig, dtype=torch.float32)
    ytr = torch.tensor(y_train,              dtype=torch.float32)
    Xva = torch.tensor((X_val   - mu) / sig, dtype=torch.float32, device=device)

    d   = Xtr.shape[1]
    net = nn.Sequential(
        nn.Linear(d, hidden), nn.SiLU(),
        nn.BatchNorm1d(hidden), nn.Dropout(0.2),
        nn.Linear(hidden, 1),
    ).to(device)
    opt  = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.HuberLoss(delta=1.0)

    n = len(Xtr)
    net.train()
    for _ in range(n_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb  = Xtr[idx].to(device, non_blocking=True)
            yb  = ytr[idx].to(device, non_blocking=True)
            loss = crit(net(xb).squeeze(1), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        preds = net(Xva).squeeze(1).cpu().numpy()
    del net, Xtr, ytr, Xva
    torch.cuda.empty_cache()
    return preds


# ── Main benchmark function ────────────────────────────────────────────────

def run_benchmark(
    model:       TargetTrace3,
    norm_stats:  dict,
    max_samples: int = 20_000,
    status_cb:   Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Rebuild val set from DB, run all variants, return (metrics_df, curves, scatter).

    metrics_df  columns: Variant, AUC, AUPRC, EF1%, EF5%, BEDROC,
                         MAE, RMSE, Spearman, Pearson, CI
    curves      {variant_label: {"roc": [(fpr,tpr)…], "pr": [(rec,prec)…]}}
    scatter     {variant_label: [(true_pic50, pred_pic50)…]}
    """
    if not _SKLEARN_OK:
        return pd.DataFrame(), {}, {}

    def _st(msg):
        if status_cb:
            status_cb(msg)

    t0 = _time.time()
    p_mean = norm_stats.get("pic50_mean", 7.0)
    p_std  = norm_stats.get("pic50_std",  1.5)

    # ── 1. Rebuild the same val split ──────────────────────────────────────
    _st("Benchmark 1/5 — Loading data from database…")
    df = get_training_data()
    df = df[df["target_name"].notna() & df["smiles"].notna()].reset_index(drop=True)
    if max_samples and len(df) > max_samples:
        df = _fast_stratified_sample(df, max_samples)
    train_df, val_df = _scaffold_split(df, val_frac=0.2, seed=42)

    tgt_seq_map = df.groupby("target_name")["protein_sequence"].first().fillna("").astype(str)
    tgt_to_key  = {t: _seq_key(s) for t, s in tgt_seq_map.items()}
    all_targets = sorted(tgt_to_key.keys())
    all_keys    = [tgt_to_key[t] for t in all_targets]
    unique_keys = list(dict.fromkeys(all_keys))
    prot_key_to_idx = {k: i for i, k in enumerate(unique_keys)}

    _st(f"Benchmark 1/5 — {len(train_df):,} train / {len(val_df):,} val  ({_time.time()-t0:.1f}s)")

    # ── 2. Load caches ─────────────────────────────────────────────────────
    _st("Benchmark 2/5 — Loading feature caches…")
    needed_smi  = set(df["smiles"].astype(str).unique())
    fp_cache    = load_fp_subset(needed_smi)
    prot_pool   = _load(_PROT_POOLED)
    needed_keys = set(unique_keys)
    residue_cache = load_residues_subset(needed_keys)
    _st(f"Benchmark 2/5 — Caches loaded  ({_time.time()-t0:.1f}s)")

    # ── 3. Build datasets ──────────────────────────────────────────────────
    _st("Benchmark 3/5 — Building datasets…")
    train_ds = TargetTrace3Dataset(train_df, p_mean, p_std, fp_cache,
                                   prot_pool.copy(), prot_key_to_idx, tgt_to_key)
    val_ds   = TargetTrace3Dataset(val_df,   p_mean, p_std, fp_cache,
                                   prot_pool,        prot_key_to_idx, tgt_to_key)

    # GPU residue tensors
    max_L       = max((r.shape[0] for r in residue_cache.values()), default=1)
    N_prot      = len(unique_keys)
    prot_res_t  = torch.zeros(N_prot, max_L, ESM_DIM, device=device, dtype=torch.float32)
    prot_mask_t = torch.ones( N_prot, max_L,           device=device, dtype=torch.bool)
    for k, i in prot_key_to_idx.items():
        r = residue_cache.get(k)
        if r is not None:
            L = r.shape[0]
            prot_res_t[i, :L]  = torch.tensor(r, dtype=torch.float32)
            prot_mask_t[i, :L] = False
    del residue_cache
    gc.collect()
    _st(f"Benchmark 3/5 — Datasets ready  ({_time.time()-t0:.1f}s)")

    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            collate_fn=_collate, num_workers=0,
                            pin_memory=(device.type == "cuda"))

    # ── 4. Collect val predictions for TargetTrace3 variants ───────────────
    _st("Benchmark 4/5 — Running TargetTrace3 variants…")

    VARIANTS = [
        ("TargetTrace3 (Full)",    "full"),
        ("No Cross-Attention",     "no_xattn"),
        ("No Hadamard Fusion",     "no_hadamard"),
        ("ECFP4 Only (2048-dim)",  "ecfp4_only"),
    ]

    variant_results: dict[str, dict] = {}

    for label, vkey in VARIANTS:
        _st(f"Benchmark 4/5 — {label}…")
        y_true, y_score, pt, pp = [], [], [], []

        for batch in val_loader:
            fp, esm_p, prot_idx, pic50s = [b.to(device, non_blocking=True) for b in batch]
            B = len(fp)

            # Positive residues
            pos_res  = prot_res_t[prot_idx]
            pos_mask = prot_mask_t[prot_idx]

            # Negatives: shuffle proteins (same logic as training)
            neg_perm = torch.randperm(B, device=device)
            same     = neg_perm == torch.arange(B, device=device)
            neg_perm[same] = (neg_perm[same] + 1) % B
            neg_res  = prot_res_t[prot_idx[neg_perm]]
            neg_mask = prot_mask_t[prot_idx[neg_perm]]

            fp2   = torch.cat([fp,    fp   ])
            esm2  = torch.cat([esm_p, esm_p])
            res2  = torch.cat([pos_res,  neg_res ])
            mask2 = torch.cat([pos_mask, neg_mask])

            scores, p50_pos, _ = _infer(model, vkey, fp2, esm2, res2, mask2,
                                        p_mean, p_std)
            y_true.extend([1] * B + [0] * B)
            y_score.extend(scores.tolist())

            for t, p in zip(pic50s.cpu().numpy(), p50_pos):
                if not np.isnan(float(t)):
                    pt.append(float(t) * p_std + p_mean)
                    pp.append(float(p))

        variant_results[label] = _compute_metrics(y_true, y_score, pt, pp, label)

    # ── 5. GPU baseline classifiers ────────────────────────────────────────
    # Vectorised feature extraction — numpy fancy-indexing, no Python loops
    _st("Benchmark 5/5 — Extracting features (vectorised)…")
    MAX_PAIRS = 30_000          # cap training pairs per set for speed
    tr_X_pos, tr_X_neg, tr_pic50_n = _build_X_fast(train_ds, MAX_PAIRS, seed=0)
    va_X_pos, va_X_neg, va_pic50_n = _build_X_fast(val_ds,   None,      seed=1)

    X_train = np.vstack([tr_X_pos, tr_X_neg]).astype(np.float32)
    y_train = np.array([1] * len(tr_X_pos) + [0] * len(tr_X_neg), dtype=np.float32)
    X_val   = np.vstack([va_X_pos, va_X_neg]).astype(np.float32)
    y_val   = np.array([1] * len(va_X_pos) + [0] * len(va_X_neg), dtype=np.int32)

    # Decode normalised pIC50 back to original scale
    tr_pic50 = tr_pic50_n * p_std + p_mean
    va_pic50 = va_pic50_n * p_std + p_mean

    _st(f"Benchmark 5/5 — {len(tr_X_pos):,} train pairs, {len(va_X_pos):,} val pairs  "
        f"({_time.time()-t0:.1f}s)")

    GPU_VARIANTS = [
        ("GPU Logistic Regression", 0),    # hidden=0 → linear model
        ("GPU MLP (2-layer)",     256),    # hidden=256 → non-linear baseline
    ]

    for label, hidden_dim in GPU_VARIANTS:
        _st(f"Benchmark 5/5 — Fitting {label} on {device}…")
        scores = _gpu_fit_predict(X_train, y_train, X_val, hidden=hidden_dim)

        # pIC50 regression on positive pairs only (GPU MLP regressor)
        pt_ml, pp_ml = [], []
        valid_tr = ~np.isnan(tr_pic50)
        valid_va = ~np.isnan(va_pic50)
        if valid_tr.sum() >= 2 and valid_va.sum() >= 1:
            preds = _gpu_fit_predict_reg(
                tr_X_pos[valid_tr], tr_pic50[valid_tr], va_X_pos
            )
            pt_ml = va_pic50[valid_va].tolist()
            pp_ml = preds[valid_va].tolist()

        variant_results[label] = _compute_metrics(
            y_val.tolist(), scores.tolist(), pt_ml, pp_ml, label
        )

    # Random baseline
    rng_scores = np.random.default_rng(42).random(len(y_val))
    variant_results["Random Baseline"] = _compute_metrics(
        y_val.tolist(), rng_scores.tolist(), [], [], "Random Baseline"
    )

    # ── Assemble outputs ───────────────────────────────────────────────────
    all_labels = ([v[0] for v in VARIANTS] +
                  [v[0] for v in GPU_VARIANTS] +
                  ["Random Baseline"])

    rows, curves, scatter = [], {}, {}
    for lbl in all_labels:
        if lbl not in variant_results:
            continue
        r = variant_results[lbl]
        rows.append(r["metrics"])
        curves[lbl]  = {"roc": r["roc"], "pr": r["pr"]}
        scatter[lbl] = r["scatter"]

    metrics_df = pd.DataFrame(rows)

    elapsed = _time.time() - t0
    _st(f"Benchmark complete — {len(rows)} variants, {elapsed:.0f}s")
    return metrics_df, curves, scatter
