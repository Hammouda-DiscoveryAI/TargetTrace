"""
Run TargetTrace on PDBbind and save per-pair scores to results/pdbbind_scores.csv.
All downstream analysis scripts load this file instead of re-running inference.
"""
import gc, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from trainer      import load_latest, device, USE_AMP, _collate
from embedders    import (_seq_key, _load, _PROT_POOLED, load_residues_subset,
                          unload_models, ESM_DIM, embed_proteins)
from features     import load_fp_subset, precompute_fps
from evaluate_external import _ExternalDataset

OUT = Path(__file__).parent / "results" / "pdbbind_scores.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)


def run(csv_path: str = "pdbbind_eval.csv") -> pd.DataFrame:
    print("Loading model…")
    model, _, _, _, norm_stats = load_latest()
    assert model is not None, "No valid checkpoint found"
    p_mean = norm_stats.get("pic50_mean", 7.0)
    p_std  = norm_stats.get("pic50_std",  1.5)
    model.eval()

    df = pd.read_csv(csv_path)
    df["seq_key"] = df["protein_sequence"].astype(str).map(_seq_key)

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
            L = r.shape[0]; res_t[i, :L] = torch.tensor(r); mask_t[i, :L] = False
    del res_cache; gc.collect()

    ds = _ExternalDataset(df, p_mean, p_std, fp_cache, prot_pool, idx_map)
    loader = DataLoader(ds, batch_size=256, shuffle=False,
                        collate_fn=_collate, num_workers=0)
    del fp_cache, prot_pool; gc.collect()

    scores, pic50_preds, pic50_true = [], [], []
    with torch.no_grad():
        for fp, esm_p, pidx, p50s in loader:
            fp, esm_p, pidx, p50s = [b.to(device) for b in [fp, esm_p, pidx, p50s]]
            with autocast(device_type="cuda", enabled=USE_AMP):
                s, p50p = model(fp, esm_p, res_t[pidx], mask_t[pidx])
            scores.extend(torch.sigmoid(s).cpu().float().tolist())
            pic50_preds.extend((p50p.cpu().float() * p_std + p_mean).tolist())
            pic50_true.extend((p50s.cpu().float() * p_std + p_mean).tolist())

    out_df = df[["smiles", "uniprot", "target_name", "active", "pic50",
                 "pdb_id", "ligand_id"]].copy()
    out_df["score"]      = scores
    out_df["pic50_pred"] = pic50_preds
    out_df["pic50_true"] = pic50_true
    out_df.to_csv(OUT, index=False)
    print(f"Saved {len(out_df)} rows → {OUT}")
    return out_df


if __name__ == "__main__":
    run()
