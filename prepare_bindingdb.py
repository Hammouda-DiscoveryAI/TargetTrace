"""
Prepare BindingDB for TargetTrace3 external evaluation.

Download the full dataset first:
  https://www.bindingdb.org/bind/downloads.jsp
  → "BindingDB_All.tsv.zip"  (~500 MB zip, ~5 GB unzipped)

Then run:
  python prepare_bindingdb.py BindingDB_All.tsv

Outputs
-------
  bindingdb_eval.csv   — ready for evaluate_external.py
    columns: smiles, protein_sequence, active, pic50

Active threshold: Ki or IC50 ≤ 10,000 nM (10 µM), consistent with ChEMBL.
pIC50 = 9 − log10(activity_nM).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

# BindingDB column names (as of 2024 release)
_SMILES_COL   = "Ligand SMILES"
_SEQ_COL      = "Target Sequence"
_KI_COL       = "Ki (nM)"
_IC50_COL     = "IC50 (nM)"
_KD_COL       = "Kd (nM)"
_EC50_COL     = "EC50 (nM)"
_TGT_COL      = "Target Name"

_ACTIVE_THRESHOLD_NM = 10_000.0   # 10 µM
_CHUNK = 200_000


def _parse_nm(val) -> float | None:
    """Parse a BindingDB activity string like '123', '>1000', '<0.5' → float nM."""
    if pd.isna(val):
        return None
    s = str(val).strip().lstrip("><~≤≥").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _best_activity(row) -> float | None:
    """Return the most reliable activity value: Ki > Kd > IC50 > EC50."""
    for col in (_KI_COL, _KD_COL, _IC50_COL, _EC50_COL):
        v = _parse_nm(row.get(col))
        if v is not None and v > 0:
            return v
    return None


def _pic50(nm: float) -> float:
    return 9.0 - math.log10(nm)


def _canonical(smi: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


def prepare(tsv_path: str | Path, out_path: str | Path = "bindingdb_eval.csv",
            active_threshold_nm: float = _ACTIVE_THRESHOLD_NM,
            max_rows: int | None = None) -> pd.DataFrame:

    tsv_path = Path(tsv_path)
    out_path = Path(out_path)

    if not tsv_path.exists():
        print(f"ERROR: {tsv_path} not found.", file=sys.stderr)
        print("Download from https://www.bindingdb.org/bind/downloads.jsp", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {tsv_path} in chunks…")
    needed_cols = [_SMILES_COL, _SEQ_COL, _KI_COL, _IC50_COL, _KD_COL, _EC50_COL]

    rows = []
    total_read = 0

    for chunk in pd.read_csv(
        tsv_path,
        sep="\t",
        usecols=lambda c: c in needed_cols,
        low_memory=False,
        chunksize=_CHUNK,
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
    ):
        chunk = chunk.rename(columns=str.strip)

        # Must have SMILES and sequence
        chunk = chunk[chunk[_SMILES_COL].notna() & chunk[_SEQ_COL].notna()]

        for _, row in chunk.iterrows():
            activity = _best_activity(row)
            if activity is None:
                continue
            smi = _canonical(str(row[_SMILES_COL]))
            if smi is None:
                continue
            seq = str(row[_SEQ_COL]).strip()
            if len(seq) < 10:
                continue
            # Only standard amino-acid sequences
            if not set(seq.upper()).issubset(set("ACDEFGHIKLMNPQRSTVWY")):
                continue

            active = int(activity <= active_threshold_nm)
            pic50  = _pic50(activity) if active else float("nan")
            rows.append({"smiles": smi, "protein_sequence": seq,
                         "active": active, "pic50": pic50})

        total_read += len(chunk)
        n_kept = len(rows)
        print(f"  Processed {total_read:,} rows → {n_kept:,} kept…", end="\r")

        if max_rows and total_read >= max_rows:
            break

    print()
    if not rows:
        print("ERROR: no valid rows extracted.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    before = len(df)

    # Deduplicate (smiles, protein_sequence) — keep row with highest activity
    df["_act_num"] = pd.to_numeric(df["pic50"], errors="coerce").fillna(0)
    df = (df.sort_values("_act_num", ascending=False)
            .drop_duplicates(subset=["smiles", "protein_sequence"])
            .drop(columns=["_act_num"])
            .reset_index(drop=True))

    n_act  = int(df["active"].sum())
    n_inact = len(df) - n_act

    print(f"\nDone. {before:,} rows → {len(df):,} unique pairs")
    print(f"  Actives  (≤ {active_threshold_nm/1000:.0f} µM): {n_act:,}")
    print(f"  Inactives:                               {n_inact:,}")
    print(f"  Saved → {out_path}")

    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", help="Path to BindingDB_All.tsv")
    ap.add_argument("--out",       default="bindingdb_eval.csv",
                    help="Output CSV path (default: bindingdb_eval.csv)")
    ap.add_argument("--threshold", type=float, default=_ACTIVE_THRESHOLD_NM,
                    help="Active threshold in nM (default: 10000)")
    ap.add_argument("--max_rows",  type=int,   default=None,
                    help="Stop after reading this many raw rows (for testing)")
    args = ap.parse_args()

    prepare(args.tsv, out_path=args.out,
            active_threshold_nm=args.threshold, max_rows=args.max_rows)
