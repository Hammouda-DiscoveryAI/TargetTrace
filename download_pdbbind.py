"""
Download PDBbind/BindingDB affinity data for exactly the targets TargetTrace was
trained on.

Strategy
--------
1. Read the 425 training targets (UniProt accession + protein sequence) from
   bioactivity.db.
2. Search RCSB PDB for entries whose polymer chains map to those UniProts AND
   that carry experimental binding-affinity annotations (Ki / Kd / IC50 from
   BindingDB, sourced through RCSB).
3. For every matching PDB entry: fetch ligand SMILES and the tightest affinity
   measurement via the RCSB GraphQL API.
4. Write pdbbind_eval.csv using the TRAINING sequence (not the PDB chain
   sequence), so the model's ESM-2 embeddings are identical to training.

Usage
-----
  python download_pdbbind.py                          # full run
  python download_pdbbind.py --out my_eval.csv
  python download_pdbbind.py --max-entries 100        # quick test
  python download_pdbbind.py --threshold 1000         # tighter activity cutoff

Output
------
  pdbbind_eval.csv — compatible with evaluate_external.py
    columns: smiles, protein_sequence, active, pic50,
             pdb_id, ligand_id, uniprot, target_name
"""

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from rdkit import Chem

# ── Config ─────────────────────────────────────────────────────────────────────

_DB_PATH        = Path(__file__).parent / "bioactivity.db"
_RCSB_GRAPHQL   = "https://data.rcsb.org/graphql"
_RCSB_SEARCH    = "https://search.rcsb.org/rcsbsearch/v2/query"
_RCSB_CHEMCOMP  = "https://data.rcsb.org/rest/v1/core/chemcomp/{cid}"

_ACTIVE_NM      = 10_000.0    # 10 µM threshold for active label
_BATCH_SIZE     = 100          # PDB IDs per GraphQL call
_PAGE_SIZE      = 250          # RCSB search results per page
_SLEEP          = 0.25         # seconds between API batches
_RETRY          = 3

# Solvent / ion / cofactor component IDs to skip
_SKIP_LIGS = {
    "HOH","EDO","GOL","PEG","PO4","SO4","MG","ZN","CA","CL","NA","K","MN",
    "FE","CU","BR","IOD","F","ACE","NH2","ACT","TRS","EPE","MES","BME","DMS",
    "MSE","SEP","TPO","PTR","FMT","IMD","CIT","ADP","ATP","GTP","GDP","FAD",
    "NAD","HEM","CO","NI","NO3","FLC","FCY","EOH","MPD","IPA","PE4","PGE",
    "1PE","P6G",
}

_UNIT_TO_NM = {"nm": 1.0, "um": 1e3, "µm": 1e3, "mm": 1e6, "pm": 1e-3}


# ── Database helpers ───────────────────────────────────────────────────────────

def load_training_targets() -> dict[str, dict]:
    """
    Return {uniprot_accession: {target_name, protein_sequence}}
    for all 425 training targets.
    """
    import sqlite3
    if not _DB_PATH.exists():
        sys.exit(f"ERROR: database not found at {_DB_PATH}")

    con = sqlite3.connect(str(_DB_PATH))
    rows = con.execute(
        """SELECT MAX(uniprot_accession)   AS uniprot,
                  target_name,
                  MAX(protein_sequence)    AS seq
           FROM bioactivity
           WHERE uniprot_accession  IS NOT NULL
             AND uniprot_accession  != ''
             AND protein_sequence   IS NOT NULL
             AND protein_sequence   != ''
           GROUP BY target_name"""
    ).fetchall()
    con.close()

    targets = {}
    for uniprot, name, seq in rows:
        if uniprot and seq:
            targets[uniprot] = {"target_name": name, "protein_sequence": seq}
    return targets


# ── RCSB search ────────────────────────────────────────────────────────────────

def search_pdb_for_targets(uniprots: list[str],
                            session: requests.Session) -> list[str]:
    """
    Find all PDB entries that:
      - are associated with at least one UniProt in `uniprots`
      - have at least one binding-affinity annotation (Ki/Kd/IC50 from BindingDB)

    Returns a de-duplicated list of PDB IDs (uppercase).
    """
    print(f"Searching RCSB for PDB entries matching {len(uniprots)} training targets…")

    query_payload = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_binding_affinity.value",
                        "operator":  "greater",
                        "value":     0,
                        "negation":  False,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers"
                            ".reference_sequence_identifiers.database_accession"
                        ),
                        "operator": "in",
                        "value":    uniprots,
                        "negation": False,
                    },
                },
            ],
        },
        "return_type": "entry",
    }

    all_ids  = []
    start    = 0

    while True:
        payload = dict(query_payload)
        payload["request_options"] = {
            "paginate": {"start": start, "rows": _PAGE_SIZE}
        }

        for attempt in range(_RETRY):
            try:
                r = session.post(_RCSB_SEARCH, json=payload, timeout=30)
                r.raise_for_status()
                data = r.json()
                break
            except Exception as exc:
                if attempt == _RETRY - 1:
                    print(f"  [WARN] search failed: {exc}", file=sys.stderr)
                    data = {}
                time.sleep(2 ** attempt)

        hits = [h["identifier"] for h in (data.get("result_set") or [])]
        all_ids.extend(hits)
        total = data.get("total_count", 0)
        print(f"  …fetched {len(all_ids)} / {total}")

        if len(all_ids) >= total or not hits:
            break
        start += _PAGE_SIZE

    return list(dict.fromkeys(all_ids))   # deduplicate, preserve order


# ── RCSB GraphQL ───────────────────────────────────────────────────────────────

_GQL = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_binding_affinity {
      type
      value
      unit
      comp_id
    }
    polymer_entities {
      rcsb_polymer_entity_container_identifiers {
        reference_sequence_identifiers {
          database_accession
          database_name
        }
      }
    }
    nonpolymer_entities {
      nonpolymer_comp {
        chem_comp { id }
      }
    }
  }
}
"""


def gql_fetch(session: requests.Session, pdb_ids: list[str]) -> dict:
    """Batch-fetch entry data for up to _BATCH_SIZE PDB IDs."""
    payload = {"query": _GQL, "variables": {"ids": [p.upper() for p in pdb_ids]}}
    for attempt in range(_RETRY):
        try:
            r = session.post(_RCSB_GRAPHQL, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            return {
                e["rcsb_id"].upper(): e
                for e in (data.get("data", {}).get("entries") or [])
                if e is not None
            }
        except Exception as exc:
            if attempt == _RETRY - 1:
                print(f"  [WARN] GraphQL failed: {exc}", file=sys.stderr)
                return {}
            time.sleep(2 ** attempt)
    return {}


def best_affinity(entry: dict) -> tuple[float, str] | None:
    """
    Return (affinity_nM, ligand_comp_id) — the tightest (smallest nM) Ki/Kd
    annotation.  IC50 / EC50 are used only when no Ki/Kd is present.
    Returns None if no valid annotation found.
    """
    _pref = {"ki": 0, "kd": 1, "ic50": 2, "ec50": 3}
    best = None   # (priority, nm, comp_id)

    for ann in (entry.get("rcsb_binding_affinity") or []):
        val  = ann.get("value")
        unit = (ann.get("unit") or "nM").lower().replace("μ", "u")
        cid  = (ann.get("comp_id") or "").upper()
        atype = (ann.get("type") or "").lower()
        if val is None or val <= 0:
            continue
        mult = _UNIT_TO_NM.get(unit, 1.0)
        nm   = float(val) * mult
        prio = _pref.get(atype, 99)
        if best is None or prio < best[0] or (prio == best[0] and nm < best[1]):
            best = (prio, nm, cid)

    return (best[1], best[2]) if best else None


def entry_uniprots(entry: dict) -> set[str]:
    """Extract all UniProt accessions referenced by polymer entities."""
    out = set()
    for pe in (entry.get("polymer_entities") or []):
        idents = pe.get("rcsb_polymer_entity_container_identifiers") or {}
        for ref in (idents.get("reference_sequence_identifiers") or []):
            if ref.get("database_name") == "UniProt":
                acc = ref.get("database_accession")
                if acc:
                    out.add(acc)
    return out


def ligand_cids(entry: dict) -> list[str]:
    """Non-solvent chemical-component IDs from nonpolymer_entities."""
    out = []
    for ent in (entry.get("nonpolymer_entities") or []):
        cid = ((ent.get("nonpolymer_comp") or {})
               .get("chem_comp") or {}).get("id", "")
        if cid and cid.upper() not in _SKIP_LIGS:
            out.append(cid.upper())
    return out


# ── SMILES helper ──────────────────────────────────────────────────────────────

def get_smiles(session: requests.Session, cid: str,
               cache: dict[str, str | None]) -> str | None:
    if cid in cache:
        return cache[cid]
    for attempt in range(_RETRY):
        try:
            r = session.get(_RCSB_CHEMCOMP.format(cid=cid), timeout=15)
            if r.status_code == 404:
                cache[cid] = None
                return None
            r.raise_for_status()
            d   = r.json()
            smi = None
            for desc in (d.get("pdbx_chem_comp_descriptor") or []):
                if (desc.get("type") or "").upper() in ("SMILES_CANONICAL", "SMILES"):
                    smi = desc.get("descriptor")
                    if smi:
                        break
            if not smi:
                for desc in (d.get("rcsb_chem_comp_descriptor") or []):
                    smi = desc.get("smiles_stereo") or desc.get("smiles")
                    if smi:
                        break
            canon = _to_canon(smi) if smi else None
            cache[cid] = canon
            return canon
        except Exception as exc:
            if attempt == _RETRY - 1:
                cache[cid] = None
                print(f"  [WARN] SMILES fetch failed for {cid}: {exc}", file=sys.stderr)
                return None
            time.sleep(1)
    return None


def _to_canon(smi: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol) if mol else None
    except Exception:
        return None


# ── Main pipeline ──────────────────────────────────────────────────────────────

def build(out_path: Path,
          active_nm: float = _ACTIVE_NM,
          max_entries: int | None = None) -> pd.DataFrame:

    session = requests.Session()
    session.headers["User-Agent"] = "TargetTrace3-eval/1.0 (academic)"

    # ── 1. Load training targets ───────────────────────────────────────────────
    targets = load_training_targets()
    print(f"Loaded {len(targets)} training targets from database.")

    uniprot_list = list(targets.keys())

    # ── 2. Search RCSB for matching PDB entries ────────────────────────────────
    pdb_ids = search_pdb_for_targets(uniprot_list, session)
    print(f"Found {len(pdb_ids)} PDB entries with binding affinity data.\n")

    if max_entries:
        pdb_ids = pdb_ids[:max_entries]
        print(f"Limiting to first {max_entries} entries (--max-entries).\n")

    # ── 3. Batch-fetch entry data via GraphQL ──────────────────────────────────
    smiles_cache: dict[str, str | None] = {}
    rows        = []
    n_total     = len(pdb_ids)

    for batch_start in range(0, n_total, _BATCH_SIZE):
        batch = pdb_ids[batch_start: batch_start + _BATCH_SIZE]
        done  = min(batch_start + len(batch), n_total)
        print(f"  Processing entries {batch_start+1}–{done}/{n_total}"
              f"  ({len(rows)} rows so far)…", end="", flush=True)

        entry_data = gql_fetch(session, batch)

        for pdb_id in batch:
            entry = entry_data.get(pdb_id.upper())
            if not entry:
                continue

            # Which of our training targets does this PDB entry cover?
            pdb_uniprots = entry_uniprots(entry) & set(uniprot_list)
            if not pdb_uniprots:
                continue

            # Binding affinity
            aff = best_affinity(entry)
            if not aff:
                continue
            aff_nm, aff_cid = aff

            # Resolve SMILES: prefer the annotated comp_id, then structural ligs
            candidate_cids = []
            if aff_cid and aff_cid not in _SKIP_LIGS:
                candidate_cids.append(aff_cid)
            for cid in ligand_cids(entry):
                if cid not in candidate_cids:
                    candidate_cids.append(cid)

            smi = None
            used_cid = ""
            for cid in candidate_cids:
                smi = get_smiles(session, cid, smiles_cache)
                if smi:
                    used_cid = cid
                    break

            if not smi:
                continue

            active = int(aff_nm <= active_nm)
            pic50  = (9.0 - math.log10(aff_nm)) if active else float("nan")

            # Create one row per matching training target
            for uniprot in sorted(pdb_uniprots):
                tgt = targets[uniprot]
                rows.append({
                    "smiles":           smi,
                    "protein_sequence": tgt["protein_sequence"],
                    "active":           active,
                    "pic50":            pic50,
                    "pdb_id":           pdb_id.lower(),
                    "ligand_id":        used_cid,
                    "uniprot":          uniprot,
                    "target_name":      tgt["target_name"],
                })

        print()
        time.sleep(_SLEEP)

    if not rows:
        print("\nERROR: no valid rows produced.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)

    # Deduplicate (smiles, protein_sequence) — keep the row with highest pIC50
    df["_sort"] = pd.to_numeric(df["pic50"], errors="coerce").fillna(-99)
    df = (df.sort_values("_sort", ascending=False)
            .drop_duplicates(subset=["smiles", "protein_sequence"])
            .drop(columns=["_sort"])
            .reset_index(drop=True))

    n_act   = int(df["active"].sum())
    n_inact = len(df) - n_act

    print(f"\nResults")
    print(f"  Unique (smiles, sequence) pairs : {len(df):,}")
    print(f"  PDB entries covered             : {df['pdb_id'].nunique():,}")
    print(f"  Training targets covered        : {df['uniprot'].nunique():,} / {len(targets)}")
    print(f"  Actives  (≤ {active_nm/1000:.0f} µM)            : {n_act:,}")
    print(f"  Inactives                       : {n_inact:,}")
    print(f"  Saved → {out_path}")

    df.to_csv(out_path, index=False)
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Download PDBbind/BindingDB data for TargetTrace's training targets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_pdbbind.py                         # full run → pdbbind_eval.csv
  python download_pdbbind.py --max-entries 50        # quick smoke-test
  python download_pdbbind.py --threshold 1000        # stricter active cutoff (1 µM)
  python download_pdbbind.py --out results/pdbbind.csv

After downloading:
  python evaluate_external.py pdbbind_eval.csv --label PDBbind
        """,
    )
    ap.add_argument("--out",         default="pdbbind_eval.csv",
                    help="Output CSV (default: pdbbind_eval.csv)")
    ap.add_argument("--threshold",   type=float, default=_ACTIVE_NM,
                    help="Active threshold in nM (default: 10000)")
    ap.add_argument("--max-entries", type=int,   default=None,
                    help="Process at most N PDB entries (for testing)")
    args = ap.parse_args()

    build(
        out_path    = Path(args.out),
        active_nm   = args.threshold,
        max_entries = args.max_entries,
    )


if __name__ == "__main__":
    main()
