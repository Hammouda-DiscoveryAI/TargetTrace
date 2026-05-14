# TargetTrace

**Drug-Target Interaction (DTI) scoring powered by cross-attention over ESM-2 protein residues.**

TargetTrace is an interactive Streamlit application that learns to predict whether a small molecule is active against a protein target and estimates its potency (pIC50). It combines multi-radius molecular fingerprints with per-residue ESM-2 protein embeddings through a molecule-queries-protein cross-attention mechanism, then scores the interaction using a fused neural network head.

---

## How It Works

```
Molecule (SMILES)                    Protein (sequence / UniProt)
     │                                         │
Multi-radius ECFP + FCFP              ESM-2 (480-dim residues)
(8,379-dim fingerprint)                        │
     │                                   Cross-Attention
     └──────────────── Hadamard fusion ──────────┘
                              │
                   128-dim joint representation
                    ┌─────────┴──────────┐
               P(active)            pIC50 estimate
```

**Key design choices:**

| Component | Detail |
|---|---|
| Molecule features | ECFP2 + ECFP4 + ECFP6 + FCFP4 + MACCS + 20 physicochemical descriptors = 8,379 dims |
| Protein features | ESM-2 (`esm2_t12_35M_UR50D`, 480-dim) — pooled CLS + per-residue tensors |
| Cross-attention | Molecule queries protein residues (4 heads) → learns binding-site context per molecule |
| Fusion | `cat[mol_enc, prot_ctx, mol_enc ⊙ prot_ctx]` → Hadamard interaction term |
| Training | Inline negative sampling + scaffold split (Murcko) to prevent data leakage |
| Targets | AUC-ROC (binary activity) + pIC50 MAE (regression) |

---

## Quick Start

### 1. Create the environment

**Conda (recommended):**
```bash
conda env create -f environment.yml
conda activate targettrace
```

**pip only:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **GPU users:** after activating the environment, replace the CPU PyTorch build:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Seed the database

Pull bioactivity data from ChEMBL for an initial set of targets (one-time step):
```bash
python bootstrap.py --limit 200
```

### 3. Launch the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Project Structure

```
TargetTrace/
├── app.py                  # Streamlit UI — scoring, training, data import
├── model.py                # TargetTrace3 neural network architecture
├── trainer.py              # Training loop, scaffold split, evaluation
├── embedders.py            # ESM-2 protein embedding + residue cache
├── features.py             # Molecular fingerprint computation + cache
├── database.py             # SQLite bioactivity database helpers
├── chembl_client.py        # ChEMBL REST API wrappers
├── bootstrap.py            # One-time DB seeding from ChEMBL CSV chunks
├── mol_gnn.py              # Auxiliary GNN module
│
├── benchmark.py            # Benchmark runner
├── evaluate_external.py    # External dataset evaluation
├── run_ablation.py         # Ablation study runner
├── compute_pdbbind_scores.py
├── download_pdbbind.py
├── prepare_bindingdb.py
├── profile_train.py
│
├── plot_*.py               
├── draw_architecture.py
├── make_all_figures.py
│
├── requirements.txt        # pip dependencies
├── environment.yml         # conda environment
└── results/                # Evaluation outputs and figures
```

---

## Features

- **Score a molecule** against all known targets or a custom protein sequence
- **Live training** — add new compound-protein pairs from ChEMBL directly in the UI and retrain on-the-fly
- **Attention visualization** — inspect which residues the model attends to for any (molecule, protein) pair
- **PDBbind / BindingDB evaluation** — scripts to benchmark against external curated datasets
- **Ablation studies** — compare architectures (no cross-attention, no Hadamard, fingerprint subsets)

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CPU or CUDA)
- RDKit 2023.3+
- See `environment.yml` or `requirements.txt` for the full list

---

## Citation

If you use TargetTrace in your research, please cite this repository.
