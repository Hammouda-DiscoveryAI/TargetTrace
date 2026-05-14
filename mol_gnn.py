"""
Graph Attention Network for molecules — pure PyTorch (no PyG required).
Molecules are represented as padded (MAX_ATOMS, feat_dim) tensors so the
entire batch can be processed with batched matrix multiplication.

GAT replaces GCN's uniform neighbourhood aggregation with learned per-edge
attention weights — each atom decides how much to weight each neighbour,
which is meaningfully better for chemistry (e.g. aromatic vs single bonds).

Graph tensors (atom features + adjacency) are cached to disk exactly like
fingerprints so Dataset.__init__ never calls smiles_to_graph more than once
per unique SMILES across all training runs.
"""
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import rdchem

_N_WORKERS = max(1, min(os.cpu_count() or 1, 8))

_GRAPH_CACHE      = Path(__file__).parent / "embed_cache" / "mol_graph.pkl"
_GRAPH_CACHE_KEYS = Path(__file__).parent / "embed_cache" / "mol_graph.keys.pkl"
_GRAPH_CACHE.parent.mkdir(exist_ok=True)

# ── Atom feature vocabulary ────────────────────────────────────────────────
_ATOM_TYPES  = ['C','N','O','S','F','P','Cl','Br','I','Se','Si','B']   # 12 + other = 13
_HYBRID      = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]                                                                        # 5 + other = 6
_BOND_VALS   = {
    rdchem.BondType.SINGLE:   1.0,
    rdchem.BondType.DOUBLE:   2.0,
    rdchem.BondType.TRIPLE:   3.0,
    rdchem.BondType.AROMATIC: 1.5,
}

# Each _one_hot adds len(choices)+1 (includes "other" bucket)
# 13 + 8 + 6 + 6 + 1 + 1 + 6 = 41
ATOM_FEAT_DIM = 13 + 8 + 6 + 6 + 1 + 1 + 6
MAX_ATOMS     = 50


def _one_hot(val, choices: list) -> list:
    vec = [0] * (len(choices) + 1)   # +1 for "other"
    try:
        vec[choices.index(val)] = 1
    except ValueError:
        vec[-1] = 1
    return vec


def _atom_feat(atom) -> np.ndarray:
    return np.array(
        _one_hot(atom.GetSymbol(),          _ATOM_TYPES)
        + _one_hot(atom.GetTotalDegree(),   list(range(7)))
        + _one_hot(atom.GetFormalCharge(),  [-2, -1, 0, 1, 2])
        + _one_hot(atom.GetHybridization(), _HYBRID)
        + [int(atom.GetIsAromatic())]
        + [int(atom.IsInRing())]
        + _one_hot(atom.GetTotalNumHs(),    list(range(5))),
        dtype=np.float32,
    )


def smiles_to_graph(smiles: str) -> tuple[np.ndarray, np.ndarray, int] | None:
    """
    Returns (node_feats, norm_adj, n_atoms) padded to MAX_ATOMS, or None for
    invalid / oversized molecules.

    Adjacency is symmetrically normalised: D^{-1/2} A D^{-1/2} with self-loops,
    so GCN reduces to a simple weighted average of neighbour features.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n = mol.GetNumAtoms()
    if n == 0 or n > MAX_ATOMS:
        return None

    nf = np.zeros((MAX_ATOMS, ATOM_FEAT_DIM), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        nf[i] = _atom_feat(atom)

    adj = np.zeros((MAX_ATOMS, MAX_ATOMS), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        v = _BOND_VALS.get(bond.GetBondType(), 1.0)
        adj[i, j] = v
        adj[j, i] = v
    for i in range(n):
        adj[i, i] = 1.0                                 # self-loop

    deg      = adj.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    adj = inv_sqrt[:, None] * adj * inv_sqrt[None, :]

    return nf, adj, n


def _load_graph_cache() -> dict:
    if _GRAPH_CACHE.exists():
        with open(_GRAPH_CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def _save_graph_keys(cache: dict) -> None:
    with open(_GRAPH_CACHE_KEYS, "wb") as f:
        pickle.dump(set(cache.keys()), f)


# Top-level so multiprocessing can pickle it
def _graph_worker(smi: str):
    return smi, smiles_to_graph(smi)


def precompute_graphs(smiles_list: list[str], status_cb=None) -> None:
    """
    Compute and cache graph tensors for all unique SMILES not already in cache.
    New SMILES are computed on all CPU cores via multiprocessing.
    """
    unique = [s for s in dict.fromkeys(smiles_list) if s]

    # Fast path: load the tiny keys file instead of the full 4 GB cache
    if _GRAPH_CACHE_KEYS.exists():
        with open(_GRAPH_CACHE_KEYS, "rb") as f:
            cached_keys = pickle.load(f)
        todo = [s for s in unique if s not in cached_keys]
        if not todo:
            return

    cache = _load_graph_cache()
    todo  = [s for s in unique if s not in cache]
    if not todo:
        _save_graph_keys(cache)
        return

    n = len(todo)
    if status_cb:
        status_cb(f"Graph tensors: computing {n:,} new SMILES on {_N_WORKERS} cores…")

    SAVE_EVERY = 10_000
    ctx = mp.get_context("fork")
    with ctx.Pool(_N_WORKERS) as pool:
        for i, (smi, graph) in enumerate(pool.imap(_graph_worker, todo, chunksize=500)):
            cache[smi] = graph
            if (i + 1) % SAVE_EVERY == 0:
                with open(_GRAPH_CACHE, "wb") as f:
                    pickle.dump(cache, f)
                if status_cb:
                    status_cb(f"Graph tensors: {i+1:,}/{n:,} ({_N_WORKERS} cores)")

    with open(_GRAPH_CACHE, "wb") as f:
        pickle.dump(cache, f)
    _save_graph_keys(cache)
    if status_cb:
        status_cb(f"Graph tensors: {n:,} new → {len(cache):,} total cached")


class MolGNN(nn.Module):
    """
    Three-layer Graph Convolutional Network with masked global mean+max pooling.
    Uses batched matrix multiply (bmm) on the normalised adjacency — extremely
    fast on GPU (no O(N²) attention allocations). SiLU activations and residual
    projections improve gradient flow vs the original ReLU GCN.
    """

    def __init__(self, in_dim: int = ATOM_FEAT_DIM,
                 hidden: int = 128, out_dim: int = 256, num_layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(dims[i + 1]) for i in range(num_layers)]
        )
        # residual projections (only needed when dimensions change)
        self.res = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1], bias=False) if dims[i] != dims[i + 1]
            else nn.Identity()
            for i in range(num_layers)
        ])
        self.drop      = nn.Dropout(0.1)
        self.pool_proj = nn.Linear(out_dim * 2, out_dim)

    def forward(self, x: torch.Tensor,
                adj: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, MAX_ATOMS, in_dim)
        adj  : (B, MAX_ATOMS, MAX_ATOMS)  — D^{-1/2} A D^{-1/2} normalised
        mask : (B, MAX_ATOMS)             — True for real atoms
        """
        for i, (conv, bn, res) in enumerate(zip(self.convs, self.bns, self.res)):
            h = conv(torch.bmm(adj, x))          # aggregate then project
            B, N, D = h.shape
            h = F.silu(bn(h.reshape(B * N, D)).reshape(B, N, D))
            x = h + res(x)                       # residual
            if i < len(self.convs) - 1:
                x = self.drop(x)

        mask_f    = mask.unsqueeze(-1).float()
        x_masked  = x * mask_f
        mean_pool = x_masked.sum(1) / mask_f.sum(1).clamp(min=1)

        # max pool: fill padding with -inf so padded atoms don't win.
        # But if ALL atoms are masked (graph=None case), -inf propagates through
        # the subsequent Linear → SiLU(-inf*w) = NaN. Clamp those rows to 0.
        has_atom = mask.any(dim=1)                          # (B,) bool
        x_maxfill = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_pool  = x_maxfill.max(1).values                 # (B, D), -inf for empty
        max_pool  = max_pool.masked_fill(~has_atom.unsqueeze(-1), 0.0)

        return self.pool_proj(torch.cat([mean_pool, max_pool], dim=1))
