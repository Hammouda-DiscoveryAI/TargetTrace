"""
TargetTrace3 — Drug-Target Interaction scorer.

Architecture improvements over vanilla concat-MLP:
  #1  Multi-radius ECFP (r=1,2,3) + FCFP4 pharmacophore fingerprints
      FP_DIM: 2225 → 8379 — captures local, medium, distal and pharmacophoric environments
  #2  Cross-attention pooling over ESM-2 residues
      Molecule embedding queries protein residues → learns which residues matter per molecule
      Falls back to prot_enc MLP when residue tensors are not passed (e.g. ad-hoc sequences)
  #3  Hadamard interaction term in fusion
      cat[mol, prot, mol*prot] → richer pairwise feature interactions without full bilinear cost

Task: DTI binary scoring — given (mol FP, protein ESM-2) → P(active) + pIC50.
Training uses inline negative sampling; inference scores mol vs all known proteins.
"""
import torch
import torch.nn as nn

from embedders import ESM_DIM
from features import FP_DIM

_MOL  = 256
_PROT = 256
_FUSE = 128


class _MolProtCrossAttn(nn.Module):
    """
    Molecule-queries-protein cross-attention.
    query  = mol embedding  (B, mol_dim)
    keys/values = ESM-2 residue hidden states  (B, L, res_dim)
    output = protein context vector  (B, out_dim)
    """
    def __init__(self, mol_dim: int = _MOL, res_dim: int = ESM_DIM,
                 n_heads: int = 4, out_dim: int = _PROT):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = out_dim // n_heads
        self.scale   = self.d_k ** -0.5

        self.q_proj   = nn.Linear(mol_dim, out_dim, bias=False)
        self.k_proj   = nn.Linear(res_dim, out_dim, bias=False)
        self.v_proj   = nn.Linear(res_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm     = nn.LayerNorm(out_dim)

    def forward(self, mol: torch.Tensor, residues: torch.Tensor,
                pad_mask: torch.Tensor | None = None,
                return_attn: bool = False):
        """
        mol      : (B, mol_dim)
        residues : (B, L, res_dim)
        pad_mask : (B, L)  True = padding (masked out)
        returns  : (B, out_dim)  or  ((B, out_dim), (B, H, L)) if return_attn
        """
        B, L, _ = residues.shape
        H, dk   = self.n_heads, self.d_k

        Q = self.q_proj(mol).view(B, H, dk)                           # (B, H, dk)
        K = self.k_proj(residues).view(B, L, H, dk).transpose(1, 2)  # (B, H, L, dk)
        V = self.v_proj(residues).view(B, L, H, dk).transpose(1, 2)  # (B, H, L, dk)

        attn = (Q.unsqueeze(2) @ K.transpose(-2, -1)) * self.scale   # (B, H, 1, L)
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask[:, None, None, :], float("-inf"))
        attn = torch.softmax(attn, dim=-1)                            # (B, H, 1, L)

        out = (attn @ V).squeeze(2).reshape(B, H * dk)               # (B, out_dim)
        out = self.norm(self.out_proj(out))
        if return_attn:
            return out, attn.squeeze(2)                               # (B, H, L)
        return out


class TargetTrace3(nn.Module):

    def __init__(self):
        super().__init__()

        # ── ① Fingerprint encoder (8379 → 1024 → 512 → 256) ───────────────
        # Two extra hidden layers to handle the larger sparse input
        self.fp_enc = nn.Sequential(
            nn.Linear(FP_DIM, 1024), nn.SiLU(), nn.BatchNorm1d(1024), nn.Dropout(0.4),
            nn.Linear(1024,    512), nn.SiLU(), nn.BatchNorm1d(512),  nn.Dropout(0.3),
            nn.Linear(512,    _MOL), nn.SiLU(),
        )

        # ── ② Protein encoder — fallback when residues not available ───────
        self.prot_enc = nn.Sequential(
            nn.Linear(ESM_DIM, 512), nn.SiLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512,    _PROT), nn.SiLU(),
        )

        # ── ② Cross-attention: two stacked layers, mol queries ESM-2 residues
        # Layer 1: mol embedding queries residues → protein context
        # Layer 2: protein context re-queries residues with updated representation
        # Residual connection preserves layer-1 signal.
        self.cross_attn   = _MolProtCrossAttn(mol_dim=_MOL,  res_dim=ESM_DIM,
                                              n_heads=4, out_dim=_PROT)
        self.cross_attn_2 = _MolProtCrossAttn(mol_dim=_PROT, res_dim=ESM_DIM,
                                              n_heads=4, out_dim=_PROT)

        # ── ③ Fusion with Hadamard interaction term (768 → 256 → 128) ──────
        # Input: cat[mol(256), prot(256), mol*prot(256)] = 768
        self.fusion = nn.Sequential(
            nn.Linear(_MOL + _PROT + _MOL, 256), nn.SiLU(),
            nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, _FUSE), nn.SiLU(),
        )

        # ── ④ Output heads ─────────────────────────────────────────────────
        self.act_head   = nn.Linear(_FUSE, 1)   # binding probability (raw logit)
        self.pic50_head = nn.Linear(_FUSE, 1)   # pIC50 (normalised)

    def forward(self, fp: torch.Tensor,
                esm_pooled: torch.Tensor | None = None,
                residues:   torch.Tensor | None = None,
                pad_mask:   torch.Tensor | None = None):
        """
        fp         : (B, FP_DIM)      — multi-radius ECFP + FCFP4 + MACCS + descriptors
        esm_pooled : (B, ESM_DIM)     — mean-pooled ESM-2; used only when residues=None
        residues   : (B, L, ESM_DIM)  — per-residue ESM-2 hidden states (optional)
        pad_mask   : (B, L)  bool     — True = padding position

        Returns:
          act_logit : (B,)  — raw logit for P(binding); apply sigmoid for probability
          pic50_pred: (B,)  — predicted pIC50 (normalised)
        """
        mol = self.fp_enc(fp)                        # (B, 256)

        if residues is not None:
            prot_1 = self.cross_attn(mol,    residues, pad_mask)   # (B, 256)
            prot   = self.cross_attn_2(prot_1, residues, pad_mask) + prot_1  # residual
        elif esm_pooled is not None:
            prot = self.prot_enc(esm_pooled)                  # (B, 256)
        else:
            raise ValueError("Either esm_pooled or residues must be provided.")

        # Hadamard interaction: explicit pairwise mol–prot feature product
        interaction = mol * prot                     # (B, 256)
        fused = self.fusion(torch.cat([mol, prot, interaction], dim=1))  # (B, 128)

        return self.act_head(fused).squeeze(1), self.pic50_head(fused).squeeze(1)
