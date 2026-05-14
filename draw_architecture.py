"""Generate publication-quality TargetTrace3 architecture figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── colour palette ──────────────────────────────────────────────────────────
C_MOL   = "#4878D0"
C_PROT  = "#EE854A"
C_ATTN  = "#4CAF50"
C_HAD   = "#D65F5F"
C_FUS   = "#7B5EA7"
C_OUT   = "#1A6FAF"
C_EDGE  = "#2A2A2A"

FIG_W, FIG_H = 20, 15
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ── helpers ─────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel="",
        color="#F0F0F0", edgecolor=C_EDGE, lw=1.4,
        fontsize=9, bold=False, radius=0.22):
    r = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle=f"round,pad=0.04,rounding_size={radius}",
                       facecolor=color, edgecolor=edgecolor,
                       linewidth=lw, zorder=3)
    ax.add_patch(r)
    dy = 0.14 if sublabel else 0
    ax.text(x, y + dy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal",
            color=C_EDGE, zorder=4)
    if sublabel:
        ax.text(x, y - 0.21, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color="#555", zorder=4, style="italic")

def arr(ax, x0, y0, x1, y1, color=C_EDGE, lw=1.5,
        cs="arc3,rad=0.0", sA=4, sB=4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=12,
                                connectionstyle=cs,
                                shrinkA=sA, shrinkB=sB), zorder=5)

def lbl(ax, x, y, t, fs=8, color="#555", ha="center", va="center"):
    ax.text(x, y, t, ha=ha, va=va, fontsize=fs, color=color, zorder=6)

def section(ax, x, y, w, h, color, alpha=0.07, title="", ty=None):
    r = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.1,rounding_size=0.3",
                       facecolor=color, edgecolor=color, linewidth=1.8,
                       alpha=alpha, zorder=1)
    ax.add_patch(r)
    if title:
        yt = ty if ty is not None else y + h + 0.18
        ax.text(x + w/2, yt, title, ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold", zorder=2, alpha=0.9)

# ════════════════════════════════════════════════════════════════════════════
# Hardcoded positions  (y increases upward; top = ~14.5, bottom = ~0.5)
# ════════════════════════════════════════════════════════════════════════════

# --- Molecule branch (left) ---
X_MOL  = 4.7
Y_SMILES  = 13.9
Y_FP_BOXES = 12.70    # six FP type boxes
Y_CONCAT  = 11.60    # concatenated 8379-dim
Y_ENC1    = 10.50    # Linear 8379→1024
Y_ENC2    = 9.40     # Linear 1024→512
Y_ENC3    = 8.30     # Linear 512→256
Y_MOL_EMB = 7.20     # mol embed 256-dim

# --- Protein branch (right) ---
X_PROT    = 16.0
Y_SEQ     = 13.9
Y_ESM     = 12.40
Y_RES     = 11.00    # residue embeddings

# --- Cross-attention (centre-right) ---
X_CATTN   = 11.5
Y_CATTN   = 8.80     # cross-attention block
Y_PROT_EMB = 7.20    # prot embed (same level as mol embed)

# --- Hadamard ---
X_HAD     = 8.0
Y_HAD     = 5.90

# --- Fusion ---
X_FUS     = 10.5
Y_CAT768  = 4.90
Y_FUS1    = 3.80
Y_FUS2    = 2.80
Y_SHARED  = 1.90

# --- Outputs ---
Y_HEADS   = 1.10
Y_OVAL    = 0.45
X_ACT     = 7.5
X_PIC     = 13.5

BH = 0.54   # standard box height

# ════════════════════════════════════════════════════════════════════════════
# Section backgrounds
# ════════════════════════════════════════════════════════════════════════════
section(ax, 1.8, 6.8, 5.8, 7.5, C_MOL, alpha=0.07, title="Molecule Branch")
section(ax, 13.5, 10.2, 5.5, 4.0, C_PROT, alpha=0.07, title="Protein Branch")
section(ax, 9.4, 6.8, 4.6, 5.5, C_ATTN, alpha=0.07, title="Cross-Attention")
section(ax, 1.8, 0.22, 16.5, 6.3, C_FUS, alpha=0.05, title="Fusion  &  Output", ty=6.6)

# ════════════════════════════════════════════════════════════════════════════
# SMILES input
# ════════════════════════════════════════════════════════════════════════════
box(ax, X_MOL, Y_SMILES, 2.8, BH, "SMILES string",
    color="#DDEEFF", edgecolor=C_MOL, lw=2, bold=True)

# ════════════════════════════════════════════════════════════════════════════
# Fingerprint fan-out
# ════════════════════════════════════════════════════════════════════════════
FP_TYPES = [
    ("ECFP2\n2048-d", "#B8D4F8"),
    ("ECFP4\n2048-d", "#B8D4F8"),
    ("ECFP6\n2048-d", "#B8D4F8"),
    ("FCFP4\n2048-d", "#C4DDFA"),
    ("MACCS\n167-d",  "#CDEAFC"),
    ("Phys.\n20-d",   "#D8F2FF"),
]
fp_xs = np.linspace(X_MOL - 2.75, X_MOL + 2.75, len(FP_TYPES))
for i, (ft, fc) in enumerate(FP_TYPES):
    box(ax, fp_xs[i], Y_FP_BOXES, 0.92, 0.74, ft,
        color=fc, edgecolor=C_MOL, lw=1.1, fontsize=7.5)
    arr(ax, X_MOL, Y_SMILES - BH/2, fp_xs[i], Y_FP_BOXES + 0.37,
        color=C_MOL, lw=0.85, sA=3, sB=2)

# concat bracket
bx0, bx1 = fp_xs[0] - 0.48, fp_xs[-1] + 0.48
by = Y_FP_BOXES - 0.37 - 0.05
for xi in [bx0, bx1]:
    ax.plot([xi, xi], [by + 0.12, by + 0.22], color=C_MOL, lw=1.0, zorder=5)
ax.plot([bx0, bx1], [by + 0.22, by + 0.22], color=C_MOL, lw=1.0, zorder=5)
ax.plot([X_MOL, X_MOL], [by + 0.02, by + 0.22], color=C_MOL, lw=1.0, zorder=5)

# concat box
box(ax, X_MOL, Y_CONCAT, 3.6, 0.48,
    "Concatenated fingerprint  (8 379-dim)",
    color="#D0E8FF", edgecolor=C_MOL, lw=1.6, fontsize=8.5, bold=True)
arr(ax, X_MOL, by, X_MOL, Y_CONCAT + 0.24, color=C_MOL, lw=1.3, sA=2, sB=2)

# ════════════════════════════════════════════════════════════════════════════
# FP Encoder
# ════════════════════════════════════════════════════════════════════════════
ENC = [
    (Y_ENC1, "Linear  8 379 → 1 024", "SiLU  ·  BN  ·  Drop(0.40)"),
    (Y_ENC2, "Linear  1 024 → 512",   "SiLU  ·  BN  ·  Drop(0.30)"),
    (Y_ENC3, "Linear  512 → 256",     "SiLU"),
]
prev = Y_CONCAT - 0.24
for (ey, lm, ls) in ENC:
    box(ax, X_MOL, ey, 3.0, BH + 0.16, lm, ls,
        color="#C4DCF8", edgecolor=C_MOL, lw=1.4, fontsize=8.5)
    arr(ax, X_MOL, prev, X_MOL, ey + (BH + 0.16)/2, color=C_MOL, lw=1.5)
    prev = ey - (BH + 0.16)/2

# mol embed
box(ax, X_MOL, Y_MOL_EMB, 2.1, BH, "mol embed", "256-dim",
    color=C_MOL, edgecolor=C_MOL, lw=2.0, fontsize=9, bold=True)
ax.text(X_MOL, Y_MOL_EMB + 0.16, "mol embed", ha="center", va="center",
        fontsize=9, fontweight="bold", color="white", zorder=5)
ax.text(X_MOL, Y_MOL_EMB - 0.14, "256-dim", ha="center", va="center",
        fontsize=7.5, color="white", zorder=5, style="italic")
arr(ax, X_MOL, prev, X_MOL, Y_MOL_EMB + BH/2, color=C_MOL, lw=1.8)

# ════════════════════════════════════════════════════════════════════════════
# Protein branch
# ════════════════════════════════════════════════════════════════════════════
box(ax, X_PROT, Y_SEQ, 3.0, BH, "Protein sequence",
    color="#FFE8D6", edgecolor=C_PROT, lw=2, bold=True)
box(ax, X_PROT, Y_ESM, 3.2, 0.84, "ESM-2  (35 M params)", "Frozen pre-trained LM",
    color="#FFD6B0", edgecolor=C_PROT, lw=1.8, fontsize=9, bold=True)
arr(ax, X_PROT, Y_SEQ - BH/2, X_PROT, Y_ESM + 0.42, color=C_PROT, lw=1.8)

box(ax, X_PROT, Y_RES, 3.0, BH + 0.1, "Residue embeddings", "L × 480-dim",
    color="#FFDDC0", edgecolor=C_PROT, lw=1.4, fontsize=8.5)
arr(ax, X_PROT, Y_ESM - 0.42, X_PROT, Y_RES + (BH + 0.1)/2, color=C_PROT, lw=1.5)

# ════════════════════════════════════════════════════════════════════════════
# Cross-Attention
# ════════════════════════════════════════════════════════════════════════════
box(ax, X_CATTN, Y_CATTN, 4.0, 1.10,
    "Multi-Head Cross-Attention",
    "Q: mol(256)   K, V: residues(L×480)   |   4 heads, dₖ=64",
    color="#C8F0C4", edgecolor=C_ATTN, lw=2.0, fontsize=9, bold=True, radius=0.3)

# mol → cross-attn (Q arrow)
arr(ax, X_MOL, Y_MOL_EMB + BH/2,
    X_CATTN - 1.5, Y_CATTN + 0.4,
    color=C_MOL, lw=1.5, cs="arc3,rad=-0.25")
lbl(ax, (X_MOL + X_CATTN)/2 - 0.6, Y_MOL_EMB + 0.6, "Q", fs=9, color=C_MOL)

# residues → cross-attn (K,V arrow)
arr(ax, X_PROT, Y_RES - (BH+0.1)/2,
    X_CATTN + 1.5, Y_CATTN + 0.4,
    color=C_PROT, lw=1.5, cs="arc3,rad=0.22")
lbl(ax, (X_PROT + X_CATTN)/2 + 0.5, Y_RES - 0.55, "K, V", fs=9, color=C_PROT)

# prot embed output
box(ax, X_CATTN, Y_PROT_EMB, 2.1, BH, "prot embed", "256-dim",
    color=C_ATTN, edgecolor=C_ATTN, lw=2.0, fontsize=9, bold=True)
ax.text(X_CATTN, Y_PROT_EMB + 0.16, "prot embed", ha="center", va="center",
        fontsize=9, fontweight="bold", color="white", zorder=5)
ax.text(X_CATTN, Y_PROT_EMB - 0.14, "256-dim", ha="center", va="center",
        fontsize=7.5, color="white", zorder=5, style="italic")
arr(ax, X_CATTN, Y_CATTN - 0.55,
    X_CATTN, Y_PROT_EMB + BH/2, color=C_ATTN, lw=1.8)

# ════════════════════════════════════════════════════════════════════════════
# Hadamard
# ════════════════════════════════════════════════════════════════════════════
box(ax, X_HAD, Y_HAD, 2.8, 0.64,
    "Hadamard interaction",
    "mol  ⊙  prot  →  256-dim",
    color="#FFD0D0", edgecolor=C_HAD, lw=1.8, fontsize=9, bold=True, radius=0.28)

arr(ax, X_MOL,   Y_MOL_EMB  - BH/2, X_HAD - 0.7, Y_HAD + 0.32,
    color=C_MOL, lw=1.3, cs="arc3,rad=0.18")
arr(ax, X_CATTN, Y_PROT_EMB - BH/2, X_HAD + 0.7, Y_HAD + 0.32,
    color=C_ATTN, lw=1.3, cs="arc3,rad=-0.18")

# ════════════════════════════════════════════════════════════════════════════
# Fusion
# ════════════════════════════════════════════════════════════════════════════
box(ax, X_FUS, Y_CAT768, 4.2, 0.50,
    "Concatenate  [ mol ‖ prot ‖ interaction ]  →  768-dim",
    color="#EAD8FC", edgecolor=C_FUS, lw=1.6, fontsize=8.5, bold=True)

# three arrows into cat node, with dim labels
for (sx, sy, conn, col, dlabel, lx, ly) in [
    (X_MOL,   Y_MOL_EMB  - BH/2, "arc3,rad=0.25",  C_MOL,  "256", X_MOL + 1.1,  Y_CAT768 + 0.85),
    (X_CATTN, Y_PROT_EMB - BH/2, "arc3,rad=-0.22", C_ATTN, "256", X_CATTN - 0.8, Y_CAT768 + 0.85),
    (X_HAD,   Y_HAD - 0.32,      "arc3,rad=0.0",   C_HAD,  "256", X_HAD - 0.6,   Y_CAT768 + 0.55),
]:
    arr(ax, sx, sy, X_FUS, Y_CAT768 + 0.25, color=col, lw=1.3, cs=conn)
    lbl(ax, lx, ly, dlabel, fs=7.5, color=col)

FLAYERS = [
    (Y_FUS1, "Linear  768 → 256", "SiLU  ·  BN  ·  Drop(0.30)"),
    (Y_FUS2, "Linear  256 → 128", "SiLU"),
]
prevf = Y_CAT768 - 0.25
for (fy, lm, ls) in FLAYERS:
    box(ax, X_FUS, fy, 3.0, BH + 0.16, lm, ls,
        color="#D8C8F0", edgecolor=C_FUS, lw=1.5, fontsize=8.5)
    arr(ax, X_FUS, prevf, X_FUS, fy + (BH + 0.16)/2, color=C_FUS, lw=1.6)
    prevf = fy - (BH + 0.16)/2

# shared repr
box(ax, X_FUS, Y_SHARED, 2.0, BH, "shared repr", "128-dim",
    color=C_FUS, edgecolor=C_FUS, lw=2.0, bold=True)
ax.text(X_FUS, Y_SHARED + 0.15, "shared repr", ha="center", va="center",
        fontsize=9, fontweight="bold", color="white", zorder=5)
ax.text(X_FUS, Y_SHARED - 0.14, "128-dim", ha="center", va="center",
        fontsize=7.5, color="white", zorder=5, style="italic")
arr(ax, X_FUS, prevf, X_FUS, Y_SHARED + BH/2, color=C_FUS, lw=1.8)

# ════════════════════════════════════════════════════════════════════════════
# Output heads
# ════════════════════════════════════════════════════════════════════════════
for (hx, hlabel, hsub) in [(X_ACT, "act_head  (128 → 1)", "Sigmoid"),
                             (X_PIC, "pic50_head  (128 → 1)", "Linear")]:
    box(ax, hx, Y_HEADS, 2.6, BH, hlabel, hsub,
        color="#C8E8F8", edgecolor="#2E86AB", lw=1.6, fontsize=8.5)
    arr(ax, X_FUS, Y_SHARED - BH/2, hx, Y_HEADS + BH/2,
        color=C_FUS, lw=1.4,
        cs=f"arc3,rad={'0.35' if hx < X_FUS else '-0.35'}")

# output ovals
for (ox, ot) in [(X_ACT, "P(active)"), (X_PIC, "pIC50")]:
    ell = mpatches.Ellipse((ox, Y_OVAL), 2.0, 0.56,
                           facecolor=C_OUT, edgecolor="#0D4A80",
                           linewidth=1.8, zorder=3)
    ax.add_patch(ell)
    ax.text(ox, Y_OVAL, ot, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=5)
    arr(ax, ox, Y_HEADS - BH/2, ox, Y_OVAL + 0.28,
        color="#2E86AB", lw=1.5)

# ════════════════════════════════════════════════════════════════════════════
# Legend
# ════════════════════════════════════════════════════════════════════════════
patches = [
    mpatches.Patch(facecolor=C_MOL,  edgecolor=C_EDGE, label="Molecule branch (FP encoder)"),
    mpatches.Patch(facecolor=C_PROT, edgecolor=C_EDGE, label="Protein branch (ESM-2)"),
    mpatches.Patch(facecolor=C_ATTN, edgecolor=C_EDGE, label="Cross-attention"),
    mpatches.Patch(facecolor=C_HAD,  edgecolor=C_EDGE, label="Hadamard interaction"),
    mpatches.Patch(facecolor=C_FUS,  edgecolor=C_EDGE, label="Fusion MLP"),
    mpatches.Patch(facecolor=C_OUT,  edgecolor=C_EDGE, label="Output heads"),
]
ax.legend(handles=patches, loc="lower left", bbox_to_anchor=(0.01, 0.01),
          fontsize=8.5, framealpha=0.92, edgecolor="#AAAAAA",
          ncol=3, handlelength=1.2)

# ── title ───────────────────────────────────────────────────────────────────
ax.text(FIG_W/2, FIG_H - 0.28,
        "TargetTrace3 — Dual-Input Drug–Target Interaction Architecture",
        ha="center", va="top", fontsize=14, fontweight="bold", color=C_EDGE)
ax.text(FIG_W/2, FIG_H - 0.72,
        ("Multi-radius fingerprint encoder  ×  ESM-2 protein LM  →  "
         "Cross-attention fusion  →  Binary classification + pIC50 regression"),
        ha="center", va="top", fontsize=8.5, color="#555555")

# ── save ─────────────────────────────────────────────────────────────────────
out = ("/home/hhn/Documents/Final-project/TargetTrace3/"
       "training results/architecture_figure.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
plt.close(fig)
