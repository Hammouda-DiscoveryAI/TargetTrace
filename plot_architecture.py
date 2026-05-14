"""
Publication-quality TargetTrace architecture diagram.
Clean vertical two-branch layout: molecule (left) | protein (right)
converging through stacked cross-attention → Hadamard fusion → dual heads.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT_DIR = Path(__file__).parent / "results" / "architecture"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
BG      = "#FFFFFF"
C_INPUT = "#E8EDF2"   # light grey-blue   — raw inputs
C_MOL   = "#3A6BAD"   # deep blue          — molecule branch
C_PROT  = "#2D8A5A"   # deep green         — protein branch
C_ATTN  = "#6B52A8"   # deep purple        — cross-attention
C_RES   = "#E8A020"   # amber              — residual
C_FUSE  = "#C95E20"   # burnt orange       — fusion / hadamard
C_HEAD  = "#B03040"   # crimson            — output heads
TXT_DRK = "#1A1A2E"   # near-black text on light boxes
TXT_LGT = "#FFFFFF"   # white text on dark boxes


def _rect(ax, cx, cy, w, h, facecolor, edgecolor="white",
          lw=1.5, radius=0.008, alpha=1.0, zorder=3):
    box = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    return box


def _label(ax, cx, cy, line1, line2="", color=TXT_LGT,
           fs1=9.5, fs2=8.0, zorder=4):
    dy = 0.013 if line2 else 0
    ax.text(cx, cy + dy, line1, ha="center", va="center",
            fontsize=fs1, fontweight="bold", color=color,
            zorder=zorder, linespacing=1.3)
    if line2:
        ax.text(cx, cy - dy, line2, ha="center", va="center",
                fontsize=fs2, color=color, alpha=0.88,
                zorder=zorder)


def _arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.5,
           hw=0.006, hl=0.012, zorder=2):
    ax.annotate("",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=f"->,head_width={hw},head_length={hl}",
            color=color, lw=lw,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=zorder,
    )


def _dim_tag(ax, x, y, text, color, ha="left"):
    ax.text(x, y, text, ha=ha, va="center",
            fontsize=7.5, color=color, style="italic",
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", edgecolor=color,
                      alpha=0.85, linewidth=0.8),
            zorder=5)


def draw():
    fig = plt.figure(figsize=(13, 17))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ── grid constants ─────────────────────────────────────────────────────
    XL   = 0.225   # molecule branch centre-x
    XR   = 0.720   # protein branch centre-x
    XCTR = 0.472   # cross-attention / fusion centre-x

    BW  = 0.20    # standard box width
    BH  = 0.052   # standard box height
    BWW = 0.34    # wide box (cross-attention)
    BWS = 0.16    # narrow box (output heads)

    # y-positions (top → bottom)
    Y = dict(
        title   = 0.970,
        subtitle= 0.945,
        input   = 0.900,
        enc     = 0.800,
        emb     = 0.702,
        ca1     = 0.590,
        ca1_mid = 0.590,
        ca2     = 0.460,
        had     = 0.342,
        fuse    = 0.235,
        heads   = 0.105,
        labels  = 0.058,
    )

    # ── title ──────────────────────────────────────────────────────────────
    ax.text(0.5, Y["title"], "TargetTrace — Model Architecture",
            ha="center", va="center", fontsize=17, fontweight="bold",
            color=TXT_DRK, zorder=5)
    ax.text(0.5, Y["subtitle"],
            "Drug-Target Interaction: P(binding) + pIC50 prediction  │  10.6 M parameters",
            ha="center", va="center", fontsize=10, color="#555555", zorder=5)

    # ── column headers ─────────────────────────────────────────────────────
    for x, lbl, c in [(XL, "Molecule Branch", C_MOL),
                      (XR, "Protein Branch",  C_PROT)]:
        ax.text(x, Y["input"] + 0.050, lbl, ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=c,
                bbox=dict(boxstyle="round,pad=0.25", facecolor=c+"22",
                          edgecolor=c, linewidth=0.8))

    # ═══════════════════════ INPUTS ════════════════════════════════════════
    # Molecule SMILES
    _rect(ax, XL, Y["input"], BW, BH, C_INPUT, edgecolor=C_MOL, lw=1.8)
    _label(ax, XL, Y["input"], "Molecule SMILES", color=TXT_DRK)

    # Protein Sequence
    _rect(ax, XR, Y["input"], BW, BH, C_INPUT, edgecolor=C_PROT, lw=1.8)
    _label(ax, XR, Y["input"], "Protein Sequence", color=TXT_DRK)

    # ─── arrows: inputs → encoders ─────────────────────────────────────────
    _arrow(ax, XL, Y["input"]-BH/2, XL, Y["enc"]+BH/2, color=C_MOL)
    _arrow(ax, XR, Y["input"]-BH/2, XR, Y["enc"]+BH/2, color=C_PROT)

    # ═══════════════════════ ENCODERS ══════════════════════════════════════
    # FP Encoder
    _rect(ax, XL, Y["enc"], BW+0.01, BH*1.55, C_MOL)
    _label(ax, XL, Y["enc"], "Fingerprint Encoder",
           "8379 → 1024 → 512 → 256")
    # FP detail annotation
    ax.text(XL - BW/2 - 0.01, Y["enc"],
            "SiLU · BN · Dropout\n(3 layers, 9.2 M params)",
            ha="right", va="center", fontsize=7.5, color=C_MOL,
            style="italic", linespacing=1.4)

    # ESM-2
    _rect(ax, XR, Y["enc"], BW+0.01, BH*1.55, C_PROT)
    _label(ax, XR, Y["enc"], "ESM-2  (650 M, frozen)",
           "residues: L × 480")
    ax.text(XR + BW/2 + 0.01, Y["enc"],
            "Pre-trained protein\nlanguage model",
            ha="left", va="center", fontsize=7.5, color=C_PROT,
            style="italic", linespacing=1.4)

    # ─── dimension tags ────────────────────────────────────────────────────
    _arrow(ax, XL, Y["enc"]-BH*0.78, XL, Y["emb"]+BH*0.55, color=C_MOL)
    _arrow(ax, XR, Y["enc"]-BH*0.78, XR, Y["emb"]+BH*0.55, color=C_PROT)
    _dim_tag(ax, XL+0.015, (Y["enc"]+Y["emb"])/2 - 0.005, "256-dim", C_MOL)
    _dim_tag(ax, XR+0.015, (Y["enc"]+Y["emb"])/2 - 0.005, "L × 480", C_PROT)

    # ═══════════════════════ EMBEDDING NODES ═══════════════════════════════
    EMB_R = 0.035
    # mol node
    mol_circ = plt.Circle((XL, Y["emb"]), EMB_R, color=C_MOL,
                            zorder=3, linewidth=1.5, ec="white")
    ax.add_patch(mol_circ)
    ax.text(XL, Y["emb"]+0.005, "mol", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=4)
    ax.text(XL, Y["emb"]-0.016, "256", ha="center", va="center",
            fontsize=7.5, color="white", zorder=4)

    # residues node
    res_circ = plt.Circle((XR, Y["emb"]), EMB_R, color=C_PROT,
                            zorder=3, linewidth=1.5, ec="white")
    ax.add_patch(res_circ)
    ax.text(XR, Y["emb"]+0.005, "residues", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white", zorder=4)
    ax.text(XR, Y["emb"]-0.016, "L × 480", ha="center", va="center",
            fontsize=7.5, color="white", zorder=4)

    # ═══════════════════════ CROSS-ATTENTION #1 ════════════════════════════
    CA1Y = Y["ca1"]
    _rect(ax, XCTR, CA1Y, BWW, BH*1.6, C_ATTN)
    _label(ax, XCTR, CA1Y, "Cross-Attention  #1",
           "Query: mol (256)  ·  Key/Value: residues (L×480)  →  prot₁ (256)")
    # 4-head annotation
    ax.text(XCTR + BWW/2 + 0.01, CA1Y,
            "4 heads\nd_k = 64\nLayerNorm",
            ha="left", va="center", fontsize=7.5, color=C_ATTN,
            style="italic", linespacing=1.5)

    # arrows into CA1
    # mol → CA1  (diagonal from left)
    _arrow(ax, XL, Y["emb"]-EMB_R, XCTR-BWW/2, CA1Y+BH*0.55,
           color=C_MOL, lw=1.6, hw=0.007, hl=0.013)
    ax.text((XL + XCTR-BWW/2)/2 - 0.01,
            (Y["emb"]-EMB_R + CA1Y+BH*0.55)/2 + 0.01,
            "Q", ha="right", va="bottom",
            fontsize=10, fontweight="bold", color=C_MOL)

    # residues → CA1 (diagonal from right)
    _arrow(ax, XR, Y["emb"]-EMB_R, XCTR+BWW/2*0.6, CA1Y+BH*0.55,
           color=C_PROT, lw=1.6, hw=0.007, hl=0.013)
    ax.text((XR + XCTR+BWW/2*0.6)/2 + 0.01,
            (Y["emb"]-EMB_R + CA1Y+BH*0.55)/2 + 0.01,
            "K, V", ha="left", va="bottom",
            fontsize=10, fontweight="bold", color=C_PROT)

    # CA1 output → prot₁
    PROT1Y = CA1Y - BH*1.1
    _arrow(ax, XCTR, CA1Y-BH*0.8, XCTR, PROT1Y+0.016, color=C_ATTN, lw=1.6)
    _dim_tag(ax, XCTR+0.01, (CA1Y-BH*0.8 + PROT1Y+0.016)/2, "prot₁  (256)", C_ATTN)

    # ═══════════════════════ RESIDUAL SPLIT ════════════════════════════════
    # prot₁ node
    p1_circ = plt.Circle((XCTR, PROT1Y), 0.024, color=C_ATTN,
                           zorder=3, ec="white", lw=1.2)
    ax.add_patch(p1_circ)
    ax.text(XCTR, PROT1Y, "p₁", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white", zorder=4)

    # residual branch: prot₁ goes right, curves down to the + node
    RES_X = XCTR + 0.225
    CA2Y  = Y["ca2"]
    # horizontal leg from p₁ to the right
    _arrow(ax, XCTR+0.024, PROT1Y, RES_X, PROT1Y,
           color=C_RES, lw=2.0, hw=0.007, hl=0.013)
    # vertical leg down to ca2 level
    ax.annotate("",
        xy=(RES_X, CA2Y - BH*0.9),
        xytext=(RES_X, PROT1Y),
        arrowprops=dict(arrowstyle="->,head_width=0.007,head_length=0.013",
                        color=C_RES, lw=2.0),
        zorder=3)
    ax.text(RES_X + 0.012, (PROT1Y + CA2Y-BH*0.9)/2,
            "Residual\nconnection",
            ha="left", va="center", fontsize=8, color=C_RES,
            fontweight="bold", linespacing=1.3)

    # ═══════════════════════ CROSS-ATTENTION #2 ════════════════════════════
    _rect(ax, XCTR, CA2Y, BWW, BH*1.6, C_ATTN)
    _label(ax, XCTR, CA2Y, "Cross-Attention  #2",
           "Query: prot₁ (256)  ·  Key/Value: residues (L×480)  →  prot₂ (256)")
    ax.text(XCTR + BWW/2 + 0.01, CA2Y,
            "4 heads\nd_k = 64\nLayerNorm",
            ha="left", va="center", fontsize=7.5, color=C_ATTN,
            style="italic", linespacing=1.5)

    # prot₁ → CA2  (main down arrow)
    _arrow(ax, XCTR, PROT1Y-0.024, XCTR, CA2Y+BH*0.8,
           color=C_ATTN, lw=1.6)

    # residues → CA2  (same residues re-used)
    _arrow(ax, XR, Y["emb"]-EMB_R,
           XCTR+BWW/2*0.65, CA2Y+BH*0.7,
           color=C_PROT, lw=1.2, hw=0.006, hl=0.011)
    ax.text(XR*0.55 + XCTR*0.45 + 0.03,
            (Y["emb"]-EMB_R + CA2Y+BH*0.7)/2 + 0.005,
            "K, V (reused)", ha="left", va="bottom",
            fontsize=8, color=C_PROT, style="italic")

    # CA2 output → prot₂
    PROT2Y = CA2Y - BH*1.1
    _arrow(ax, XCTR, CA2Y-BH*0.8, XCTR, PROT2Y+0.016, color=C_ATTN, lw=1.6)
    _dim_tag(ax, XCTR+0.01, (CA2Y-BH*0.8 + PROT2Y+0.016)/2, "prot₂  (256)", C_ATTN)

    # + node (add residual)
    PLUS_R = 0.024
    ax.add_patch(plt.Circle((XCTR, PROT2Y), PLUS_R,
                              color="white", ec=C_RES, lw=2.2, zorder=4))
    ax.text(XCTR, PROT2Y, "+", ha="center", va="center",
            fontsize=14, fontweight="bold", color=C_RES, zorder=5)

    # residual arrives at + node (from right)
    _arrow(ax, RES_X, CA2Y-BH*0.9, XCTR+PLUS_R, PROT2Y,
           color=C_RES, lw=2.0, hw=0.007, hl=0.013)

    # prot = prot₂ + prot₁  label
    ax.text(XCTR + 0.040, PROT2Y + 0.002,
            "prot  =  prot₂  +  prot₁",
            ha="left", va="center", fontsize=8.5, color=C_ATTN,
            fontstyle="italic")

    # ═══════════════════════ HADAMARD ══════════════════════════════════════
    HADY  = Y["had"]
    # prot arrow down
    _arrow(ax, XCTR, PROT2Y-PLUS_R, XCTR, HADY+BH*0.55, color=C_ATTN, lw=1.6)

    # mol arrow: long diagonal from mol circle down to hadamard
    _arrow(ax, XL + EMB_R*0.7, Y["emb"] - EMB_R*0.7,
           XCTR - BWW/2 + 0.02, HADY + BH*0.4,
           color=C_MOL, lw=1.6, hw=0.007, hl=0.013)
    ax.text(XL + 0.06, (Y["emb"] + HADY)/2 + 0.01,
            "mol (256)", ha="left", va="bottom",
            fontsize=8, color=C_MOL, style="italic")

    _rect(ax, XCTR, HADY, BWW*0.72, BH*1.3, C_FUSE)
    _label(ax, XCTR, HADY,
           "Hadamard Interaction",
           "mol  ⊙  prot  →  interaction  (256)")

    # ═══════════════════════ FUSION MLP ════════════════════════════════════
    FUSEY = Y["fuse"]
    _arrow(ax, XCTR, HADY-BH*0.65, XCTR, FUSEY+BH*0.75, color=C_FUSE, lw=1.6)

    _rect(ax, XCTR, FUSEY, BWW*0.88, BH*1.8, C_FUSE)
    _label(ax, XCTR, FUSEY,
           "Fusion MLP",
           "cat[ mol  ‖  prot  ‖  mol⊙prot ]   768 → 256 → 128")
    ax.text(XCTR + BWW*0.88/2 + 0.01, FUSEY,
            "SiLU · BN · Dropout(0.3)",
            ha="left", va="center", fontsize=7.5, color=C_FUSE,
            style="italic")

    # bracket showing three components entering fusion
    BX = XCTR - BWW*0.88/2 - 0.03
    for dy, lbl, c in [(+0.025, "mol  (256)",         C_MOL),
                       (  0.0,  "prot  (256)",         C_ATTN),
                       (-0.025, "mol ⊙ prot  (256)",   C_FUSE)]:
        ax.text(BX, FUSEY+dy, f"• {lbl}", ha="right", va="center",
                fontsize=8, color=c)

    # dimension out
    _dim_tag(ax, XCTR+0.01, (FUSEY-BH*0.9 + Y["heads"]+BH*0.55)/2,
             "fused  (128)", C_FUSE)

    # ═══════════════════════ OUTPUT HEADS ══════════════════════════════════
    HEADSY = Y["heads"]
    XH_L   = XCTR - 0.145
    XH_R   = XCTR + 0.145
    _arrow(ax, XCTR, FUSEY-BH*0.9, XH_L+0.0, HEADSY+BH*0.65,
           color="#888888", lw=1.4, hw=0.006, hl=0.011)
    _arrow(ax, XCTR, FUSEY-BH*0.9, XH_R-0.0, HEADSY+BH*0.65,
           color="#888888", lw=1.4, hw=0.006, hl=0.011)

    # act_head
    _rect(ax, XH_L, HEADSY, BWS*1.05, BH*1.4, C_HEAD)
    _label(ax, XH_L, HEADSY, "act_head", "128 → 1")

    # pic50_head
    _rect(ax, XH_R, HEADSY, BWS*1.05, BH*1.4, C_HEAD)
    _label(ax, XH_R, HEADSY, "pic50_head", "128 → 1")

    # output labels
    ax.text(XH_L, Y["labels"],
            "P(binding)\n sigmoid(logit)",
            ha="center", va="center", fontsize=9, color=C_HEAD,
            fontweight="bold", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C_HEAD+"18",
                      edgecolor=C_HEAD, linewidth=0.8))
    ax.text(XH_R, Y["labels"],
            "pIC50\n (normalised)",
            ha="center", va="center", fontsize=9, color=C_HEAD,
            fontweight="bold", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C_HEAD+"18",
                      edgecolor=C_HEAD, linewidth=0.8))

    # ═══════════════════════ FINGERPRINT INFO BOX ══════════════════════════
    fp_txt = ("FP composition (8,379-dim)\n"
              "─────────────────────────\n"
              "ECFP2   2,048 bits  (r=1)\n"
              "ECFP4   2,048 bits  (r=2)\n"
              "ECFP6   2,048 bits  (r=3)\n"
              "FCFP4   2,048 bits\n"
              "MACCS     167 bits\n"
              "RDKit desc.  200-dim")
    ax.text(0.025, Y["enc"] - 0.07, fp_txt,
            ha="left", va="top", fontsize=7.8, color=C_MOL,
            family="monospace", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#EEF3FA",
                      edgecolor=C_MOL, linewidth=1.0, alpha=0.95))

    # ═══════════════════════ LEGEND ════════════════════════════════════════
    legend_items = [
        mpatches.Patch(facecolor=C_MOL,  edgecolor="white", label="Molecule branch"),
        mpatches.Patch(facecolor=C_PROT, edgecolor="white", label="Protein branch (ESM-2)"),
        mpatches.Patch(facecolor=C_ATTN, edgecolor="white", label="Cross-attention"),
        mpatches.Patch(facecolor=C_FUSE, edgecolor="white", label="Fusion / Hadamard"),
        mpatches.Patch(facecolor=C_HEAD, edgecolor="white", label="Output heads"),
        mpatches.Patch(facecolor=C_RES,  edgecolor="white", label="Residual connection"),
    ]
    ax.legend(handles=legend_items,
              loc="lower right", bbox_to_anchor=(0.99, 0.01),
              fontsize=9, framealpha=0.95, edgecolor="#BBBBBB",
              title="Component", title_fontsize=9)

    # ═══════════════════════ STAGE ANNOTATIONS ═════════════════════════════
    stages = [
        (Y["input"],  "① Input"),
        (Y["enc"],    "② Encoding"),
        (Y["ca1"],    "③ Cross-Attention"),
        (Y["ca2"],    ""),
        (Y["had"],    "④ Interaction"),
        (Y["fuse"],   "⑤ Fusion"),
        (Y["heads"],  "⑥ Output"),
    ]
    for y, lbl in stages:
        if lbl:
            ax.text(0.008, y, lbl, ha="left", va="center",
                    fontsize=8, color="#888888", fontweight="bold")

    # light horizontal dividers between stages
    for y in [0.855, 0.745, 0.640, 0.400, 0.290, 0.175]:
        ax.axhline(y, xmin=0.01, xmax=0.99,
                   color="#EEEEEE", linewidth=0.8, zorder=0)

    out = OUT_DIR / "architecture.png"
    plt.savefig(out, dpi=200, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    draw()
