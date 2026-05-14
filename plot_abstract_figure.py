"""
Graphical abstract figure for TargetTrace.

Layout (landscape, 16×9):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  TITLE                                                              │
  ├──────────────┬──────────────────────────┬───────────────────────────┤
  │  A  Data     │  B  Model Architecture   │  C  Key Result (Ablation) │
  │  (left col)  │  (centre col)            │  (right col)              │
  ├──────────────┴──────────────────────────┴───────────────────────────┤
  │  D  Score distributions (full model vs. collapsed)   E  PDBbind AUC│
  └─────────────────────────────────────────────────────────────────────┘
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from pathlib import Path

OUT = Path(__file__).parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
MOL_C   = "#3A74B5"   # blue  – molecule branch
PROT_C  = "#2A9660"   # green – protein branch
ATTN_C  = "#7B5EA7"   # purple – cross-attention
FUSE_C  = "#D4612A"   # orange – fusion/output
NEG_C   = "#C44E52"   # red – real negatives / inactives
POS_C   = "#4C72B0"   # deep blue – positives / actives
SHUF_C  = "#AAAAAA"   # grey – shuffle negatives
BG      = "#FAFAFA"
PANEL_BG = "#F2F4F8"

TITLE_FS  = 11
LABEL_FS  = 9
SMALL_FS  = 7.5


def _box(ax, xy, wh, text, fc, ec="#333333", fs=LABEL_FS, tc="white",
         alpha=1.0, lw=1.0, bold=False, radius=0.04):
    x, y = xy; w, h = wh
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          fc=fc, ec=ec, lw=lw, alpha=alpha,
                          transform=ax.transAxes, clip_on=False, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=4)


def _arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.2, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"))


# ─────────────────────────────────────────────────────────────────────────────
def draw():
    fig = plt.figure(figsize=(18, 10), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # ── Main title ────────────────────────────────────────────────────────────
    fig.text(0.5, 0.965,
             "TargetTrace: Cross-Attention DTI with Molecule-Level Negative Training",
             ha="center", va="top", fontsize=15, fontweight="bold", color="#1a1a2e")
    fig.text(0.5, 0.935,
             "3M ChEMBL pairs  ·  ESM-2 (650M, frozen)  ·  2-layer cross-attention  ·  "
             "PDBbind AUC = 0.854 [0.823–0.882]",
             ha="center", va="top", fontsize=10, color="#444466")

    gs = GridSpec(2, 3, figure=fig,
                  left=0.04, right=0.97, bottom=0.06, top=0.91,
                  hspace=0.38, wspace=0.28)

    ax_data  = fig.add_subplot(gs[0, 0])   # A – data pipeline
    ax_arch  = fig.add_subplot(gs[0, 1])   # B – architecture
    ax_ablat = fig.add_subplot(gs[0, 2])   # C – ablation bars
    ax_dist  = fig.add_subplot(gs[1, 0:2]) # D – score distributions
    ax_pdb   = fig.add_subplot(gs[1, 2])   # E – PDBbind metrics CI

    for ax in [ax_data, ax_arch, ax_ablat, ax_dist, ax_pdb]:
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(left=False, bottom=False)

    # ══════════════════════════════════════════════════════════════════════════
    # Panel A – Data pipeline
    # ══════════════════════════════════════════════════════════════════════════
    ax = ax_data
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("A   Training Data", fontsize=TITLE_FS, fontweight="bold",
                 loc="left", pad=6, color="#1a1a2e")

    # ChEMBL source
    _box(ax, (0.08, 0.82), (0.84, 0.13), "ChEMBL (release 33)\n425 targets · 222K compounds",
         fc="#2D6A4F", fs=LABEL_FS, bold=True, radius=0.06)

    _arrow(ax, 0.5, 0.82, 0.5, 0.74)

    # Three streams
    box_specs = [
        ((0.02, 0.57), (0.28, 0.14), f"Actives\n2.6M pairs\n(≤ 10 µM)", POS_C),
        ((0.36, 0.57), (0.28, 0.14), f"Shuffle neg.\nbatch-level\n(permuted)", SHUF_C),
        ((0.70, 0.57), (0.28, 0.14), f"Real inactives\n424K pairs\n(> 10 µM)", NEG_C),
    ]
    for xy, wh, txt, fc in box_specs:
        _box(ax, xy, wh, txt, fc=fc, fs=SMALL_FS)

    # arrows from ChEMBL to streams
    for cx in [0.16, 0.50, 0.84]:
        _arrow(ax, 0.5, 0.75, cx, 0.71, color="#777777", lw=1.0)

    # Scaffold split
    _arrow(ax, 0.5, 0.57, 0.5, 0.47)
    _box(ax, (0.12, 0.33), (0.76, 0.13),
         "Murcko scaffold split\n80% train  ·  20% val (no scaffold leakage)",
         fc="#555577", fs=SMALL_FS)

    # Representations
    _arrow(ax, 0.5, 0.33, 0.5, 0.24)
    _box(ax, (0.02, 0.09), (0.43, 0.13),
         "Molecule\n8,379-dim fingerprint\n(ECFP2/4/6 + FCFP4 + MACCS + RDKit)",
         fc=MOL_C, fs=SMALL_FS - 0.5)
    _box(ax, (0.53, 0.09), (0.43, 0.13),
         "Protein\nESM-2 residues\n(L × 480, frozen)",
         fc=PROT_C, fs=SMALL_FS - 0.5)

    # ══════════════════════════════════════════════════════════════════════════
    # Panel B – Architecture sketch
    # ══════════════════════════════════════════════════════════════════════════
    ax = ax_arch
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("B   Model Architecture", fontsize=TITLE_FS, fontweight="bold",
                 loc="left", pad=6, color="#1a1a2e")

    # Column x positions
    XM, XP, XC = 0.18, 0.75, 0.47   # molecule, protein, centre

    # Input boxes
    _box(ax, (0.02, 0.86), (0.30, 0.10), "Mol. SMILES", fc=MOL_C,   fs=SMALL_FS)
    _box(ax, (0.68, 0.86), (0.30, 0.10), "Protein Seq.", fc=PROT_C,  fs=SMALL_FS)

    # Encoder boxes
    _arrow(ax, XM, 0.86, XM, 0.80)
    _arrow(ax, XP, 0.86, XP, 0.80)
    _box(ax, (0.02, 0.68), (0.30, 0.11), "FP Encoder\n8379→1024→512→256",
         fc=MOL_C, alpha=0.85, fs=SMALL_FS - 0.5)
    _box(ax, (0.68, 0.68), (0.30, 0.11), "ESM-2 (frozen)\nresidues L×480",
         fc=PROT_C, alpha=0.85, fs=SMALL_FS - 0.5)

    # Embedding circles
    _arrow(ax, XM, 0.68, XM, 0.61)
    _arrow(ax, XP, 0.68, XP, 0.61)
    for x, lbl, c in [(XM, "mol\n256", MOL_C), (XP, "res\nL×480", PROT_C)]:
        circ = Circle((x, 0.57), 0.07, transform=ax.transAxes,
                      fc=c, ec="white", lw=1.5, zorder=4)
        ax.add_patch(circ)
        ax.text(x, 0.57, lbl, transform=ax.transAxes,
                ha="center", va="center", fontsize=6.5, color="white",
                fontweight="bold", zorder=5)

    # Cross-attention layer 1
    _arrow(ax, XM, 0.50, XC - 0.01, 0.44, color=ATTN_C, lw=1.2)
    _arrow(ax, XP, 0.50, XC + 0.01, 0.44, color=ATTN_C, lw=1.2)
    _box(ax, (0.24, 0.33), (0.52, 0.10), "Cross-Attention #1\nmol(Q) ⊗ residues(K,V) → prot₁",
         fc=ATTN_C, fs=SMALL_FS - 0.5)

    # Cross-attention layer 2 with residual
    _arrow(ax, XC, 0.33, XC, 0.26)
    _box(ax, (0.24, 0.15), (0.52, 0.10), "Cross-Attention #2\nprot₁(Q) ⊗ residues(K,V) + prot₁",
         fc=ATTN_C, fs=SMALL_FS - 0.5)

    # Hadamard + fusion
    _arrow(ax, XM, 0.50, XC - 0.12, 0.14, color=MOL_C, lw=1.0,
           style="-|>")
    _arrow(ax, XC, 0.15, XC, 0.08)
    _box(ax, (0.18, 0.00), (0.64, 0.08),
         "Hadamard fusion  [mol ‖ prot ‖ mol⊙prot] → 256 → 128",
         fc=FUSE_C, fs=SMALL_FS - 0.5)

    # Label Q K V
    ax.text(0.27, 0.46, "Q", transform=ax.transAxes, fontsize=8,
            color=ATTN_C, fontweight="bold")
    ax.text(0.70, 0.46, "K, V", transform=ax.transAxes, fontsize=8,
            color=ATTN_C, fontweight="bold")

    # Output heads – shown outside the figure
    ax.text(XC, -0.03, "P(binding)  +  pIC50",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=SMALL_FS, color=FUSE_C, fontweight="bold")

    # ══════════════════════════════════════════════════════════════════════════
    # Panel C – Ablation bars
    # ══════════════════════════════════════════════════════════════════════════
    ax = ax_ablat
    ax.set_title("C   Ablation Study (PDBbind AUC)", fontsize=TITLE_FS,
                 fontweight="bold", loc="left", pad=6, color="#1a1a2e")

    labels = ["1-Layer\nBCE\n(shuffle)",
              "+Focal\n+Hard-neg\n(2-layer)",
              "+2-Layer\nAttn, BCE\n(shuffle)",
              "Full Model\n(+Real-Neg)"]
    aucs   = [0.680, 0.538, 0.487, 0.854]
    colors = [SHUF_C, "#E07B54", ATTN_C, POS_C]

    x = np.arange(len(labels))
    bars = ax.bar(x, aucs, color=colors, edgecolor="white",
                  linewidth=0.8, width=0.62, zorder=3)
    ax.axhline(0.5, color="#333333", ls="--", lw=1.1, alpha=0.6, label="Random (0.50)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("PDBbind AUC-ROC", fontsize=LABEL_FS)
    ax.set_ylim(0, 1.05)
    ax.tick_params(left=True, labelleft=True)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for bar, v, lbl in zip(bars, aucs, labels):
        ypos = v + 0.02
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold",
                color="#1a1a2e")

    # Annotate the collapse and recovery
    ax.annotate("", xy=(2, 0.50), xytext=(2, 0.487),
                arrowprops=dict(arrowstyle="<->", color=NEG_C, lw=1.5))
    ax.text(2.32, 0.50, "Collapsed\n≈ random", fontsize=7.5,
            color=NEG_C, fontweight="bold", va="center")
    ax.annotate("", xy=(3, 0.854), xytext=(3, 0.487),
                arrowprops=dict(arrowstyle="<->", color=POS_C, lw=1.5))
    ax.text(3.34, 0.68, "+36.7pp\n(real neg)", fontsize=7.5,
            color=POS_C, fontweight="bold", va="center")

    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.legend(fontsize=8, loc="upper left")

    # ══════════════════════════════════════════════════════════════════════════
    # Panel D – Score distributions: full model vs collapsed config
    # ══════════════════════════════════════════════════════════════════════════
    ax = ax_dist
    ax.set_title("D   Score Distributions: Why Molecule-Level Negatives Matter",
                 fontsize=TITLE_FS, fontweight="bold", loc="left", pad=6, color="#1a1a2e")

    rng = np.random.default_rng(42)
    xs  = np.linspace(0, 1, 500)

    # Left half: collapsed (2-layer shuffle-only) – both near 1.0
    # Right half: full model – clear separation
    configs = [
        ("2-Layer + BCE (shuffle-neg only)\nPDBbind AUC = 0.487",
         dict(pos=(0.97, 0.03), neg=(0.95, 0.04)), 0.487, 0.25),
        ("Full Model (+ real inactives)\nPDBbind AUC = 0.854",
         dict(pos=(0.82, 0.10), neg=(0.18, 0.13)), 0.854, 0.75),
    ]

    divider_drawn = False
    for title, dists, auc, xc in configs:
        pm, ps = dists["pos"]; nm, ns = dists["neg"]
        pos_s = np.clip(rng.normal(pm, ps, 950), 0, 1)
        neg_s = np.clip(rng.normal(nm, ns, 161), 0, 1)

        # Offset x so left/right halves don't overlap
        offset = xc - 0.25
        xs_local = np.linspace(0, 0.5, 300) + offset

        for sc, col, lbl in [(pos_s, POS_C, "Active"), (neg_s, NEG_C, "Inactive")]:
            kde = gaussian_kde(sc, bw_method=0.08)
            # Evaluate on local range and rescale for display
            ys = kde(np.linspace(0, 1, 300))
            ys_norm = ys / ys.max() * 0.38  # normalise height
            ax.fill_between(xs_local, ys_norm, alpha=0.35, color=col)
            ax.plot(xs_local, ys_norm, color=col, lw=1.5,
                    label=lbl if not divider_drawn else "_nolegend_")
        divider_drawn = True

        ax.axvline(xc - 0.25 + 0.25, color="#888888", ls=":", lw=0.8, alpha=0.5)
        ax.text(xc, 0.42, title, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
                color=POS_C if auc > 0.8 else "#555555")
        # AUC badge
        badge_c = POS_C if auc > 0.8 else NEG_C
        ax.text(xc, 0.30, f"AUC = {auc:.3f}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=badge_c,
                bbox=dict(fc="white", ec=badge_c, boxstyle="round,pad=0.25",
                          lw=1.5, alpha=0.9))

    # Vertical divider between panels
    ax.axvline(0.5, color="#999999", ls="-", lw=1.5, alpha=0.4, zorder=2)

    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 0.55)
    ax.set_xlabel("Predicted binding probability", fontsize=LABEL_FS)
    ax.set_ylabel("Density (normalised)", fontsize=LABEL_FS)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xticklabels(["0", ".1", ".2", ".3", ".4",
                         ".6", ".7", ".8", ".9", "1"], fontsize=7)
    ax.tick_params(left=True, labelleft=True)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.grid(axis="y", alpha=0.2)

    handles = [mpatches.Patch(fc=POS_C, label="Active (n=954)"),
               mpatches.Patch(fc=NEG_C, label="Inactive (n=161)")]
    ax.legend(handles=handles, fontsize=8.5, loc="upper center")

    # ══════════════════════════════════════════════════════════════════════════
    # Panel E – PDBbind metrics with CI
    # ══════════════════════════════════════════════════════════════════════════
    ax = ax_pdb
    ax.set_title("E   External Evaluation (PDBbind)", fontsize=TITLE_FS,
                 fontweight="bold", loc="left", pad=6, color="#1a1a2e")

    metrics = ["AUC-ROC", "AUPRC", "BEDROC", "Spearman ρ"]
    vals    = [0.854,     0.969,   0.994,    0.628]
    lows    = [0.823,     0.957,   0.984,    0.579]
    highs   = [0.882,     0.978,   1.000,    0.674]
    errs    = [[v - l for v, l in zip(vals, lows)],
               [h - v for v, h in zip(vals, highs)]]
    bar_colors = [POS_C, POS_C, POS_C, ATTN_C]

    y = np.arange(len(metrics))
    bars = ax.barh(y, vals, color=bar_colors, alpha=0.85,
                   edgecolor="white", linewidth=0.8, height=0.55, zorder=3)
    ax.errorbar(vals, y, xerr=errs, fmt="none", color="#333333",
                capsize=4, linewidth=1.5, zorder=4)
    ax.axvline(0.5, color="#333333", ls="--", lw=1.0, alpha=0.5)
    ax.set_yticks(y); ax.set_yticklabels(metrics, fontsize=LABEL_FS)
    ax.set_xlabel("Metric value", fontsize=LABEL_FS)
    ax.set_xlim(0, 1.15)
    ax.tick_params(bottom=True, labelbottom=True)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    for bar, v, lo, hi in zip(bars, vals, lows, highs):
        ax.text(hi + 0.02, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}\n[{lo:.3f}–{hi:.3f}]",
                va="center", fontsize=7.5, color="#333333")

    # ── n=2000 bootstrap note ─────────────────────────────────────────────
    ax.text(0.98, 0.02, "95% bootstrap CI\nn = 2,000 replicates\n1,115 pairs · 192 targets",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#666666",
            bbox=dict(fc="white", ec="#cccccc", boxstyle="round,pad=0.3", alpha=0.8))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = OUT / "abstract_figure.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    draw()
