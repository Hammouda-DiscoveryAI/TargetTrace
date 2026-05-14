"""
Master script — run all figure-generation scripts after training completes.

Creates:
  results/training/      training_curves.png + individual panels
  results/evaluation/    roc_curve.png, pr_curve.png, pic50_scatter.png,
                         val_metrics_bar.png, evaluation_combined.png
  results/external/      pdbbind_evaluation.png, pdbbind_roc.png
  results/ablation/      ablation_pdbbind_auc.png,
                         ablation_internal_vs_external.png,
                         ablation_heatmap.png
  results/architecture/  architecture.png

Usage:
    python make_all_figures.py [--output_file PATH_TO_TRAINING_OUTPUT]
"""
import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))


def _find_training_output() -> str | None:
    base = Path("/tmp/claude-1000")
    candidates = sorted(
        base.rglob("bxluywrry.output"),
        key=lambda p: p.stat().st_mtime, reverse=True
    )
    if candidates:
        return str(candidates[0])
    # fallback: any output with training data
    for p in sorted(base.rglob("*.output"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            txt = p.read_text()
            if "Epoch" in txt and "AUC" in txt and "Training complete" in txt:
                return str(p)
        except Exception:
            pass
    return None


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_file", default=None,
                    help="Path to training output log file")
    ap.add_argument("--csv", default="pdbbind_eval.csv",
                    help="PDBbind evaluation CSV")
    args = ap.parse_args()

    t_start = time.time()

    # ── 1. Training curves ─────────────────────────────────────────────────
    section("1 / 5  Training Curves")
    out_file = args.output_file or _find_training_output()
    if out_file:
        print(f"Using training log: {out_file}")
        from plot_training import parse_log, plot as plot_training
        text = Path(out_file).read_text()
        rows = parse_log(text)
        print(f"  {len(rows)} epoch entries found")
        plot_training(rows, title_suffix=" (Real-Neg Model)")
    else:
        print("WARNING: No training output file found — skipping training plots.")

    # ── 2. Evaluation (shuffle-neg, validation-style) ──────────────────────
    section("2 / 5  Validation Evaluation (Shuffle-Neg Protocol)")
    from plot_evaluation import run_shuffle_eval, make_plots as make_eval_plots
    eval_res = run_shuffle_eval(args.csv)
    make_eval_plots(eval_res)

    # ── 3. External evaluation (PDBbind explicit negatives) ────────────────
    section("3 / 5  External Evaluation (PDBbind Explicit Negatives)")
    from plot_external import run_explicit_eval, make_plots as make_ext_plots
    ext_res = run_explicit_eval(args.csv)
    make_ext_plots(ext_res)

    # ── 4. Ablation ────────────────────────────────────────────────────────
    section("4 / 5  Ablation Study")
    # Patch in live results from external eval (avoid re-running inference)
    from run_ablation import ABLATION_TABLE, _get_val_auc_from_output, make_ablation_plots
    table = [r.copy() for r in ABLATION_TABLE]

    # val AUC from training output
    if out_file:
        val = _get_val_auc_from_output(out_file)
        if val:
            table[3]["val_auc"] = val
            print(f"  Full-model val AUC = {val:.4f}")

    # Use PDBbind explicit-neg AUC already computed above
    table[3]["pdbbind_auc"] = ext_res["auc"]
    table[3]["spearman"]    = ext_res["spearman"]
    table[3]["mae"]         = ext_res["mae"]
    print(f"  Full-model PDBbind AUC = {ext_res['auc']:.4f}")

    make_ablation_plots(table)

    # ── 5. Architecture diagram ────────────────────────────────────────────
    section("5 / 5  Architecture Diagram")
    from plot_architecture import draw as draw_arch
    draw_arch()

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    section("Done")
    print(f"All figures generated in {elapsed/60:.1f} min")
    print()
    print("Output directories:")
    for d in ["training", "evaluation", "external", "ablation", "architecture"]:
        dpath = BASE / "results" / d
        files = list(dpath.glob("*.png"))
        print(f"  results/{d}/   ({len(files)} files)")
    print()
    print("Key results:")
    print(f"  Validation AUC (shuffle-neg):  {eval_res['auc']:.4f}")
    print(f"  PDBbind AUC (explicit-neg):    {ext_res['auc']:.4f}")
    print(f"  pIC50 Spearman:                {ext_res['spearman']:.4f}")
    print(f"  pIC50 MAE:                     {ext_res['mae']:.4f}")


if __name__ == "__main__":
    main()
