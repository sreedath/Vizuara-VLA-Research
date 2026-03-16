#!/bin/bash
# Generate all paper figures using paperbanana
# Usage: GEMINI_API_KEY=<your_key> bash generate_figures.sh

set -e

if [ -z "$GEMINI_API_KEY" ]; then
    echo "Error: GEMINI_API_KEY not set"
    echo "Usage: GEMINI_API_KEY=<your_key> bash generate_figures.sh"
    exit 1
fi

cd "$(dirname "$0")/.."
source .venv/bin/activate

OUTDIR="paper/latex/figures_pb"
DATADIR="paper/data"

echo "=== Generating Figure 1: Pipeline Diagram ==="
paperbanana generate \
  --input "$DATADIR/fig1_method.txt" \
  --caption "Overview of the hidden-state monitoring pipeline for VLA OOD detection. During calibration (left), a centroid is computed from clean image embeddings at an early transformer layer. During deployment (right), cosine distance from the centroid provides near-zero-cost OOD detection while output confidence remains uninformative." \
  --output "$OUTDIR/fig1_pipeline.png" \
  --iterations 2 &

echo "=== Generating Figure 2: Main Results Bar Chart ==="
paperbanana plot \
  --data "$DATADIR/fig2_main_results.csv" \
  --intent "Grouped bar chart comparing OOD detection AUROC across 6 methods (MSP, Energy Score, Entropy, Top-k Probability, Feature Norm, and Cosine Distance which is ours) for 4 corruption types (Fog, Night, Noise, Blur) plus Mean. Our method achieves perfect 1.0 AUROC across all corruptions while baselines fail catastrophically on fog. Use professional NeurIPS academic styling with clear legend. Highlight our method with a distinct darker color." \
  --output "$OUTDIR/fig2_main_results.png" \
  --iterations 2 &

echo "=== Generating Figure 3: Calibration Efficiency ==="
paperbanana plot \
  --data "$DATADIR/fig3_calibration_size.csv" \
  --intent "Line plot showing AUROC vs number of calibration images for 4 corruption types (Fog, Night, Noise, Blur). X-axis: calibration images (1,2,5,10,20). Y-axis: AUROC 0.94 to 1.01. Fog, Night, Blur are flat at 1.0. Noise starts at 0.96 and converges to 1.0. Professional NeurIPS styling with markers and gridlines." \
  --output "$OUTDIR/fig3_calibration.png" \
  --iterations 2 &

echo "=== Generating Figure 4: Severity Curves ==="
paperbanana plot \
  --data "$DATADIR/fig4_severity.csv" \
  --intent "Log-scale line plot showing cosine distance vs corruption severity for Fog, Night, Noise, Blur with a horizontal dashed threshold line at 8.03e-5. Y-axis log scale 1e-6 to 1e-1. Blur and Night cross threshold early, Noise barely crosses at 0.75. Shade region below threshold. Professional NeurIPS academic styling." \
  --output "$OUTDIR/fig4_severity.png" \
  --iterations 2 &

echo "=== Generating Figure 5: Layer Sweep ==="
paperbanana plot \
  --data "$DATADIR/fig5_layer_sweep.csv" \
  --intent "Dual-axis plot: left y-axis AUROC (0-1.05, blue), right y-axis Separation Ratio (0-280, red). X-axis: Layer 0-32. Layer 0 has AUROC=0. Layer 1 peaks at 256.9x separation. Layers 1-9 have AUROC=1.0. Middle layers dip slightly. Professional NeurIPS dual-axis styling." \
  --output "$OUTDIR/fig5_layer_sweep.png" \
  --iterations 2 &

echo "=== Generating Figure 6: Mechanism Diagram ==="
paperbanana generate \
  --input "$DATADIR/fig7_mechanism.txt" \
  --caption "Mechanistic analysis: attention patterns are nearly unchanged at the detection layer (cosine similarity >0.9998) while hidden-state cosine distances show orders-of-magnitude separation between clean and corrupted inputs. The OOD signal is in geometry, not attention." \
  --output "$OUTDIR/fig6_mechanism.png" \
  --iterations 2 &

echo "All 6 figures launched in parallel. Waiting..."
wait
echo "=== All figures generated! ==="
ls -la "$OUTDIR"/*.png 2>/dev/null || echo "Check $OUTDIR/run_*/ for outputs"
