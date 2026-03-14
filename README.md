# Vizuara-VLA-Research

## Uncertainty-Aware Vision-Language-Action Models for Safe Autonomous Driving

**Authors:** Raj Dandekar, Rajat Dandekar, Sreedath Panat, Claude Code

Vision-Language-Action (VLA) models show remarkable promise for autonomous driving by unifying perception, reasoning, and action in a single framework. However, a critical gap remains: **VLAs don't know what they don't know.** Current driving VLAs produce confident action predictions even in unfamiliar or dangerous scenarios, lacking the ability to quantify their own uncertainty — a fundamental requirement for safety-critical deployment.

This repository presents **CalibDrive**, the first systematic framework for uncertainty quantification and calibration of VLA models in autonomous driving. We study how well driving VLAs are calibrated, develop methods to improve their uncertainty estimates, and demonstrate that uncertainty-aware selective prediction significantly reduces collision rates in long-tail driving scenarios.

## Key Contributions

1. **CalibDrive Benchmark**: First systematic benchmark for evaluating VLA calibration in autonomous driving, spanning normal, adverse, and long-tail scenarios
2. **Uncertainty Quantification Framework**: Comprehensive study of uncertainty methods (MC Dropout, Deep Ensembles, Temperature Scaling, Conformal Prediction) applied to driving VLAs
3. **Selective Prediction for Safety**: Uncertainty-aware action selection that enables VLAs to abstain, slow down, or request human intervention when confidence is low
4. **Empirical Analysis**: Extensive experiments showing calibrated VLAs achieve measurably lower collision rates while maintaining driving performance

## Project Structure

```
src/
├── models/          # VLA model wrappers and interfaces
├── calibration/     # Uncertainty quantification methods
├── evaluation/      # Calibration metrics and driving metrics
├── data/            # Data loading for driving benchmarks
└── utils/           # Shared utilities
experiments/         # Experiment configs and tracking
configs/             # Model and training configurations
paper/               # LaTeX source and figures
notebooks/           # Analysis and visualization
scripts/             # Setup and utility scripts
```

## Research Methodology

This project follows the [autoresearch](https://github.com/karpathy/autoresearch) philosophy of autonomous experimentation: systematic hypothesis testing with tracked results, keeping improvements and discarding failures.

## Setup

```bash
# Clone the repository
git clone https://github.com/sreedath/Vizuara-VLA-Research.git
cd Vizuara-VLA-Research

# Install dependencies
pip install -r requirements.txt

# Run experiments
python scripts/run_experiment.py --config configs/baseline.yaml
```

## Results

See `experiments/results.tsv` for the full experiment log.

## License

MIT
