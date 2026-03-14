# CalibDrive Experiment Plan

## Phase 1: Baseline Characterization (Weeks 1-2)

### Experiment 1.1: Baseline VLA Calibration Assessment
- **Goal**: Measure out-of-the-box calibration of driving VLAs
- **Models**: OpenVLA-7B, smaller VLA variants
- **Dataset**: NAVSIM validation set (1000 scenarios)
- **Metrics**: ECE, MCE, Brier Score, reliability diagrams
- **Expected outcome**: VLAs are significantly miscalibrated (ECE > 0.15)

### Experiment 1.2: Calibration vs. Scenario Difficulty
- **Goal**: Analyze how calibration degrades with scenario difficulty
- **Setup**: Split validation by scenario type (normal, adverse, long-tail)
- **Hypothesis**: Calibration degrades significantly in hard/long-tail scenarios
- **Metrics**: Per-scenario ECE, confidence distributions

### Experiment 1.3: Overconfidence in OOD Scenarios
- **Goal**: Demonstrate systematic overconfidence in out-of-distribution inputs
- **Setup**: Evaluate on held-out scenario categories not in training
- **Metrics**: Confidence on OOD vs ID, separation quality (AUROC)

## Phase 2: Uncertainty Quantification Methods (Weeks 3-5)

### Experiment 2.1: MC Dropout
- **Setup**: Enable dropout (p=0.1), N={5, 10, 20, 50} forward passes
- **Metrics**: ECE improvement, computation overhead, AUROC
- **Ablation**: Dropout rate {0.05, 0.1, 0.2, 0.3}

### Experiment 2.2: Temperature Scaling
- **Setup**: Learn T on 30% of validation set, evaluate on rest
- **Metrics**: ECE before/after, NLL improvement
- **Ablation**: Calibration set size {10%, 20%, 30%, 50%}

### Experiment 2.3: Deep Ensembles
- **Setup**: M={3, 5, 7} independently fine-tuned VLAs
- **Metrics**: ECE, Brier, AUROC, ensemble disagreement analysis
- **Note**: Most compute-intensive experiment

### Experiment 2.4: Conformal Prediction
- **Setup**: Split conformal with alpha={0.05, 0.1, 0.2}
- **Metrics**: Empirical coverage, prediction set size, efficiency
- **Variant**: Adaptive conformal (uncertainty-weighted sets)

### Experiment 2.5: Evidential Deep Learning
- **Setup**: Replace output head with Dirichlet parameterization
- **Metrics**: ECE, aleatoric vs epistemic decomposition
- **Training**: Fine-tune with evidential loss

### Experiment 2.6: Method Comparison
- **Goal**: Head-to-head comparison across all methods
- **Axes**: Calibration quality, compute cost, failure detection, ease of use
- **Output**: Main comparison table for paper

## Phase 3: Selective Prediction for Safety (Weeks 6-8)

### Experiment 3.1: Uncertainty-Threshold Policies
- **Policies**: Abstain, Slow-Down, Human Handoff
- **Setup**: Sweep confidence thresholds [0.1, 0.2, ..., 0.9]
- **Metrics**: Coverage vs collision rate Pareto curve

### Experiment 3.2: Collision Rate Reduction
- **Goal**: Quantify safety improvement from selective prediction
- **Setup**: Compare collision rate with/without selective prediction
- **Target**: >30% collision rate reduction at >80% coverage

### Experiment 3.3: Failure Prediction
- **Goal**: Show uncertainty predicts failures before they occur
- **Setup**: Look at uncertainty N timesteps before collision
- **Metrics**: AUROC for failure prediction at t-1, t-3, t-5

### Experiment 3.4: Optimal Policy Search
- **Goal**: Learn optimal selective prediction policy
- **Approach**: Constrained optimization (minimize collision rate s.t. coverage >= C)

## Phase 4: Long-Tail Analysis (Weeks 9-10)

### Experiment 4.1: Scenario-Specific Calibration
- **Goal**: Detailed calibration analysis per scenario category
- **Categories**: 8 categories from CalibDrive benchmark
- **Metrics**: Per-category ECE, AUROC, confidence distributions

### Experiment 4.2: Distributional Shift Detection
- **Goal**: Use uncertainty to detect distributional shift
- **Setup**: Gradually increase scenario difficulty, track uncertainty
- **Metrics**: Shift detection AUROC, response time

### Experiment 4.3: Combining Methods
- **Goal**: Explore combinations (e.g., ensemble + conformal)
- **Setup**: Stack methods, measure marginal improvement
- **Hypothesis**: Combinations outperform individual methods

## Compute Requirements

| Experiment | GPU Hours | GPUs | Priority |
|-----------|----------|------|----------|
| Phase 1 | 20-40 | 1 | HIGH |
| Phase 2 | 100-200 | 1-4 | HIGH |
| Phase 3 | 40-80 | 1 | HIGH |
| Phase 4 | 60-100 | 1-2 | MEDIUM |
| **Total** | **220-420** | | |

## Key Ablations for Paper

1. Effect of number of MC samples (5, 10, 20, 50)
2. Effect of ensemble size (3, 5, 7)
3. Effect of calibration set size (10%, 20%, 30%, 50%)
4. Coverage vs safety trade-off curves
5. Per-scenario uncertainty analysis
6. Computational overhead comparison
