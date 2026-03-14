# CalibDrive: Uncertainty-Aware VLA for Safe Autonomous Driving

## Research Program

This document defines the autonomous research program for studying uncertainty quantification in Vision-Language-Action models for autonomous driving, following the [autoresearch](https://github.com/karpathy/autoresearch) methodology.

## Research Questions

1. **RQ1 (Calibration Gap):** How well-calibrated are current driving VLAs? Do they exhibit systematic overconfidence in out-of-distribution or long-tail scenarios?
2. **RQ2 (UQ Methods):** Which uncertainty quantification methods (MC Dropout, Deep Ensembles, Temperature Scaling, Conformal Prediction) best improve VLA calibration for driving?
3. **RQ3 (Selective Prediction):** Can uncertainty-aware selective prediction (abstain/slow-down/handoff) measurably reduce collision rates while maintaining throughput?
4. **RQ4 (Failure Detection):** Can VLA uncertainty scores predict impending driving failures before they occur?

## Experimental Design

### Models Under Study
- **OpenVLA** (7B) — open-source baseline VLA
- **Driving-specific VLAs** — adapted from DriveVLM / OpenDriveVLA architectures
- **Lightweight VLA** — smaller models for ablation studies

### Uncertainty Methods
1. **MC Dropout**: Enable dropout at inference, run N forward passes, measure variance
2. **Deep Ensembles**: Train M independent models, aggregate predictions
3. **Temperature Scaling**: Post-hoc calibration on held-out validation set
4. **Conformal Prediction**: Distribution-free prediction intervals with coverage guarantees
5. **Evidential Deep Learning**: Dirichlet-based uncertainty for single-pass inference

### Datasets and Benchmarks
- **NAVSIM**: Realistic driving simulation benchmark
- **nuScenes**: Real-world driving dataset with diverse scenarios
- **CARLA**: Controllable simulation for long-tail scenario generation

### Metrics
**Calibration Metrics:**
- Expected Calibration Error (ECE)
- Brier Score
- Negative Log-Likelihood (NLL)
- Reliability Diagrams

**Safety Metrics:**
- AUROC for failure prediction
- Collision rate (overall and per-scenario)
- False abstention rate
- Selective prediction coverage vs. safety trade-off curves

**Driving Performance Metrics:**
- PDMS (Predictive Driver Model Score) on NAVSIM
- Planning metrics (L2 displacement error, heading error)

## Experiment Loop

Adapted from autoresearch methodology:

```
LOOP:
1. Formulate hypothesis about uncertainty/calibration improvement
2. Implement the change in src/
3. Run experiment with fixed evaluation protocol
4. Record results in experiments/results.tsv
5. If metrics improve → keep (git commit)
6. If metrics worsen → discard (git reset)
7. Analyze results, generate next hypothesis
```

### Results Format (results.tsv)

```
commit	ece	brier	auroc	collision_rate	status	description
```

## Phases

### Phase 1: Baseline Characterization
- Establish baseline VLA performance on driving benchmarks
- Measure out-of-the-box calibration (ECE, reliability diagrams)
- Identify systematic miscalibration patterns

### Phase 2: Uncertainty Methods
- Implement and evaluate each UQ method
- Compare calibration improvement across methods
- Analyze computational cost vs. calibration trade-off

### Phase 3: Selective Prediction
- Design uncertainty-threshold policies (abstain/slow/handoff)
- Evaluate collision rate reduction under selective prediction
- Optimize coverage vs. safety Pareto frontier

### Phase 4: Long-Tail Analysis
- Generate/curate challenging driving scenarios
- Study uncertainty behavior in OOD conditions
- Demonstrate failure prediction capability

## Never Stop

Once experiments begin, continue autonomously. If stuck, try:
- Combine methods (e.g., ensemble + conformal)
- Ablate components
- Try different model sizes
- Vary scenario difficulty
- Explore new calibration metrics
