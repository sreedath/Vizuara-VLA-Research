# CalibDrive: Uncertainty-Aware Vision-Language-Action Models for Safe Autonomous Driving

## Paper Outline (NeurIPS 2026)

### Title Options
1. **CalibDrive: Are Driving VLAs Calibrated? A Benchmark for Uncertainty-Aware Autonomous Driving**
2. **When VLAs Don't Know What They Don't Know: Uncertainty Quantification for Safe Autonomous Driving**
3. **Calibrate Before You Drive: Uncertainty-Aware Vision-Language-Action Models for Safety-Critical Navigation**

### Abstract (Target: 200 words)
- VLAs are promising for autonomous driving but lack uncertainty awareness
- We introduce CalibDrive: first benchmark for VLA calibration in driving
- Key finding: driving VLAs are systematically miscalibrated (ECE = X.XX), especially in long-tail scenarios
- We evaluate 5 UQ methods and show [best method] reduces ECE by X%
- Selective prediction with calibrated VLAs reduces collision rate by X% at Y% coverage
- Code and benchmark released at [URL]

---

### 1. Introduction (1.5 pages)
- VLAs unify perception, reasoning, and action for driving (cite DriveVLM, AutoVLA, Alpamayo-R1)
- Critical gap: VLAs don't quantify uncertainty — confident even in unfamiliar scenarios
- Only one prior study (Zollo et al., 2025) on VLA calibration, focused on robotics (LIBERO)
- No calibration study for driving VLAs exists
- This matters: safety-critical deployment requires knowing when to abstain
- **Contributions:**
  1. CalibDrive benchmark (8 scenario categories, 3 difficulty levels)
  2. First systematic calibration study of driving VLAs
  3. Comprehensive evaluation of 5 UQ methods for driving VLAs
  4. Selective prediction framework reducing collision rate by X%

### 2. Related Work (1 page)
#### 2.1 VLA Models for Autonomous Driving
- End-to-end: OpenVLA, AutoVLA, LatentVLA, ELF-VLA
- Dual-system: DriveVLM, InsightDrive
- With reasoning: Alpamayo-R1, FutureSightDrive, CoC-VLA

#### 2.2 Uncertainty Quantification in Deep Learning
- MC Dropout (Gal & Ghahramani, 2016)
- Deep Ensembles (Lakshminarayanan et al., 2017)
- Conformal Prediction (Angelopoulos & Bates, 2023)
- Temperature Scaling (Guo et al., 2017)

#### 2.3 Calibration of Foundation Models
- LLM calibration (Kadavath et al., 2022; Tian et al., 2023)
- VLM calibration studies
- VLA calibration: Zollo et al. (2025) — only on LIBERO robotics

#### 2.4 Safety in Autonomous Driving
- Selective prediction, human handoff
- Safe RL under uncertainty

### 3. CalibDrive Benchmark (1.5 pages)
#### 3.1 Design Principles
- Scenario diversity (8 categories from normal to long-tail)
- Difficulty stratification (easy/medium/hard)
- Standardized evaluation protocol
- Model-agnostic (works with any driving VLA)

#### 3.2 Scenario Categories
Table: 8 categories with description, difficulty, sample count, expected challenges

#### 3.3 Evaluation Protocol
- Open-loop trajectory prediction on NAVSIM
- Closed-loop evaluation on CARLA/Bench2Drive
- Fixed evaluation metrics (ECE, Brier, AUROC, collision rate)

#### 3.4 Metrics
- Calibration: ECE, MCE, Brier Score, NLL, reliability diagrams
- Uncertainty quality: AUROC (failure detection), AUPRC, AUSE (sparsification)
- Driving: ADE, FDE, collision rate, PDMS
- Safety: selective coverage, selective collision rate

### 4. Uncertainty Quantification Methods (2 pages)
#### 4.1 Problem Formulation
- VLA as f: (images, text) → trajectory
- Calibration: P(correct | confidence = p) = p
- Goal: uncertainty u(x) that correlates with prediction error

#### 4.2 Baseline: Raw Softmax Confidence
- Extract token-level probabilities from VLA
- Aggregate into trajectory-level confidence

#### 4.3 MC Dropout
- Enable dropout at inference, N forward passes
- Trajectory variance as uncertainty

#### 4.4 Temperature Scaling
- Learn scalar T on calibration set
- Apply T to token logits before decoding

#### 4.5 Deep Ensembles
- M independently trained VLAs
- Epistemic uncertainty from inter-model disagreement

#### 4.6 Conformal Prediction
- Distribution-free coverage guarantees
- Adaptive prediction sets based on uncertainty

#### 4.7 Prompt Ensembles (Novel for Driving)
- Multiple prompts describing same scenario differently
- Variance across prompt-conditioned predictions

### 5. Selective Prediction for Safety (1.5 pages)
#### 5.1 Selective Prediction Framework
- Three decisions: PROCEED, SLOW_DOWN, ABSTAIN
- Threshold-based: compare uncertainty to learned thresholds
- Learned policy: optimize coverage vs. safety Pareto frontier

#### 5.2 Coverage-Safety Trade-off
- Pareto curve: coverage on x-axis, collision rate on y-axis
- Show that calibrated VLAs dominate uncalibrated in this space

#### 5.3 Failure Prediction
- Can uncertainty predict failures N timesteps ahead?
- AUROC at t-1, t-3, t-5 before collision

### 6. Experiments (3 pages)
#### 6.1 Experimental Setup
- Models: OpenVLA-7B, [additional driving VLAs]
- Datasets: NAVSIM, [additional]
- Hardware, training details, hyperparameters

#### 6.2 How Calibrated Are Driving VLAs?
- Table: Baseline ECE across models and scenarios
- **Key finding**: ECE is X.XX overall, Y.YY on long-tail scenarios
- Reliability diagrams showing overconfidence

#### 6.3 UQ Method Comparison
- Main table: ECE, Brier, AUROC, compute cost for each method
- **Key finding**: [Best method] achieves X% ECE reduction

#### 6.4 Selective Prediction Results
- Coverage-collision Pareto curves
- **Key finding**: X% collision rate reduction at Y% coverage

#### 6.5 Per-Scenario Analysis
- Heatmap: calibration quality across scenario categories
- Long-tail scenarios show worst calibration

#### 6.6 Ablation Studies
- Number of MC samples
- Ensemble size
- Calibration set size
- Prompt variation effect

### 7. Discussion (0.5 pages)
- Why driving VLAs are miscalibrated (training objective mismatch)
- Practical deployment implications
- Limitations: compute cost of ensembles, need for calibration data
- Connection to regulatory requirements for AD

### 8. Conclusion (0.5 pages)
- First systematic study of driving VLA calibration
- CalibDrive benchmark enables reproducible evaluation
- Selective prediction demonstrates concrete safety improvement
- Open source: code, benchmark, and results

---

## Key Figures (to be generated with paperbanana)

1. **Fig 1**: Teaser — VLA making overconfident prediction in dangerous scenario + calibrated version abstaining
2. **Fig 2**: CalibDrive benchmark overview — scenario categories, difficulty levels, evaluation pipeline
3. **Fig 3**: Reliability diagrams — baseline vs. calibrated models
4. **Fig 4**: UQ method comparison radar chart — ECE, AUROC, compute, coverage
5. **Fig 5**: Coverage-collision Pareto curves — main result figure
6. **Fig 6**: Per-scenario calibration heatmap
7. **Fig 7**: Failure prediction AUROC vs. time-before-collision

## Key Tables

1. **Table 1**: Baseline VLA calibration across models and scenarios
2. **Table 2**: UQ method comparison (main results)
3. **Table 3**: Selective prediction results
4. **Table 4**: Ablation studies
