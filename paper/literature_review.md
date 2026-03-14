# Literature Review: Uncertainty-Aware Vision-Language-Action Models for Safe Autonomous Driving

**Prepared for NeurIPS-level submission**
**Last updated: March 2026**

---

## Table of Contents

1. [Vision-Language-Action Models for Driving](#1-vision-language-action-models-for-driving)
2. [Uncertainty Quantification in Deep Learning](#2-uncertainty-quantification-in-deep-learning)
3. [Calibration of Large Language Models and Vision-Language Models](#3-calibration-of-large-language-models-and-vision-language-models)
4. [VLA Calibration (The Gap)](#4-vla-calibration-the-gap)
5. [Safety-Critical Decision Making Under Uncertainty](#5-safety-critical-decision-making-under-uncertainty)
6. [Benchmarks and Datasets](#6-benchmarks-and-datasets)

---

## 1. Vision-Language-Action Models for Driving

Vision-Language-Action (VLA) models represent a paradigm shift in autonomous driving, integrating visual perception, natural language understanding, and control within a single policy. Recent surveys (Jiang et al., 2025; World Bench et al., 2025) identify four successive waves of evolution: Pre-VLA Explainers, Modular VLA4AD, End-to-End VLA4AD, and Reasoning-Augmented VLA4AD. Below, we review the major VLA models relevant to autonomous driving.

### 1.1 DriveVLM

- **Authors:** Tian, X., Gu, J., et al.
- **Year:** 2024
- **Venue:** CoRL 2024 / arXiv 2402.12289
- **Title:** *DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models*

**Key Contribution:** DriveVLM is a pioneering system that leverages Vision-Language Models (VLMs) for enhanced scene understanding and planning in autonomous driving, particularly targeting complex and long-tail scenarios such as challenging road conditions and delicate human behaviors.

**Architecture:** The system employs a Chain-of-Thought (CoT) process with three sequential modules: (1) scene description, (2) scene analysis, and (3) hierarchical planning. It maps visual observations into language space and uses Qwen-VL (9.6B parameters) as the default vision-language backbone. **DriveVLM-Dual** extends this by combining the VLM reasoning with a traditional autonomous driving pipeline (VAD), addressing VLM limitations in spatial reasoning and real-time inference.

**Key Results:** State-of-the-art performance on the nuScenes planning task when combined with VAD. Deployed on a production vehicle, confirming real-world effectiveness.

**Limitations:** VLMs exhibit poor spatial reasoning capability; high computational requirements make real-time inference challenging without the Dual hybrid architecture; relies on open-loop evaluation which may not fully reflect closed-loop driving performance.

**Relevance:** Establishes the foundational VLM-for-driving paradigm but does not address uncertainty quantification in its planning outputs, creating a gap our work aims to fill.

---

### 1.2 AutoVLA

- **Authors:** UCLA Mobility Group
- **Year:** 2025
- **Venue:** NeurIPS 2025 / arXiv 2506.13757
- **Title:** *AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning*

**Key Contribution:** AutoVLA unifies reasoning and action generation within a single autoregressive generation model, performing semantic reasoning and trajectory planning directly from raw visual inputs and language instructions.

**Architecture:** The model tokenizes continuous trajectories into discrete, feasible actions for integration into the language model. It employs supervised fine-tuning with dual thinking modes: (1) **fast thinking** (trajectory-only for simple scenarios) and (2) **slow thinking** (enhanced with chain-of-thought reasoning for complex situations). Reinforcement fine-tuning based on Group Relative Policy Optimization (GRPO) reduces unnecessary reasoning in straightforward scenarios.

**Key Results:** Top score in the RFS Spotlight metric in the Waymo Vision-based End-to-End Driving Challenge (May 2025). Highly ranked in both RFS Overall and ADE metrics.

**Limitations:** Autoregressive action tokenization introduces quantization artifacts; adaptive reasoning allocation between fast/slow modes may not always be optimal; no explicit uncertainty quantification mechanism.

**Relevance:** The dual thinking mode paradigm suggests that different driving scenarios require different levels of cognitive engagement, which aligns with our hypothesis that uncertainty-aware models should allocate more careful reasoning to high-uncertainty situations.

---

### 1.3 OpenDriveVLA

- **Authors:** DriveVLA Team
- **Year:** 2025
- **Venue:** AAAI 2026 / arXiv 2503.23463
- **Title:** *OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Model*

**Key Contribution:** An open-source VLA model for end-to-end autonomous driving that builds upon pre-trained large VLMs to generate reliable driving actions conditioned on 3D environmental perception, ego vehicle states, and driver commands.

**Architecture:** Proposes a hierarchical vision-language alignment process, projecting both 2D and 3D structured visual tokens into a unified semantic space. Models agent-environment-ego interactions through an autoregressive process, ensuring spatially and behaviorally informed trajectory planning. Autoregressively produces trajectory waypoints from structured vision-language tokens.

**Key Results:** State-of-the-art results on nuScenes across open-loop trajectory planning and driving-related question-answering tasks.

**Limitations:** Open-loop evaluation only; no closed-loop validation; no uncertainty estimation for generated trajectories.

**Relevance:** As an open-source model, OpenDriveVLA is a strong candidate for applying post-hoc uncertainty quantification methods, making it directly relevant as a baseline for our calibration study.

---

### 1.4 Alpamayo-R1

- **Authors:** NVIDIA Autonomous Vehicle Research Group (Wang, Luo, et al.)
- **Year:** 2025
- **Venue:** arXiv 2511.00088
- **Title:** *Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail*

**Key Contribution:** Integrates Chain of Causation (CoC) reasoning with trajectory planning for complex driving scenarios. Part of NVIDIA's Alpamayo family of open-source AI models.

**Architecture:** Three key innovations: (1) the **Chain of Causation (CoC) dataset** built through a hybrid auto-labeling and human-in-the-loop pipeline, producing 700,000 decision-grounded, causally linked reasoning traces; (2) a **modular VLA architecture** combining Cosmos-Reason (a VLM pre-trained for Physical AI) with a diffusion-based trajectory decoder using flow matching for real-time dynamically feasible plans; (3) a **multi-stage training strategy** using supervised fine-tuning followed by reinforcement learning to optimize reasoning quality.

**Key Results:** Up to 12% improvement in planning accuracy on challenging cases compared to trajectory-only baseline; 35% reduction in close encounter rate in closed-loop simulation.

**Limitations:** Reasoning traces are post-hoc rationalizations, not verified causal explanations; diffusion-based action generation may introduce multi-modality that complicates uncertainty quantification; computational cost of full reasoning chain.

**Relevance:** The CoC reasoning framework provides structured decision rationale that could be combined with uncertainty quantification -- uncertain scenarios could trigger deeper causal reasoning chains. The diffusion-based trajectory generation naturally supports multi-modal trajectory distributions.

---

### 1.5 ELF-VLA

- **Authors:** Not specified
- **Year:** 2025/2026
- **Venue:** arXiv 2603.01063
- **Title:** *Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures*

**Key Contribution:** A framework that augments reinforcement learning with structured diagnostic feedback to address performance plateaus in VLA models during RL optimization.

**Architecture:** When VLA models hit RL performance plateaus, ELF-VLA generates detailed, interpretable failure reports that identify the specific failure mode (incorrect planning, flawed reasoning, or poor trajectory execution) instead of relying on vague scalar rewards. This structured feedback enables targeted refinement of the model.

**Key Results:** PDMS of 91.0 on NAVSIMv1 under vision-only setting, establishing a new state-of-the-art (0.7 PDMS improvement over the previous best vision-only method). State-of-the-art on NAVSIM for PDMS, EPDMS score, and high-level planning accuracy.

**Limitations:** The failure diagnosis is performed offline during training, not during inference; requires extensive computational resources for RL training; does not provide runtime uncertainty estimates.

**Relevance:** The failure mode taxonomy (planning failures vs. reasoning failures vs. execution failures) directly informs how uncertainty should be decomposed in driving VLAs -- epistemic uncertainty may manifest differently in each failure category.

---

### 1.6 FastDriveVLA

- **Authors:** XPENG & Peking University
- **Year:** 2025
- **Venue:** AAAI 2026 / arXiv 2507.23318
- **Title:** *FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning*

**Key Contribution:** A novel reconstruction-based vision token pruning framework designed specifically for autonomous driving VLAs, addressing the computational bottleneck of processing high-resolution visual inputs.

**Architecture:** Introduces **ReconPruner**, a plug-and-play visual token pruner that prioritizes foreground information through MAE-style pixel reconstruction, with an adversarial foreground-background reconstruction strategy for training. The pruner selectively removes redundant visual tokens while preserving driving-critical information.

**Key Results:** Achieves approximately 4.86 FPS, significantly faster than unpruned baselines while maintaining competitive driving performance.

**Limitations:** Token pruning may inadvertently remove safety-critical information in edge cases; the adversarial strategy may not generalize to all driving scenarios; still below real-time requirements for deployment.

**Relevance:** Efficiency improvements are critical for deploying uncertainty-aware VLAs in real-time driving, as uncertainty quantification methods (ensembles, MC dropout) multiply computational cost. FastDriveVLA's pruning could be combined with lightweight uncertainty methods.

---

### 1.7 DriveLM

- **Authors:** OpenDriveLab (Sima, C., et al.)
- **Year:** 2024
- **Venue:** ECCV 2024 (Oral) / arXiv 2312.14150
- **Title:** *DriveLM: Driving with Graph Visual Question Answering*

**Key Contribution:** Introduces Graph VQA as a proxy task for modeling the human reasoning process in driving, with graph-structured reasoning through perception, prediction, and planning question-answer pairs.

**Architecture:** Develops explicit chain-based reasoning pipelines for data construction. The DriveLM-Data dataset is built upon nuScenes and CARLA with structured QA graphs. DriveLM-Agent serves as a VLM-based baseline for jointly performing Graph VQA and end-to-end driving. A key insight: human drivers reason in multiple steps -- localization of key objects, estimation of interactions, then action decisions.

**Key Results:** Competitive end-to-end autonomous driving compared to state-of-the-art driving-specific architectures. Benefits pronounced in zero-shot evaluation on unseen objects or sensor configurations. Featured as a main track in the CVPR 2024 Autonomous Driving Challenge.

**Limitations:** Graph VQA introduces overhead; open-loop evaluation; reasoning graphs may not capture all safety-critical aspects.

**Relevance:** The structured Graph VQA framework provides interpretable intermediate representations where uncertainty could be quantified at each reasoning stage (perception uncertainty, prediction uncertainty, planning uncertainty).

---

### 1.8 SimLingo

- **Authors:** Renz, K., et al.
- **Year:** 2025
- **Venue:** CVPR 2025 (Spotlight) / arXiv 2503.09594
- **Title:** *SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment*

**Key Contribution:** A VLA model achieving state-of-the-art closed-loop driving performance on both the CARLA Leaderboard and Bench2Drive, while simultaneously supporting language capabilities (VQA, commentary, instruction following).

**Architecture:** Built on InternVL2-1B and Qwen2-0.5B vision-language backbones. Uses a disentangled output representation separating temporal speed waypoints from geometric path waypoints. The breakthrough innovation is **Action Dreaming**: each input frame is augmented with multiple potential future trajectories (turning left, speeding up, braking), each paired with a unique instruction, enabling language-action alignment without the model ignoring language inputs.

**Key Results:** Leading driving score of 85.94 on Bench2Drive; winning entry at the CARLA Challenge 2024; state-of-the-art on closed-loop benchmarks.

**Limitations:** Relatively small backbone models may limit reasoning depth; action dreaming requires careful trajectory augmentation; vision-only approach lacks LiDAR-based depth information.

**Relevance:** SimLingo's disentangled speed/path representation could allow dimension-specific uncertainty quantification, similar to the action-wise calibration proposed in Zollo et al. (2025) but applied to driving-specific action decomposition.

---

### 1.9 LatentVLA

- **Authors:** Not specified
- **Year:** 2025
- **Venue:** arXiv 2601.05611
- **Title:** *LatentVLA: Efficient Vision-Language Models for Autonomous Driving via Latent Action Prediction*

**Key Contribution:** A framework using self-supervised latent action prediction to train VLA models without language annotations, eliminating linguistic bias while learning rich driving representations from unlabeled trajectory data.

**Architecture:** Two-stage training: (1) ego-centric latent action tokens generated by a trained Latent Action Model serve as supervision for the VLM to predict latent actions; (2) VLM parameters are frozen and visual/action embeddings are combined using multi-head attention pooling. Continuous action tokens are discretized via VQ-VAE, yielding quantized representations indexed by a codebook. Knowledge distillation transfers VLA capabilities to efficient vision-based networks for real-time deployment.

**Key Results:** State-of-the-art on NAVSIM with PDMS score of 92.4; strong zero-shot generalization on nuScenes.

**Limitations:** Discretization via VQ-VAE may lose fine-grained action information; self-supervised training may miss safety-critical corner cases present in language-annotated data; latent space interpretability is limited.

**Relevance:** The VQ-VAE codebook provides a natural discrete distribution over latent actions where entropy can serve as an uncertainty proxy. This is a compelling architecture for studying how latent-space uncertainty relates to driving safety.

---

### 1.10 InsightDrive

- **Authors:** Not specified
- **Year:** 2025
- **Venue:** arXiv 2503.13047
- **Title:** *InsightDrive: Insight Scene Representation for End-to-End Autonomous Driving*

**Key Contribution:** Integrates causal language reasoning with Model Predictive Control (MPC), assigning "why" to the VLM and "how" to the planner.

**Architecture:** Introduces an Insight scene representation that jointly models attention-centric explicit scene representation and reasoning-centric implicit scene representation, aligning scene understanding with human cognitive patterns. Models human reasoning with Chain-of-Thought (CoT) instructions and injects this knowledge into scene features via a task-level Mixture-of-Experts (MoE) adapter.

**Key Results:** Competitive performance with improved interpretability compared to pure end-to-end approaches.

**Limitations:** MPC planner may be conservative; MoE routing adds complexity; the separation of "why" and "how" may introduce misalignment between reasoning and execution.

**Relevance:** The explicit separation of reasoning (VLM) and execution (MPC) creates natural intervention points for uncertainty-aware decision making: high reasoning uncertainty could trigger conservative MPC behavior or human handoff.

---

### 1.11 FutureSightDrive

- **Authors:** Not specified
- **Year:** 2025
- **Venue:** NeurIPS 2025 / arXiv 2505.17685
- **Title:** *FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving*

**Key Contribution:** Proposes Spatio-Temporal Visual Chain-of-Thought, enabling VLMs to "think with images" and function as inverse dynamics models for trajectory planning.

**Architecture:** The spatio-temporal CoT acts as an intermediate reasoning step between current observations and future predictions, establishing an end-to-end visual reasoning pipeline that enables direct visual causal inference. This eliminates semantic gaps caused by cross-modal conversions (e.g., converting visual perceptions into textual descriptions for reasoning).

**Key Results:** Demonstrates that visual chain-of-thought reasoning can outperform textual chain-of-thought for spatial planning tasks.

**Limitations:** Computational cost of generating visual future predictions; reliance on accurate world model for future frame generation; no explicit uncertainty handling for predicted futures.

**Relevance:** Predicted future frames inherently carry uncertainty; incorporating uncertainty over multiple predicted futures could directly improve planning safety. The visual CoT framework could be extended with uncertainty-weighted future reasoning.

---

### 1.12 CoC-VLA

- **Authors:** Not specified
- **Year:** 2025
- **Venue:** arXiv 2511.19914
- **Title:** *CoC-VLA: Delving into Adversarial Domain Transfer for Explainable Autonomous Driving via Chain-of-Causality Visual-Language-Action Model*

**Key Contribution:** A VLM-guided, end-to-end adversarial transfer framework that transfers long-tail handling capabilities from simulation to real-world deployment.

**Architecture:** Both teacher and student VLM models use a shared Chain-of-Causality Visual-Language Model (CoC VLM) architecture, integrating temporal information via an end-to-end text adapter. Supports chain-of-thought reasoning for complex driving logic inference. Adversarial domain adaptation bridges the sim-to-real gap.

**Key Results:** Improved transfer of long-tail scenario handling from simulation to real-world driving.

**Limitations:** Adversarial training can be unstable; sim-to-real gap may persist for rare real-world events; computational overhead of dual-model architecture.

**Relevance:** Domain transfer introduces a natural source of distributional uncertainty. Uncertainty-aware models could identify when inputs fall outside the training distribution (simulation vs. real world), triggering appropriate safety responses.

---

### 1.13 DriveVLA-W0

- **Authors:** BraveGroup
- **Year:** 2025
- **Venue:** arXiv 2510.12796
- **Title:** *DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving*

**Key Contribution:** A training paradigm that employs world modeling to predict future images, addressing the "supervision deficit" where vast VLA model capacity is supervised by sparse, low-dimensional actions.

**Architecture:** Implements world modeling as an additional training objective for two dominant VLA archetypes: (1) an **autoregressive world model** for VLAs using discrete visual tokens, and (2) a **diffusion world model** for those operating on continuous visual features. A lightweight action expert handles inference latency for real-time deployment.

**Key Results:** Significantly outperforms BEV and VLA baselines on NAVSIM v1/v2 and a 680x larger in-house dataset. Crucially, amplifies the data scaling law -- performance gains accelerate as training data increases.

**Limitations:** World model training is computationally expensive; future image prediction accuracy degrades over longer horizons; the lightweight action expert may sacrifice some world model benefits at inference time.

**Relevance:** World model predictions carry inherent uncertainty about future states. Uncertainty in world model predictions could serve as a leading indicator for planning uncertainty, enabling proactive safety interventions before dangerous situations develop.

---

### Summary Table: VLA Models for Autonomous Driving

| Model | Year | Venue | Key Innovation | Benchmark | Safety/Uncertainty |
|-------|------|-------|---------------|-----------|-------------------|
| DriveVLM | 2024 | CoRL 2024 | CoT scene understanding | nuScenes | None |
| AutoVLA | 2025 | NeurIPS 2025 | Dual thinking + GRPO | Waymo E2E | None |
| OpenDriveVLA | 2025 | AAAI 2026 | 3D hierarchical alignment | nuScenes | None |
| Alpamayo-R1 | 2025 | arXiv | Chain of Causation | NAVSIM | Causal reasoning |
| ELF-VLA | 2025 | arXiv | Failure diagnosis + RL | NAVSIM | Failure modes |
| FastDriveVLA | 2025 | AAAI 2026 | Token pruning | NAVSIM | None |
| DriveLM | 2024 | ECCV 2024 | Graph VQA | nuScenes, CARLA | None |
| SimLingo | 2025 | CVPR 2025 | Action Dreaming | Bench2Drive, CARLA | None |
| LatentVLA | 2025 | arXiv | Self-supervised latent actions | NAVSIM | None |
| InsightDrive | 2025 | arXiv | VLM + MPC separation | nuScenes | Indirect |
| FutureSightDrive | 2025 | NeurIPS 2025 | Visual CoT | NAVSIM | None |
| CoC-VLA | 2025 | arXiv | Adversarial domain transfer | CARLA | None |
| DriveVLA-W0 | 2025 | arXiv | World model training | NAVSIM | None |

**Critical Observation:** None of the 13 surveyed VLA models for driving include explicit uncertainty quantification or calibration mechanisms. This represents the central gap our work addresses.

---

## 2. Uncertainty Quantification in Deep Learning

This section reviews the foundational methods for uncertainty quantification (UQ) in deep learning, with emphasis on their applicability to sequence prediction and action generation -- the core operations of VLA models.

### 2.1 MC Dropout

- **Authors:** Gal, Y. & Ghahramani, Z.
- **Year:** 2016
- **Venue:** ICML 2016
- **Title:** *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*

**Key Contribution:** Established the theoretical framework casting dropout training in deep neural networks as approximate Bayesian inference in deep Gaussian processes. This provides principled tools to model uncertainty without sacrificing computational complexity or test accuracy.

**Method:** Uncertainty estimation is obtained by computing the variance across multiple stochastic forward passes with different dropout masks at inference time. The optimization of a neural network with a dropout layer is shown to be equivalent to approximating a Bayesian neural network with variational inference.

**Key Results:** Demonstrated that existing dropout networks can be used for uncertainty estimation with no additional training cost. The predictive variance decomposes into aleatoric (data) and epistemic (model) uncertainty.

**Application to VLAs:** MC Dropout can be applied to VLA models by keeping dropout active during inference and performing multiple forward passes to generate a distribution over predicted trajectories. The variance across trajectories indicates regions of high uncertainty. **Challenge for VLAs:** Autoregressive token generation means uncertainty compounds across the sequence; each token's dropout-induced variance propagates through subsequent tokens. This compounding effect makes trajectory-level uncertainty estimation non-trivial.

**Relevance:** Most lightweight UQ method applicable to existing VLAs without retraining. A natural first baseline for our proposed calibration study.

---

### 2.2 Deep Ensembles

- **Authors:** Lakshminarayanan, B., Pritzel, A., & Blundell, C.
- **Year:** 2017
- **Venue:** NeurIPS 2017
- **Title:** *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*

**Key Contribution:** Proposed training multiple neural networks in parallel on the same training dataset and using the disagreement among ensemble members to estimate predictive uncertainty.

**Method:** Each ensemble member is trained with random initialization, leading to diverse solutions in the loss landscape. Predictive uncertainty is captured through the variance of predictions across ensemble members. The method is simple, parallelizable, and requires no changes to the model architecture.

**Key Results:** Deep ensembles generate well-calibrated uncertainty estimates. As the number of models increases, entropy increases much faster than MC-dropout, indicating better handling of unseen test examples. Consistently outperforms MC dropout on out-of-distribution uncertainty benchmarks.

**Application to VLAs:** For driving VLAs, ensembles of trajectory predictions can capture multi-modal driving intentions (e.g., the ensemble might predict both "continue straight" and "turn left" at an intersection). High disagreement signals genuine ambiguity in the scene. **Challenge for VLAs:** Training multiple VLA models (typically >1B parameters each) is prohibitively expensive. Partial ensemble strategies (e.g., ensembling only the action head while sharing the vision-language backbone) offer a practical compromise.

**Relevance:** Gold standard for UQ but computationally challenging for large VLAs. Motivates our investigation of lightweight alternatives that approximate ensemble diversity.

---

### 2.3 Temperature Scaling

- **Authors:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q.
- **Year:** 2017
- **Venue:** ICML 2017
- **Title:** *On Calibration of Modern Neural Networks*

**Key Contribution:** Discovered that modern neural networks, unlike their predecessors, are poorly calibrated -- they tend to be significantly overconfident. Identified that depth, width, weight decay, and Batch Normalization all influence calibration.

**Method:** Temperature scaling is a single-parameter post-processing technique that divides logits by a learned scalar temperature T > 1 to soften the output distribution. It is a special case of Platt Scaling and is learned on a held-out validation set by minimizing the negative log-likelihood.

**Key Results:** Temperature scaling is surprisingly effective at restoring calibration on most datasets, often matching or outperforming more complex calibration methods. The Expected Calibration Error (ECE) metric is established as the standard measure.

**Application to VLAs:** For VLA models that generate discrete action tokens via softmax over a vocabulary, temperature scaling can be directly applied to the action token logits. **Challenge for VLAs:** VLAs generate sequences of tokens, and the optimal temperature may vary across the action sequence (early tokens vs. late tokens). This motivates position-dependent temperature scaling, which we explore in our work.

**Relevance:** The simplest and most practical post-hoc calibration method. Directly applicable to autoregressive VLAs as a baseline calibration technique.

---

### 2.4 Conformal Prediction

- **Authors:** Angelopoulos, A.N. & Bates, S.
- **Year:** 2023
- **Venue:** Foundations and Trends in Machine Learning, Vol. 16, No. 4, pp. 494-591
- **Title:** *Conformal Prediction: A Gentle Introduction*

**Key Contribution:** A comprehensive tutorial on conformal prediction -- a distribution-free, model-agnostic framework for constructing prediction sets with guaranteed coverage probability.

**Method:** Given a pre-trained model and a user-specified error rate alpha, conformal prediction produces prediction sets guaranteed to contain the ground truth with probability 1-alpha. The method requires only exchangeability of the calibration data (weaker than i.i.d. assumption). It works by computing nonconformity scores on a calibration set and using quantiles of these scores to construct prediction sets at test time.

**Key Results:** Provides finite-sample coverage guarantees without distributional or model assumptions. Extensions cover structured outputs, distribution shift, time-series, outlier detection, and models that abstain.

**Application to VLAs:** Conformal prediction can construct trajectory prediction regions -- sets of plausible future trajectories with guaranteed coverage. For driving, this means producing trajectory uncertainty tubes that contain the true trajectory with probability 1-alpha. **Challenge for VLAs:** The exchangeability assumption is violated in sequential driving data. Adaptive conformal prediction methods (e.g., ACI by Gibbs & Candes, 2021) can handle temporal distribution shifts but have not been applied to VLA trajectory generation.

**Relevance:** Provides the strongest theoretical guarantees for uncertainty quantification. A conformal prediction layer on VLA trajectories could provide formal safety guarantees -- a critical requirement for autonomous driving certification.

---

### 2.5 Evidential Deep Learning

- **Authors:** Sensoy, M., Kaplan, L., & Kandemir, M.
- **Year:** 2018
- **Venue:** NeurIPS 2018
- **Title:** *Evidential Deep Learning to Quantify Classification Uncertainty*

**Key Contribution:** Proposes explicit modeling of prediction uncertainty using the theory of subjective logic, by placing a Dirichlet distribution on class probabilities to treat neural network predictions as subjective opinions.

**Method:** The standard softmax output is replaced with parameters of a Dirichlet distribution. The network outputs "evidence" for each class via a ReLU activation (instead of softmax), and the Dirichlet parameters are set to 1 + evidence. This models second-order probabilities -- uncertainty about the uncertainty -- enabling clean separation of aleatoric and epistemic uncertainty in a single forward pass.

**Key Results:** Unprecedented success on detection of out-of-distribution queries and robustness against adversarial perturbations. Requires only a single forward pass (unlike MC Dropout or Ensembles).

**Application to VLAs:** For VLAs with discrete action spaces, each action token prediction can be modeled as a Dirichlet distribution over the vocabulary. Low total evidence indicates high epistemic uncertainty (unfamiliar input), while high entropy under the expected categorical distribution indicates aleatoric uncertainty (genuinely ambiguous scenario). **Challenge for VLAs:** Extending to continuous action spaces requires the Normal-Inverse-Gamma (NIG) variant; composing evidential predictions across autoregressive steps remains an open problem.

**Relevance:** Single-forward-pass UQ is highly attractive for real-time driving applications. The evidence accumulation metaphor aligns well with how driving decisions are informed by accumulating perceptual evidence over time.

---

### 2.6 Additional UQ Methods Relevant to Action Generation

**Semantic Entropy for Autoregressive Models:** Recent work (Kuhn et al., 2023, ICLR 2023) introduces semantic entropy, which groups semantically equivalent output sequences to avoid overestimating uncertainty due to linguistic variation. For VLAs, semantically equivalent trajectories (same driving intention, slightly different waypoints) should be grouped when computing uncertainty.

**Quantile-Based Uncertainty in Forecasting:** Autoregressive multi-quantile decoding (2024-2025) preserves forecast uncertainty while maintaining tractable inference, producing calibrated prediction intervals for time-series and trajectory prediction.

**Copula Conformal Prediction for Multi-Step Forecasting:** Stankeviciute et al. (ICLR 2024) propose CopulaCPTS, a copula-based conformal prediction approach for time-series forecasting that handles dependencies between prediction steps -- directly relevant to multi-waypoint trajectory prediction in driving.

---

## 3. Calibration of Large Language Models and Vision-Language Models

### 3.1 Are LLMs Well-Calibrated?

#### 3.1.1 Language Models (Mostly) Know What They Know

- **Authors:** Kadavath, S., Conerly, T., Askell, A., et al.
- **Year:** 2022
- **Venue:** arXiv 2207.05221
- **Title:** *Language Models (Mostly) Know What They Know*

**Key Contribution:** Studies whether language models can evaluate the validity of their own claims and predict which questions they will be able to answer correctly.

**Key Findings:** (1) Larger models are well-calibrated on diverse multiple choice and true/false questions when provided in the right format. (2) Calibration is highly sensitive to input formatting -- lettered options (A, B, C, D) with single-letter output produce the best calibration. (3) P(True) -- asking models to evaluate the probability that their sampled answer is correct -- shows encouraging performance and scaling. (4) P(IK) -- the probability that "I know" the answer -- can be trained and shows strong calibration properties.

**Relevance:** Demonstrates that self-evaluation capabilities exist in large models and scale with model size, suggesting VLA models may also possess latent self-evaluation capabilities that can be elicited for uncertainty estimation.

---

#### 3.1.2 Just Ask for Calibration

- **Authors:** Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., & Manning, C.
- **Year:** 2023
- **Venue:** EMNLP 2023
- **Title:** *Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback*

**Key Contribution:** Demonstrates that RLHF-tuned language models produce verbalized confidence scores that are better calibrated than their internal conditional probabilities.

**Key Findings:** For RLHF-LMs (ChatGPT, GPT-4, Claude), verbalized confidences emitted as output tokens are typically better-calibrated than the model's conditional probabilities on TriviaQA, SciQ, and TruthfulQA, often reducing ECE by a relative 50%. This highlights a novel capability: RLHF-LMs can directly communicate uncertainty in a quantitatively meaningful way.

**Relevance:** Suggests that verbalized confidence may be a viable uncertainty estimation strategy for VLAs, especially those built on RLHF-tuned language model backbones. Our work could explore whether driving VLAs can verbalize trajectory confidence.

---

#### 3.1.3 Can LLMs Express Their Uncertainty?

- **Authors:** Xiong, M., Hu, Z., et al.
- **Year:** 2024
- **Venue:** ICLR 2024
- **Title:** *Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs*

**Key Contribution:** Defines a systematic framework for LLM uncertainty elicitation with three components: prompting strategies, sampling methods, and aggregation techniques.

**Key Findings:** (1) LLMs, when verbalizing confidence, tend to be overconfident, potentially imitating human patterns. (2) Human-inspired prompts, multi-response consistency, and better aggregation strategies mitigate overconfidence. (3) Verbalized confidence is model-agnostic and low-overhead. (4) Fine-tuning widens the confidence score range, enabling more nuanced uncertainty expression.

**Relevance:** The overconfidence finding is particularly concerning for safety-critical driving applications. If VLAs inherit LLM overconfidence, they may express unwarranted certainty in dangerous situations. Our work specifically measures and addresses this concern.

---

### 3.2 How Does Calibration Change with Model Size?

#### 3.2.1 The Dunning-Kruger Effect in Large Language Models

- **Authors:** Not specified
- **Year:** 2026
- **Venue:** arXiv 2603.09985
- **Title:** *The Dunning-Kruger Effect in Large Language Models: An Empirical Study of Confidence Calibration*

**Key Contribution:** Reveals systematic overconfidence patterns in LLMs that parallel the Dunning-Kruger effect -- models are most overconfident on topics where they perform worst.

**Key Findings:** Larger models show improved calibration on factual questions but exhibit new forms of miscalibration. Extended thinking (chain-of-thought reasoning) may improve confidence calibration. Models fail to achieve calibration even when directly instructed to do so.

**Relevance:** Suggests that simply scaling VLA models will not automatically solve calibration problems. Domain-specific calibration interventions are necessary, particularly for safety-critical driving tasks.

---

#### 3.2.2 Survey of Confidence Estimation and Calibration in LLMs

- **Authors:** Geng, J., et al.
- **Year:** 2024
- **Venue:** NAACL 2024
- **Title:** *A Survey of Confidence Estimation and Calibration in Large Language Models*

**Key Contribution:** Comprehensive survey covering the landscape of LLM confidence estimation methods.

**Key Findings:** Methods fall into two categories: (1) **Logit-based methods** that derive confidence from token probability distributions, using entropy, perplexity, or trained classifiers on hidden states; (2) **Verbalized methods** that prompt models to output confidence scores as text tokens. Logit-based methods require white-box access but are more stable; verbalized methods are model-agnostic but suffer from systematic overconfidence. Post-hoc calibration (temperature scaling, Platt scaling) can improve both approaches.

**Relevance:** Provides the methodological toolkit for studying VLA calibration. Our work applies both logit-based and verbalized confidence approaches to driving VLAs.

---

### 3.3 Verbalized Confidence vs. Logit-Based Confidence

#### 3.3.1 On Verbalized Confidence Scores for LLMs

- **Authors:** Not specified
- **Year:** 2025
- **Venue:** ICLR 2025
- **Title:** *On Verbalized Confidence Scores for LLMs*

**Key Findings:** In zero-shot settings, LLMs predominantly assign high confidence scores, reflecting overconfidence attributed to supervised pretraining that favors confident expressions. After fine-tuning, confidence scores span a wider range, including lower values, indicating more nuanced uncertainty expression. Token-based scoring substantially outperforms self-verbalized confidence for precision-sensitive deployments.

---

#### 3.3.2 Do LLMs Estimate Uncertainty Well?

- **Authors:** Not specified
- **Year:** 2025
- **Venue:** ICLR 2025
- **Title:** *Do LLMs Estimate Uncertainty Well in Instruction Following?*

**Key Findings:** Better-performing LLMs show more aligned overall confidence levels, but even the most accurate models still show minimal variation in confidence between correct and incorrect predictions. This "flat confidence" phenomenon is particularly dangerous for safety-critical applications where discriminating between certain and uncertain predictions is essential.

---

### 3.4 VLM-Specific Calibration

#### 3.4.1 Object-Level Verbalized Confidence Calibration in VLMs

- **Authors:** Not specified
- **Year:** 2025
- **Venue:** arXiv 2504.14848
- **Title:** *Object-Level Verbalized Confidence Calibration in Vision-Language Models via Semantic Perturbation*

**Key Contribution:** Introduces CSP (Calibration through Semantic Perturbation), a framework for training VLMs for better object-level calibration with verbalized confidence.

**Method:** Creates a perturbed dataset that modifies key visual elements based on different confidence levels, allowing the model to learn explicit mappings between visual uncertainties and verbalized confidence.

**Relevance:** Directly applicable to driving VLAs where object-level confidence (e.g., confidence in pedestrian detection) should inform planning uncertainty.

---

#### 3.4.2 Confidence Calibration in Contrastive VLMs

- **Authors:** Not specified
- **Year:** 2025
- **Venue:** Springer
- **Title:** *Confidence Calibration in Contrastive Vision-Language Models*

**Key Contribution:** Studies calibration in zero-shot and fine-tuned contrastive VLMs (e.g., CLIP), finding that confidence calibration can significantly undermine reliability.

**Relevance:** Contrastive VLMs serve as the visual backbone for many driving VLAs; backbone miscalibration propagates to downstream action predictions.

---

## 4. VLA Calibration (The Gap)

### 4.1 The Only Known VLA Calibration Study

#### Zollo et al. (2025): Confidence Calibration in Vision-Language-Action Models

- **Authors:** Zollo, T., et al.
- **Year:** 2025
- **Venue:** arXiv 2507.17383
- **Title:** *Confidence Calibration in Vision-Language-Action Models*

**Key Contribution:** The first (and to date, only) systematic study of confidence calibration in VLA foundation models.

**Experimental Setup:** Uses models derived from OpenVLA, fine-tuned on three task suites from the LIBERO robotics benchmark: Spatial, Object, and Goal. LIBERO is a simulation environment for language-conditioned robot manipulation tasks inspired by human activities.

**Key Findings:**
1. **VLAs are miscalibrated.** The models exhibit systematic over-confidence and under-confidence across different action dimensions and task stages.
2. **Calibration evolves over the task horizon.** Early actions tend to be differently calibrated than later actions within the same task episode.
3. **Dimension-specific miscalibration.** Different action dimensions (e.g., x-translation, y-translation, rotation, gripper) exhibit distinct calibration profiles, with some dimensions systematically overconfident and others underconfident.

**Proposed Solutions:**
1. **Prompt Ensembles:** A lightweight, Bayesian-style method that averages a VLA's confidence across multiple semantically equivalent rephrasings of the instruction. Consistently improves calibration, cutting Expected Calibration Error (ECE) by more than 20% on average.
2. **Action-wise Platt Scaling:** Independently recalibrates each action dimension, producing more reliable per-dimension confidence estimates.

**Limitations of This Study:**
- **Robotics-only evaluation.** All experiments are conducted on tabletop manipulation tasks (LIBERO), not driving.
- **Limited action space.** Manipulation actions (6-DoF + gripper) differ fundamentally from driving actions (trajectory waypoints, speed, steering).
- **Static environments.** Manipulation tasks have static or slowly changing environments, unlike the highly dynamic driving domain.
- **No safety analysis.** The study does not connect calibration quality to downstream safety outcomes (collision avoidance, human handoff, etc.).
- **Single model family.** Only OpenVLA variants are studied; calibration properties may vary across architectures.

---

### 4.2 What Is Missing: The Driving Domain Gap

The following aspects of VLA calibration remain completely unexplored:

| Gap | Description | Why It Matters |
|-----|-------------|----------------|
| **Driving-domain calibration** | No study has measured VLA calibration on autonomous driving benchmarks (NAVSIM, nuScenes, Bench2Drive, Waymo) | Driving has fundamentally different safety requirements than manipulation |
| **Multi-waypoint trajectory calibration** | Calibration over sequences of trajectory waypoints (4-8 seconds into the future) | Uncertainty compounds over prediction horizons; early calibration does not guarantee late calibration |
| **Speed-dependent calibration** | Calibration at different driving speeds and in different driving contexts | High-speed calibration failures are far more dangerous than low-speed ones |
| **Cross-architecture comparison** | Calibration comparison across DriveVLM, AutoVLA, SimLingo, LatentVLA, etc. | Calibration may be architecture-dependent |
| **Safety-calibration connection** | Formal connection between calibration quality and safety metrics (collision rate, TTC, comfort) | Calibration is only useful if it improves safety outcomes |
| **Calibration under distribution shift** | How VLA calibration degrades with novel weather, road conditions, or geographic regions | Autonomous vehicles encounter distributional novelty daily |
| **Uncertainty-triggered intervention** | Using calibrated uncertainty to trigger human handoff or conservative driving | The ultimate purpose of calibration in safety-critical systems |

**This gap defines the primary contribution of our work.** We propose the first comprehensive study of VLA calibration in the autonomous driving domain, connecting calibration quality to safety outcomes.

---

## 5. Safety-Critical Decision Making Under Uncertainty

### 5.1 Selective Prediction and Abstention

#### 5.1.1 Know Your Limits: A Survey of Abstention in Large Language Models

- **Authors:** Wen, B., et al.
- **Year:** 2025
- **Venue:** TACL 2025
- **Title:** *Know Your Limits: A Survey of Abstention in Large Language Models*

**Key Contribution:** Comprehensive survey on the ability of language models to abstain from answering uncertain queries.

**Key Findings:** Abstention is a critical mechanism for reducing hallucination effects and improving safety in high-risk applications. Three categories of abstention mechanisms: (1) threshold-based on confidence scores, (2) self-reflection-based where models assess their own competence, and (3) training-based where abstention is learned during fine-tuning.

**Relevance:** For driving VLAs, abstention maps directly to the "request human takeover" action. Our work proposes uncertainty-triggered abstention thresholds calibrated to minimize collision risk.

---

#### 5.1.2 SelectLLM: Calibrating LLMs for Selective Prediction

- **Authors:** Not specified
- **Year:** 2025
- **Venue:** OpenReview
- **Title:** *SelectLLM: Calibrating LLMs for Selective Prediction: Balancing Coverage and Risk*

**Key Contribution:** An end-to-end method integrating selective prediction into fine-tuning, enhancing LLMs' ability to recognize and express uncertainty.

**Key Results:** Significantly outperforms standard baselines on TriviaQA, CommonsenseQA, and MedConceptsQA, improving abstention behavior while maintaining high accuracy.

**Relevance:** The coverage-risk tradeoff framework directly applies to autonomous driving: higher coverage means the system drives more (fewer interventions), but higher risk means more potential accidents. Our work formalizes this tradeoff for driving VLAs.

---

#### 5.1.3 Selective "Selective Prediction" for Vision-Language Reasoning

- **Authors:** Not specified
- **Year:** 2024
- **Venue:** ACL Findings 2024 / arXiv 2402.15610
- **Title:** *Selective "Selective Prediction": Reducing Unnecessary Abstention in Vision-Language Reasoning*

**Key Contribution:** Addresses the problem of unnecessary abstention in vision-language models, where models abstain even when they would have been correct.

**Relevance:** In driving, unnecessary abstention (requesting human takeover when the system would have driven safely) degrades user trust and system utility. Balancing false positive (unnecessary) and false negative (dangerous) abstention is critical.

---

### 5.2 Safe Reinforcement Learning Under Uncertainty

#### 5.2.1 Game-Theoretic Risk-Shaped RL for Autonomous Driving

- **Authors:** Not specified
- **Year:** 2025
- **Venue:** arXiv 2510.10960
- **Title:** *Game-Theoretic Risk-Shaped Reinforcement Learning for Safe Autonomous Driving*

**Key Contribution:** A safe RL framework combining game-theoretic world modeling, uncertainty-aware reachability modeling, and long-term risk constraints.

**Key Innovation:** Explicitly captures both epistemic and aleatoric uncertainty to guide constrained policy optimization with multi-level reasoning and adaptive horizon adjustment.

**Relevance:** Demonstrates how uncertainty can be formally incorporated into RL-based driving policies. Our work extends this to VLA-based policies where uncertainty emerges from the language-vision-action pipeline.

---

#### 5.2.2 Uncertainty-Aware Safe-Evolving RL for Highway Driving

- **Authors:** Not specified
- **Year:** 2025
- **Venue:** Engineering Applications of AI
- **Title:** *An Uncertainty-Aware Safe-Evolving Reinforcement Learning Algorithm for Decision-Making and Control in Highway Autonomous Driving*

**Key Contribution:** The UA-MBRL method evaluates environmental model uncertainty by constructing an action-conditioned ensemble model and dynamically adjusts the prediction horizon based on an adaptive truncation mechanism.

**Relevance:** The concept of dynamically adjusting planning horizon based on uncertainty is directly applicable to driving VLAs -- when uncertain, plan only for the near future; when confident, plan further ahead.

---

### 5.3 Conformal Prediction for Robotics and Driving Safety

#### 5.3.1 Formal Verification and Control with Conformal Prediction

- **Authors:** Not specified
- **Year:** 2024
- **Venue:** IEEE Transactions on Automatic Control
- **Title:** *Formal Verification and Control With Conformal Prediction: Practical Safety Guarantees For Autonomous Systems*

**Key Contribution:** Addresses the fragility of learning-enabled components in safety-critical autonomous systems through formal verification combined with conformal prediction.

**Key Innovation:** Provides provable false negative rates for warning systems -- among situations where an alert should be issued, fewer than epsilon cases occur without an alert, while achieving low false positive rates.

**Relevance:** Provides the theoretical foundation for constructing formally verified safety guarantees around VLA trajectory predictions using conformal prediction.

---

#### 5.3.2 Copula Conformal Prediction for Multi-Step Time Series

- **Authors:** Stankeviciute, K., et al.
- **Year:** 2024
- **Venue:** ICLR 2024
- **Title:** *Copula Conformal Prediction for Multi-Step Time Series Forecasting*

**Key Contribution:** A copula-based approach for multi-step time series forecasting that handles dependencies between prediction steps.

**Key Results:** Validated on the Argoverse autonomous vehicle motion forecasting dataset for 3-second trajectory prediction.

**Relevance:** Multi-step trajectory prediction in driving requires handling temporal dependencies between waypoints. Copula conformal prediction provides a principled way to construct joint prediction regions with coverage guarantees.

---

#### 5.3.3 Safe Control for Learning-Enabled Autonomy with Conformal Prediction

- **Authors:** Not specified
- **Year:** 2024
- **Venue:** UCSD Seminar
- **Title:** *Safe Control for Learning-Enabled Autonomy with Conformal Prediction*

**Key Contribution:** Develops methods using conformal prediction for safe control of autonomous systems operating in dynamic environments with uncontrollable agents.

**Relevance:** Directly addresses the challenge of providing safety guarantees when VLA-predicted trajectories interact with other road users whose behavior is uncertain.

---

### 5.4 Uncertainty-Driven Reliability

#### Uncertainty-Driven Reliability: Selective Prediction and Safety

- **Authors:** Not specified
- **Year:** 2025
- **Venue:** arXiv 2508.07556
- **Title:** *Uncertainty-Driven Reliability: Selective Prediction and Safety*

**Key Contribution:** Combines uncertainty quantification with selective prediction to improve reliability in safety-critical applications.

**Relevance:** Provides a framework for connecting calibrated uncertainty estimates to selective prediction decisions in safety-critical domains like autonomous driving.

---

### 5.5 Risk Assessment in Autonomous Driving

#### Comprehensive Survey of Risk Assessment

- **Authors:** Not specified
- **Year:** 2025
- **Venue:** Autonomous Intelligent Systems (Springer)
- **Title:** *Risk Assessment in Autonomous Driving: A Comprehensive Survey of Risk Sources, Methodologies, and System Architectures*

**Key Findings:** Three major challenges: (1) absence of a unified risk metric system, (2) limited consideration of system capability in risk estimation, and (3) scarcity of high-quality labeled risk data.

**Relevance:** Motivates the need for VLA-internal risk assessment via calibrated uncertainty, addressing the gap of considering the AI system's own uncertainty as a risk factor.

---

### 5.6 Uncertainty Quantification for SOTIF in Autonomous Driving

#### Comprehensive Survey on UQ for SOTIF

- **Authors:** Not specified
- **Year:** 2025
- **Venue:** ResearchGate
- **Title:** *Uncertainty Quantification for Safety of the Intended Functionality of Autonomous Driving: A Comprehensive Survey*

**Key Contribution:** Reviews uncertainty quantification specifically in the context of SOTIF (Safety of the Intended Functionality, ISO 21448), the automotive industry standard for ensuring that AI-based driving functions operate safely even in their intended operating conditions.

**Relevance:** Connects our VLA calibration work to automotive industry safety standards, strengthening the practical impact of our research.

---

## 6. Benchmarks and Datasets

### 6.1 NAVSIM

- **Authors:** Dauner, D., Hallgarten, M., et al.
- **Year:** 2024
- **Venue:** NeurIPS 2024 Datasets and Benchmarks Track / CoRL 2025
- **Title:** *NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking*

**Description:** A middle ground between open-loop evaluation and full simulation, using large real-world datasets with non-reactive simulation to enable scalable benchmarking.

**Key Metrics:**
- **PDMS (Predictive Driver Model Score):** Composite metric integrating safety, comfort, and progress
- **NC (No Collision):** Binary collision detection
- **DAC (Drivable Area Compliance):** Staying within road boundaries
- **EP (Ego Progress):** Forward progress along route
- **TTC (Time-to-Collision):** Safety margin metric
- **Comfort:** Acceleration/jerk bounds

**Evaluation Protocol:** 4-second trajectory prediction with LQR-controlled rollouts. Non-reactive simulation where the evaluated policy and environment do not influence each other.

**Recent Developments:** NAVSIM v2 (2025) introduces pseudo-simulation achieving strong correlation with traditional closed-loop simulations while requiring 6x less compute. CVPR 2024 competition: 143 teams, 463 entries. Key finding: simpler well-optimized architectures can match larger models; ensemble/sampling-based policy selection emerged as competitive.

**Relevance:** Primary benchmark for evaluating driving VLAs. Our calibration study uses NAVSIM to measure how VLA confidence correlates with PDMS, NC, and TTC metrics. The finding that ensembles are competitive directly supports our uncertainty quantification approach.

---

### 6.2 nuScenes

- **Authors:** Caesar, H., Bankiti, V., et al.
- **Year:** 2020
- **Venue:** CVPR 2020
- **Title:** *nuScenes: A Multimodal Dataset for Autonomous Driving*

**Description:** One of the most widely-used autonomous driving datasets, containing 1000 driving scenes in Boston and Singapore with full sensor suite (6 cameras, 1 LiDAR, 5 radars, GPS, IMU).

**Key Metrics:**
- **L2 Error:** Average displacement error of predicted trajectory
- **Collision Rate:** Percentage of scenarios where predicted trajectory collides with other agents
- **Open-loop evaluation protocol** (log-replay)

**Limitations:** Open-loop evaluation cannot fully reflect driving performance, as recently acknowledged by the community. L2 error penalizes novel but safe trajectories (e.g., taking a different but equally valid lane).

**Relevance:** Standard evaluation dataset for VLA models (DriveVLM, OpenDriveVLA, LatentVLA all report nuScenes results). Our work uses nuScenes to study calibration under diverse urban driving conditions (Boston vs. Singapore present natural distribution shift).

---

### 6.3 CARLA

- **Authors:** Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., & Koltun, V.
- **Year:** 2017
- **Venue:** CoRL 2017
- **Title:** *CARLA: An Open Urban Driving Simulator*

**Description:** An open-source simulator for autonomous driving research supporting flexible specification of sensor suites, environmental conditions, and dynamic agents.

**Key Metrics:**
- **Driving Score (DS):** Route completion * infraction penalty
- **Route Completion (RC):** Percentage of route completed
- **Infraction Score:** Penalty for collisions, traffic violations, etc.
- **CARLA Leaderboard:** Online evaluation platform

**Data for Bench2Drive:** Bench2Drive uses CARLA v2 to collect 2 million fully annotated frames from 10,000 short clips under 44 interactive scenarios, 23 weather conditions, and 12 towns.

**Relevance:** Closed-loop evaluation in CARLA enables measuring how uncertainty-aware interventions (e.g., stopping when uncertain) affect actual driving safety metrics, not just trajectory prediction accuracy.

---

### 6.4 Bench2Drive

- **Authors:** Jia, X., et al.
- **Year:** 2024
- **Venue:** NeurIPS 2024 Datasets and Benchmarks Track
- **Title:** *Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-to-End Autonomous Driving*

**Description:** The first benchmark for evaluating end-to-end autonomous driving systems' multiple abilities in a closed-loop manner.

**Key Features:**
- 2 million fully annotated frames from 10,000 short clips
- 44 interactive scenarios, 23 weather conditions, 12 towns (CARLA v2)
- Each clip approximately 150 meters with a single specific scenario
- Dev10 tiny validation set for rapid development (proposed in DriveTransformer, ICLR 2025)

**Evaluation Protocol:** E2E-AD models must pass scenarios across different locations and weather conditions, totaling 220 routes. Comprehensive and disentangled assessment of driving capability under different situations.

**Key Results:** SimLingo achieves leading driving score of 85.94; Think2Drive (world model RL expert) is the only expert able to solve all 44 scenarios.

**Relevance:** The diverse scenario coverage (44 scenarios x 23 weathers) creates natural variation in difficulty, making it ideal for studying how VLA uncertainty correlates with scenario difficulty. We propose using Bench2Drive to measure whether calibrated uncertainty predicts scenario failure before it occurs.

---

### 6.5 Waymo Open Dataset

- **Authors:** Sun, P., et al.
- **Year:** 2020/2025
- **Venue:** CVPR 2020 (original); continuously updated
- **Title:** *Waymo Open Dataset*

**Description:** Three interconnected datasets: (1) Perception Dataset with 2,030 segments, (2) Motion Dataset with 103,354 segments and 3D maps, (3) End-to-End Driving Dataset with 5,000 segments including camera images and routing instructions.

**Key Metrics:**
- **Rater Feedback Score (RFS):** Human-centric evaluation metric aligning predicted trajectories with safe, legal, and efficient behaviors under critical circumstances, based on rater-annotated trajectory preference labels
- **ADE (Average Displacement Error):** Standard trajectory accuracy metric
- **Semantic & Panoptic Segmentation:** PQ, mIoU, STQ, wSTQ metrics

**Recent Developments (2025):**
- **WOD-E2E:** 4,021 high-difficulty, long-tail scenario segments (approx 12 hours), focusing on rare real-world situations occurring with less than 0.03% frequency
- **Risk-Based Filtering:** Annotates high-risk first- and second-order driving situations through probabilistic collision-risk models
- **RFS Spotlight:** Focus on the most challenging scenarios

**Relevance:** The Waymo RFS metric is unique in incorporating human judgment of driving quality, making it the most relevant metric for evaluating whether uncertainty-aware driving policies produce human-acceptable behavior. WOD-E2E's long-tail scenarios are ideal for studying calibration under distributional novelty.

---

### Summary Table: Benchmarks and Datasets

| Benchmark | Type | Eval Mode | Key Safety Metrics | Scale | VLA Papers Using It |
|-----------|------|-----------|-------------------|-------|-------------------|
| NAVSIM | Real data + pseudo-sim | Open-loop (pseudo-closed) | PDMS, NC, TTC, DAC | Large | ELF-VLA, LatentVLA, DriveVLA-W0, FutureSightDrive |
| nuScenes | Real data | Open-loop | L2 error, collision rate | 1000 scenes | DriveVLM, OpenDriveVLA, LatentVLA, DriveLM |
| CARLA | Simulation | Closed-loop | DS, RC, infraction score | Unlimited | SimLingo, CoC-VLA, DriveLM |
| Bench2Drive | CARLA v2 simulation | Closed-loop | DS across 44 scenarios | 10K clips | SimLingo |
| Waymo | Real data | Open-loop | RFS, ADE | 100K+ segments | AutoVLA |

---

## Summary of Key Research Gaps

Based on this comprehensive review, we identify the following critical gaps that motivate our work:

1. **No driving-domain VLA calibration study exists.** Zollo et al. (2025) studied VLA calibration only on robotic manipulation (LIBERO). No work has measured whether driving VLAs are calibrated.

2. **Uncertainty quantification is absent from all driving VLAs.** All 13 surveyed driving VLA models lack explicit uncertainty estimation mechanisms.

3. **No formal connection between VLA confidence and driving safety metrics.** While UQ methods exist (MC Dropout, Ensembles, Conformal Prediction) and driving safety metrics exist (PDMS, TTC, collision rate), no work connects VLA-generated confidence to these safety outcomes.

4. **Selective prediction/abstention is unexplored for driving VLAs.** While LLM abstention is an active research area, no work has applied uncertainty-triggered human handoff or conservative driving to VLA-based systems.

5. **Calibration under driving-specific distribution shifts is unstudied.** Weather changes, novel road geometries, and geographic transfer (e.g., Boston to Singapore) create distribution shifts that may degrade VLA calibration in safety-critical ways.

6. **No benchmark protocol exists for evaluating VLA calibration in driving.** Current benchmarks (NAVSIM, nuScenes, Bench2Drive) measure trajectory quality but not confidence quality.

Our proposed work addresses gaps 1-6 by introducing the first comprehensive framework for uncertainty-aware Vision-Language-Action models in autonomous driving, connecting calibration quality to safety outcomes, and establishing evaluation protocols for future research.

---

## References

### Section 1: Vision-Language-Action Models for Driving
- Tian, X., Gu, J., et al. (2024). DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models. *CoRL 2024*. arXiv:2402.12289.
- UCLA Mobility Group. (2025). AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning. *NeurIPS 2025*. arXiv:2506.13757.
- DriveVLA Team. (2025). OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Model. *AAAI 2026*. arXiv:2503.23463.
- Wang, Luo, et al. / NVIDIA. (2025). Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail. arXiv:2511.00088.
- ELF-VLA Authors. (2025). Unleashing VLA Potentials in Autonomous Driving via Explicit Learning from Failures. arXiv:2603.01063.
- XPENG & Peking University. (2025). FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning. *AAAI 2026*. arXiv:2507.23318.
- Sima, C., et al. / OpenDriveLab. (2024). DriveLM: Driving with Graph Visual Question Answering. *ECCV 2024 (Oral)*. arXiv:2312.14150.
- Renz, K., et al. (2025). SimLingo: Vision-Only Closed-Loop Autonomous Driving with Language-Action Alignment. *CVPR 2025 (Spotlight)*. arXiv:2503.09594.
- LatentVLA Authors. (2025). LatentVLA: Efficient Vision-Language Models for Autonomous Driving via Latent Action Prediction. arXiv:2601.05611.
- InsightDrive Authors. (2025). InsightDrive: Insight Scene Representation for End-to-End Autonomous Driving. arXiv:2503.13047.
- FutureSightDrive Authors. (2025). FutureSightDrive: Thinking Visually with Spatio-Temporal CoT for Autonomous Driving. *NeurIPS 2025*. arXiv:2505.17685.
- CoC-VLA Authors. (2025). CoC-VLA: Delving into Adversarial Domain Transfer for Explainable Autonomous Driving via Chain-of-Causality Visual-Language-Action Model. arXiv:2511.19914.
- BraveGroup. (2025). DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving. arXiv:2510.12796.
- Jiang et al. (2025). A Survey on Vision-Language-Action Models for Autonomous Driving. *ICCV 2025 Workshop*. arXiv:2506.24044.
- World Bench et al. (2025). Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future. arXiv:2512.16760.

### Section 2: Uncertainty Quantification in Deep Learning
- Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *ICML 2016*. arXiv:1506.02142.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS 2017*. arXiv:1612.01474.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*. arXiv:1706.04599.
- Angelopoulos, A.N. & Bates, S. (2023). Conformal Prediction: A Gentle Introduction. *Foundations and Trends in Machine Learning*, 16(4), 494-591. arXiv:2107.07511.
- Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential Deep Learning to Quantify Classification Uncertainty. *NeurIPS 2018*. arXiv:1806.01768.
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs. *ICLR 2023*.
- Stankeviciute, K., et al. (2024). Copula Conformal Prediction for Multi-Step Time Series Forecasting. *ICLR 2024*.

### Section 3: Calibration of LLMs and VLMs
- Kadavath, S., Conerly, T., Askell, A., et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.
- Tian, K., Mitchell, E., Zhou, A., et al. (2023). Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback. *EMNLP 2023*. arXiv:2305.14975.
- Xiong, M., Hu, Z., et al. (2024). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. *ICLR 2024*. arXiv:2306.13063.
- Geng, J., et al. (2024). A Survey of Confidence Estimation and Calibration in Large Language Models. *NAACL 2024*.
- Various authors. (2026). The Dunning-Kruger Effect in Large Language Models: An Empirical Study of Confidence Calibration. arXiv:2603.09985.
- Various authors. (2025). On Verbalized Confidence Scores for LLMs. *ICLR 2025*.
- Various authors. (2025). Do LLMs Estimate Uncertainty Well in Instruction Following? *ICLR 2025*.
- Various authors. (2025). Object-Level Verbalized Confidence Calibration in Vision-Language Models via Semantic Perturbation. arXiv:2504.14848.
- Various authors. (2025). Confidence Calibration in Contrastive Vision-Language Models. *Springer*.

### Section 4: VLA Calibration
- Zollo, T., et al. (2025). Confidence Calibration in Vision-Language-Action Models. arXiv:2507.17383.

### Section 5: Safety-Critical Decision Making Under Uncertainty
- Wen, B., et al. (2025). Know Your Limits: A Survey of Abstention in Large Language Models. *TACL 2025*.
- SelectLLM Authors. (2025). SelectLLM: Calibrating LLMs for Selective Prediction: Balancing Coverage and Risk. *OpenReview*.
- Various authors. (2024). Selective "Selective Prediction": Reducing Unnecessary Abstention in Vision-Language Reasoning. *ACL Findings 2024*. arXiv:2402.15610.
- GTR2L Authors. (2025). Game-Theoretic Risk-Shaped Reinforcement Learning for Safe Autonomous Driving. arXiv:2510.10960.
- UA-MBRL Authors. (2025). An Uncertainty-Aware Safe-Evolving Reinforcement Learning Algorithm for Decision-Making and Control in Highway Autonomous Driving. *Engineering Applications of AI*.
- Various authors. (2024). Formal Verification and Control With Conformal Prediction: Practical Safety Guarantees For Autonomous Systems. *IEEE Transactions on Automatic Control*.
- Various authors. (2025). Uncertainty Quantification for Safety of the Intended Functionality of Autonomous Driving: A Comprehensive Survey.
- Various authors. (2025). Risk Assessment in Autonomous Driving: A Comprehensive Survey. *Autonomous Intelligent Systems (Springer)*.
- Various authors. (2025). Uncertainty Quantification for Safe and Reliable Autonomous Vehicles: A Review. *IEEE TITS*.

### Section 6: Benchmarks and Datasets
- Dauner, D., Hallgarten, M., et al. (2024). NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking. *NeurIPS 2024*. arXiv:2406.15349.
- Caesar, H., Bankiti, V., et al. (2020). nuScenes: A Multimodal Dataset for Autonomous Driving. *CVPR 2020*.
- Dosovitskiy, A., Ros, G., Codevilla, F., Lopez, A., & Koltun, V. (2017). CARLA: An Open Urban Driving Simulator. *CoRL 2017*.
- Jia, X., et al. (2024). Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-to-End Autonomous Driving. *NeurIPS 2024*. arXiv:2406.03877.
- Sun, P., et al. (2020). Waymo Open Dataset. *CVPR 2020*.
- WOD-E2E Authors. (2025). WOD-E2E: Waymo Open Dataset for End-to-End Driving in Challenging Long-tail Scenarios. arXiv:2510.26125.
