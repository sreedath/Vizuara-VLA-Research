"""
Large-Scale OOD Detection Evaluation for VLA Paper
Experiment: 200 clean + 200x4 corrupted images, proper metrics
"""
import os, json, time, datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Image generation - more realistic synthetic driving scenes
SIZE = (224, 224)  # OpenVLA's expected input size

def create_clean_scene(idx, variant=0):
    """Create more realistic driving scene with gradient sky, road texture, lane markings."""
    rng = np.random.default_rng(idx * 7919 + variant * 101)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    H, W = SIZE
    horizon = H // 2 + rng.integers(-20, 21)

    # Sky with gradient and clouds
    for row in range(horizon):
        t = row / horizon
        base = [int(135 - t*30), int(206 - t*20), int(235 - t*10)]
        noise = rng.integers(-8, 9, 3)
        img[row] = np.clip(np.array(base) + noise, 0, 255)

    # Road with perspective and texture
    for row in range(horizon, H):
        t = (row - horizon) / (H - horizon)
        base = [int(70 + t*20), int(70 + t*20), int(70 + t*20)]
        noise = rng.integers(-10, 11, 3)
        img[row] = np.clip(np.array(base) + noise, 0, 255)

    # Lane markings with perspective (narrower at top, wider at bottom)
    lane_width = max(2, int(5 * (1 + (horizon / H))))
    cx = W // 2
    for row in range(horizon, H, 20):
        t = (row - horizon) / (H - horizon)
        lw = max(1, int(lane_width * t))
        if (row // 20) % 2 == 0:
            img[row:min(row+10, H), max(0,cx-lw):min(W,cx+lw)] = [255, 255, 200]

    # Roadside elements (trees/buildings as vertical rectangles)
    if rng.random() > 0.3:
        bldg_h = rng.integers(H//4, H//2)
        bldg_w = rng.integers(20, 60)
        bldg_x = rng.integers(0, W//3)
        color = [int(c) for c in rng.integers(60, 180, 3)]
        img[horizon-bldg_h:horizon, bldg_x:bldg_x+bldg_w] = color

    return img

def apply_fog(img, severity):
    img_f = img.astype(float)
    return np.clip(img_f * (1 - 0.6*severity) + 255 * 0.6*severity, 0, 255).astype(np.uint8)

def apply_night(img, severity):
    return np.clip(img.astype(float) * max(0.01, 1 - 0.95*severity), 0, 255).astype(np.uint8)

def apply_noise(img, severity):
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.3*severity*255, img.shape)
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

def apply_blur(img, severity):
    from PIL import ImageFilter
    pil_img = Image.fromarray(img)
    radius = max(1, int(10 * severity))
    return np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=radius)))

# Load OpenVLA-7B
print("Loading OpenVLA-7B...", flush=True)
t0 = time.time()
from transformers import AutoProcessor, AutoModelForVision2Seq
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", trust_remote_code=True,
    torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

def get_embedding(img_array, prompt="In: What action should the robot take to drive safely?\nOut:"):
    """Extract layer-3 hidden state embedding."""
    img = Image.fromarray(img_array)
    inputs = processor(prompt, img, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    h3 = outputs.hidden_states[3][0, -1, :].float().cpu().numpy()
    return h3

def get_output_confidence(img_array, prompt="In: What action should the robot take to drive safely?\nOut:"):
    """Get softmax confidence of predicted action token."""
    img = Image.fromarray(img_array)
    inputs = processor(prompt, img, return_tensors="pt").to("cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    return probs.max().item(), probs.cpu().numpy()

def cosine_dist(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

# Generate embeddings for all images
print("\n=== Generating embeddings ===", flush=True)
N_CLEAN = 200
N_CORRUPTED = 200
SEVERITY = 1.0
PROMPT = "In: What action should the robot take to drive safely?\nOut:"

# Clean images - use first 50 for calibration, rest for evaluation
print(f"Processing {N_CLEAN} clean images...", flush=True)
clean_embeddings = []
clean_confidences = []
for i in range(N_CLEAN):
    if i % 20 == 0: print(f"  Clean {i}/{N_CLEAN}", flush=True)
    img = create_clean_scene(i, variant=i % 5)
    emb = get_embedding(img, PROMPT)
    clean_embeddings.append(emb)
    conf, _ = get_output_confidence(img, PROMPT)
    clean_confidences.append(conf)
clean_embeddings = np.array(clean_embeddings)

# Calibration: first 50 clean images
N_CAL = 50
cal_embeddings = clean_embeddings[:N_CAL]
centroid = cal_embeddings.mean(axis=0)
cal_dists = np.array([cosine_dist(e, centroid) for e in cal_embeddings])
threshold = cal_dists.mean() + 3 * cal_dists.std()
print(f"Threshold (3sigma): {threshold:.6e}", flush=True)

# Test clean (last 150)
test_clean_embeddings = clean_embeddings[N_CAL:]
test_clean_dists = np.array([cosine_dist(e, centroid) for e in test_clean_embeddings])
fpr_clean = (test_clean_dists > threshold).mean()
print(f"FPR on clean (n={len(test_clean_dists)}): {fpr_clean:.4f}", flush=True)

# Corrupted images
corruption_results = {}
for name, apply_fn in [("fog", apply_fog), ("night", apply_night), ("noise", apply_noise), ("blur", apply_blur)]:
    print(f"\nProcessing {N_CORRUPTED} {name} images...", flush=True)
    ood_dists = []
    ood_confs = []
    for i in range(N_CORRUPTED):
        if i % 20 == 0: print(f"  {name} {i}/{N_CORRUPTED}", flush=True)
        img = create_clean_scene(i + 1000, variant=(i % 5) + 5)
        corrupted = apply_fn(img, SEVERITY)
        emb = get_embedding(corrupted, PROMPT)
        d = cosine_dist(emb, centroid)
        ood_dists.append(d)
        conf, _ = get_output_confidence(corrupted, PROMPT)
        ood_confs.append(conf)

    ood_dists = np.array(ood_dists)

    # AUROC
    y_true = np.concatenate([np.zeros(len(test_clean_dists)), np.ones(len(ood_dists))])
    y_score = np.concatenate([test_clean_dists, ood_dists])
    auroc = roc_auc_score(y_true, y_score)

    # FPR@95TPR
    tpr_threshold = np.percentile(ood_dists, 5)  # 95% of OOD detected
    fpr_at_95 = (test_clean_dists > tpr_threshold).mean()

    # Bootstrap CI
    boot_aurocs = []
    rng = np.random.default_rng(42)
    for _ in range(1000):
        idx_clean = rng.choice(len(test_clean_dists), len(test_clean_dists), replace=True)
        idx_ood = rng.choice(len(ood_dists), len(ood_dists), replace=True)
        y_b = np.concatenate([np.zeros(len(idx_clean)), np.ones(len(idx_ood))])
        s_b = np.concatenate([test_clean_dists[idx_clean], ood_dists[idx_ood]])
        boot_aurocs.append(roc_auc_score(y_b, s_b))

    # Cohen's d
    d_prime = (ood_dists.mean() - test_clean_dists.mean()) / np.sqrt(
        (test_clean_dists.std()**2 + ood_dists.std()**2) / 2 + 1e-10)

    corruption_results[name] = {
        "auroc": float(auroc),
        "auroc_ci_lower": float(np.percentile(boot_aurocs, 2.5)),
        "auroc_ci_upper": float(np.percentile(boot_aurocs, 97.5)),
        "fpr_at_95tpr": float(fpr_at_95),
        "d_prime": float(d_prime),
        "ood_mean_dist": float(ood_dists.mean()),
        "ood_std_dist": float(ood_dists.std()),
        "clean_mean_conf": float(np.mean(clean_confidences)),
        "ood_mean_conf": float(np.mean(ood_confs)),
        "n_clean": len(test_clean_dists),
        "n_ood": len(ood_dists),
    }
    print(f"  AUROC={auroc:.4f} [{np.percentile(boot_aurocs,2.5):.4f},{np.percentile(boot_aurocs,97.5):.4f}], FPR@95={fpr_at_95:.4f}, d'={d_prime:.2f}", flush=True)

# Feature Norm baseline
print("\nFeature Norm baseline...", flush=True)
clean_norms = np.linalg.norm(test_clean_embeddings, axis=1)
fn_results = {}
for name, apply_fn in [("fog", apply_fog), ("night", apply_night), ("noise", apply_noise), ("blur", apply_blur)]:
    ood_norms = []
    for i in range(N_CORRUPTED):
        img = create_clean_scene(i + 1000, variant=(i % 5) + 5)
        corrupted = apply_fn(img, SEVERITY)
        emb = get_embedding(corrupted, PROMPT)
        ood_norms.append(np.linalg.norm(emb))
    ood_norms = np.array(ood_norms)
    y_true = np.concatenate([np.zeros(len(clean_norms)), np.ones(len(ood_norms))])
    y_score = np.concatenate([clean_norms, ood_norms])
    try:
        auroc = roc_auc_score(y_true, y_score)
    except:
        auroc = roc_auc_score(y_true, -y_score)
    fn_results[name] = float(auroc)
    print(f"  FeatureNorm {name}: {auroc:.4f}", flush=True)

# MSP baseline
print("\nMSP baseline...", flush=True)
msp_results = {}
for name, apply_fn in [("fog", apply_fog), ("night", apply_night), ("noise", apply_noise), ("blur", apply_blur)]:
    ood_confs = []
    for i in range(N_CORRUPTED):
        img = create_clean_scene(i + 1000, variant=(i % 5) + 5)
        corrupted = apply_fn(img, SEVERITY)
        conf, _ = get_output_confidence(corrupted, PROMPT)
        ood_confs.append(conf)
    y_true = np.concatenate([np.zeros(len(clean_confidences[N_CAL:])), np.ones(len(ood_confs))])
    y_score = np.concatenate([-np.array(clean_confidences[N_CAL:]), -np.array(ood_confs)])
    try:
        auroc = roc_auc_score(y_true, y_score)
    except:
        auroc = 0.5
    msp_results[name] = float(auroc)
    print(f"  MSP {name}: {auroc:.4f}", flush=True)

# Save results
results = {
    "experiment": "large_scale_ood_detection",
    "timestamp": datetime.datetime.now().isoformat(),
    "n_calibration": N_CAL,
    "n_test_clean": len(test_clean_dists),
    "n_test_ood_per_type": N_CORRUPTED,
    "severity": SEVERITY,
    "threshold_3sigma": float(threshold),
    "fpr_on_clean": float(fpr_clean),
    "cosine_distance": corruption_results,
    "feature_norm": fn_results,
    "msp": msp_results,
    "clean_dist_stats": {
        "mean": float(test_clean_dists.mean()),
        "std": float(test_clean_dists.std()),
        "max": float(test_clean_dists.max()),
    }
}

fname = f"/workspace/Vizuara-VLA-Research/experiments/large_scale_ood_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(fname, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {fname}", flush=True)
print("\n=== FINAL RESULTS ===")
for name, r in corruption_results.items():
    print(f"{name}: AUROC={r['auroc']:.4f} CI=[{r['auroc_ci_lower']:.4f},{r['auroc_ci_upper']:.4f}] FPR@95={r['fpr_at_95tpr']:.4f} d'={r['d_prime']:.2f}")
print("===EXPERIMENT_COMPLETE===")
