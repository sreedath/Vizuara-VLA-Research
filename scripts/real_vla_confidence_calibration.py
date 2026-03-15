#!/usr/bin/env python3
"""Experiment 156: Confidence calibration — OOD distance vs logit entropy.

Measures whether the model's own token-level uncertainty (logit entropy)
correlates with our external OOD detector distance. If yes, we can use
entropy as a complementary or standalone confidence signal.
"""

import json, os, sys, datetime
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageFilter

SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent
EXPERIMENTS_DIR = REPO_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR = str(EXPERIMENTS_DIR)

SIZE = (256, 256)
rng = np.random.RandomState(42)

def create_highway(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]; img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    return np.clip(img.astype(np.int16) + rng.randint(-5, 6, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def create_urban(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]; img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]; img[SIZE[0]//2:] = [60, 60, 60]
    return np.clip(img.astype(np.int16) + rng.randint(-5, 6, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def create_rural(idx):
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [100, 180, 255]; img[SIZE[0]//3:SIZE[0]*2//3] = [34, 139, 34]; img[SIZE[0]*2//3:] = [90, 90, 90]
    return np.clip(img.astype(np.int16) + rng.randint(-8, 9, img.shape).astype(np.int16), 0, 255).astype(np.uint8)

def apply_fog(a, alpha):
    return np.clip(a*(1-alpha)+np.full_like(a,[200,200,210])*alpha, 0, 255).astype(np.uint8)
def apply_night(a): return np.clip(a*0.15, 0, 255).astype(np.uint8)
def apply_blur(a, r=8): return np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius=r)))
def apply_noise(a, s=50): return np.clip(a.astype(np.float32)+np.random.normal(0,s,a.shape), 0, 255).astype(np.uint8)
def apply_occlusion(a):
    o=a.copy(); h,w=o.shape[:2]; o[h//4:3*h//4, w//4:3*w//4]=128; return o

def cosine_distance(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def extract_hidden_and_logits(model, processor, image, prompt, layers):
    """Extract hidden states AND next-token logits for entropy computation."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    
    hiddens = {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}
    
    # Get logits for the last token position (next-token prediction)
    logits = fwd.logits[0, -1, :].float().cpu()
    
    # Full vocabulary entropy
    probs = torch.softmax(logits, dim=0)
    full_entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)))
    
    # Action token entropy (only over action token range 31744-31999)
    action_logits = logits[31744:32000]
    action_probs = torch.softmax(action_logits, dim=0)
    action_entropy = float(-torch.sum(action_probs * torch.log(action_probs + 1e-10)))
    
    # Top-1 action token probability (confidence)
    top1_prob = float(action_probs.max())
    top1_token = int(action_probs.argmax()) + 31744
    
    # Top-5 action token probabilities
    top5_vals, top5_idx = torch.topk(action_probs, 5)
    top5_info = [(int(idx) + 31744, float(val)) for idx, val in zip(top5_idx, top5_vals)]
    
    return hiddens, {
        "full_entropy": full_entropy,
        "action_entropy": action_entropy,
        "top1_prob": top1_prob,
        "top1_token": top1_token,
        "top5": top5_info,
    }

def main():
    print("=" * 60)
    print("Experiment 156: Confidence Calibration (Distance vs Entropy)")
    print("=" * 60, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"
    layers = [3, 32]

    creators = [create_highway, create_urban, create_rural]
    n_cal = 8
    n_test = 6

    # Calibration
    print("\n--- Calibrating centroids ---", flush=True)
    cal_arrs = [creators[i%3](i) for i in range(n_cal)]
    centroids = {}
    for l in layers:
        cal_embs = []
        for arr in cal_arrs:
            h, _ = extract_hidden_and_logits(model, processor, Image.fromarray(arr), prompt, layers)
            cal_embs.append(h[l])
        centroids[l] = np.array(cal_embs).mean(axis=0)
    print("  Centroids computed.", flush=True)

    # Test images
    test_arrs = [creators[(i+n_cal)%3](i+n_cal) for i in range(n_test)]

    # Corruptions with varying severity
    corruption_suite = {
        "clean": lambda a: a,
        "fog_10": lambda a: apply_fog(a, 0.1),
        "fog_20": lambda a: apply_fog(a, 0.2),
        "fog_30": lambda a: apply_fog(a, 0.3),
        "fog_40": lambda a: apply_fog(a, 0.4),
        "fog_50": lambda a: apply_fog(a, 0.5),
        "fog_60": lambda a: apply_fog(a, 0.6),
        "fog_70": lambda a: apply_fog(a, 0.7),
        "fog_80": lambda a: apply_fog(a, 0.8),
        "night": apply_night,
        "blur_2": lambda a: apply_blur(a, 2),
        "blur_4": lambda a: apply_blur(a, 4),
        "blur_8": lambda a: apply_blur(a, 8),
        "blur_16": lambda a: apply_blur(a, 16),
        "noise_15": lambda a: apply_noise(a, 15),
        "noise_30": lambda a: apply_noise(a, 30),
        "noise_50": lambda a: apply_noise(a, 50),
        "noise_100": lambda a: apply_noise(a, 100),
        "occlusion": apply_occlusion,
    }

    all_points = []  # (condition, image_idx, dist_L3, dist_L32, full_ent, action_ent, top1_prob)
    results = {}

    for cname, cfn in corruption_suite.items():
        print(f"\n--- {cname} ---", flush=True)
        dists_l3 = []; dists_l32 = []
        full_ents = []; action_ents = []; top1_probs = []

        for i, arr in enumerate(test_arrs):
            corr_arr = cfn(arr)
            h, ent_info = extract_hidden_and_logits(model, processor, Image.fromarray(corr_arr), prompt, layers)
            d3 = cosine_distance(h[3], centroids[3])
            d32 = cosine_distance(h[32], centroids[32])
            dists_l3.append(d3)
            dists_l32.append(d32)
            full_ents.append(ent_info["full_entropy"])
            action_ents.append(ent_info["action_entropy"])
            top1_probs.append(ent_info["top1_prob"])
            all_points.append({
                "condition": cname, "image": i,
                "dist_L3": d3, "dist_L32": d32,
                "full_entropy": ent_info["full_entropy"],
                "action_entropy": ent_info["action_entropy"],
                "top1_prob": ent_info["top1_prob"],
            })

        entry = {
            "dist_L3_mean": float(np.mean(dists_l3)),
            "dist_L3_std": float(np.std(dists_l3)),
            "dist_L32_mean": float(np.mean(dists_l32)),
            "dist_L32_std": float(np.std(dists_l32)),
            "full_entropy_mean": float(np.mean(full_ents)),
            "full_entropy_std": float(np.std(full_ents)),
            "action_entropy_mean": float(np.mean(action_ents)),
            "action_entropy_std": float(np.std(action_ents)),
            "top1_prob_mean": float(np.mean(top1_probs)),
            "top1_prob_std": float(np.std(top1_probs)),
        }
        results[cname] = entry
        print(f"  dist_L3={entry['dist_L3_mean']:.6f} dist_L32={entry['dist_L32_mean']:.4f} "
              f"act_ent={entry['action_entropy_mean']:.3f} top1={entry['top1_prob_mean']:.4f}", flush=True)

    # Compute correlations
    pts = all_points
    d3_arr = np.array([p["dist_L3"] for p in pts])
    d32_arr = np.array([p["dist_L32"] for p in pts])
    ae_arr = np.array([p["action_entropy"] for p in pts])
    fe_arr = np.array([p["full_entropy"] for p in pts])
    t1_arr = np.array([p["top1_prob"] for p in pts])

    def pearson(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    correlations = {
        "dist_L3_vs_action_entropy": pearson(d3_arr, ae_arr),
        "dist_L3_vs_full_entropy": pearson(d3_arr, fe_arr),
        "dist_L3_vs_top1_prob": pearson(d3_arr, t1_arr),
        "dist_L32_vs_action_entropy": pearson(d32_arr, ae_arr),
        "dist_L32_vs_full_entropy": pearson(d32_arr, fe_arr),
        "dist_L32_vs_top1_prob": pearson(d32_arr, t1_arr),
        "action_entropy_vs_full_entropy": pearson(ae_arr, fe_arr),
    }

    print("\n" + "=" * 80)
    print("CORRELATION MATRIX")
    for k, v in correlations.items():
        print(f"  {k}: r = {v:.4f}")

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Condition':<15} {'dist_L3':>10} {'dist_L32':>10} {'act_ent':>10} {'top1_p':>10}")
    for cname, entry in results.items():
        print(f"{cname:<15} {entry['dist_L3_mean']:10.6f} {entry['dist_L32_mean']:10.4f} "
              f"{entry['action_entropy_mean']:10.3f} {entry['top1_prob_mean']:10.4f}")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "confidence_calibration",
        "experiment_number": 156,
        "timestamp": ts,
        "n_cal": n_cal, "n_test": n_test,
        "layers": layers,
        "conditions": list(corruption_suite.keys()),
        "results": results,
        "correlations": correlations,
        "all_points": all_points,
    }
    path = os.path.join(RESULTS_DIR, f"confidence_calibration_{ts}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {path}")

if __name__ == "__main__":
    main()
