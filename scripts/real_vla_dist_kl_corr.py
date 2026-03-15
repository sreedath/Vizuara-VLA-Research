"""Experiment 262: Distance-KL Correlation
Measures correlation between cosine distance and KL divergence across
multiple severity levels to test if they capture the same information.
"""
import torch, json, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageFilter
import time

print("=" * 60)
print("Experiment 262: Distance-KL Correlation")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
base_img = Image.fromarray(np.random.RandomState(42).randint(0, 256, (256, 256, 3), dtype=np.uint8))

def extract_both(image, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    h = fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()
    logits = fwd.logits[0, -1, :].float().cpu()
    probs = torch.softmax(logits, dim=-1).numpy()
    action_probs = probs[31744:32000]
    action_probs = action_probs / action_probs.sum()
    return h, action_probs

def cosine_dist(a, b):
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def kl_div(p, q):
    p, q = np.clip(p, 1e-10, 1), np.clip(q, 1e-10, 1)
    return float(np.sum(p * np.log(p / q)))

# Corruptions at 10 severity levels
severities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

def apply_night(img, s):
    return Image.fromarray((np.array(img).astype(np.float32)*(1-s*0.85)).clip(0,255).astype(np.uint8))

def apply_blur(img, s):
    r = s * 5
    return img.filter(ImageFilter.GaussianBlur(radius=max(r, 0.1)))

# Reference
ref_h, ref_dist = extract_both(base_img)

results = {}
for name, fn in [('night', apply_night), ('blur', apply_blur)]:
    print(f"\n--- {name} ---")
    cos_dists = []
    kl_divs = []
    entropies = []
    top1_probs = []
    
    for sev in severities:
        img = fn(base_img, sev)
        h, dist = extract_both(img)
        cd = cosine_dist(ref_h, h)
        kl = kl_div(ref_dist, dist)
        ent = float(-np.sum(dist * np.log(dist + 1e-10)))
        top1 = float(np.max(dist))
        
        cos_dists.append(round(cd, 8))
        kl_divs.append(round(kl, 6))
        entropies.append(round(ent, 6))
        top1_probs.append(round(top1, 6))
        print(f"  sev={sev:.1f}: cos_d={cd:.6f}, KL={kl:.4f}, entropy={ent:.4f}, top1={top1:.4f}")
    
    corr_d_kl = float(np.corrcoef(cos_dists, kl_divs)[0, 1])
    corr_d_ent = float(np.corrcoef(cos_dists, entropies)[0, 1])
    
    results[name] = {
        "cosine_distances": cos_dists,
        "kl_divergences": kl_divs,
        "entropies": entropies,
        "top1_probs": top1_probs,
        "corr_distance_kl": round(corr_d_kl, 4),
        "corr_distance_entropy": round(corr_d_ent, 4)
    }
    print(f"  Corr(cos_d, KL): {corr_d_kl:.4f}")
    print(f"  Corr(cos_d, entropy): {corr_d_ent:.4f}")

output = {
    "experiment": "dist_kl_correlation",
    "experiment_number": 262,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "results": results
}

path = f"/workspace/Vizuara-VLA-Research/experiments/dist_kl_corr_{output['timestamp']}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")
