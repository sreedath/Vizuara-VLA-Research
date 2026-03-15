"""Experiment 260: KL Divergence Between Output Distributions
Measures KL divergence between clean and corrupted next-token probability
distributions to quantify how corruption affects model confidence.
"""
import torch, json, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageFilter
import time

print("=" * 60)
print("Experiment 260: KL Divergence Analysis")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
base_img = Image.fromarray(np.random.RandomState(42).randint(0, 256, (256, 256, 3), dtype=np.uint8))

def get_logit_distribution(image):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs)
    logits = fwd.logits[0, -1, :].float().cpu()  # Last token logits
    probs = torch.softmax(logits, dim=-1).numpy()
    # Focus on action tokens (31744-31999)
    action_probs = probs[31744:32000]
    action_probs = action_probs / action_probs.sum()  # Renormalize
    return action_probs, probs

def kl_divergence(p, q):
    """KL(P || Q) where P is clean, Q is corrupt"""
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return float(np.sum(p * np.log(p / q)))

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

# Corruptions
def apply_fog(img): return Image.fromarray((np.array(img).astype(np.float32)*0.6+120*0.4).clip(0,255).astype(np.uint8))
def apply_night(img): return Image.fromarray((np.array(img).astype(np.float32)*0.15).clip(0,255).astype(np.uint8))
def apply_noise(img):
    a = np.array(img).astype(np.float32); a += np.random.RandomState(99).randn(*a.shape)*50
    return Image.fromarray(a.clip(0,255).astype(np.uint8))
def apply_blur(img): return img.filter(ImageFilter.GaussianBlur(radius=5))

# Get clean distribution
clean_action_dist, clean_full = get_logit_distribution(base_img)
clean_top_idx = int(np.argmax(clean_action_dist))
clean_top_prob = float(clean_action_dist[clean_top_idx])
clean_entropy = float(-np.sum(clean_action_dist * np.log(clean_action_dist + 1e-10)))
print(f"Clean: top={clean_top_idx+31744}, prob={clean_top_prob:.4f}, entropy={clean_entropy:.4f}")

corruptions = {'fog': apply_fog, 'night': apply_night, 'noise': apply_noise, 'blur': apply_blur}
results = {
    "clean": {
        "top_token": clean_top_idx + 31744,
        "top_prob": round(clean_top_prob, 6),
        "entropy": round(clean_entropy, 6),
        "top5": [(int(i+31744), round(float(clean_action_dist[i]), 6)) 
                 for i in np.argsort(clean_action_dist)[-5:][::-1]]
    }
}

for name, fn in corruptions.items():
    cimg = fn(base_img)
    corrupt_dist, corrupt_full = get_logit_distribution(cimg)
    
    top_idx = int(np.argmax(corrupt_dist))
    top_prob = float(corrupt_dist[top_idx])
    entropy = float(-np.sum(corrupt_dist * np.log(corrupt_dist + 1e-10)))
    
    kl = kl_divergence(clean_action_dist, corrupt_dist)
    js = js_divergence(clean_action_dist, corrupt_dist)
    
    results[name] = {
        "top_token": top_idx + 31744,
        "top_prob": round(top_prob, 6),
        "entropy": round(entropy, 6),
        "kl_divergence": round(kl, 6),
        "js_divergence": round(js, 6),
        "top5": [(int(i+31744), round(float(corrupt_dist[i]), 6))
                 for i in np.argsort(corrupt_dist)[-5:][::-1]]
    }
    print(f"\n  {name}:")
    print(f"    top={top_idx+31744}, prob={top_prob:.4f}, entropy={entropy:.4f}")
    print(f"    KL(clean||corrupt)={kl:.4f}, JS={js:.4f}")
    print(f"    Top5: {results[name]['top5']}")

output = {
    "experiment": "kl_divergence",
    "experiment_number": 260,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "results": results
}

path = f"/workspace/Vizuara-VLA-Research/experiments/kl_div_{output['timestamp']}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")
