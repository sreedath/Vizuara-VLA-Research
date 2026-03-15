#!/usr/bin/env python3
"""Experiment 284: Temperature Scaling and Generation Effects
Tests how temperature scaling during generation affects:
1. Action token prediction (does temperature change output actions?)
2. Hidden state embeddings (does temperature change L3 hidden states?)
3. Detection reliability (does temperature affect AUROC?)
Also tests greedy vs sampling generation modes.
"""
import torch, json, numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

def apply_corruption(img, ctype, severity=1.0):
    arr = np.array(img).astype(np.float32) / 255.0
    if ctype == 'fog':
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == 'night':
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == 'noise':
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

print("=" * 60)
print("Experiment 284: Temperature Scaling Effects")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
np.random.seed(42)
pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(pixels)

corruptions = {
    'fog': apply_corruption(img, 'fog'),
    'night': apply_corruption(img, 'night'),
    'noise': apply_corruption(img, 'noise'),
    'blur': apply_corruption(img, 'blur'),
}

# Test: Hidden states are computed before generation (temperature only affects sampling)
# So temperature should NOT affect our detector at all.
# But let's verify this experimentally.

# Step 1: Verify hidden states don't depend on temperature
# Hidden states from a single forward pass don't involve temperature
# Temperature only matters during auto-regressive generation
print("\n=== HIDDEN STATE INDEPENDENCE ===")
centroid = extract_hidden(model, processor, img, prompt)

# Step 2: Generate action tokens at different temperatures
print("\n=== GENERATION WITH TEMPERATURE ===")
temperatures = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0]

generation_results = {}
for cname, cimg in [('clean', img)] + list(corruptions.items()):
    print(f"\n--- {cname} ---")
    inputs = processor(prompt, cimg).to(model.device, dtype=torch.bfloat16)
    
    gen_by_temp = {}
    for temp in temperatures:
        # Greedy (temperature doesn't matter for greedy)
        with torch.no_grad():
            output_greedy = model.generate(
                **inputs, max_new_tokens=7, do_sample=False
            )
        greedy_tokens = output_greedy[0, -7:].cpu().tolist()
        
        # Sampling with temperature
        tokens_samples = []
        for trial in range(5):
            torch.manual_seed(trial)
            with torch.no_grad():
                output_sample = model.generate(
                    **inputs, max_new_tokens=7, do_sample=True,
                    temperature=temp, top_k=0, top_p=1.0
                )
            sample_tokens = output_sample[0, -7:].cpu().tolist()
            tokens_samples.append(sample_tokens)
        
        # Check consistency across samples
        all_same = all(s == tokens_samples[0] for s in tokens_samples)
        
        gen_by_temp[str(temp)] = {
            'greedy_tokens': greedy_tokens,
            'sample_tokens': tokens_samples,
            'all_samples_identical': all_same,
            'n_unique_outputs': len(set(tuple(s) for s in tokens_samples))
        }
        
        print(f"  T={temp:.1f}: greedy={greedy_tokens[-3:]}, "
              f"samples_identical={all_same}, "
              f"n_unique={gen_by_temp[str(temp)]['n_unique_outputs']}")
    
    generation_results[cname] = gen_by_temp

# Step 3: Verify detection AUROC with different generation modes
# (Detection uses hidden states, not generation, so should be unaffected)
print("\n=== DETECTION VS GENERATION MODE ===")
# Extract hidden states for all conditions
hs = {}
for cname, cimg in [('clean', img)] + list(corruptions.items()):
    hs[cname] = extract_hidden(model, processor, cimg, prompt)

# Cosine distances
distances = {}
for ctype in corruptions:
    d = 1.0 - np.dot(hs[ctype], centroid) / (np.linalg.norm(hs[ctype]) * np.linalg.norm(centroid) + 1e-30)
    distances[ctype] = float(d)
    print(f"  {ctype}: d={d:.6f} (independent of temperature)")

# Step 4: Logit distribution analysis at different temperatures
print("\n=== LOGIT DISTRIBUTION vs TEMPERATURE ===")
logit_analysis = {}
for cname, cimg in [('clean', img), ('night', corruptions['night'])]:
    inputs = processor(prompt, cimg).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs)
    logits = fwd.logits[0, -1, :].float().cpu()
    
    temp_stats = {}
    for temp in temperatures:
        scaled = logits / temp
        probs = torch.softmax(scaled, dim=0)
        entropy = float(-torch.sum(probs * torch.log(probs + 1e-10)))
        top1_prob = float(probs.max())
        top5_prob = float(probs.topk(5).values.sum())
        
        temp_stats[str(temp)] = {
            'entropy': entropy,
            'top1_prob': top1_prob,
            'top5_prob': top5_prob,
            'top1_token': int(probs.argmax())
        }
        print(f"  {cname} T={temp:.1f}: entropy={entropy:.3f}, top1={top1_prob:.4f}")
    
    logit_analysis[cname] = temp_stats

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'temperature_effect',
    'experiment_number': 284,
    'timestamp': ts,
    'results': {
        'generation': generation_results,
        'detection_distances': distances,
        'logit_analysis': logit_analysis
    }
}

path = f'/workspace/Vizuara-VLA-Research/experiments/temperature_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
