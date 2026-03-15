#!/usr/bin/env python3
"""Experiment 277: Vision Encoder vs Language Model Attribution
Determines where the OOD signal originates by extracting representations
at three stages: (1) vision encoder output, (2) projection layer output,
(3) language model hidden states. Tests whether the signal is created
by the vision encoder or amplified/transformed by the language backbone.
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

print("=" * 60)
print("Experiment 277: Vision Encoder vs Language Model Attribution")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

# Explore model architecture
print("\n=== MODEL ARCHITECTURE ===")
print(f"Model type: {type(model).__name__}")
print(f"Top-level modules:")
for name, module in model.named_children():
    print(f"  {name}: {type(module).__name__}")

# Try to find vision backbone and projector
vision_backbone = None
projector = None
llm_backbone = None

for name, module in model.named_children():
    name_lower = name.lower()
    if 'vision' in name_lower or 'vit' in name_lower or 'image' in name_lower:
        vision_backbone = (name, module)
        print(f"\nVision backbone found: {name}")
        print(f"  Type: {type(module).__name__}")
    if 'project' in name_lower or 'connector' in name_lower or 'mlp' in name_lower:
        projector = (name, module)
        print(f"\nProjector found: {name}")
        print(f"  Type: {type(module).__name__}")
    if 'language' in name_lower or 'llm' in name_lower or 'model' in name_lower:
        llm_backbone = (name, module)
        print(f"\nLLM backbone found: {name}")
        print(f"  Type: {type(module).__name__}")

# Dig deeper into structure
print("\n=== DETAILED STRUCTURE ===")
for name, param in model.named_parameters():
    if any(k in name for k in ['vision', 'vit', 'project', 'embed']):
        print(f"  {name}: {param.shape}")
        if len([n for n, _ in model.named_parameters() if 'vision' in n or 'vit' in n]) > 20:
            break

prompt = "In: What action should the robot take to pick up the object?\nOut:"
np.random.seed(42)
pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(pixels)

corruptions = {'fog': apply_corruption(img, 'fog'),
               'night': apply_corruption(img, 'night'),
               'noise': apply_corruption(img, 'noise'),
               'blur': apply_corruption(img, 'blur')}

results = {}

# Stage 1: Full model hidden states (what we already know works)
print("\n=== STAGE 1: Full Model Hidden States ===")
for cname, cimg in [('clean', img)] + list(corruptions.items()):
    inputs = processor(prompt, cimg).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    # Get all layer hidden states at last token
    n_layers = len(fwd.hidden_states)
    hs = {}
    for l in range(n_layers):
        h = fwd.hidden_states[l][0, -1, :].float().cpu().numpy()
        hs[l] = h
    results[f'hidden_{cname}'] = {l: h.tolist() for l, h in hs.items()}
    print(f"  {cname}: {n_layers} layers, dim={fwd.hidden_states[0].shape[-1]}")

# Compute cosine distances at each layer
print("\n=== COSINE DISTANCES PER LAYER ===")
distance_by_layer = {}
for ctype in ['fog', 'night', 'noise', 'blur']:
    dists = {}
    for l in range(n_layers):
        clean_h = np.array(results[f'hidden_clean'][l])
        corr_h = np.array(results[f'hidden_{ctype}'][l])
        d = 1.0 - np.dot(clean_h, corr_h) / (np.linalg.norm(clean_h) * np.linalg.norm(corr_h) + 1e-30)
        dists[l] = float(d)
    distance_by_layer[ctype] = dists
    print(f"  {ctype}: L0={dists[0]:.6f}, L3={dists[3]:.6f}, L15={dists[15]:.6f}, L31={dists[31]:.6f}")

# Stage 2: Try to extract vision encoder outputs via hooks
print("\n=== STAGE 2: Vision Encoder Attribution via Hooks ===")
hook_outputs = {}

def make_hook(name):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hook_outputs[name] = output[0].detach().float().cpu()
        elif isinstance(output, torch.Tensor):
            hook_outputs[name] = output.detach().float().cpu()
    return hook_fn

# Register hooks on various modules
hooks = []
for name, module in model.named_modules():
    # Look for vision encoder output, projector, and embedding layers
    if name.endswith(('.vision_backbone', '.vision_tower', '.featurizer', '.fused_featurizer')):
        hooks.append(module.register_forward_hook(make_hook(f'vision_{name}')))
        print(f"  Hook registered: {name}")
    elif 'projector' in name and '.' not in name.split('projector')[-1]:
        hooks.append(module.register_forward_hook(make_hook(f'proj_{name}')))
        print(f"  Hook registered: {name}")
    elif name in ['model.embed_tokens', 'embed_tokens']:
        hooks.append(module.register_forward_hook(make_hook(f'embed_{name}')))
        print(f"  Hook registered: {name}")

# Also try common OpenVLA-specific module names
for name, module in model.named_modules():
    short = name.split('.')[-1] if '.' in name else name
    if short in ['backbone', 'encoder', 'visual']:
        hooks.append(module.register_forward_hook(make_hook(f'vis_{name}')))
        print(f"  Hook registered: {name}")

vision_results = {}
for cname, cimg in [('clean', img)] + list(corruptions.items()):
    hook_outputs.clear()
    inputs = processor(prompt, cimg).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)

    vision_results[cname] = {}
    for hook_name, tensor in hook_outputs.items():
        vision_results[cname][hook_name] = {
            'shape': list(tensor.shape),
            'mean': float(tensor.mean()),
            'std': float(tensor.std()),
            'norm': float(tensor.norm()),
        }
        # Store flattened representation for distance computation
        vision_results[cname][f'{hook_name}_flat'] = tensor.reshape(-1).numpy()
        print(f"  {cname} {hook_name}: shape={list(tensor.shape)}, norm={tensor.norm():.4f}")

# Clean up hooks
for h in hooks:
    h.remove()

# Compute vision encoder distances
print("\n=== VISION ENCODER DISTANCES ===")
vision_distances = {}
for ctype in ['fog', 'night', 'noise', 'blur']:
    vd = {}
    for hook_name in vision_results.get('clean', {}):
        if hook_name.endswith('_flat'):
            continue
        flat_key = f'{hook_name}_flat'
        if flat_key in vision_results['clean'] and flat_key in vision_results.get(ctype, {}):
            clean_v = vision_results['clean'][flat_key]
            corr_v = vision_results[ctype][flat_key]
            cos_d = 1.0 - np.dot(clean_v, corr_v) / (np.linalg.norm(clean_v) * np.linalg.norm(corr_v) + 1e-30)
            l2_d = float(np.linalg.norm(clean_v - corr_v))
            vd[hook_name] = {'cosine_distance': float(cos_d), 'l2_distance': l2_d}
            print(f"  {ctype} {hook_name}: cos_d={cos_d:.6f}, l2_d={l2_d:.4f}")
    vision_distances[ctype] = vd

# Stage 3: Analyze where the signal amplifies
print("\n=== SIGNAL AMPLIFICATION ACROSS LAYERS ===")
amplification = {}
for ctype in ['fog', 'night', 'noise', 'blur']:
    layer_dists = distance_by_layer[ctype]
    # Find first layer with nonzero distance
    first_nonzero = None
    for l in sorted(layer_dists.keys()):
        if layer_dists[l] > 1e-10:
            first_nonzero = l
            break

    # Signal growth from L0 to L31
    d0 = layer_dists[0]
    d3 = layer_dists[3]
    d31 = layer_dists[max(layer_dists.keys())]

    amplification[ctype] = {
        'first_nonzero_layer': first_nonzero,
        'L0_distance': float(d0),
        'L3_distance': float(d3),
        'L31_distance': float(d31),
        'L0_to_L3_ratio': float(d3 / (d0 + 1e-30)),
        'L0_to_L31_ratio': float(d31 / (d0 + 1e-30)),
        'all_layer_distances': {int(l): float(d) for l, d in layer_dists.items()}
    }
    print(f"  {ctype}: L0={d0:.6f} → L3={d3:.6f} ({d3/(d0+1e-30):.1f}×) → L31={d31:.6f} ({d31/(d0+1e-30):.1f}×)")

# Compile results
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
# Remove numpy arrays from vision_results for serialization
clean_vision_results = {}
for cname, vr in vision_results.items():
    clean_vision_results[cname] = {k: v for k, v in vr.items() if not k.endswith('_flat')}

out = {
    'experiment': 'vision_encoder_attribution',
    'experiment_number': 277,
    'timestamp': ts,
    'model_type': type(model).__name__,
    'results': {
        'distance_by_layer': distance_by_layer,
        'vision_hook_stats': clean_vision_results,
        'vision_distances': vision_distances,
        'amplification': amplification
    }
}

path = f'/workspace/Vizuara-VLA-Research/experiments/attribution_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
