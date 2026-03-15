#!/usr/bin/env python3
"""Experiment 278: Attention Head Decomposition of OOD Signal
Decomposes the L3 hidden state into individual attention head contributions
to identify which heads carry the OOD signal. Uses activation patching:
replace each head's output individually and measure the impact on detection.
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
print("Experiment 278: Attention Head Decomposition")
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

# Analyze attention head outputs at layers 1-7
# For LLaMA, each layer has: attention (32 heads) + MLP
# The residual stream is: x + attn(x) + mlp(x + attn(x))

target_layers = [1, 2, 3, 5, 7]

# Collect per-head attention outputs using hooks
results = {}

for layer_idx in target_layers:
    print(f"\n=== Layer {layer_idx} ===")

    # Find the attention module for this layer
    # OpenVLA uses LLaMA architecture: model.language_model.model.layers[i].self_attn
    attn_module = None
    mlp_module = None

    for name, module in model.named_modules():
        if f'layers.{layer_idx}.self_attn' in name and name.endswith('self_attn'):
            attn_module = (name, module)
        if f'layers.{layer_idx}.mlp' in name and name.endswith('mlp'):
            mlp_module = (name, module)

    if attn_module is None:
        print(f"  Could not find attention module for layer {layer_idx}")
        # Try alternative naming
        for name, module in model.named_modules():
            if f'.{layer_idx}.' in name and 'attn' in name.lower():
                print(f"  Found alternative: {name}")
                if attn_module is None:
                    attn_module = (name, module)
        if attn_module is None:
            continue

    print(f"  Attention module: {attn_module[0]}")

    # Hook to capture attention output and per-head projections
    attn_outputs = {}
    attn_input_store = {}

    def make_attn_hook(store_key):
        def hook_fn(module, input, output):
            # LLaMA self_attn returns (attn_output, attn_weights, past_key_value)
            if isinstance(output, tuple):
                attn_outputs[store_key] = output[0].detach().float().cpu()
            else:
                attn_outputs[store_key] = output.detach().float().cpu()
        return hook_fn

    def make_input_hook(store_key):
        def hook_fn(module, input, output):
            try:
                if isinstance(input, tuple) and len(input) > 0:
                    attn_input_store[store_key] = input[0].detach().float().cpu()
                elif isinstance(input, torch.Tensor):
                    attn_input_store[store_key] = input.detach().float().cpu()
            except Exception as e:
                print(f"    Input hook error: {e}")
        return hook_fn

    layer_results = {}

    for cname, cimg in [('clean', img)] + list(corruptions.items()):
        # Register hooks
        h1 = attn_module[1].register_forward_hook(make_attn_hook(cname))

        inputs = processor(prompt, cimg).to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            fwd = model(**inputs, output_hidden_states=True, output_attentions=True)

        h1.remove()

        # Get the full hidden state at this layer
        hs = fwd.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
        layer_results[f'hidden_{cname}'] = hs

        # Get attention output at last token position
        if cname in attn_outputs:
            attn_out = attn_outputs[cname][0, -1, :].numpy()  # (hidden_dim,)
            layer_results[f'attn_out_{cname}'] = attn_out
            print(f"  {cname}: attn_out norm={np.linalg.norm(attn_out):.4f}, hs norm={np.linalg.norm(hs):.4f}")

        # Extract attention weights for this layer (if available)
        if fwd.attentions is not None and layer_idx < len(fwd.attentions):
            attn_weights = fwd.attentions[layer_idx][0].float().cpu().numpy()  # (n_heads, seq_len, seq_len)
            n_heads = attn_weights.shape[0]

            # Per-head attention at last token position
            head_attns = attn_weights[:, -1, :]  # (n_heads, seq_len)

            # Entropy per head
            head_entropies = []
            head_max_weights = []
            head_img_fracs = []
            for h in range(n_heads):
                p = head_attns[h]
                p_pos = p[p > 0]
                ent = -np.sum(p_pos * np.log(p_pos))
                head_entropies.append(float(ent))
                head_max_weights.append(float(p.max()))
                # Image tokens roughly at positions 1-256
                head_img_fracs.append(float(p[1:257].sum()))

            layer_results[f'head_entropy_{cname}'] = head_entropies
            layer_results[f'head_max_weight_{cname}'] = head_max_weights
            layer_results[f'head_img_frac_{cname}'] = head_img_fracs

    # Compute per-component distances (attn output)
    print(f"\n  --- Attention Output Distances ---")
    attn_distances = {}
    for ctype in ['fog', 'night', 'noise', 'blur']:
        clean_attn = layer_results.get(f'attn_out_clean')
        corr_attn = layer_results.get(f'attn_out_{ctype}')
        if clean_attn is not None and corr_attn is not None:
            d = 1.0 - np.dot(clean_attn, corr_attn) / (np.linalg.norm(clean_attn) * np.linalg.norm(corr_attn) + 1e-30)
            attn_distances[ctype] = float(d)
            print(f"    {ctype}: attn cos_d = {d:.6f}")

    # Compute hidden state distances
    hs_distances = {}
    for ctype in ['fog', 'night', 'noise', 'blur']:
        clean_hs = layer_results[f'hidden_clean']
        corr_hs = layer_results[f'hidden_{ctype}']
        d = 1.0 - np.dot(clean_hs, corr_hs) / (np.linalg.norm(clean_hs) * np.linalg.norm(corr_hs) + 1e-30)
        hs_distances[ctype] = float(d)

    # Per-head entropy divergence
    print(f"\n  --- Per-Head Entropy Changes ---")
    head_divergences = {}
    for ctype in ['fog', 'night', 'noise', 'blur']:
        clean_ent = layer_results.get(f'head_entropy_clean', [])
        corr_ent = layer_results.get(f'head_entropy_{ctype}', [])
        if clean_ent and corr_ent:
            changes = [(c - cl) / (cl + 1e-10) * 100 for cl, c in zip(clean_ent, corr_ent)]
            head_divergences[ctype] = {
                'entropy_pct_changes': [float(c) for c in changes],
                'mean_change': float(np.mean(changes)),
                'max_change': float(np.max(np.abs(changes))),
                'most_affected_head': int(np.argmax(np.abs(changes)))
            }
            top3 = np.argsort(np.abs(changes))[-3:][::-1]
            print(f"    {ctype}: mean={np.mean(changes):.1f}%, top heads: {[(int(h), f'{changes[h]:.1f}%') for h in top3]}")

    # Per-head image attention fraction changes
    img_frac_changes = {}
    for ctype in ['fog', 'night', 'noise', 'blur']:
        clean_frac = layer_results.get(f'head_img_frac_clean', [])
        corr_frac = layer_results.get(f'head_img_frac_{ctype}', [])
        if clean_frac and corr_frac:
            changes = [c - cl for cl, c in zip(clean_frac, corr_frac)]
            img_frac_changes[ctype] = {
                'frac_changes': [float(c) for c in changes],
                'mean_change': float(np.mean(changes)),
                'most_affected_head': int(np.argmax(np.abs(changes)))
            }

    results[f'L{layer_idx}'] = {
        'attn_distances': attn_distances,
        'hidden_distances': hs_distances,
        'head_divergences': head_divergences,
        'img_frac_changes': img_frac_changes,
        'n_heads': n_heads if 'n_heads' in dir() else 32
    }

ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out = {
    'experiment': 'head_decomposition',
    'experiment_number': 278,
    'timestamp': ts,
    'target_layers': target_layers,
    'results': results
}

path = f'/workspace/Vizuara-VLA-Research/experiments/heads_{ts}.json'
with open(path, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {path}")
