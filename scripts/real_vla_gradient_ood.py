"""
Gradient-Based OOD Detection.

Tests whether gradient norms w.r.t. the predicted action token
can distinguish ID from OOD inputs. The hypothesis is that OOD
inputs produce larger/more erratic gradients.

Signals:
1. Input gradient norm (w.r.t. input embeddings)
2. Last-layer gradient norm
3. Gradient entropy

Experiment 76 in the CalibDrive series.
"""
import os
import json
import datetime
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "/workspace/Vizuara-VLA-Research/experiments"
os.makedirs(RESULTS_DIR, exist_ok=True)
SIZE = (256, 256)


def create_highway(idx):
    rng = np.random.default_rng(idx * 5001)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//2] = [135, 206, 235]
    img[SIZE[0]//2:] = [80, 80, 80]
    img[SIZE[0]//2:, SIZE[1]//2-3:SIZE[1]//2+3] = [255, 255, 255]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_urban(idx):
    rng = np.random.default_rng(idx * 5002)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:SIZE[0]//3] = [135, 206, 235]
    img[SIZE[0]//3:SIZE[0]//2] = [139, 119, 101]
    img[SIZE[0]//2:] = [60, 60, 60]
    noise = rng.integers(-5, 6, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_noise(idx):
    rng = np.random.default_rng(idx * 5003)
    return rng.integers(0, 256, (*SIZE, 3), dtype=np.uint8)

def create_indoor(idx):
    rng = np.random.default_rng(idx * 5004)
    img = np.zeros((*SIZE, 3), dtype=np.uint8)
    img[:] = [200, 180, 160]
    img[SIZE[0]//2:, :] = [100, 80, 60]
    img[:SIZE[0]//3, SIZE[1]//3:2*SIZE[1]//3] = [150, 200, 255]
    noise = rng.integers(-10, 11, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def create_blackout(idx):
    return np.zeros((*SIZE, 3), dtype=np.uint8)


def extract_gradient_signals(model, processor, image, prompt):
    """Extract gradient-based OOD signals."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)

    # Enable gradient computation
    model.zero_grad()

    # Get embeddings with gradient tracking
    input_ids = inputs['input_ids']

    # Forward pass with gradients
    outputs = model(**inputs, output_hidden_states=True)
    logits = outputs.logits[0, -1, :]

    # Compute loss w.r.t. top predicted token
    top_token = logits.argmax()
    loss = logits[top_token]

    # Backward pass
    loss.backward()

    result = {}

    # Collect gradient norms from model parameters
    grad_norms = []
    last_layer_grad_norm = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            gnorm = param.grad.float().norm().item()
            grad_norms.append(gnorm)

            # Track last transformer layer specifically
            if 'layers.31' in name or 'model.layers.31' in name:
                last_layer_grad_norm += gnorm

    result['total_grad_norm'] = float(np.sum(grad_norms)) if grad_norms else 0
    result['mean_grad_norm'] = float(np.mean(grad_norms)) if grad_norms else 0
    result['max_grad_norm'] = float(np.max(grad_norms)) if grad_norms else 0
    result['last_layer_grad_norm'] = float(last_layer_grad_norm)
    result['n_params_with_grad'] = len(grad_norms)

    # Also collect hidden state for cosine comparison
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        result['hidden'] = outputs.hidden_states[-1][0, -1, :].float().detach().cpu().numpy()

    model.zero_grad()
    return result


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("GRADIENT-BASED OOD DETECTION", flush=True)
    print("=" * 70, flush=True)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b", trust_remote_code=True,
    )
    # Don't call model.eval() — we need gradients
    print("Model loaded.", flush=True)

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"

    # Calibration (need centroid for cosine baseline)
    print("\nCalibrating...", flush=True)
    cal_hidden = []
    for fn in [create_highway, create_urban]:
        for i in range(8):
            sig = extract_gradient_signals(model, processor,
                                           Image.fromarray(fn(i + 9000)), prompt)
            if 'hidden' in sig:
                cal_hidden.append(sig['hidden'])
    centroid = np.mean(cal_hidden, axis=0)
    print(f"  Calibrated with {len(cal_hidden)} samples", flush=True)

    # Test
    test_fns = {
        'highway': (create_highway, False, 8),
        'urban': (create_urban, False, 8),
        'noise': (create_noise, True, 6),
        'indoor': (create_indoor, True, 6),
        'blackout': (create_blackout, True, 4),
    }

    all_data = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())
    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            sig = extract_gradient_signals(model, processor,
                                           Image.fromarray(fn(i + 500)), prompt)
            sig['scenario'] = scene
            sig['is_ood'] = is_ood
            if 'hidden' in sig:
                sig['cosine'] = cosine_dist(sig['hidden'], centroid)
            all_data.append(sig)
            if cnt % 5 == 0:
                print(f"  [{cnt}/{total}] {scene}: total_grad={sig['total_grad_norm']:.4f} "
                      f"mean_grad={sig['mean_grad_norm']:.6f}", flush=True)

    print(f"\nCollected {len(all_data)} samples.", flush=True)

    # Analysis
    print("\n" + "=" * 70, flush=True)
    print("RESULTS", flush=True)
    print("=" * 70, flush=True)

    id_data = [d for d in all_data if not d['is_ood']]
    ood_data = [d for d in all_data if d['is_ood']]
    labels = [0]*len(id_data) + [1]*len(ood_data)

    results = {}
    for metric in ['total_grad_norm', 'mean_grad_norm', 'max_grad_norm',
                    'last_layer_grad_norm', 'cosine']:
        scores = [d.get(metric, 0) for d in id_data] + [d.get(metric, 0) for d in ood_data]
        auroc = roc_auc_score(labels, scores)
        results[metric] = float(auroc)

        id_vals = [d.get(metric, 0) for d in id_data]
        ood_vals = [d.get(metric, 0) for d in ood_data]
        print(f"  {metric:<25}: AUROC={auroc:.3f}  "
              f"ID={np.mean(id_vals):.6f}±{np.std(id_vals):.6f}  "
              f"OOD={np.mean(ood_vals):.6f}±{np.std(ood_vals):.6f}", flush=True)

    # Per-scenario
    print("\n  Per-scenario gradient norms:", flush=True)
    per_scenario = {}
    for scene in ['highway', 'urban', 'noise', 'indoor', 'blackout']:
        sd = [d for d in all_data if d['scenario'] == scene]
        per_scenario[scene] = {
            'total': float(np.mean([d['total_grad_norm'] for d in sd])),
            'mean': float(np.mean([d['mean_grad_norm'] for d in sd])),
        }
        print(f"    {scene:<12}: total={per_scenario[scene]['total']:.4f} "
              f"mean={per_scenario[scene]['mean']:.6f}", flush=True)

    # Save (remove numpy arrays)
    for d in all_data:
        if 'hidden' in d:
            del d['hidden']

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'gradient_ood',
        'experiment_number': 76,
        'timestamp': timestamp,
        'n_samples': len(all_data),
        'auroc_by_metric': results,
        'per_scenario': per_scenario,
    }
    output_path = os.path.join(RESULTS_DIR, f"gradient_ood_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
