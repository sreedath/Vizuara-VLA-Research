"""
Token Position Analysis.

Examines how the hidden state OOD signal varies across different
token positions in the sequence (not just the last token).

Hypothesis: visual tokens carry the strongest OOD signal since
they directly process the image, while text tokens may carry
weaker or no signal.

Experiment 77 in the CalibDrive series.
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


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a / (np.linalg.norm(a) + 1e-10),
                               b / (np.linalg.norm(b) + 1e-10)))


def main():
    print("=" * 70, flush=True)
    print("TOKEN POSITION ANALYSIS", flush=True)
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
    model.eval()
    print("Model loaded.", flush=True)

    prompt = "In: What action should the robot take to drive forward at 25 m/s safely?\nOut:"

    # Sample positions to test (relative to sequence)
    # We'll test: first 5%, 25%, 50%, 75%, 90%, 95%, last token
    position_fracs = [0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]

    test_fns = {
        'highway': (create_highway, False, 8),
        'urban': (create_urban, False, 8),
        'noise': (create_noise, True, 6),
        'indoor': (create_indoor, True, 6),
        'blackout': (create_blackout, True, 4),
    }

    # Calibrate at each position
    print("\nCalibrating at all positions...", flush=True)
    cal_hidden_by_pos = {frac: [] for frac in position_fracs}

    for fn in [create_highway, create_urban]:
        for i in range(10):
            img = Image.fromarray(fn(i + 9000))
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)

            if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
                last_hs = fwd.hidden_states[-1][0]  # [seq_len, hidden_dim]
                seq_len = last_hs.shape[0]
                for frac in position_fracs:
                    pos = min(int(frac * seq_len) - 1, seq_len - 1)
                    pos = max(pos, 0)
                    h = last_hs[pos].float().cpu().numpy()
                    cal_hidden_by_pos[frac].append(h)

    centroids = {frac: np.mean(cal_hidden_by_pos[frac], axis=0) for frac in position_fracs}
    print(f"  Calibrated {len(position_fracs)} positions", flush=True)

    # Test
    print("\nTesting...", flush=True)
    test_hidden_by_pos = {frac: [] for frac in position_fracs}
    test_labels = []
    cnt = 0
    total = sum(v[2] for v in test_fns.values())

    for scene, (fn, is_ood, n) in test_fns.items():
        for i in range(n):
            cnt += 1
            img = Image.fromarray(fn(i + 500))
            inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)

            if hasattr(fwd, 'hidden_states') and fwd.hidden_states:
                last_hs = fwd.hidden_states[-1][0]
                seq_len = last_hs.shape[0]
                for frac in position_fracs:
                    pos = min(int(frac * seq_len) - 1, seq_len - 1)
                    pos = max(pos, 0)
                    h = last_hs[pos].float().cpu().numpy()
                    test_hidden_by_pos[frac].append(h)

            test_labels.append(1 if is_ood else 0)
            if cnt % 10 == 0:
                print(f"  [{cnt}/{total}] {scene}", flush=True)

    # Compute AUROC at each position
    print("\n" + "=" * 70, flush=True)
    print("RESULTS", flush=True)
    print("=" * 70, flush=True)

    results = {}
    for frac in position_fracs:
        scores = [cosine_dist(h, centroids[frac]) for h in test_hidden_by_pos[frac]]
        auroc = roc_auc_score(test_labels, scores)

        id_scores = [s for s, l in zip(scores, test_labels) if l == 0]
        ood_scores = [s for s, l in zip(scores, test_labels) if l == 1]
        pooled_std = np.sqrt((np.std(id_scores)**2 + np.std(ood_scores)**2) / 2)
        cohens_d = abs(np.mean(id_scores) - np.mean(ood_scores)) / (pooled_std + 1e-10)

        results[str(frac)] = {
            'auroc': float(auroc),
            'cohens_d': float(cohens_d),
            'id_mean': float(np.mean(id_scores)),
            'ood_mean': float(np.mean(ood_scores)),
        }
        pct = int(frac * 100)
        print(f"  Position {pct:>3}%: AUROC={auroc:.3f}  d={cohens_d:.2f}  "
              f"ID={np.mean(id_scores):.4f}  OOD={np.mean(ood_scores):.4f}", flush=True)

    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'experiment': 'token_position',
        'experiment_number': 77,
        'timestamp': timestamp,
        'position_fracs': position_fracs,
        'n_test': len(test_labels),
        'results': results,
    }
    output_path = os.path.join(RESULTS_DIR, f"token_position_{timestamp}.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
