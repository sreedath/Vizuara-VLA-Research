"""
Experiment 218: Embedding Trajectory Analysis
How does the embedding evolve as corruption severity increases continuously?
Traces the embedding path from clean to fully corrupted.
"""
import torch, json, numpy as np, os
from datetime import datetime
from PIL import Image, ImageFilter

def make_driving_image(w=256, h=256):
    img = Image.new('RGB', (w, h))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            if y < h // 2:
                b = int(180 + 75 * (1 - y / (h / 2)))
                pixels[x, y] = (100, 150, b)
            else:
                g = int(80 + 40 * ((y - h/2) / (h/2)))
                pixels[x, y] = (g, g + 10, g - 10)
    return img

def apply_corruption_at_severity(img, name, severity):
    """Apply corruption at a specific severity level (0.0 = clean, 1.0 = full)."""
    arr = np.array(img, dtype=np.float32)
    clean_arr = arr.copy()
    
    if name == 'fog':
        fog = np.full_like(arr, 200)
        alpha = severity * 0.8
        arr = arr * (1 - alpha) + fog * alpha
    elif name == 'night':
        brightness = 1.0 - severity * 0.9
        arr = arr * brightness
    elif name == 'blur':
        radius = severity * 8
        if radius > 0.1:
            blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
            return blurred
        return img
    elif name == 'noise':
        sigma = severity * 60
        rng = np.random.default_rng(42)
        arr = arr + rng.normal(0, sigma, arr.shape)
    
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def extract_hidden(model, processor, image, prompt, layers):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return {l: fwd.hidden_states[l][0, -1, :].float().cpu().numpy() for l in layers}

def cosine_dist(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def main():
    print("=" * 60)
    print("Experiment 218: Embedding Trajectory Analysis")
    print("=" * 60)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    layers = [1, 3]
    prompt = "In: What action should the robot take to drive forward?\nOut:"
    base_img = make_driving_image()
    corruption_types = ['fog', 'night', 'blur', 'noise']
    severities = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Get clean embedding (centroid)
    clean_h = extract_hidden(model, processor, base_img, prompt, layers)
    
    print("\n--- Tracing trajectories ---")
    results = {}
    for ctype in corruption_types:
        trajectory = {f"L{l}": [] for l in layers}
        prev_h = {l: clean_h[l].copy() for l in layers}
        
        for sev in severities:
            img = apply_corruption_at_severity(base_img, ctype, sev)
            h = extract_hidden(model, processor, img, prompt, layers)
            
            for l in layers:
                dist_from_clean = cosine_dist(h[l], clean_h[l])
                # Step distance from previous severity
                step_dist = cosine_dist(h[l], prev_h[l])
                
                trajectory[f"L{l}"].append({
                    "severity": sev,
                    "dist_from_clean": round(dist_from_clean, 6),
                    "step_dist": round(step_dist, 6),
                })
                prev_h[l] = h[l].copy()
            
            print(f"  {ctype} sev={sev:.2f}: L1={trajectory['L1'][-1]['dist_from_clean']:.6f}, L3={trajectory['L3'][-1]['dist_from_clean']:.6f}")
        
        results[ctype] = trajectory

    # Compute trajectory properties
    print("\n--- Trajectory properties ---")
    properties = {}
    for ctype in corruption_types:
        props = {}
        for l_key in [f"L{l}" for l in layers]:
            traj = results[ctype][l_key]
            dists = [t['dist_from_clean'] for t in traj]
            
            # Find threshold crossing (first severity where dist > 0.0001)
            threshold_sev = None
            for t in traj:
                if t['dist_from_clean'] > 0.0001:
                    threshold_sev = t['severity']
                    break
            
            # Linearity: correlation between severity and distance
            sevs = [t['severity'] for t in traj]
            corr = np.corrcoef(sevs, dists)[0, 1] if np.std(dists) > 0 else 0
            
            props[l_key] = {
                "max_dist": round(max(dists), 6),
                "threshold_severity": threshold_sev,
                "linearity": round(float(corr), 4),
            }
        properties[ctype] = props
        print(f"  {ctype}: {props}")

    output = {
        "experiment": "embedding_trajectory",
        "experiment_number": 218,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "severities": severities,
        "layers": [1, 3],
        "trajectories": results,
        "properties": properties,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/trajectory_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
