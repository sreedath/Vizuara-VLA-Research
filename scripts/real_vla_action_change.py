"""
Experiment 224: Action Token Corruption Analysis
Do the predicted action tokens actually CHANGE under visual corruption?
Measures action deviation to understand if the model produces wrong actions
when corrupted — establishing that OOD detection prevents real harm.
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

def apply_corruption(img, name, rng):
    arr = np.array(img, dtype=np.float32)
    if name == 'fog':
        fog = np.full_like(arr, 200)
        arr = arr * 0.4 + fog * 0.6
    elif name == 'night':
        arr = arr * 0.15
    elif name == 'blur':
        return img.filter(ImageFilter.GaussianBlur(radius=5))
    elif name == 'noise':
        arr = arr + rng.normal(0, 40, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

def extract_actions(model, processor, image, prompt):
    """Extract predicted action tokens from OpenVLA."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=7,
            do_sample=False,
        )
    # Get the generated token IDs (excluding input tokens)
    input_len = inputs['input_ids'].shape[1]
    gen_ids = output[0, input_len:].cpu().numpy()

    # OpenVLA action tokens are in range [31744, 31999] (256 bins per dimension)
    # Convert to continuous actions: (token_id - 31744) / 255 mapped to action range
    actions = []
    for tid in gen_ids:
        if 31744 <= tid <= 31999:
            actions.append(int(tid))
        else:
            actions.append(int(tid))  # non-action token

    return gen_ids.tolist(), actions

def cosine_dist(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_auroc(id_scores, ood_scores):
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    n_id, n_ood = len(id_scores), len(ood_scores)
    if n_id == 0 or n_ood == 0:
        return 0.5
    count = sum(float(np.sum(o > id_scores) + 0.5 * np.sum(o == id_scores)) for o in ood_scores)
    return count / (n_id * n_ood)

def main():
    print("=" * 60)
    print("Experiment 224: Action Token Corruption Analysis")
    print("=" * 60)

    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("Loading OpenVLA-7B...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    prompt = "In: What action should the robot take to drive forward?\nOut:"
    n_images = 8
    rng = np.random.default_rng(42)
    base_imgs = [make_driving_image() for _ in range(n_images)]
    corruption_types = ['fog', 'night', 'blur', 'noise']

    # Get clean actions
    print("\n--- Clean image actions ---")
    clean_actions = []
    for i in range(n_images):
        gen_ids, actions = extract_actions(model, processor, base_imgs[i], prompt)
        clean_actions.append(actions)
        print(f"  Image {i}: tokens={gen_ids[:7]}")

    # Get corrupted actions
    print("\n--- Corrupted image actions ---")
    results = {}
    for ctype in corruption_types:
        corr_actions = []
        n_changed = 0
        total_deviation = 0
        per_dim_changes = [0] * 7

        for i in range(n_images):
            img = apply_corruption(base_imgs[i], ctype, rng)
            gen_ids, actions = extract_actions(model, processor, img, prompt)
            corr_actions.append(actions)

            # Compare to clean
            n_dims = min(len(clean_actions[i]), len(actions), 7)
            changed = False
            for d in range(n_dims):
                if clean_actions[i][d] != actions[d]:
                    changed = True
                    per_dim_changes[d] += 1
                    deviation = abs(clean_actions[i][d] - actions[d])
                    total_deviation += deviation

            if changed:
                n_changed += 1

            print(f"  {ctype} img {i}: tokens={gen_ids[:7]} | changed={changed}")

        results[ctype] = {
            "n_images_with_changed_actions": n_changed,
            "fraction_changed": round(n_changed / n_images, 4),
            "per_dim_change_count": per_dim_changes,
            "mean_total_deviation": round(total_deviation / n_images, 4),
            "sample_clean": clean_actions[0][:7],
            "sample_corrupted": corr_actions[0][:7],
        }
        print(f"  Summary: {n_changed}/{n_images} images had action changes")

    # Also extract hidden states for comparison
    print("\n--- Hidden state distances for context ---")
    layers = [1, 3]
    hidden_results = {}
    for l in layers:
        for i in range(min(3, n_images)):
            inputs = processor(prompt, base_imgs[i]).to(model.device, dtype=torch.bfloat16)
            with torch.no_grad():
                fwd = model(**inputs, output_hidden_states=True)
            clean_h = fwd.hidden_states[l][0, -1, :].float().cpu().numpy()

            for ctype in corruption_types:
                img = apply_corruption(base_imgs[i], ctype, rng)
                inputs = processor(prompt, img).to(model.device, dtype=torch.bfloat16)
                with torch.no_grad():
                    fwd = model(**inputs, output_hidden_states=True)
                corr_h = fwd.hidden_states[l][0, -1, :].float().cpu().numpy()
                d = cosine_dist(clean_h, corr_h)
                if f"L{l}" not in hidden_results:
                    hidden_results[f"L{l}"] = {}
                if ctype not in hidden_results[f"L{l}"]:
                    hidden_results[f"L{l}"][ctype] = []
                hidden_results[f"L{l}"][ctype].append(round(d, 6))

    # Average hidden distances
    for lk in hidden_results:
        for ctype in hidden_results[lk]:
            vals = hidden_results[lk][ctype]
            hidden_results[lk][ctype] = {"distances": vals, "mean": round(float(np.mean(vals)), 6)}

    output = {
        "experiment": "action_token_analysis",
        "experiment_number": 224,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_images": n_images,
        "corruption_types": corruption_types,
        "action_results": results,
        "hidden_state_distances": hidden_results,
    }

    out_path = f"/workspace/Vizuara-VLA-Research/experiments/action_tokens_{output['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
