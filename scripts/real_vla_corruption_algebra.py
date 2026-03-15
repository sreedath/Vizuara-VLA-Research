#!/usr/bin/env python3
"""Experiment 323: Corruption Composition Algebra (Real OpenVLA-7B)

Tests whether corruption effects compose linearly in embedding space:
1. Vector additivity: Is fog+noise ≈ fog_vec + noise_vec?
2. Scalar linearity: Is 2×fog ≈ 2×fog_vec?
3. Commutativity: Is fog→noise = noise→fog?
4. Cancellation: Can anti-corruption vectors be constructed?
5. Superposition decomposition: Can composite corruptions be unmixed?
6. Basis representation: Do corruption vectors form a useful basis?
"""

import json, time, os, sys
import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor

def extract_hidden(model, processor, image, prompt, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def apply_corruption(image, ctype, severity=1.0):
    arr = np.array(image).astype(np.float32) / 255.0
    if ctype == 'fog':
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == 'night':
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == 'noise':
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == 'blur':
        return image.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))

def apply_sequential(image, corruptions):
    """Apply corruptions sequentially: [(type, severity), ...]"""
    result = image.copy()
    for ctype, sev in corruptions:
        result = apply_corruption(result, ctype, sev)
    return result

def cosine_dist(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return 1.0 - dot / (na * nb)

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return np.dot(a, b) / (na * nb)

def main():
    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model.eval()

    np.random.seed(42)
    pixels = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    base_img = Image.fromarray(pixels)
    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    results = {}

    # ========== 1. Extract basis vectors ==========
    print("\n=== Extracting corruption basis vectors ===")
    clean_emb = extract_hidden(model, processor, base_img, prompt)

    ctypes = ['fog', 'night', 'noise', 'blur']
    severities = [0.25, 0.5, 0.75, 1.0]

    # Get embeddings at multiple severities
    emb_map = {}
    vec_map = {}  # displacement vectors from clean
    for ct in ctypes:
        emb_map[ct] = {}
        vec_map[ct] = {}
        for sev in severities:
            img = apply_corruption(base_img, ct, sev)
            emb = extract_hidden(model, processor, img, prompt)
            emb_map[ct][sev] = emb
            vec_map[ct][sev] = emb - clean_emb
            print(f"  {ct} sev={sev}: d={cosine_dist(clean_emb, emb):.6f}")

    # ========== 2. Scalar linearity ==========
    print("\n=== Scalar Linearity ===")
    scalar_results = {}
    for ct in ctypes:
        v25 = vec_map[ct][0.25]
        v50 = vec_map[ct][0.5]
        v75 = vec_map[ct][0.75]
        v100 = vec_map[ct][1.0]

        # Test: is v50 ≈ 2*v25? v75 ≈ 3*v25? v100 ≈ 4*v25?
        pred_50 = 2 * v25
        pred_75 = 3 * v25
        pred_100 = 4 * v25

        sim_50 = cosine_sim(v50, pred_50)
        sim_75 = cosine_sim(v75, pred_75)
        sim_100 = cosine_sim(v100, pred_100)

        # Ratio of norms
        nr_50 = np.linalg.norm(v50) / (2 * np.linalg.norm(v25)) if np.linalg.norm(v25) > 0 else 0
        nr_75 = np.linalg.norm(v75) / (3 * np.linalg.norm(v25)) if np.linalg.norm(v25) > 0 else 0
        nr_100 = np.linalg.norm(v100) / (4 * np.linalg.norm(v25)) if np.linalg.norm(v25) > 0 else 0

        scalar_results[ct] = {
            'cos_sim_2x': float(sim_50),
            'cos_sim_3x': float(sim_75),
            'cos_sim_4x': float(sim_100),
            'norm_ratio_2x': float(nr_50),
            'norm_ratio_3x': float(nr_75),
            'norm_ratio_4x': float(nr_100),
        }
        print(f"  {ct}: sim(v50, 2*v25)={sim_50:.4f}, sim(v75, 3*v25)={sim_75:.4f}, sim(v100, 4*v25)={sim_100:.4f}")
        print(f"         norm_ratio: 2x={nr_50:.4f}, 3x={nr_75:.4f}, 4x={nr_100:.4f}")

    results['scalar_linearity'] = scalar_results

    # ========== 3. Vector additivity ==========
    print("\n=== Vector Additivity ===")
    additivity_results = {}
    pairs = [('fog', 'noise'), ('fog', 'night'), ('fog', 'blur'),
             ('night', 'noise'), ('night', 'blur'), ('noise', 'blur')]

    for ct1, ct2 in pairs:
        sev = 0.5
        # Predicted composite = clean + v1 + v2
        v1 = vec_map[ct1][sev]
        v2 = vec_map[ct2][sev]
        predicted_emb = clean_emb + v1 + v2

        # Actual composite: apply ct1 then ct2
        comp_img_12 = apply_sequential(base_img, [(ct1, sev), (ct2, sev)])
        actual_emb_12 = extract_hidden(model, processor, comp_img_12, prompt)
        actual_vec_12 = actual_emb_12 - clean_emb

        # Apply ct2 then ct1
        comp_img_21 = apply_sequential(base_img, [(ct2, sev), (ct1, sev)])
        actual_emb_21 = extract_hidden(model, processor, comp_img_21, prompt)
        actual_vec_21 = actual_emb_21 - clean_emb

        pred_vec = v1 + v2

        # Direction similarity
        dir_sim_12 = cosine_sim(pred_vec, actual_vec_12)
        dir_sim_21 = cosine_sim(pred_vec, actual_vec_21)

        # Commutativity: is ct1→ct2 = ct2→ct1?
        commute_sim = cosine_sim(actual_vec_12, actual_vec_21)

        # Norm ratios
        pred_norm = np.linalg.norm(pred_vec)
        actual_norm_12 = np.linalg.norm(actual_vec_12)
        actual_norm_21 = np.linalg.norm(actual_vec_21)

        additivity_results[f"{ct1}+{ct2}"] = {
            'direction_sim_12': float(dir_sim_12),
            'direction_sim_21': float(dir_sim_21),
            'commutativity_sim': float(commute_sim),
            'norm_pred': float(pred_norm),
            'norm_actual_12': float(actual_norm_12),
            'norm_actual_21': float(actual_norm_21),
            'norm_ratio_12': float(actual_norm_12 / pred_norm) if pred_norm > 0 else 0,
        }
        print(f"  {ct1}+{ct2}: dir_sim={dir_sim_12:.4f}/{dir_sim_21:.4f}, commute={commute_sim:.4f}, norm_ratio={actual_norm_12/pred_norm:.4f}")

    results['vector_additivity'] = additivity_results

    # ========== 4. Basis representation ==========
    print("\n=== Corruption Basis Representation ===")
    # Use unit corruption vectors as a basis
    basis_vecs = []
    basis_labels = []
    for ct in ctypes:
        v = vec_map[ct][1.0]
        norm = np.linalg.norm(v)
        if norm > 0:
            basis_vecs.append(v / norm)
            basis_labels.append(ct)

    basis_matrix = np.array(basis_vecs)  # 4 x 4096

    # Project composite corruption vectors onto this basis
    basis_proj = {}
    for ct1, ct2 in pairs:
        sev = 0.5
        comp_img = apply_sequential(base_img, [(ct1, sev), (ct2, sev)])
        comp_emb = extract_hidden(model, processor, comp_img, prompt)
        comp_vec = comp_emb - clean_emb

        # Project onto corruption basis
        coeffs = basis_matrix @ comp_vec  # 4 coefficients

        # Reconstruction
        recon = basis_matrix.T @ coeffs
        recon_error = np.linalg.norm(comp_vec - recon) / np.linalg.norm(comp_vec) if np.linalg.norm(comp_vec) > 0 else 0

        basis_proj[f"{ct1}+{ct2}"] = {
            'coefficients': {bl: float(c) for bl, c in zip(basis_labels, coeffs)},
            'reconstruction_error': float(recon_error),
            'dominant_component': basis_labels[int(np.argmax(np.abs(coeffs)))],
        }
        print(f"  {ct1}+{ct2}: coeffs={dict(zip(basis_labels, [f'{c:.4f}' for c in coeffs]))}, recon_err={recon_error:.4f}")

    results['basis_representation'] = basis_proj

    # ========== 5. Cancellation experiment ==========
    print("\n=== Cancellation Experiment ===")
    cancel_results = {}
    for ct in ctypes:
        # Can we construct an "anti-corruption" that cancels the effect?
        v = vec_map[ct][1.0]

        # The anti-embedding would be clean - v (going in opposite direction)
        anti_emb = clean_emb - v

        # Distance from anti_emb to actual corrupted embedding
        corrupt_emb = emb_map[ct][1.0]
        d_anti_to_corrupt = cosine_dist(anti_emb, corrupt_emb)
        d_anti_to_clean = cosine_dist(anti_emb, clean_emb)
        d_corrupt_to_clean = cosine_dist(corrupt_emb, clean_emb)

        # Angle between corruption vector and its "reflection"
        anti_vec = anti_emb - clean_emb  # = -v
        angle = np.degrees(np.arccos(np.clip(cosine_sim(v, anti_vec), -1, 1)))

        cancel_results[ct] = {
            'd_anti_to_corrupt': float(d_anti_to_corrupt),
            'd_anti_to_clean': float(d_anti_to_clean),
            'd_corrupt_to_clean': float(d_corrupt_to_clean),
            'angle_v_antiv': float(angle),
            'symmetry_ratio': float(d_anti_to_clean / d_corrupt_to_clean) if d_corrupt_to_clean > 0 else 0,
        }
        print(f"  {ct}: d(anti,corrupt)={d_anti_to_corrupt:.6f}, d(anti,clean)={d_anti_to_clean:.6f}, angle={angle:.1f}°")

    results['cancellation'] = cancel_results

    # ========== 6. Interpolation paths ==========
    print("\n=== Interpolation Between Corruptions ===")
    interp_results = {}
    for ct1, ct2 in [('fog', 'night'), ('fog', 'blur'), ('noise', 'blur')]:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        path = []
        for alpha in alphas:
            # Linear interpolation in corruption vector space
            interp_vec = (1 - alpha) * vec_map[ct1][1.0] + alpha * vec_map[ct2][1.0]
            interp_emb = clean_emb + interp_vec

            d_to_clean = cosine_dist(clean_emb, interp_emb)
            d_to_ct1 = cosine_dist(emb_map[ct1][1.0], interp_emb)
            d_to_ct2 = cosine_dist(emb_map[ct2][1.0], interp_emb)

            path.append({
                'alpha': float(alpha),
                'd_clean': float(d_to_clean),
                'd_ct1': float(d_to_ct1),
                'd_ct2': float(d_to_ct2),
            })

        interp_results[f"{ct1}_to_{ct2}"] = path
        print(f"  {ct1}→{ct2}: d_clean at α=0.5: {path[2]['d_clean']:.6f}")

    results['interpolation'] = interp_results

    # ========== 7. Gram matrix of corruption vectors ==========
    print("\n=== Corruption Vector Gram Matrix ===")
    gram = np.zeros((4, 4))
    for i, ct1 in enumerate(ctypes):
        for j, ct2 in enumerate(ctypes):
            gram[i, j] = cosine_sim(vec_map[ct1][1.0], vec_map[ct2][1.0])

    # Eigendecomposition of gram matrix
    eigvals = np.linalg.eigvalsh(gram)

    results['gram_matrix'] = {
        'matrix': {ct1: {ct2: float(gram[i, j]) for j, ct2 in enumerate(ctypes)}
                   for i, ct1 in enumerate(ctypes)},
        'eigenvalues': [float(e) for e in sorted(eigvals, reverse=True)],
        'condition_number': float(max(eigvals) / max(min(eigvals), 1e-10)),
        'rank': int(np.sum(eigvals > 1e-6)),
    }

    print("  Gram matrix:")
    for ct in ctypes:
        row = [f"{gram[ctypes.index(ct), ctypes.index(ct2)]:.4f}" for ct2 in ctypes]
        print(f"    {ct}: {row}")
    print(f"  Eigenvalues: {sorted(eigvals, reverse=True)}")

    # ========== 8. Triple corruption ==========
    print("\n=== Triple Corruption Composition ===")
    triple_results = {}
    triples = [('fog', 'night', 'noise'), ('fog', 'night', 'blur'),
               ('fog', 'noise', 'blur'), ('night', 'noise', 'blur')]

    for ct1, ct2, ct3 in triples:
        sev = 0.5
        # Predicted: sum of individual vectors
        pred_vec = vec_map[ct1][sev] + vec_map[ct2][sev] + vec_map[ct3][sev]

        # Actual: apply sequentially
        triple_img = apply_sequential(base_img, [(ct1, sev), (ct2, sev), (ct3, sev)])
        actual_emb = extract_hidden(model, processor, triple_img, prompt)
        actual_vec = actual_emb - clean_emb

        dir_sim = cosine_sim(pred_vec, actual_vec)
        norm_ratio = np.linalg.norm(actual_vec) / np.linalg.norm(pred_vec) if np.linalg.norm(pred_vec) > 0 else 0

        # Detection: is this still detectable?
        d = cosine_dist(clean_emb, actual_emb)

        triple_results[f"{ct1}+{ct2}+{ct3}"] = {
            'direction_sim': float(dir_sim),
            'norm_ratio': float(norm_ratio),
            'cosine_dist': float(d),
            'detected': bool(d > 0),
        }
        print(f"  {ct1}+{ct2}+{ct3}: dir_sim={dir_sim:.4f}, norm_ratio={norm_ratio:.4f}, d={d:.6f}")

    results['triple_corruption'] = triple_results

    # ========== 9. Subspace angles between corruption types ==========
    print("\n=== Subspace Angles ===")
    angle_results = {}
    for ct1 in ctypes:
        angle_results[ct1] = {}
        for ct2 in ctypes:
            v1 = vec_map[ct1][1.0]
            v2 = vec_map[ct2][1.0]
            sim = cosine_sim(v1, v2)
            angle = np.degrees(np.arccos(np.clip(sim, -1, 1)))
            angle_results[ct1][ct2] = float(angle)

    results['subspace_angles'] = angle_results
    print("  Pairwise angles (degrees):")
    for ct in ctypes:
        angles = [f"{angle_results[ct][ct2]:.1f}" for ct2 in ctypes]
        print(f"    {ct}: {angles}")

    # ========== 10. Severity algebra ==========
    print("\n=== Severity Algebra ===")
    sev_algebra = {}
    for ct in ctypes:
        # Is sev(0.25) + sev(0.25) ≈ sev(0.50) in direction?
        v25 = vec_map[ct][0.25]
        v50 = vec_map[ct][0.5]
        v75 = vec_map[ct][0.75]

        # Direction consistency across severities
        if np.linalg.norm(v25) > 0 and np.linalg.norm(v50) > 0:
            dir_25_50 = cosine_sim(v25, v50)
        else:
            dir_25_50 = 0
        if np.linalg.norm(v25) > 0 and np.linalg.norm(v75) > 0:
            dir_25_75 = cosine_sim(v25, v75)
        else:
            dir_25_75 = 0
        if np.linalg.norm(v50) > 0 and np.linalg.norm(v75) > 0:
            dir_50_75 = cosine_sim(v50, v75)
        else:
            dir_50_75 = 0

        # Norm scaling: is ||v(s)|| ∝ s?
        norms = [np.linalg.norm(vec_map[ct][s]) for s in severities]
        norm_ratios = [n / norms[0] if norms[0] > 0 else 0 for n in norms]

        sev_algebra[ct] = {
            'direction_consistency': {
                '25_vs_50': float(dir_25_50),
                '25_vs_75': float(dir_25_75),
                '50_vs_75': float(dir_50_75),
            },
            'norms': {str(s): float(n) for s, n in zip(severities, norms)},
            'norm_ratios': {str(s): float(r) for s, r in zip(severities, norm_ratios)},
            'linear_scaling': float(np.corrcoef(severities, norms)[0, 1]) if norms[0] > 0 else 0,
        }
        print(f"  {ct}: dir_consistency={dir_25_50:.4f}/{dir_25_75:.4f}/{dir_50_75:.4f}, norm_corr={sev_algebra[ct]['linear_scaling']:.4f}")

    results['severity_algebra'] = sev_algebra

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"experiments/corruption_algebra_{ts}.json"

    def convert(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def recursive_convert(d):
        if isinstance(d, dict):
            return {k: recursive_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [recursive_convert(x) for x in d]
        return convert(d)

    results = recursive_convert(results)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
