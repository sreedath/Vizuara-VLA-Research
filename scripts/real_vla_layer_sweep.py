"""
Comprehensive Layer Sweep for OOD Detection.

Previous experiments used layer 3 for OOD detection. This experiment
systematically evaluates ALL transformer layers (0-32) to determine
which layers are best for detection, and whether combining multiple
layers improves detection.

Analyses:
1. Per-Layer AUROC: cosine distance from clean centroid for each layer
2. Per-Layer Separation Ratio: mean_between / mean_within
3. Multi-Layer Fusion: concatenated embeddings for selected layer pairs/triplets
4. Layer-Specific Corruption Sensitivity: which corruption is easiest at each layer
5. Low-Severity Detection (severity=0.1): which layers detect subtle corruptions

Experiment 450 in the CalibDrive series.
"""
import torch
import json
import os
import sys
import numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1000, 2000, 3000, 4000]
CORRUPTIONS = ["fog", "night", "noise", "blur"]
SEVERITY_MAIN = 1.0
SEVERITY_LOW = 0.1

# Predefined fusion combinations; best+second_best is added dynamically.
FUSION_PAIRS = [
    ("L0+L3",        [0, 3]),
    ("L3+L15",       [3, 15]),
    ("L3+L31",       [3, 31]),
    ("L0+L15+L31",   [0, 15, 31]),
]


# ---------------------------------------------------------------------------
# Image generation helpers
# ---------------------------------------------------------------------------

def make_image(seed=42):
    rng = np.random.RandomState(seed)
    return Image.fromarray(rng.randint(0, 255, (224, 224, 3), dtype=np.uint8))


def apply_corruption(image, ctype, severity=1.0):
    arr = np.array(image).astype(np.float32) / 255.0
    if ctype == "fog":
        arr = arr * (1 - 0.6 * severity) + 0.6 * severity
    elif ctype == "night":
        arr = arr * max(0.01, 1.0 - 0.95 * severity)
    elif ctype == "noise":
        arr = arr + np.random.RandomState(42).randn(*arr.shape) * 0.3 * severity
        arr = np.clip(arr, 0, 1)
    elif ctype == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=10 * severity))
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    print("Loading openvla/openvla-7b ...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b", trust_remote_code=True
    )
    model.eval()
    print("Model loaded.", flush=True)
    return model, processor


# ---------------------------------------------------------------------------
# Efficient single-pass layer extraction
# ---------------------------------------------------------------------------

def extract_all_layers(model, processor, image, prompt):
    """Single forward pass returning last-token embedding for every layer."""
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    # fwd.hidden_states is tuple of (batch, seq, dim) for each layer
    # Return last token from each layer
    all_layers = []
    for h in fwd.hidden_states:
        all_layers.append(h[0, -1, :].float().cpu().numpy())
    return all_layers  # list of length n_layers, each (dim,)


# ---------------------------------------------------------------------------
# Metrics (no sklearn dependency)
# ---------------------------------------------------------------------------

def cosine_distance(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
    return float(1.0 - np.dot(a, b) / denom)


def auroc(scores, labels):
    labels = np.array(labels)
    scores = np.array(scores)
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted) / n_pos
    fp = np.cumsum(1 - labels_sorted) / n_neg
    fp = np.concatenate([[0.0], fp])
    tp = np.concatenate([[0.0], tp])
    return float(np.trapezoid(tp, fp)) if hasattr(np, "trapezoid") else float(np.trapz(tp, fp))


def compute_auroc_for_layer(clean_embeds, corrupt_embeds_by_type):
    """
    Returns overall AUROC and per-corruption AUROC for a single layer.

    clean_embeds: list of np arrays (ID samples)
    corrupt_embeds_by_type: dict {ctype: list of np arrays} (OOD samples)
    """
    centroid = np.mean(clean_embeds, axis=0)
    clean_dists = [cosine_distance(e, centroid) for e in clean_embeds]

    per_corruption_auroc = {}
    all_ood_dists = []

    for ctype, ood_embeds in corrupt_embeds_by_type.items():
        ood_dists = [cosine_distance(e, centroid) for e in ood_embeds]
        scores = np.array(clean_dists + ood_dists)
        labels = np.array([0] * len(clean_dists) + [1] * len(ood_dists))
        per_corruption_auroc[ctype] = auroc(scores, labels)
        all_ood_dists.extend(ood_dists)

    all_scores = np.array(clean_dists + all_ood_dists)
    all_labels = np.array([0] * len(clean_dists) + [1] * len(all_ood_dists))
    overall = auroc(all_scores, all_labels)

    return overall, per_corruption_auroc


def compute_separation_ratio(clean_embeds, all_ood_embeds):
    """mean_between_dist / mean_within_dist. Higher = better separation."""
    centroid = np.mean(clean_embeds, axis=0)
    within_dists = [cosine_distance(e, centroid) for e in clean_embeds]
    between_dists = [cosine_distance(e, centroid) for e in all_ood_embeds]
    mean_within = float(np.mean(within_dists)) if within_dists else 1e-10
    mean_between = float(np.mean(between_dists)) if between_dists else 0.0
    return mean_between / (mean_within + 1e-10)


# ---------------------------------------------------------------------------
# Data collection (all layers in one forward pass per image)
# ---------------------------------------------------------------------------

def collect_embeddings(model, processor, prompt, seeds, corruptions, severity):
    """
    Runs a single forward pass per image and collects hidden states for all layers.

    Returns:
        clean_by_layer: list[n_layers] of list of np arrays (ID embeddings)
        corrupt_by_layer_by_type: list[n_layers] of dict {ctype: list of np arrays}
        n_layers: int
    """
    clean_images = [make_image(seed) for seed in seeds]

    print(f"  Collecting clean embeddings ({len(seeds)} scenes)...", flush=True)
    clean_by_layer = None
    for i, img in enumerate(clean_images):
        layer_embeds = extract_all_layers(model, processor, img, prompt)
        if clean_by_layer is None:
            clean_by_layer = [[] for _ in range(len(layer_embeds))]
        for li, emb in enumerate(layer_embeds):
            clean_by_layer[li].append(emb)
        if (i + 1) % 4 == 0:
            print(f"    clean scene {i + 1}/{len(seeds)}", flush=True)

    n_layers = len(clean_by_layer)
    corrupt_by_layer_by_type = [{} for _ in range(n_layers)]

    for ctype in corruptions:
        print(f"  Collecting '{ctype}' corruption (severity={severity})...", flush=True)
        for li in range(n_layers):
            corrupt_by_layer_by_type[li][ctype] = []

        for i, (seed, img) in enumerate(zip(seeds, clean_images)):
            corrupted = apply_corruption(img, ctype, severity)
            layer_embeds = extract_all_layers(model, processor, corrupted, prompt)
            for li, emb in enumerate(layer_embeds):
                corrupt_by_layer_by_type[li][ctype].append(emb)
            if (i + 1) % 4 == 0:
                print(f"    {ctype} scene {i + 1}/{len(seeds)}", flush=True)

    return clean_by_layer, corrupt_by_layer_by_type, n_layers


# ---------------------------------------------------------------------------
# Analysis 1: Per-Layer AUROC
# ---------------------------------------------------------------------------

def analysis_per_layer_auroc(clean_by_layer, corrupt_by_layer_by_type, n_layers):
    print("\n[Analysis 1] Per-Layer AUROC (cosine distance from centroid)", flush=True)
    results = {}
    for li in range(n_layers):
        overall, per_corr = compute_auroc_for_layer(
            clean_by_layer[li], corrupt_by_layer_by_type[li]
        )
        results[li] = {
            "layer": li,
            "overall_auroc": overall,
            "per_corruption_auroc": per_corr,
        }
        corr_str = "  ".join(f"{k}={v:.3f}" for k, v in per_corr.items())
        print(f"  Layer {li:2d}: overall={overall:.4f}  [{corr_str}]", flush=True)

    best_layer = max(results, key=lambda x: results[x]["overall_auroc"])
    print(
        f"  --> Best single layer: {best_layer} "
        f"(AUROC={results[best_layer]['overall_auroc']:.4f})",
        flush=True,
    )
    return results


# ---------------------------------------------------------------------------
# Analysis 2: Per-Layer Separation Ratio
# ---------------------------------------------------------------------------

def analysis_separation_ratio(clean_by_layer, corrupt_by_layer_by_type, n_layers):
    print("\n[Analysis 2] Per-Layer Separation Ratio (mean_between / mean_within)", flush=True)
    results = {}
    for li in range(n_layers):
        all_ood = []
        for embeds in corrupt_by_layer_by_type[li].values():
            all_ood.extend(embeds)
        ratio = compute_separation_ratio(clean_by_layer[li], all_ood)
        results[li] = {"layer": li, "separation_ratio": ratio}
        print(f"  Layer {li:2d}: sep_ratio={ratio:.4f}", flush=True)

    best = max(results, key=lambda x: results[x]["separation_ratio"])
    print(
        f"  --> Best separation: Layer {best} "
        f"(ratio={results[best]['separation_ratio']:.4f})",
        flush=True,
    )
    return results


# ---------------------------------------------------------------------------
# Analysis 3: Multi-Layer Fusion
# ---------------------------------------------------------------------------

def analysis_multi_layer_fusion(
    clean_by_layer, corrupt_by_layer_by_type, n_layers, auroc_results
):
    print("\n[Analysis 3] Multi-Layer Fusion (concatenated embeddings)", flush=True)

    # Determine best and second-best layers from analysis 1
    sorted_by_auroc = sorted(
        auroc_results.keys(),
        key=lambda x: auroc_results[x]["overall_auroc"],
        reverse=True,
    )
    best_layer = sorted_by_auroc[0]
    second_best = sorted_by_auroc[1] if len(sorted_by_auroc) > 1 else best_layer

    combos = list(FUSION_PAIRS)
    combos.append(
        (f"best+second_best (L{best_layer}+L{second_best})", [best_layer, second_best])
    )

    results = {}
    for combo_name, layer_indices in combos:
        valid = [li for li in layer_indices if li < n_layers]
        if len(valid) < len(layer_indices):
            print(f"  [{combo_name}] skipped (layer index out of range)", flush=True)
            continue

        n_scenes = len(clean_by_layer[0])
        clean_concat = [
            np.concatenate([clean_by_layer[li][i] for li in valid])
            for i in range(n_scenes)
        ]

        corrupt_concat_by_type = {}
        for ctype in corrupt_by_layer_by_type[0].keys():
            n_samples = len(corrupt_by_layer_by_type[0][ctype])
            corrupt_concat_by_type[ctype] = [
                np.concatenate([corrupt_by_layer_by_type[li][ctype][i] for li in valid])
                for i in range(n_samples)
            ]

        overall, per_corr = compute_auroc_for_layer(clean_concat, corrupt_concat_by_type)
        dim = sum(clean_by_layer[li][0].shape[0] for li in valid)
        results[combo_name] = {
            "layers": valid,
            "dim": dim,
            "overall_auroc": overall,
            "per_corruption_auroc": per_corr,
        }
        corr_str = "  ".join(f"{k}={v:.3f}" for k, v in per_corr.items())
        print(
            f"  [{combo_name}] dim={dim}: overall={overall:.4f}  [{corr_str}]",
            flush=True,
        )

    return results


# ---------------------------------------------------------------------------
# Analysis 4: Layer-Specific Corruption Sensitivity
# ---------------------------------------------------------------------------

def analysis_corruption_sensitivity(auroc_results, corruptions):
    print("\n[Analysis 4] Layer-Specific Corruption Sensitivity (rank by AUROC)", flush=True)
    per_layer = {}
    for li, res in auroc_results.items():
        per_corr = res["per_corruption_auroc"]
        ranked = sorted(per_corr.items(), key=lambda x: x[1], reverse=True)
        per_layer[li] = {
            "layer": li,
            "corruption_ranking": [(c, float(v)) for c, v in ranked],
            "easiest": ranked[0][0] if ranked else None,
            "hardest": ranked[-1][0] if ranked else None,
        }

    print("  Best layer per corruption type:", flush=True)
    best_layer_per_corruption = {}
    for ctype in corruptions:
        best_li = max(
            auroc_results.keys(),
            key=lambda li: auroc_results[li]["per_corruption_auroc"].get(ctype, 0.0),
        )
        best_score = auroc_results[best_li]["per_corruption_auroc"].get(ctype, 0.0)
        best_layer_per_corruption[ctype] = {"best_layer": best_li, "auroc": best_score}
        print(f"    {ctype}: Layer {best_li} (AUROC={best_score:.4f})", flush=True)

    return {"per_layer": per_layer, "best_layer_per_corruption": best_layer_per_corruption}


# ---------------------------------------------------------------------------
# Analysis 5: Low-Severity Detection per Layer
# ---------------------------------------------------------------------------

def analysis_low_severity(model, processor, prompt, seeds, corruptions):
    print(
        f"\n[Analysis 5] Low-Severity Detection (severity={SEVERITY_LOW}), "
        "reveals which layers catch subtle corruptions",
        flush=True,
    )
    clean_by_layer, corrupt_by_layer_by_type, n_layers = collect_embeddings(
        model, processor, prompt, seeds, corruptions, SEVERITY_LOW
    )

    results = {}
    for li in range(n_layers):
        overall, per_corr = compute_auroc_for_layer(
            clean_by_layer[li], corrupt_by_layer_by_type[li]
        )
        results[li] = {
            "layer": li,
            "overall_auroc": overall,
            "per_corruption_auroc": per_corr,
        }
        corr_str = "  ".join(f"{k}={v:.3f}" for k, v in per_corr.items())
        print(f"  Layer {li:2d}: overall={overall:.4f}  [{corr_str}]", flush=True)

    best_layer = max(results, key=lambda x: results[x]["overall_auroc"])
    print(
        f"  --> Best layer at low severity: {best_layer} "
        f"(AUROC={results[best_layer]['overall_auroc']:.4f})",
        flush=True,
    )
    return results, n_layers


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70, flush=True)
    print("EXPERIMENT 450: COMPREHENSIVE LAYER SWEEP FOR OOD DETECTION", flush=True)
    print("=" * 70, flush=True)
    print(f"Scenes (seeds): {SEEDS}", flush=True)
    print(f"Corruptions: {CORRUPTIONS}", flush=True)
    print(f"Main severity: {SEVERITY_MAIN}  |  Low severity: {SEVERITY_LOW}", flush=True)
    print(
        f"Predefined fusion combos: {[c[0] for c in FUSION_PAIRS]} + best+second_best",
        flush=True,
    )
    print(flush=True)

    model, processor = load_model()
    prompt = "In: What action should the robot take to pick up the object?\nOut:"

    # -----------------------------------------------------------------------
    # Main data collection at full severity.
    # A single forward pass per image extracts all layers simultaneously.
    # -----------------------------------------------------------------------
    print("\n--- Main Data Collection (severity=1.0) ---", flush=True)
    clean_by_layer, corrupt_by_layer_by_type, n_layers = collect_embeddings(
        model, processor, prompt, SEEDS, CORRUPTIONS, SEVERITY_MAIN
    )
    print(f"Total layers detected: {n_layers}", flush=True)

    # -----------------------------------------------------------------------
    # Analyses on full-severity data
    # -----------------------------------------------------------------------
    auroc_results = analysis_per_layer_auroc(
        clean_by_layer, corrupt_by_layer_by_type, n_layers
    )

    sep_results = analysis_separation_ratio(
        clean_by_layer, corrupt_by_layer_by_type, n_layers
    )

    fusion_results = analysis_multi_layer_fusion(
        clean_by_layer, corrupt_by_layer_by_type, n_layers, auroc_results
    )

    sensitivity_results = analysis_corruption_sensitivity(auroc_results, CORRUPTIONS)

    # -----------------------------------------------------------------------
    # Low-severity analysis (separate data collection at severity=0.1)
    # -----------------------------------------------------------------------
    low_sev_results, _ = analysis_low_severity(
        model, processor, prompt, SEEDS, CORRUPTIONS
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    sorted_layers = sorted(
        auroc_results.keys(),
        key=lambda li: auroc_results[li]["overall_auroc"],
        reverse=True,
    )
    print("Top 5 layers (main severity, overall AUROC):", flush=True)
    for rank, li in enumerate(sorted_layers[:5], 1):
        print(
            f"  #{rank}: Layer {li:2d}  AUROC={auroc_results[li]['overall_auroc']:.4f}  "
            f"sep_ratio={sep_results[li]['separation_ratio']:.4f}",
            flush=True,
        )

    sorted_fusion = sorted(
        fusion_results.items(),
        key=lambda x: x[1]["overall_auroc"],
        reverse=True,
    )
    print("\nTop fusion combos:", flush=True)
    for name, res in sorted_fusion[:3]:
        print(f"  [{name}]: AUROC={res['overall_auroc']:.4f}", flush=True)

    sorted_low = sorted(
        low_sev_results.keys(),
        key=lambda li: low_sev_results[li]["overall_auroc"],
        reverse=True,
    )
    print("\nBest layer at low severity:", flush=True)
    if sorted_low:
        li = sorted_low[0]
        print(
            f"  Layer {li:2d}: AUROC={low_sev_results[li]['overall_auroc']:.4f}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # Persist results as valid JSON
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "layer_sweep",
        "experiment_number": 450,
        "timestamp": timestamp,
        "config": {
            "seeds": SEEDS,
            "corruptions": CORRUPTIONS,
            "severity_main": SEVERITY_MAIN,
            "severity_low": SEVERITY_LOW,
            "n_layers": n_layers,
            "prompt": prompt,
        },
        "analysis_1_per_layer_auroc": {
            str(li): {
                "layer": v["layer"],
                "overall_auroc": v["overall_auroc"],
                "per_corruption_auroc": v["per_corruption_auroc"],
            }
            for li, v in auroc_results.items()
        },
        "analysis_2_separation_ratio": {
            str(li): v for li, v in sep_results.items()
        },
        "analysis_3_multi_layer_fusion": fusion_results,
        "analysis_4_corruption_sensitivity": {
            "per_layer": {
                str(li): v
                for li, v in sensitivity_results["per_layer"].items()
            },
            "best_layer_per_corruption": sensitivity_results["best_layer_per_corruption"],
        },
        "analysis_5_low_severity": {
            str(li): {
                "layer": v["layer"],
                "overall_auroc": v["overall_auroc"],
                "per_corruption_auroc": v["per_corruption_auroc"],
            }
            for li, v in low_sev_results.items()
        },
        "summary": {
            "top5_layers_main": [
                {
                    "layer": li,
                    "overall_auroc": auroc_results[li]["overall_auroc"],
                    "separation_ratio": sep_results[li]["separation_ratio"],
                }
                for li in sorted_layers[:5]
            ],
            "top_fusion_combos": [
                {
                    "name": name,
                    "layers": res["layers"],
                    "overall_auroc": res["overall_auroc"],
                }
                for name, res in sorted_fusion[:3]
            ],
            "best_layer_low_severity": {
                "layer": sorted_low[0] if sorted_low else None,
                "overall_auroc": (
                    low_sev_results[sorted_low[0]]["overall_auroc"] if sorted_low else None
                ),
            },
        },
    }

    output_path = os.path.join(RESULTS_DIR, f"layer_sweep_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
