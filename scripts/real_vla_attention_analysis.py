"""
Attention Pattern Analysis under Corruption.

Experiment 449: Analyzes how corruption changes the attention patterns in the
VLA model, specifically examining whether corrupted images cause different
attention distributions across visual and text tokens.

Analyses:
  1. Attention Entropy per Layer        — last-token entropy at selected layers,
                                          clean vs corrupted
  2. Visual vs Text Attention Split     — fraction of last-token attention that
                                          goes to visual tokens vs text tokens
  3. Attention Concentration            — effective number of attended tokens
                                          (exp of entropy) per layer
  4. Cross-Layer Attention Shift        — cosine similarity of last-token
                                          attention pattern (clean vs corrupted)
  5. Head-Level Analysis at Layer 3     — per-head entropy change and cosine
                                          shift for the detection layer
  6. Attention to Corruption Regions    — does the model attend more/less to
                                          visual tokens under each corruption?

Settings:
  - 8 scenes (seeds 42, 123, 456, 789, 1000, 2000, 3000, 4000)
  - 4 corruption types at severity 1.0: fog, night, noise, blur
  - Key layers: 0, 3, 15, 31 (first, detection, middle, last)
"""

import torch, json, os, sys, numpy as np
from PIL import Image, ImageFilter
from transformers import AutoModelForVision2Seq, AutoProcessor
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.join(REPO_DIR, "experiments")
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 1000, 2000, 3000, 4000]
CORRUPTIONS = ["fog", "night", "noise", "blur"]
SEVERITY = 1.0
# Layers to analyse: first, detection, middle, last
ANALYSIS_LAYERS = [0, 3, 15, 31]
HEAD_ANALYSIS_LAYER = 3  # Layer used for detailed head-level analysis
# Number of visual tokens produced by the ViT patch encoder (224/14)^2 = 256
N_VISUAL_TOKENS = 256
PROMPT = "In: What action should the robot take to pick up the object?\nOut:"
EPS = 1e-10

# ---------------------------------------------------------------------------
# Image helpers
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
# Attention extraction — memory-safe
# ---------------------------------------------------------------------------

def extract_last_token_attention(model, processor, image, prompt, layers):
    """
    Run a single forward pass with output_attentions=True, extract only the
    last-token row of the attention matrix for the requested layers, and
    immediately discard the full tensors to save GPU memory.

    Returns:
        dict[int -> np.ndarray]  layer_index -> (num_heads, seq_len) float32
        int  seq_len
    """
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_attentions=True)

    result = {}
    seq_len = None
    for l in layers:
        if l < len(fwd.attentions):
            # fwd.attentions[l]: (batch=1, num_heads, seq_len, seq_len)
            attn_last = fwd.attentions[l][0, :, -1, :]  # (num_heads, seq_len)
            arr = attn_last.float().cpu().numpy()
            result[l] = arr
            if seq_len is None:
                seq_len = arr.shape[1]
    # Explicitly free the full attention tuple before returning
    del fwd
    return result, seq_len


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def last_token_entropy(attn_last):
    """
    Compute per-head Shannon entropy of the last-token attention distribution.

    Args:
        attn_last: (num_heads, seq_len) float32

    Returns:
        (num_heads,) float64  entropy per head
    """
    a = attn_last.astype(np.float64)
    return -np.sum(a * np.log(a + EPS), axis=1)


def effective_n(entropy_per_head):
    """Effective number of attended tokens = exp(entropy)."""
    return np.exp(entropy_per_head)


def visual_attention_fraction(attn_last, n_visual):
    """
    Fraction of attention mass in the last-token row that falls on visual
    tokens (assumed to occupy positions 0..n_visual-1 of the key dimension).

    Args:
        attn_last: (num_heads, seq_len) float32
        n_visual:  int  number of visual tokens

    Returns:
        (num_heads,) float64  per-head visual fraction
    """
    a = attn_last.astype(np.float64)
    vis = np.sum(a[:, :n_visual], axis=1)
    total = np.sum(a, axis=1)
    return vis / (total + EPS)


def cosine_similarity_rows(a, b):
    """
    Compute cosine similarity between two (num_heads, seq_len) arrays,
    treating each head's row as a vector.

    Returns:
        (num_heads,) float64 per-head cosine similarity
    """
    a, b = a.astype(np.float64), b.astype(np.float64)
    dot = np.sum(a * b, axis=1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na * nb
    denom = np.where(denom < 1e-12, 1.0, denom)
    return dot / denom


# ---------------------------------------------------------------------------
# Per-scene analysis — processes one scene and returns summary stats only
# ---------------------------------------------------------------------------

def analyse_scene(model, processor, seed, layers):
    """
    Extract and compute summary statistics for one scene (clean + all corruptions).
    All attention tensors are deleted after stat extraction.

    Returns dict with keys: seq_len, num_heads, clean, corrupted
    """
    image = make_image(seed)

    # --- clean ---
    clean_attns, seq_len = extract_last_token_attention(
        model, processor, image, PROMPT, layers
    )
    num_heads = next(iter(clean_attns.values())).shape[0]

    scene_stats = {
        "seq_len": seq_len,
        "num_heads": num_heads,
        "clean": {},
        "corrupted": {c: {} for c in CORRUPTIONS},
    }

    for l in layers:
        if l not in clean_attns:
            continue
        ca = clean_attns[l]
        ent = last_token_entropy(ca)
        vis_frac = visual_attention_fraction(ca, N_VISUAL_TOKENS)
        eff = effective_n(ent)

        scene_stats["clean"][l] = {
            "entropy": ent,           # (num_heads,)
            "vis_frac": vis_frac,     # (num_heads,)
            "eff_n": eff,             # (num_heads,)
            "attn": ca,               # kept temporarily for cosine vs corrupted
        }

    # --- corrupted ---
    for ctype in CORRUPTIONS:
        c_image = apply_corruption(image, ctype, SEVERITY)
        c_attns, _ = extract_last_token_attention(
            model, processor, c_image, PROMPT, layers
        )

        for l in layers:
            if l not in c_attns or l not in clean_attns:
                continue
            ca_c = c_attns[l]
            ca_cl = clean_attns[l]

            ent_c = last_token_entropy(ca_c)
            vis_frac_c = visual_attention_fraction(ca_c, N_VISUAL_TOKENS)
            eff_c = effective_n(ent_c)
            cos_sim = cosine_similarity_rows(ca_cl, ca_c)

            scene_stats["corrupted"][ctype][l] = {
                "entropy": ent_c,
                "vis_frac": vis_frac_c,
                "eff_n": eff_c,
                "cos_sim": cos_sim,   # (num_heads,) similarity to clean
            }
        del c_attns

    # Discard raw attention arrays — keep only computed stats
    for l in layers:
        if l in scene_stats["clean"]:
            del scene_stats["clean"][l]["attn"]

    del clean_attns, c_image
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scene_stats


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_scenes(all_scene_stats, layers):
    """
    Aggregate per-scene statistics across all scenes into mean/std summaries.

    Returns structured dict ready for JSON serialisation.
    """
    agg = {
        "layers": layers,
        "n_scenes": len(all_scene_stats),
        "seq_len": all_scene_stats[0]["seq_len"],
        "num_heads": all_scene_stats[0]["num_heads"],
        "clean": {},
        "corrupted": {c: {} for c in CORRUPTIONS},
    }

    for l in layers:
        # --- clean ---
        clean_ents = np.array([s["clean"][l]["entropy"] for s in all_scene_stats
                               if l in s["clean"]])             # (scenes, heads)
        clean_vis = np.array([s["clean"][l]["vis_frac"] for s in all_scene_stats
                              if l in s["clean"]])
        clean_eff = np.array([s["clean"][l]["eff_n"] for s in all_scene_stats
                              if l in s["clean"]])

        agg["clean"][str(l)] = {
            "entropy_mean": float(np.mean(clean_ents)),
            "entropy_std": float(np.std(clean_ents)),
            "entropy_per_head_mean": clean_ents.mean(axis=0).tolist(),
            "vis_frac_mean": float(np.mean(clean_vis)),
            "vis_frac_std": float(np.std(clean_vis)),
            "vis_frac_per_head_mean": clean_vis.mean(axis=0).tolist(),
            "eff_n_mean": float(np.mean(clean_eff)),
            "eff_n_std": float(np.std(clean_eff)),
        }

        # --- corrupted ---
        for ctype in CORRUPTIONS:
            c_ents = np.array([s["corrupted"][ctype][l]["entropy"]
                               for s in all_scene_stats
                               if l in s["corrupted"].get(ctype, {})])
            c_vis = np.array([s["corrupted"][ctype][l]["vis_frac"]
                              for s in all_scene_stats
                              if l in s["corrupted"].get(ctype, {})])
            c_eff = np.array([s["corrupted"][ctype][l]["eff_n"]
                              for s in all_scene_stats
                              if l in s["corrupted"].get(ctype, {})])
            c_cos = np.array([s["corrupted"][ctype][l]["cos_sim"]
                              for s in all_scene_stats
                              if l in s["corrupted"].get(ctype, {})])

            # Delta = corrupted - clean (arrays aligned by scene)
            delta_ent = c_ents - clean_ents[:len(c_ents)]
            delta_vis = c_vis - clean_vis[:len(c_vis)]
            delta_eff = c_eff - clean_eff[:len(c_eff)]

            agg["corrupted"][ctype][str(l)] = {
                "entropy_mean": float(np.mean(c_ents)),
                "entropy_std": float(np.std(c_ents)),
                "entropy_delta_mean": float(np.mean(delta_ent)),
                "entropy_delta_std": float(np.std(delta_ent)),
                "vis_frac_mean": float(np.mean(c_vis)),
                "vis_frac_std": float(np.std(c_vis)),
                "vis_frac_delta_mean": float(np.mean(delta_vis)),
                "vis_frac_delta_std": float(np.std(delta_vis)),
                "eff_n_mean": float(np.mean(c_eff)),
                "eff_n_std": float(np.std(c_eff)),
                "eff_n_delta_mean": float(np.mean(delta_eff)),
                "eff_n_delta_std": float(np.std(delta_eff)),
                # Cosine similarity of last-token attention to clean baseline
                "attn_cos_sim_mean": float(np.mean(c_cos)),
                "attn_cos_sim_std": float(np.std(c_cos)),
                "attn_cos_sim_per_head_mean": c_cos.mean(axis=0).tolist(),
            }

    return agg


# ---------------------------------------------------------------------------
# Analysis 5: Head-level at layer HEAD_ANALYSIS_LAYER
# ---------------------------------------------------------------------------

def head_level_analysis(all_scene_stats):
    """
    At HEAD_ANALYSIS_LAYER, compute per-head:
      - entropy change (corrupted - clean), mean across scenes
      - attention cosine shift (1 - cos_sim), mean across scenes

    Returns dict keyed by corruption type, with per-head arrays.
    """
    l = HEAD_ANALYSIS_LAYER
    result = {"analysis_layer": l, "corruptions": {}}

    # Clean per-head entropy, shape (scenes, heads)
    clean_ents = np.array([s["clean"][l]["entropy"] for s in all_scene_stats
                           if l in s["clean"]])

    for ctype in CORRUPTIONS:
        c_ents = np.array([s["corrupted"][ctype][l]["entropy"]
                           for s in all_scene_stats
                           if l in s["corrupted"].get(ctype, {})])
        c_cos = np.array([s["corrupted"][ctype][l]["cos_sim"]
                          for s in all_scene_stats
                          if l in s["corrupted"].get(ctype, {})])

        n = min(len(clean_ents), len(c_ents))
        delta_ent_per_head = (c_ents[:n] - clean_ents[:n]).mean(axis=0)  # (heads,)
        cos_shift_per_head = (1.0 - c_cos[:n]).mean(axis=0)              # (heads,)

        # Rank heads by absolute entropy change
        ranked_by_ent = np.argsort(np.abs(delta_ent_per_head))[::-1]
        # Rank heads by cosine shift
        ranked_by_cos = np.argsort(cos_shift_per_head)[::-1]

        result["corruptions"][ctype] = {
            "entropy_delta_per_head": delta_ent_per_head.tolist(),
            "entropy_delta_mean": float(np.mean(delta_ent_per_head)),
            "entropy_delta_std": float(np.std(delta_ent_per_head)),
            "cosine_shift_per_head": cos_shift_per_head.tolist(),
            "cosine_shift_mean": float(np.mean(cos_shift_per_head)),
            "cosine_shift_std": float(np.std(cos_shift_per_head)),
            "top5_heads_by_entropy_delta": [
                {"head": int(ranked_by_ent[i]),
                 "delta": float(delta_ent_per_head[ranked_by_ent[i]])}
                for i in range(min(5, len(ranked_by_ent)))
            ],
            "top5_heads_by_cosine_shift": [
                {"head": int(ranked_by_cos[i]),
                 "shift": float(cos_shift_per_head[ranked_by_cos[i]])}
                for i in range(min(5, len(ranked_by_cos)))
            ],
        }

    return result


# ---------------------------------------------------------------------------
# Analysis 6: Corruption visual-attention direction
# ---------------------------------------------------------------------------

def visual_attention_direction(aggregated):
    """
    Summarise whether each corruption causes MORE or LESS visual-token attention
    relative to clean, across all analysed layers.

    Returns a per-corruption summary.
    """
    result = {}
    for ctype in CORRUPTIONS:
        deltas = []
        for l in ANALYSIS_LAYERS:
            key = str(l)
            if key in aggregated["corrupted"].get(ctype, {}):
                deltas.append(aggregated["corrupted"][ctype][key]["vis_frac_delta_mean"])
        if deltas:
            mean_delta = float(np.mean(deltas))
            direction = "more" if mean_delta > 0 else "less"
            result[ctype] = {
                "mean_vis_frac_delta_across_layers": mean_delta,
                "direction": direction,
                "per_layer_deltas": {
                    str(ANALYSIS_LAYERS[i]): float(deltas[i])
                    for i in range(len(deltas))
                },
            }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 72, flush=True)
    print("EXPERIMENT 449: Attention Pattern Analysis under Corruption", flush=True)
    print(f"Timestamp : {timestamp}", flush=True)
    print(f"Seeds     : {SEEDS}", flush=True)
    print(f"Corruptions: {CORRUPTIONS}", flush=True)
    print(f"Severity  : {SEVERITY}", flush=True)
    print(f"Layers    : {ANALYSIS_LAYERS}", flush=True)
    print("=" * 72, flush=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("\nLoading model...", flush=True)
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

    # ------------------------------------------------------------------
    # Process scenes one at a time
    # ------------------------------------------------------------------
    all_scene_stats = []
    for idx, seed in enumerate(SEEDS):
        print(f"\n[Scene {idx + 1}/{len(SEEDS)}] seed={seed}", flush=True)
        stats = analyse_scene(model, processor, seed, ANALYSIS_LAYERS)
        # Print quick summary for this scene
        l0 = ANALYSIS_LAYERS[0]
        if l0 in stats["clean"]:
            clean_ent = float(np.mean(stats["clean"][l0]["entropy"]))
            print(f"  Layer {l0} clean entropy (mean over heads): {clean_ent:.4f}", flush=True)
        for ctype in CORRUPTIONS:
            if l0 in stats["corrupted"].get(ctype, {}):
                c_ent = float(np.mean(stats["corrupted"][ctype][l0]["entropy"]))
                c_cos = float(np.mean(stats["corrupted"][ctype][l0]["cos_sim"]))
                print(f"  {ctype:6s}: L{l0} entropy={c_ent:.4f}  cos_sim={c_cos:.4f}", flush=True)
        all_scene_stats.append(stats)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    seq_len = all_scene_stats[0]["seq_len"]
    num_heads = all_scene_stats[0]["num_heads"]
    print(f"\nModel dims: seq_len={seq_len}, num_heads={num_heads}", flush=True)
    print(f"Visual tokens assumed: {N_VISUAL_TOKENS}", flush=True)

    # ------------------------------------------------------------------
    # Analysis 1-4 and 6: aggregate across scenes
    # ------------------------------------------------------------------
    print("\nAggregating results...", flush=True)
    aggregated = aggregate_scenes(all_scene_stats, ANALYSIS_LAYERS)

    # Print per-layer summary
    print("\n--- Analysis 1: Attention Entropy per Layer ---", flush=True)
    for l in ANALYSIS_LAYERS:
        key = str(l)
        clean_ent = aggregated["clean"][key]["entropy_mean"]
        print(f"  Layer {l:2d}: clean={clean_ent:.4f}", end="", flush=True)
        for ctype in CORRUPTIONS:
            if key in aggregated["corrupted"].get(ctype, {}):
                d = aggregated["corrupted"][ctype][key]["entropy_delta_mean"]
                print(f"  {ctype}Δ={d:+.4f}", end="", flush=True)
        print(flush=True)

    print("\n--- Analysis 2: Visual Attention Fraction per Layer ---", flush=True)
    for l in ANALYSIS_LAYERS:
        key = str(l)
        clean_vis = aggregated["clean"][key]["vis_frac_mean"]
        print(f"  Layer {l:2d}: clean={clean_vis:.4f}", end="", flush=True)
        for ctype in CORRUPTIONS:
            if key in aggregated["corrupted"].get(ctype, {}):
                d = aggregated["corrupted"][ctype][key]["vis_frac_delta_mean"]
                print(f"  {ctype}Δ={d:+.4f}", end="", flush=True)
        print(flush=True)

    print("\n--- Analysis 3: Effective N (attention concentration) per Layer ---", flush=True)
    for l in ANALYSIS_LAYERS:
        key = str(l)
        clean_eff = aggregated["clean"][key]["eff_n_mean"]
        print(f"  Layer {l:2d}: clean={clean_eff:.2f}", end="", flush=True)
        for ctype in CORRUPTIONS:
            if key in aggregated["corrupted"].get(ctype, {}):
                d = aggregated["corrupted"][ctype][key]["eff_n_delta_mean"]
                print(f"  {ctype}Δ={d:+.2f}", end="", flush=True)
        print(flush=True)

    print("\n--- Analysis 4: Cross-Layer Attention Shift (cos_sim to clean) ---", flush=True)
    for l in ANALYSIS_LAYERS:
        key = str(l)
        print(f"  Layer {l:2d}:", end="", flush=True)
        for ctype in CORRUPTIONS:
            if key in aggregated["corrupted"].get(ctype, {}):
                cs = aggregated["corrupted"][ctype][key]["attn_cos_sim_mean"]
                print(f"  {ctype}={cs:.4f}", end="", flush=True)
        print(flush=True)

    # ------------------------------------------------------------------
    # Analysis 5: Head-level at HEAD_ANALYSIS_LAYER
    # ------------------------------------------------------------------
    print(f"\n--- Analysis 5: Head-Level Analysis (Layer {HEAD_ANALYSIS_LAYER}) ---", flush=True)
    head_analysis = head_level_analysis(all_scene_stats)
    for ctype in CORRUPTIONS:
        info = head_analysis["corruptions"][ctype]
        print(f"  {ctype}: entropy_delta={info['entropy_delta_mean']:+.4f}  "
              f"cos_shift={info['cosine_shift_mean']:.4f}", flush=True)
        top_head = info["top5_heads_by_entropy_delta"][0]
        print(f"    Most affected head (entropy): {top_head['head']} "
              f"delta={top_head['delta']:+.4f}", flush=True)

    # ------------------------------------------------------------------
    # Analysis 6: Visual attention direction
    # ------------------------------------------------------------------
    print("\n--- Analysis 6: Visual Attention Direction ---", flush=True)
    vis_direction = visual_attention_direction(aggregated)
    for ctype, info in vis_direction.items():
        print(f"  {ctype}: mean_delta={info['mean_vis_frac_delta_across_layers']:+.5f} "
              f"({info['direction']} visual attention)", flush=True)

    # ------------------------------------------------------------------
    # Assemble final results
    # ------------------------------------------------------------------
    results = {
        "experiment": 449,
        "title": "Attention Pattern Analysis under Corruption",
        "timestamp": timestamp,
        "config": {
            "seeds": SEEDS,
            "corruptions": CORRUPTIONS,
            "severity": SEVERITY,
            "analysis_layers": ANALYSIS_LAYERS,
            "head_analysis_layer": HEAD_ANALYSIS_LAYER,
            "n_visual_tokens_assumed": N_VISUAL_TOKENS,
        },
        "model_dims": {
            "seq_len": seq_len,
            "num_heads": num_heads,
        },
        "analysis_1_attention_entropy": {
            l_str: {
                "clean": aggregated["clean"][l_str]["entropy_mean"],
                "clean_std": aggregated["clean"][l_str]["entropy_std"],
                **{
                    f"{c}_mean": aggregated["corrupted"][c][l_str]["entropy_mean"]
                    for c in CORRUPTIONS if l_str in aggregated["corrupted"].get(c, {})
                },
                **{
                    f"{c}_delta": aggregated["corrupted"][c][l_str]["entropy_delta_mean"]
                    for c in CORRUPTIONS if l_str in aggregated["corrupted"].get(c, {})
                },
            }
            for l_str in [str(l) for l in ANALYSIS_LAYERS]
            if l_str in aggregated["clean"]
        },
        "analysis_2_visual_attention_fraction": {
            l_str: {
                "clean": aggregated["clean"][l_str]["vis_frac_mean"],
                "clean_std": aggregated["clean"][l_str]["vis_frac_std"],
                **{
                    f"{c}_mean": aggregated["corrupted"][c][l_str]["vis_frac_mean"]
                    for c in CORRUPTIONS if l_str in aggregated["corrupted"].get(c, {})
                },
                **{
                    f"{c}_delta": aggregated["corrupted"][c][l_str]["vis_frac_delta_mean"]
                    for c in CORRUPTIONS if l_str in aggregated["corrupted"].get(c, {})
                },
            }
            for l_str in [str(l) for l in ANALYSIS_LAYERS]
            if l_str in aggregated["clean"]
        },
        "analysis_3_effective_n": {
            l_str: {
                "clean": aggregated["clean"][l_str]["eff_n_mean"],
                "clean_std": aggregated["clean"][l_str]["eff_n_std"],
                **{
                    f"{c}_mean": aggregated["corrupted"][c][l_str]["eff_n_mean"]
                    for c in CORRUPTIONS if l_str in aggregated["corrupted"].get(c, {})
                },
                **{
                    f"{c}_delta": aggregated["corrupted"][c][l_str]["eff_n_delta_mean"]
                    for c in CORRUPTIONS if l_str in aggregated["corrupted"].get(c, {})
                },
            }
            for l_str in [str(l) for l in ANALYSIS_LAYERS]
            if l_str in aggregated["clean"]
        },
        "analysis_4_cross_layer_attention_shift": {
            l_str: {
                f"{c}_cos_sim": aggregated["corrupted"][c][l_str]["attn_cos_sim_mean"]
                for c in CORRUPTIONS if l_str in aggregated["corrupted"].get(c, {})
            }
            for l_str in [str(l) for l in ANALYSIS_LAYERS]
            if l_str in aggregated["clean"]
        },
        "analysis_5_head_level": head_analysis,
        "analysis_6_visual_attention_direction": vis_direction,
        "full_aggregated_stats": aggregated,
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = os.path.join(
        EXPERIMENTS_DIR,
        f"attention_pattern_corruption_{timestamp}.json",
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
