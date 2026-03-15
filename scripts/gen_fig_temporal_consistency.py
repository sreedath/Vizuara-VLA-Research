import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/temporal_consistency_20260315_014016.json") as f:
    data = json.load(f)

results = data["results"]
seqs = list(results.keys())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

titles = {
    "fog_onset": "Fog Onset (0% → 90%)",
    "night_fall": "Nightfall (100% → 10% brightness)",
    "blur_increase": "Blur Increase (0 → 15 radius)",
    "noise_increase": "Noise Increase (0 → 100 std)",
}

for idx, seq_name in enumerate(seqs):
    ax = axes[idx // 2][idx % 2]
    sr = results[seq_name]
    levels = sr["levels"]
    d_l3 = sr["distances"]["L3"]
    d_l32 = sr["distances"]["L32"]
    t_l32 = sr["thresholds"]["L32"]
    t_l3 = sr["thresholds"]["L3"]

    ax.plot(levels, d_l32, 'o-', color='#FF9800', label='L32 cosine dist', markersize=4, linewidth=2)
    ax.plot(levels, d_l3, 's-', color='#2196F3', label='L3 cosine dist', markersize=4, linewidth=2)
    ax.axhline(y=t_l32, color='#FF9800', linestyle='--', alpha=0.5, label=f'L32 threshold ({t_l32:.4f})')
    ax.axhline(y=t_l3, color='#2196F3', linestyle='--', alpha=0.5, label=f'L3 threshold ({t_l3:.6f})')

    ax.set_xlabel(sr["description"].split(" ")[0] + " level")
    ax.set_ylabel("Cosine Distance")
    ax.set_title(titles.get(seq_name, seq_name))
    ax.legend(fontsize=7, loc='upper left')

    # Annotate monotonicity
    l32_monotonic = all(d_l32[i] <= d_l32[i+1] for i in range(len(d_l32)-1) if levels[i] < levels[i+1])
    ax.text(0.95, 0.05, f"L32 monotonic: {'Yes' if l32_monotonic else 'No'}\nJitter: {sr['jitter_after_first_ood']}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle("Temporal Consistency: Distance Profiles Under Gradual Corruption", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig139_temporal_consistency.png", dpi=150, bbox_inches='tight')
print("Saved fig139")
