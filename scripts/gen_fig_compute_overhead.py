"""Generate Figure 117: Computational Overhead."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/experiments/compute_overhead_20260315_000301.json") as f:
    data = json.load(f)

t = data['timings_ms']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Latency breakdown
ax = axes[0]
components = ['Preprocess', 'Forward\n(no hidden)', 'Forward\n(+hidden)', 'OOD\nPipeline', 'Action\nGeneration']
times = [t['preprocess']['mean'], t['forward']['mean'], t['forward_hidden']['mean'],
         t['pipeline']['mean'], t['generate']['mean']]
stds = [t['preprocess']['std'], t['forward']['std'], t['forward_hidden']['std'],
        t['pipeline']['std'], t['generate']['std']]
colors = ['#9E9E9E', '#2196F3', '#4CAF50', '#FF9800', '#F44336']
bars = ax.bar(range(len(components)), times, color=colors, alpha=0.8)
ax.errorbar(range(len(components)), times, yerr=stds, fmt='none', color='black', capsize=3)
ax.set_xticks(range(len(components)))
ax.set_xticklabels(components, fontsize=8)
ax.set_ylabel("Latency (ms)")
ax.set_title("(A) Latency Breakdown")
ax.grid(True, alpha=0.3, axis='y')
for bar, time_val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f"{time_val:.0f}ms", ha='center', fontsize=8, fontweight='bold')

# Panel B: Overhead analysis
ax = axes[1]
forward_ms = t['forward']['mean']
hidden_ms = t['forward_hidden']['mean']
cosine_us = t['cosine_us']['mean']
pipeline_ms = t['pipeline']['mean']
generate_ms = t['generate']['mean']

overhead_hidden = hidden_ms - forward_ms
overhead_pct = data['overhead']['hidden_state_pct']

categories = ['Hidden State\nOverhead', 'Cosine\nDistance', 'Total OOD\nOverhead']
values = [max(0, overhead_hidden), cosine_us/1000, max(0, overhead_hidden) + cosine_us/1000]
bars = ax.bar(range(len(categories)), values, color=['#4CAF50', '#2196F3', '#FF9800'], alpha=0.8)
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, fontsize=8)
ax.set_ylabel("Overhead (ms)")
ax.set_title("(B) OOD Detection Overhead")
ax.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{v:.2f}ms" if v > 0.01 else f"{v*1000:.0f}μs", ha='center', fontsize=9, fontweight='bold')

ax.text(0.5, 0.95, f"Overhead: {overhead_pct:.1f}% of forward pass",
        transform=ax.transAxes, fontsize=10, ha='center', va='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Panel C: Pipeline comparison
ax = axes[2]
scenarios = ['Standard\nInference\n(generate only)', 'OOD + Generate\n(sequential)', 'OOD + Generate\n(parallel)']
s_times = [generate_ms, pipeline_ms + generate_ms, max(pipeline_ms, generate_ms)]
bars = ax.bar(range(len(scenarios)), s_times, color=['#2196F3', '#FF9800', '#4CAF50'], alpha=0.8)
ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels(scenarios, fontsize=7)
ax.set_ylabel("Total Latency (ms)")
ax.set_title("(C) Deployment Scenarios")
ax.grid(True, alpha=0.3, axis='y')
for bar, st in zip(bars, s_times):
    pct = (st - generate_ms) / generate_ms * 100
    label = f"{st:.0f}ms\n(+{pct:.0f}%)" if pct > 0 else f"{st:.0f}ms\n(baseline)"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            label, ha='center', fontsize=8, fontweight='bold')

plt.suptitle(f"Computational Overhead (Exp 131) — NVIDIA A40\nOOD detection adds ~0ms overhead; full pipeline 2.5× faster than generation",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/latex/figures/fig117_compute_overhead.png",
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig117")
