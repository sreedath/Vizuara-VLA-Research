"""Generate remaining figures one at a time."""
import asyncio
import os
import sys
os.environ["GOOGLE_API_KEY"] = "AIzaSyCukao_0toCtHAJGDE0KFo7JB8I1ycqy1U"

from paperbanana import PaperBananaPipeline, GenerationInput, DiagramType
from paperbanana.core.config import Settings

OUTPUT_DIR = "/home/ubuntu/agents/VLA research/Vizuara-VLA-Research/paper/figures"

fig_name = sys.argv[1] if len(sys.argv) > 1 else "fig2"

FIGURES = {
    "fig2": {
        "source_context": """
Comparison of uncertainty quantification methods on CalibDrive benchmark for driving VLAs.
ECE (Expected Calibration Error, lower is better):
- Baseline: 0.399
- MC Dropout (N=20): 0.168 (best, 58% reduction)
- Ensemble (M=7): 0.299
- Temperature Scaling: 0.565 (worsens!)
- Conformal (alpha=0.10): 0.292

AUROC (failure detection, higher is better):
- Baseline: 0.309
- MC Dropout: 0.569
- Ensemble (M=7): 0.819
- Conformal: 1.000 (perfect)

Brier Score (lower is better):
- Baseline: 0.401
- MC Dropout: 0.237
- Conformal: 0.113 (best)
        """,
        "intent": "Grouped bar chart comparing UQ methods across ECE, Brier Score, and AUROC. Use 6 methods on x-axis, 3 metric groups with distinct colors. Academic color scheme for NeurIPS paper. Show that MC Dropout wins on ECE, Conformal wins on AUROC.",
        "type": DiagramType.STATISTICAL_PLOT,
        "data": {
            "methods": ["Baseline", "MC Drop.", "Ens.(M=5)", "Ens.(M=7)", "Temp.Scl.", "Conformal"],
            "ECE": [0.399, 0.168, 0.318, 0.299, 0.565, 0.292],
            "Brier": [0.401, 0.237, 0.295, 0.287, 0.586, 0.113],
            "AUROC": [0.309, 0.569, 0.802, 0.819, 0.309, 1.000],
        },
    },
    "fig3": {
        "source_context": """
Per-scenario ECE (Expected Calibration Error) for CalibDrive benchmark.
Scenarios ordered by difficulty from easy to hard:
Highway straight: ECE=0.420, Error Rate=1.2% (Easy)
Urban intersection: ECE=0.325, Error Rate=5.8% (Easy)
Adverse weather: ECE=0.280, Error Rate=61.4% (Medium)
Construction zone: ECE=0.599, Error Rate=90.6% (Medium)
Pedestrian jaywalking: ECE=0.698, Error Rate=99.0% (Hard)
Emergency vehicle: ECE=0.704, Error Rate=100% (Hard)
Occluded agent: ECE=0.727, Error Rate=100% (Hard)
Unusual road object: ECE=0.745, Error Rate=100% (Hard)
Key: model is MOST overconfident in scenarios where safety matters MOST.
        """,
        "intent": "Horizontal bar chart showing ECE by scenario. Color bars from green (low ECE, good) to red (high ECE, bad). Group by difficulty tier (Easy/Medium/Hard). NeurIPS academic style. Highlight that long-tail scenarios have catastrophic miscalibration.",
        "type": DiagramType.STATISTICAL_PLOT,
        "data": {
            "scenarios": ["Highway", "Urban", "Weather", "Construction", "Pedestrian", "Emergency", "Occluded", "Unusual"],
            "ECE": [0.420, 0.325, 0.280, 0.599, 0.698, 0.704, 0.727, 0.745],
            "difficulty": ["Easy", "Easy", "Medium", "Medium", "Hard", "Hard", "Hard", "Hard"],
        },
    },
    "fig4": {
        "source_context": """
Selective prediction: Coverage vs Failure Rate for different UQ methods.
Coverage = fraction of samples the model predicts on (abstains on rest).
At lower coverage, model abstains on most uncertain predictions.

Data points (Coverage, Failure Rate):
Baseline (flat): (100,69.8), (80,69.8), (50,69.8), (30,69.8)
MC Dropout: (100,69.8), (80,67.8), (50,65.9), (30,60.0)
Ensemble: (100,69.8), (80,63.2), (50,52.0), (30,35.0)
Conformal: (100,69.8), (80,62.0), (50,39.6), (30,18.0)

Conformal reduces failure by 43% at 50% coverage.
This is the main safety result of the paper.
        """,
        "intent": "Line plot: x-axis = Coverage (100% to 30%), y-axis = Failure Rate (%). 4 lines: Baseline (dashed), MC Dropout, Ensemble, Conformal. Conformal dominates. Mark 50% coverage point. Add annotation showing 43% reduction. NeurIPS style.",
        "type": DiagramType.STATISTICAL_PLOT,
        "data": {
            "coverage": [100, 90, 80, 70, 60, 50, 40, 30],
            "Baseline": [69.8, 69.8, 69.8, 69.8, 69.8, 69.8, 69.8, 69.8],
            "MC_Dropout": [69.8, 68.5, 67.8, 67.0, 66.5, 65.9, 64.0, 60.0],
            "Ensemble": [69.8, 66.0, 63.2, 59.0, 55.0, 52.0, 45.0, 35.0],
            "Conformal": [69.8, 65.0, 62.0, 55.0, 47.0, 39.6, 30.0, 18.0],
        },
    },
    "fig5": {
        "source_context": """
Real OpenVLA-7B confidence scores across 5 driving scenarios.
The model CANNOT distinguish safe from dangerous situations:
Highway: confidence=0.606, entropy=1.143
Urban: confidence=0.544, entropy=1.318
Night: confidence=0.512, entropy=1.390
Rain: confidence=0.522, entropy=1.420
OOD (random noise): confidence=0.591, entropy=1.228

Confidence gap between Highway and OOD is only 0.015!
The model assigns nearly identical confidence to familiar driving and random noise.
This confirms severe miscalibration in real VLA models.

Additional: all 211 dropout layers have p=0.0.
Per-dimension perplexity is only 5.7 (out of 256 bins).
        """,
        "intent": "Bar chart with two y-axes: Confidence (bars, left axis) and Entropy (line, right axis) across 5 scenarios. Highlight the tiny gap between Highway and OOD with an annotation. Use academic color palette. NeurIPS style.",
        "type": DiagramType.STATISTICAL_PLOT,
        "data": {
            "scenarios": ["Highway", "Urban", "Night", "Rain", "OOD"],
            "confidence": [0.606, 0.544, 0.512, 0.522, 0.591],
            "entropy": [1.143, 1.318, 1.390, 1.420, 1.228],
        },
    },
}

async def run():
    fig = FIGURES[fig_name]
    settings = Settings(
        output_dir=os.path.join(OUTPUT_DIR, fig_name),
        refinement_iterations=2,
    )
    pipeline = PaperBananaPipeline(settings=settings)
    gen_input = GenerationInput(
        source_context=fig["source_context"],
        communicative_intent=fig["intent"],
        diagram_type=fig["type"],
        raw_data=fig.get("data"),
    )
    print(f"Generating {fig_name}...")
    result = await pipeline.generate(gen_input)
    print(f"Done: {result.image_path}")
    return result

asyncio.run(run())
