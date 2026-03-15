"""Experiment 256: Pairwise Embedding Similarity Matrix
Computes full cosine similarity matrix between clean and all corruption types,
including cross-corruption similarities to map the embedding landscape.
"""
import torch, json, numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image, ImageFilter
import time

print("=" * 60)
print("Experiment 256: Pairwise Embedding Similarity")
print("=" * 60)

print("Loading OpenVLA-7B...")
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", torch_dtype=torch.bfloat16,
    device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
model.eval()

prompt = "In: What action should the robot take to pick up the object?\nOut:"
base_img = Image.fromarray(np.random.RandomState(42).randint(0, 256, (256, 256, 3), dtype=np.uint8))

def extract_hidden(image, layer=3):
    inputs = processor(prompt, image).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        fwd = model(**inputs, output_hidden_states=True)
    return fwd.hidden_states[layer][0, -1, :].float().cpu().numpy()

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# All corruptions
def apply_fog(img): return Image.fromarray((np.array(img).astype(np.float32)*0.6+120*0.4).clip(0,255).astype(np.uint8))
def apply_night(img): return Image.fromarray((np.array(img).astype(np.float32)*0.15).clip(0,255).astype(np.uint8))
def apply_noise(img):
    a = np.array(img).astype(np.float32); a += np.random.RandomState(99).randn(*a.shape)*50
    return Image.fromarray(a.clip(0,255).astype(np.uint8))
def apply_blur(img): return img.filter(ImageFilter.GaussianBlur(radius=5))
def apply_snow(img):
    a = np.array(img).astype(np.float32)
    mask = np.random.RandomState(55).random(a.shape[:2]) < 0.3; a[mask] = 255
    return Image.fromarray(a.clip(0,255).astype(np.uint8))
def apply_rain(img):
    a = np.array(img).astype(np.float32)
    for i in range(80):
        x = np.random.RandomState(i+200).randint(0, a.shape[1])
        y0 = np.random.RandomState(i+300).randint(0, a.shape[0]-20)
        a[y0:y0+15, max(0,x-1):x+1, :] = a[y0:y0+15, max(0,x-1):x+1, :] * 0.7 + 200*0.3
    return Image.fromarray(a.clip(0,255).astype(np.uint8))

# Also test different clean images
conditions = {
    'clean': base_img,
    'clean_s100': Image.fromarray(np.random.RandomState(100).randint(0,256,(256,256,3),dtype=np.uint8)),
    'clean_s200': Image.fromarray(np.random.RandomState(200).randint(0,256,(256,256,3),dtype=np.uint8)),
    'fog': apply_fog(base_img),
    'night': apply_night(base_img),
    'noise': apply_noise(base_img),
    'blur': apply_blur(base_img),
    'snow': apply_snow(base_img),
    'rain': apply_rain(base_img),
    'fog+night': apply_night(apply_fog(base_img)),
    'fog+noise': apply_noise(apply_fog(base_img)),
}

# Extract embeddings
embeddings = {}
for name, img in conditions.items():
    embeddings[name] = extract_hidden(img)
    print(f"  Extracted: {name}")

# Compute full pairwise similarity matrix
names = list(conditions.keys())
n = len(names)
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = cosine_sim(embeddings[names[i]], embeddings[names[j]])

print("\n--- Pairwise Cosine Similarity ---")
for i, ni in enumerate(names):
    for j, nj in enumerate(names):
        if j > i:
            print(f"  {ni} vs {nj}: {sim_matrix[i,j]:.6f}")

output = {
    "experiment": "pairwise_similarity",
    "experiment_number": 256,
    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    "layer": 3,
    "conditions": names,
    "similarity_matrix": [[round(float(sim_matrix[i,j]), 8) for j in range(n)] for i in range(n)],
    "distance_matrix": [[round(float(1-sim_matrix[i,j]), 8) for j in range(n)] for i in range(n)]
}

path = f"/workspace/Vizuara-VLA-Research/experiments/pairwise_sim_{output['timestamp']}.json"
with open(path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {path}")
