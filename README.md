# SmoothGrad: Removing Noise by Adding Noise

**Course:** Fundamentals of Computer Vision (AI4002) — Spring 2026  
**Project:** Reproduction of Smilkov et al., 2017 ([arXiv:1706.03825](https://arxiv.org/abs/1706.03825))  
**Group Members:** Salman Haider (22K-4574) · Asfandyar Khanzada (22K-4626) · M. Hamza Naeem (22K-4424)

---

## What This Project Does

When a deep neural network classifies an image, it is nearly impossible to know *why* it made that decision just by looking at the network's weights. One way to peek inside is to compute a **saliency map** — a heatmap that highlights which pixels in the image most influenced the model's prediction.

This project implements and compares two approaches:

- **Vanilla Gradient** — the simplest saliency technique, fast but visually noisy
- **SmoothGrad** — an improved technique that averages gradients over many noisy copies of the image, producing cleaner and more interpretable maps

We use a pretrained **ResNet-50** (ImageNet) and run four experiments that reproduce the key figures from the original paper.

---

## Background

### How Gradient Saliency Works

Given an image `x` and a target class `c`, a neural network computes a class score `Sc(x)`. The gradient of that score with respect to every input pixel — `∂Sc/∂x` — tells us how sensitive the score is to each pixel. Pixels with a high gradient are "important" to the decision; pixels with a low gradient are not.

Visualised as a grayscale heatmap: **bright = important, dark = irrelevant.**

### The Problem with Vanilla Gradients

The decision surface of a deep network fluctuates sharply, even over imperceptibly small pixel changes. This means the gradient at a single point is noisy and unreliable — the resulting heatmap highlights scattered random pixels alongside the genuinely important ones, making it hard to interpret.

### The SmoothGrad Fix

SmoothGrad computes the gradient not at one exact point, but as an **average over a small neighbourhood** around the image. This is done stochastically:

```
M̂c(x) = (1/n) × Σ ∂Sc(x + N(0, σ²)) / ∂x
```

- Add Gaussian noise `N(0, σ²)` to the image → compute gradient → repeat `n` times → average
- Random noise-driven fluctuations cancel out across iterations
- Pixels that are consistently important survive the averaging

Two hyperparameters control the result:
- **σ (noise level)** — standard deviation of the noise, expressed as a fraction of the image's pixel range
- **n (sample count)** — how many noisy copies to average over

---

## Project Structure

```
FCV project/
├── main.py           ← runs all 4 experiments end to end
├── smoothgrad.py     ← vanilla_grad() and smooth_grad() functions
├── utils.py          ← image downloading, preprocessing, postprocessing
├── visualize.py      ← all figure generation and saving
├── images/           ← downloaded sample images (auto-created on first run)
├── outputs/          ← saved experiment figures (auto-created on first run)
└── README.md
```

---

## How to Run

### Requirements
- Python 3.10
- PyTorch, torchvision, matplotlib, numpy, Pillow, requests

### Run

```bash
cd "E:\FCV project"
python main.py
```

That's all. On first run it will:
1. Download the pretrained ResNet-50 weights (~98 MB, cached after first run)
2. Download 6 sample images from public datasets
3. Run all 4 experiments
4. Save 5 figures to `outputs/`

**Estimated runtime: ~5 minutes on CPU.** No GPU required.

### Adjusting Parameters

Edit the constants near the top of `main.py`:

```python
N_FIXED      = 50      # fixed sample count for Experiment 1
SIGMA_FIXED  = 0.10    # fixed noise level (10%) for Experiments 2 & 3

SIGMA_VALUES = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50]  # Experiment 1 sweep
N_VALUES     = [2, 5, 20, 50, 100]                     # Experiment 2 sweep
```

---

## The 4 Experiments

### Experiment 1 — Effect of Noise Level
**Output:** `outputs/exp1_noise_level.png`  
**Replicates:** Figure 3 of the paper

Fix `n = 50`. Vary σ across **0%, 5%, 10%, 20%, 30%, 50%** for 5 images.

- σ = 0% is equivalent to vanilla gradient — noisy and scattered
- As σ increases, the object's outline becomes progressively cleaner
- At σ = 50% the map over-smooths and important detail is lost
- The sweet spot in the paper is around σ = 10–20%

---

### Experiment 2 — Effect of Sample Size
**Output:** `outputs/exp2_sample_size.png`  
**Replicates:** Figure 4 of the paper

Fix `σ = 10%`. Vary n across **2, 5, 20, 50, 100** for 5 images.

- At n = 2 the map is still very noisy (too few samples to average out noise)
- Quality improves steadily up to n = 50
- Diminishing returns after n = 50 — n = 100 looks almost identical to n = 50
- The paper recommends n = 50 as a practical choice

---

### Experiment 3 — Vanilla Gradient vs SmoothGrad
**Output:** `outputs/exp3_comparison.png`  
**Replicates:** Figure 5 of the paper

For each image, show five panels side by side:

| Original | Vanilla Gradient | SmoothGrad | Vanilla × Input | SmoothGrad × Input |
|---|---|---|---|---|

The **× Input** variant multiplies the gradient map by the original image pixel values. A pixel must be both sensitive (high gradient) and present (high pixel value) to appear bright — this further sharpens the map around the object.

This is the most visually compelling experiment. The scattered, hard-to-interpret vanilla map versus the clean, object-focused SmoothGrad map is immediately obvious.

---

### Experiment 4 — Discriminativity
**Output:** `outputs/exp4_discriminativity.png` (SmoothGrad)  
**Output:** `outputs/exp4_discriminativity_vanilla.png` (Vanilla, for comparison)  
**Replicates:** Figure 6 of the paper

Takes one image and computes saliency maps for **two different classes** — Cat (class 281) and Dog (class 207). Both maps are normalised to [0, 1], then subtracted:

```
difference = scale(map_cat) − scale(map_dog)
```

Plotted on a red-blue diverging colormap:
- **Red** → pixels the model uses to identify cats
- **Blue** → pixels it would use if looking for dogs
- **White/grey** → neutral, used by neither class specifically

A discriminative model should show clear spatial separation. SmoothGrad produces cleaner class separation than vanilla gradient.

---

## Code Details

### `smoothgrad.py`

The entire algorithm in two functions. No I/O, no plotting — pure computation.

`vanilla_grad()` enables `requires_grad=True` on the input tensor, does a single forward pass, calls `score.backward()`, and returns `input.grad` as a numpy array.

`smooth_grad()` runs that same process `n` times in a loop, adding fresh Gaussian noise each iteration. Sigma is scaled as `noise_level × (input_max − input_min)` so it adapts to the normalised pixel range of the tensor rather than assuming raw [0, 255] values. Returns the accumulated gradients divided by `n`.

### `utils.py`

**Downloading:** Each sample image has two fallback URLs (COCO dataset first, Wikimedia second). Uses `requests` for reliable HTTP handling, verifies the response is a valid image before saving, generates a synthetic placeholder as a last resort.

**Preprocessing:** Standard ImageNet pipeline — `Resize(256) → CenterCrop(224) → ToTensor() → Normalize(mean, std)`. ResNet-50 was trained with these exact values and will produce incorrect predictions without them.

**Postprocessing:** Converts raw gradient arrays (3-channel, signed, with extreme outliers) into clean [0, 1] grayscale maps:
1. Absolute value — we care about magnitude, not direction
2. Sum across RGB channels — collapse to (H, W)
3. Cap at 99th percentile — prevents outliers from washing out the colour scale
4. Normalise to [0, 1]

### `visualize.py`

Four functions corresponding to the four experiments. Each receives already-computed saliency maps and arranges them using matplotlib `subplots`. All figures call `plt.close()` after saving to avoid memory accumulation.

### `main.py`

Top-level script. Loads the model, downloads images, then runs each experiment in sequence. The data flow is the same every time:

```
PIL Image → preprocess() → vanilla_grad() or smooth_grad() → postprocess_gradient() → visualize function → PNG
```

GPU is used automatically if available (`torch.cuda.is_available()`), otherwise falls back to CPU silently.

---

## Sample Images Used

| Name | ImageNet Class | Class Index |
|---|---|---|
| Cat | Tabby Cat | 281 |
| Dog | Golden Retriever | 207 |
| Elephant | Indian Elephant | 385 |
| Bird | Robin | 15 |
| Butterfly | Monarch Butterfly | 323 |
| Cat+Dog (Exp 4) | Cat vs Dog | 281 vs 207 |

Images are downloaded automatically from the COCO val2017 dataset with Wikimedia Commons as fallback. No manual download required.

---

## Key Takeaway

Vanilla gradients show *where the network is sensitive*, but the signal is buried in noise. SmoothGrad recovers the true underlying sensitivity by averaging out local fluctuations — at the cost of `n` forward+backward passes instead of one. The result is a heatmap that genuinely reflects the features the model is using, making the network's decisions interpretable to a human.
