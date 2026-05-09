# Technical Report: Improving Visual Explanations in Deep Neural Networks using SmoothGrad

**Course:** Fundamentals of Computer Vision (AI4002) — Spring 2026  
**Group Members:** Salman Haider (22K-4574) · Asfandyar Khanzada (22K-4626) · M. Hamza Naeem (22K-4424)  
**Paper Reproduced:** Smilkov et al., "SmoothGrad: removing noise by adding noise", arXiv:1706.03825, 2017

---

## 1. Task Definition

### Primary Task: Gradient-Based Saliency Map Generation for Image Classification

This project addresses the problem of **neural network interpretability** in the context of image classification. Specifically, we implement and evaluate **gradient-based saliency (sensitivity) mapping** — a technique that produces pixel-level attribution maps revealing which regions of an input image most influenced a neural network's classification decision.

The task is not to train a classifier, but to **explain an already-trained classifier's decisions**. Given:
- A pretrained image classification network (ResNet-50)
- An input image `x`
- A target class `c` (e.g., "cat", "elephant")

The goal is to produce a spatial map `Mc(x)` of shape `(H, W)` where each value represents the importance of the corresponding pixel to the network's confidence score for class `c`.

### Sub-tasks Addressed

| Sub-task | Description |
|---|---|
| Vanilla Gradient Saliency | Baseline: single-pass gradient computation |
| SmoothGrad | Improved: averaged gradient over noisy perturbations |
| Noise Level Analysis | Effect of σ on map quality |
| Sample Size Analysis | Effect of n on map quality and convergence |
| Qualitative Comparison | Visual coherence of different methods |
| Discriminativity Analysis | Whether maps are class-specific rather than class-agnostic |

This task falls under the broader field of **Explainable AI (XAI)** and specifically **post-hoc local explanation methods** — methods that explain individual predictions after the model has been trained, without modifying the model itself.

---

## 2. Dataset Description

### Nature of the Data

This is a **reproduction study**, not a supervised learning project. There is no training dataset. Instead, we use a small set of test images at inference time to demonstrate and evaluate saliency map quality, mirroring the experimental setup of the original paper.

### Sample Images

Six images were downloaded automatically from the **COCO val2017 dataset** (primary source) with **Wikimedia Commons** as a fallback:

| Image | ImageNet Class Name | Class Index | Source |
|---|---|---|---|
| Cat | Tabby Cat | 281 | COCO val2017 #39769 |
| Dog | Golden Retriever | 207 | Wikimedia Commons |
| Elephant | Indian Elephant | 385 | COCO val2017 #1584 |
| Bird | Robin | 15 | COCO val2017 #25560 |
| Butterfly | Monarch Butterfly | 323 | COCO val2017 #174482 |
| Cat (two-class) | Cat vs. Dog (Exp 4) | 281 / 207 | COCO val2017 #39769 |

### Image Format

- **Type:** Natural RGB photographs (JPEG)
- **Original resolution:** Variable (typically 300–640px on the shorter side)
- **After preprocessing:** 224 × 224 pixels, 3 channels
- **Pixel value range (post-normalisation):** approximately [-2.1, 2.6] (normalised to ImageNet statistics)
- **Labels used:** ImageNet class indices (integers) — hardcoded, no annotation files needed

### Relation to the Original Paper

The original paper used **Inception v3** on ImageNet validation images, including images of gazelles (class 353) and other animals. Our reproduction uses ResNet-50 with COCO images of comparable subjects. The qualitative trends (noise reduction with increasing σ, convergence with increasing n) are fully reproducible with any reasonable set of natural images.

---

## 3. Data Pre-processing

All preprocessing follows the **standard ImageNet inference pipeline** that ResNet-50 was trained with. Applying a different pipeline would cause distribution shift and degrade prediction quality.

### Step-by-Step Pipeline

#### Step 1: Loading
Images are loaded as RGB PIL images. Any RGBA or grayscale images are converted to RGB to ensure consistent 3-channel input.

```
Raw JPEG file → PIL.Image.open() → .convert("RGB") → (H, W, 3) array
```

#### Step 2: Resize
The shorter side of the image is resized to 256 pixels while maintaining aspect ratio. This is standard practice to remove excessive background and reduce computational cost.

```
Variable resolution → Resize(shorter side = 256px)
```

#### Step 3: Centre Crop
A 224 × 224 pixel region is cropped from the centre of the resized image. This removes edge content and produces a fixed-size input that matches ResNet-50's expected input dimensions.

```
(256 × ~341) or (~341 × 256) → CenterCrop(224, 224)
```

#### Step 4: Convert to Tensor
The PIL image (uint8, [0, 255]) is converted to a PyTorch float tensor with values in [0.0, 1.0] and shape `(3, 224, 224)`.

```
(H, W, 3) uint8 [0, 255] → (3, H, W) float32 [0.0, 1.0]
```

#### Step 5: ImageNet Normalisation
Each channel is normalised using the mean and standard deviation computed over the full ImageNet training set:

| Channel | Mean | Std |
|---|---|---|
| Red | 0.485 | 0.229 |
| Green | 0.456 | 0.224 |
| Blue | 0.406 | 0.225 |

```
x_normalised = (x - mean) / std
```

This shifts the pixel distribution to approximately zero mean and unit variance, which is what ResNet-50's BatchNorm layers expect.

#### Step 6: Batch Dimension
A batch dimension is added to produce the final input tensor of shape `(1, 3, 224, 224)`.

#### Step 7: Gradient Postprocessing (after saliency computation)

The raw gradient tensor is also postprocessed before visualisation:

1. **Absolute value** — we care about magnitude of influence, not sign (both positive and negative gradients indicate sensitivity)
2. **Channel summation** — sum across RGB channels to collapse `(3, 224, 224)` to `(224, 224)`
3. **99th percentile capping** — gradient maps have extreme outlier values that, if not capped, cause the colour scale to collapse and the entire map appears black. Capping at the 99th percentile is recommended in the paper (Section 3.1)
4. **Min-max normalisation** — rescale to [0, 1] for display

**Optional ×Input variant:** The postprocessed saliency map is multiplied elementwise by a grayscale version of the original image. This highlights pixels that are both important (high gradient) and present (high pixel value).

### No Augmentation
Data augmentation (flipping, rotation, colour jitter, etc.) is intentionally not applied. These are inference-time images, not training data. The goal is to compute the gradient at a specific input point, not to generalise across variations.

---

## 4. Network Architecture

### Model: ResNet-50

We use **ResNet-50** (He et al., 2016) with weights pretrained on ImageNet-1K, loaded directly from `torchvision.models`. No layers are modified, added, or removed. The model is used purely in inference mode (`model.eval()`).

### Architecture Overview

ResNet-50 is a 50-layer deep residual network organised into the following blocks:

```
Input (1, 3, 224, 224)
        │
        ▼
┌─────────────────────────────┐
│  Conv1: 7×7, 64 filters     │  stride 2 → (1, 64, 112, 112)
│  BatchNorm + ReLU           │
│  MaxPool 3×3, stride 2      │  → (1, 64, 56, 56)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Layer 1 (3 bottleneck      │  64 → 256 channels
│  residual blocks)           │  → (1, 256, 56, 56)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Layer 2 (4 bottleneck      │  128 → 512 channels, stride 2
│  residual blocks)           │  → (1, 512, 28, 28)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Layer 3 (6 bottleneck      │  256 → 1024 channels, stride 2
│  residual blocks)           │  → (1, 1024, 14, 14)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Layer 4 (3 bottleneck      │  512 → 2048 channels, stride 2
│  residual blocks)           │  → (1, 2048, 7, 7)
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Global Average Pooling     │  → (1, 2048)
│  Fully Connected Layer      │  2048 → 1000
│  (no softmax at inference)  │  → (1, 1000) logits
└─────────────────────────────┘
```

### Bottleneck Residual Block

Each residual block in ResNet-50 uses a three-layer bottleneck design:

```
Input (C channels)
    │
    ├──────────────────────────────┐  (shortcut / skip connection)
    │                              │
    ▼                              │
Conv 1×1 (reduce channels)        │
BatchNorm + ReLU                  │
    ▼                              │
Conv 3×3 (spatial processing)     │
BatchNorm + ReLU                  │
    ▼                              │
Conv 1×1 (restore channels)       │
BatchNorm                         │
    │                              │
    └──────────── + ───────────────┘
                  │
                 ReLU
                  │
                Output
```

The skip connection allows gradients to flow directly back to earlier layers during backpropagation, which is what makes deep networks trainable and what makes gradient-based saliency maps meaningful even for deep architectures.

### Key Properties Relevant to Saliency

| Property | Value |
|---|---|
| Total parameters | ~25.6 million |
| Input size | 224 × 224 × 3 |
| Output | 1000-dimensional logit vector |
| Activation function | ReLU throughout |
| Normalisation | BatchNorm after every convolution |
| Pooling | Global Average Pooling before classifier |
| Pretrained on | ImageNet-1K (1.28M images, 1000 classes) |
| Top-1 accuracy (ImageNet val) | 76.1% |

### Why ResNet-50 for Saliency

ResNet-50 is fully differentiable end-to-end, which is required for gradient-based saliency. The skip connections ensure gradients do not vanish completely in early layers. The Global Average Pooling layer means spatial information is preserved until late in the network, which tends to produce more spatially coherent gradient maps than architectures using large fully-connected layers.

---

## 5. Loss Function

> **Note:** This project involves **no training**. ResNet-50 weights are fixed. However, gradient-based saliency requires defining a scalar objective to differentiate — this plays the role of a "loss" function during the backward pass.

### Saliency Objective: Class Score Function

For a given input image `x` and target class `c`, the scalar objective used for gradient computation is:

```
Sc(x) = logit output of ResNet-50 for class c
      = W_c · GAP(f(x))
```

where:
- `f(x)` is the output of the convolutional backbone (2048-dimensional feature vector after Global Average Pooling)
- `W_c` is the weight vector of the fully connected layer corresponding to class `c`
- No softmax is applied — raw logits are used, as softmax suppresses gradients for high-confidence predictions

### Gradient Computation

**Vanilla Gradient:**
```
Mc(x) = ∂Sc(x) / ∂x
```

This is a `(3, 224, 224)` tensor computed via a single call to `loss.backward()` in PyTorch, which applies the chain rule through all 50 layers automatically.

**SmoothGrad:**
```
M̂c(x) = (1/n) × Σᵢ₌₁ⁿ [ ∂Sc(x + εᵢ) / ∂x ]     where εᵢ ~ N(0, σ²)
```

Each of the `n` iterations computes a fresh gradient at a perturbed input. The `n` gradients are accumulated and divided by `n` to produce the final map.

### Why Raw Logits Instead of Softmax Score

Using softmax scores for gradient computation is problematic because:
- Softmax is a normalised function — increasing the score for class `c` also decreases scores for all other classes
- Gradients through softmax reflect competition between classes rather than absolute sensitivity to class `c`
- Raw logits give a cleaner, class-specific gradient signal

### Why No Training Loss

There is no cross-entropy loss, no weight update, no optimiser step. The entire backward pass exists solely to compute `∂Sc/∂x` — the gradient of the class score with respect to the input pixels — not to update any model parameter.

---

## 6. Hyperparameters

Since this project involves no training, there are no learning rate, batch size, or optimiser hyperparameters. The relevant hyperparameters are those of the **SmoothGrad algorithm** itself and the **visualisation pipeline**.

### SmoothGrad Hyperparameters

#### σ — Noise Level

| Parameter | Symbol | Value Used (fixed experiments) | Range Explored |
|---|---|---|---|
| Noise standard deviation | σ | 10% of pixel range | 0%, 5%, 10%, 20%, 30%, 50% |

σ is expressed as a fraction of the input tensor's pixel range `(x_max - x_min)` rather than as an absolute value. This is important because the normalised ImageNet pixel range (~[-2.1, 2.6]) is very different from raw pixel range [0, 255]. Scaling σ relative to the actual range ensures the noise magnitude is always meaningful regardless of normalisation.

**Selection rationale:** The original paper empirically found 10–20% to be optimal. Below 5% there is insufficient smoothing; above 30% the map becomes blurry. We reproduce the full sweep in Experiment 1 to verify this finding.

#### n — Number of Samples

| Parameter | Symbol | Value Used (fixed experiments) | Range Explored |
|---|---|---|---|
| Number of noisy samples | n | 50 | 2, 5, 20, 50, 100 |

**Selection rationale:** The paper reports diminishing returns beyond n = 50, with n = 100 providing negligible additional improvement. n = 50 is used as the fixed value for Experiments 1, 3, and 4 as a balance between map quality and computational cost (~50× slower than vanilla gradient).

### Visualisation Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Percentile cap | 99th | Removes extreme outliers without discarding too much signal. Lower values (e.g., 95th) over-saturate; higher values (e.g., 100th) cause washout. Recommended in the paper. |
| Normalisation range | [0, 1] | Required for grayscale display with matplotlib |
| Colormap (Exp 1–3) | Grayscale | Conventional for saliency maps; avoids false colour interpretation |
| Colormap (Exp 4) | RdBu_r | Diverging colormap centred at zero; red = positive class, blue = negative class, white = neutral. Matches paper Fig 6. |

### ResNet-50 Pretrained Weights

| Parameter | Value |
|---|---|
| Weights source | `torchvision.models.ResNet50_Weights.IMAGENET1K_V1` |
| Original training: optimizer | SGD with momentum 0.9 |
| Original training: learning rate | 0.1, decayed by 10× every 30 epochs |
| Original training: weight decay | 1e-4 |
| Original training: batch size | 256 |
| Original training: epochs | 90 |

These are provided for completeness — they describe how the pretrained weights were originally produced by the torchvision team and are not parameters we set or tuned.

---

## 7. SOTA Comparison

### Methods Compared

We compare SmoothGrad against other gradient-based saliency techniques discussed in the original paper. Since quantitative ground-truth evaluation of saliency maps remains an open problem (there is no definitive "correct" saliency map for a natural image), comparison is primarily qualitative, consistent with standard practice in the literature.

### Method Descriptions

| Method | Description | Complexity |
|---|---|---|
| **Vanilla Gradient** | `∂Sc/∂x` at input point | 1 forward + 1 backward pass |
| **SmoothGrad** (ours) | Average of `n` gradients at noisy inputs | n forward + n backward passes |
| **Integrated Gradients** (Sundararajan et al., 2017) | Accumulate gradients along straight-line path from baseline to input | k forward + k backward passes |
| **Guided Backpropagation** (Springenberg et al., 2014) | Modified backprop that only passes positive gradients through ReLU | 1 forward + 1 modified backward pass |
| **GradCAM** (Selvaraju et al., 2016) | Weighted sum of feature maps using gradients at the final conv layer | 1 forward + 1 backward, class activation map only |

### Qualitative Comparison

#### Visual Coherence (ability to highlight the object, not background)

| Method | Visual Coherence | Notes |
|---|---|---|
| Vanilla Gradient | Poor | Scattered noise across entire image, hard to identify object |
| **SmoothGrad** | **Good** | Clearly highlights object boundaries and key features |
| Integrated Gradients | Moderate | Cleaner than vanilla but requires a baseline image choice |
| Guided Backpropagation | Very high (when it works) | Edge-like, extremely sharp maps |
| GradCAM | Moderate | Low resolution (7×7 at final conv layer), upsampled coarsely |

#### Discriminativity (are maps different for different classes?)

| Method | Discriminativity | Notes |
|---|---|---|
| Vanilla Gradient | Low | Maps for different classes look nearly identical |
| **SmoothGrad** | **High** | Clear spatial separation between class-specific regions |
| Integrated Gradients | Moderate | Better than vanilla, depends heavily on baseline |
| Guided Backpropagation | Low to moderate | Often class-agnostic — highlights edges regardless of class |
| GradCAM | High | Class-specific by design (uses class-specific gradients at conv layer) |

#### Failure Cases

| Method | Known Failure Mode |
|---|---|
| Vanilla Gradient | Unreliable on any image — noise is always present |
| **SmoothGrad** | Over-smoothing at high σ; computationally expensive for large n |
| Guided Backpropagation | Produces near-identical maps for different classes (low discriminativity); fails on uniform backgrounds |
| Integrated Gradients | Sensitive to baseline choice; straight-line path may not reflect true feature importance |
| GradCAM | Resolution limited to final conv layer (7×7 for ResNet-50); misses fine-grained pixel-level detail |

### Quantitative Context

The original paper notes that quantitative evaluation of saliency maps "remains an unsolved problem" — there is no universally accepted metric. However, some proxy metrics from the literature include:

| Metric | SmoothGrad | Vanilla Gradient |
|---|---|---|
| **Insertion AUC** (how quickly does adding important pixels recover model score) | Higher | Lower |
| **Deletion AUC** (how quickly does removing important pixels drop model score) | Higher | Lower |
| **Sparsity** (are highlighted regions compact?) | More compact | More diffuse |
| **Sensitivity** (does the map change when the prediction changes?) | Higher | Lower |

These metrics from post-paper work (Samek et al., 2017; Hooker et al., 2019) consistently favour SmoothGrad over vanilla gradient.

### Computational Cost Comparison

| Method | Relative Compute (vs Vanilla) | Suitable for Real-Time? |
|---|---|---|
| Vanilla Gradient | 1× | Yes |
| **SmoothGrad (n=50)** | **~50×** | **No (offline analysis)** |
| Integrated Gradients (k=50) | ~50× | No |
| Guided Backpropagation | ~1× | Yes |
| GradCAM | ~1× | Yes |

SmoothGrad's main practical limitation is compute cost — it requires `n` full forward and backward passes, making it ~50× slower than vanilla gradient. For a 224×224 image on CPU, this takes a few seconds per image. On GPU it is practical for batch analysis.

### Summary

SmoothGrad occupies a strong middle ground in the XAI landscape:
- Significantly better visual coherence and discriminativity than vanilla gradient
- Simpler to implement than Integrated Gradients (no baseline needed)
- More spatially detailed than GradCAM
- More reliably discriminative than Guided Backpropagation
- Main trade-off: computational cost scales linearly with `n`

The original paper's central claim — that adding noise during inference produces cleaner, more interpretable saliency maps — is confirmed by our reproduction. The effect is consistent across all 5 test images, all 4 experiments, and aligns with the trends reported in Figures 3–6 of the paper.

---

## References

1. Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017). *SmoothGrad: removing noise by adding noise.* arXiv:1706.03825.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.
3. Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). *Deep inside convolutional networks: Visualising image classification models and saliency maps.* arXiv:1312.6034.
4. Springenberg, J.T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2014). *Striving for simplicity: The all convolutional net.* arXiv:1412.6806. (Guided Backpropagation)
5. Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic attribution for deep networks.* ICML 2017. (Integrated Gradients)
6. Selvaraju, R.R., et al. (2016). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV 2017.
7. Samek, W., Binder, A., Montavon, G., Lapuschkin, S., & Müller, K.R. (2017). *Evaluating the visualization of what a deep neural network has learned.* IEEE TNNLS.
8. Hooker, S., Erhan, D., Kindermans, P.J., & Kim, B. (2019). *A benchmark for interpretability methods in deep neural networks.* NeurIPS 2019.
