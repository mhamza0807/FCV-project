"""
main.py
Entry point — reproduces key results from:
  SmoothGrad: removing noise by adding noise (Smilkov et al., 2017)
  arXiv:1706.03825

Experiments:
  1. Effect of noise level (sigma) — replicates Fig 3
  2. Effect of sample size  (n)    — replicates Fig 4
  3. Vanilla Gradient vs SmoothGrad side-by-side — replicates Fig 5
  4. Discriminativity difference map — replicates Fig 6

All figures are saved to outputs/.
"""

import os
import sys
import torch
import torchvision.models as models

from utils import (
    SAMPLE_IMAGES,
    download_images,
    load_image,
    preprocess,
    postprocess_gradient,
)
from smoothgrad import vanilla_grad, smooth_grad
from visualize import (
    save_noise_level_grid,
    save_sample_size_grid,
    save_comparison_grid,
    save_discriminativity_map,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FIXED = 50          # fixed n for Experiment 1
SIGMA_FIXED = 0.10    # fixed sigma (10 %) for Experiments 2 & 3

# Experiment 1 — sigma sweep
SIGMA_VALUES  = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50]
SIGMA_LABELS  = ["0%", "5%", "10%", "20%", "30%", "50%"]

# Experiment 2 — n sweep
N_VALUES      = [2, 5, 20, 50, 100]
N_LABELS      = [f"n={n}" for n in N_VALUES]

IMG_DIR       = "images"
PERCENTILE    = 99


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def section(title):
    bar = "=" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def compute_maps(model, pil_img, class_idx, device):
    """Return vanilla and SmoothGrad saliency maps (postprocessed)."""
    inp = preprocess(pil_img, device)

    grad_v = vanilla_grad(model, inp, class_idx)
    grad_s = smooth_grad(model, inp, class_idx,
                         n_samples=N_FIXED, noise_level=SIGMA_FIXED)

    smap_v,  smap_v_xi  = postprocess_gradient(grad_v, pil_img, PERCENTILE)
    smap_sg, smap_sg_xi = postprocess_gradient(grad_s, pil_img, PERCENTILE)
    return smap_v, smap_v_xi, smap_sg, smap_sg_xi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    os.makedirs("outputs", exist_ok=True)

    # ------------------------------------------------------------------
    # Load model (ResNet-50, pretrained on ImageNet, frozen)
    # ------------------------------------------------------------------
    section("Loading pretrained ResNet-50")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(DEVICE)
    print("Model loaded — no training performed.")

    # ------------------------------------------------------------------
    # Download sample images
    # ------------------------------------------------------------------
    section("Downloading sample images")
    img_paths = download_images(IMG_DIR)

    # Identify single-object images (not the two-class image)
    single_infos = [info for info in SAMPLE_IMAGES if not info.get("two_class")]
    two_class_info = next((i for i in SAMPLE_IMAGES if i.get("two_class")), None)

    # Load all single-object PIL images
    single_pils = []
    for info in single_infos:
        pil = load_image(img_paths[info["name"]])
        single_pils.append(pil)
        print(f"  Loaded: {info['name']} (class {info['class_idx']})")

    # ==================================================================
    # EXPERIMENT 1 — Effect of noise level (Fig 3)
    # ==================================================================
    section("Experiment 1: Effect of Noise Level")
    print(f"  n fixed = {N_FIXED}, sigma in {SIGMA_LABELS}")

    exp1_results = []
    for idx, (info, pil_img) in enumerate(zip(single_infos, single_pils)):
        row = {"img_name": info["class_name"]}
        inp = preprocess(pil_img, DEVICE)
        for sigma, label in zip(SIGMA_VALUES, SIGMA_LABELS):
            print(f"  [{info['name']}] sigma={label} ...", end=" ", flush=True)
            if sigma == 0.0:
                # sigma=0 is equivalent to vanilla gradient
                grad = vanilla_grad(model, inp, info["class_idx"])
            else:
                grad = smooth_grad(model, inp, info["class_idx"],
                                   n_samples=N_FIXED, noise_level=sigma)
            smap, _ = postprocess_gradient(grad, percentile=PERCENTILE)
            row[label] = smap
            print("done")
        exp1_results.append(row)

    save_noise_level_grid(exp1_results, SIGMA_LABELS)

    # ==================================================================
    # EXPERIMENT 2 — Effect of sample size (Fig 4)
    # ==================================================================
    section("Experiment 2: Effect of Sample Size")
    print(f"  sigma fixed = {SIGMA_FIXED*100:.0f}%, n in {N_VALUES}")

    exp2_results = []
    for idx, (info, pil_img) in enumerate(zip(single_infos, single_pils)):
        row = {"img_name": info["class_name"]}
        inp = preprocess(pil_img, DEVICE)
        for n_val, label in zip(N_VALUES, N_LABELS):
            print(f"  [{info['name']}] {label} ...", end=" ", flush=True)
            grad = smooth_grad(model, inp, info["class_idx"],
                               n_samples=n_val, noise_level=SIGMA_FIXED)
            smap, _ = postprocess_gradient(grad, percentile=PERCENTILE)
            row[label] = smap
            print("done")
        exp2_results.append(row)

    save_sample_size_grid(exp2_results, N_LABELS)

    # ==================================================================
    # EXPERIMENT 3 — Vanilla vs SmoothGrad comparison (Fig 5)
    # ==================================================================
    section("Experiment 3: Vanilla Gradient vs SmoothGrad")
    print(f"  n={N_FIXED}, sigma={SIGMA_FIXED*100:.0f}%")

    exp3_entries = []
    for info, pil_img in zip(single_infos, single_pils):
        print(f"  [{info['name']}] computing maps ...", end=" ", flush=True)
        sv, sv_xi, ssg, ssg_xi = compute_maps(
            model, pil_img, info["class_idx"], DEVICE
        )
        exp3_entries.append({
            "img_name":    info["name"],
            "class_name":  info["class_name"],
            "original":    pil_img,
            "vanilla":     sv,
            "smoothgrad":  ssg,
            "vanilla_xi":  sv_xi,
            "smoothgrad_xi": ssg_xi,
        })
        print("done")

    save_comparison_grid(exp3_entries)

    # ==================================================================
    # EXPERIMENT 4 — Discriminativity (Fig 6)
    # Uses the two-object image (cat + dog)
    # ==================================================================
    section("Experiment 4: Discriminativity Map")

    if two_class_info is not None and two_class_info["name"] in img_paths:
        tc_pil = load_image(img_paths[two_class_info["name"]])
        inp_tc  = preprocess(tc_pil, DEVICE)

        class_a = two_class_info["class_idx_a"]
        class_b = two_class_info["class_idx_b"]
        name_a  = two_class_info["class_name_a"]
        name_b  = two_class_info["class_name_b"]

        print(f"  SmoothGrad for class {name_a} (idx {class_a}) ...",
              end=" ", flush=True)
        grad_a = smooth_grad(model, inp_tc, class_a,
                             n_samples=N_FIXED, noise_level=SIGMA_FIXED)
        map_a, _ = postprocess_gradient(grad_a, percentile=PERCENTILE)
        print("done")

        print(f"  SmoothGrad for class {name_b} (idx {class_b}) ...",
              end=" ", flush=True)
        grad_b = smooth_grad(model, inp_tc, class_b,
                             n_samples=N_FIXED, noise_level=SIGMA_FIXED)
        map_b, _ = postprocess_gradient(grad_b, percentile=PERCENTILE)
        print("done")

        save_discriminativity_map(
            tc_pil, map_a, map_b, name_a, name_b,
            method_label="SmoothGrad"
        )

        # Also show vanilla gradient discriminativity for comparison
        print("  Vanilla Gradient discriminativity ...", end=" ", flush=True)
        grad_va = vanilla_grad(model, inp_tc, class_a)
        grad_vb = vanilla_grad(model, inp_tc, class_b)
        vmap_a, _ = postprocess_gradient(grad_va, percentile=PERCENTILE)
        vmap_b, _ = postprocess_gradient(grad_vb, percentile=PERCENTILE)
        print("done")

        save_discriminativity_map(
            tc_pil, vmap_a, vmap_b, name_a, name_b,
            method_label="Vanilla Gradient",
            out_name="exp4_discriminativity_vanilla.png"
        )
    else:
        print("  Two-class image unavailable — skipping Experiment 4.")

    # ==================================================================
    section("All experiments complete")
    print("  Output figures saved to:  outputs/")
    print("  exp1_noise_level.png          — Fig 3 replica")
    print("  exp2_sample_size.png          — Fig 4 replica")
    print("  exp3_comparison.png           — Fig 5 replica")
    print("  exp4_discriminativity.png     — Fig 6 replica (SmoothGrad)")
    print("  exp4_discriminativity_vanilla.png  — Fig 6 (Vanilla, for comparison)")


if __name__ == "__main__":
    main()
