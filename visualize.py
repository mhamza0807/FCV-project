"""
visualize.py
All figure generation functions for SmoothGrad experiments.
Figures are saved as PNGs into the outputs/ folder.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed — works in headless mode
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

OUTPUT_DIR = "outputs"


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _imshow_gray(ax, smap, title="", fontsize=9):
    """Helper: display a (H,W) float [0,1] array as a grayscale heatmap."""
    ax.imshow(smap, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=fontsize, pad=3)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Experiment 1 — Effect of noise level (Fig 3 of paper)
# Rows = images, Columns = sigma values
# ---------------------------------------------------------------------------
def save_noise_level_grid(results, sigma_labels, out_name="exp1_noise_level.png"):
    """
    Args:
        results     : list of dicts, one per image.
                      Each dict maps sigma_label (str) -> (H,W) saliency map.
        sigma_labels: ordered list of sigma label strings (e.g. ['0%','5%',...])
        out_name    : output filename inside outputs/
    """
    _ensure_output_dir()
    n_rows = len(results)
    n_cols = len(sigma_labels)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.2, n_rows * 2.2))
    fig.suptitle("Experiment 1 — Effect of Noise Level (n=50 samples)\n"
                 "Columns: sigma  |  Rows: images",
                 fontsize=11, y=1.01)

    for row_idx, row_data in enumerate(results):
        img_name = row_data.get("img_name", f"Image {row_idx+1}")
        for col_idx, label in enumerate(sigma_labels):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            smap = row_data[label]
            title = label if row_idx == 0 else ""
            if col_idx == 0:
                ax.set_ylabel(img_name, fontsize=8)
            _imshow_gray(ax, smap, title=title)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Experiment 2 — Effect of sample size (Fig 4 of paper)
# Rows = images, Columns = n values
# ---------------------------------------------------------------------------
def save_sample_size_grid(results, n_labels, out_name="exp2_sample_size.png"):
    """
    Args:
        results  : list of dicts, one per image.
                   Each dict maps n_label (str) -> (H,W) saliency map.
        n_labels : ordered list of n label strings (e.g. ['n=2','n=5',...])
    """
    _ensure_output_dir()
    n_rows = len(results)
    n_cols = len(n_labels)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.2, n_rows * 2.2))
    fig.suptitle("Experiment 2 — Effect of Sample Size (sigma=10%)\n"
                 "Columns: n  |  Rows: images",
                 fontsize=11, y=1.01)

    for row_idx, row_data in enumerate(results):
        img_name = row_data.get("img_name", f"Image {row_idx+1}")
        for col_idx, label in enumerate(n_labels):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]
            smap = row_data[label]
            title = label if row_idx == 0 else ""
            if col_idx == 0:
                ax.set_ylabel(img_name, fontsize=8)
            _imshow_gray(ax, smap, title=title)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Experiment 3 — Vanilla vs SmoothGrad side-by-side (Fig 5 of paper)
# Rows = images
# Columns = [original, vanilla, SmoothGrad, vanilla×input, SmoothGrad×input]
# ---------------------------------------------------------------------------
def save_comparison_grid(entries, out_name="exp3_comparison.png"):
    """
    Args:
        entries: list of dicts, each containing:
            'img_name'         : str
            'original'         : PIL.Image (224x224 RGB)
            'vanilla'          : (H,W) saliency
            'smoothgrad'       : (H,W) saliency
            'vanilla_xi'       : (H,W) saliency × input  (or None)
            'smoothgrad_xi'    : (H,W) saliency × input  (or None)
            'class_name'       : str
    """
    _ensure_output_dir()
    n_rows = len(entries)
    col_titles = ["Original", "Vanilla\nGradient", "SmoothGrad",
                  "Vanilla\n×Input", "SmoothGrad\n×Input"]
    n_cols = len(col_titles)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.2, n_rows * 2.4))
    fig.suptitle("Experiment 3 — Vanilla Gradient vs SmoothGrad",
                 fontsize=12, y=1.01)

    for row_idx, entry in enumerate(entries):
        row_axes = axes[row_idx] if n_rows > 1 else axes

        # Column 0: original image
        ax = row_axes[0]
        ax.imshow(np.array(entry["original"].resize((224, 224))))
        ax.axis("off")
        if row_idx == 0:
            ax.set_title(col_titles[0], fontsize=9, pad=3)
        ax.set_ylabel(f"{entry['img_name']}\n({entry['class_name']})",
                      fontsize=7)

        # Columns 1–4: gradient maps
        maps = [entry["vanilla"], entry["smoothgrad"],
                entry.get("vanilla_xi"), entry.get("smoothgrad_xi")]
        for col_idx, smap in enumerate(maps, start=1):
            ax = row_axes[col_idx]
            if smap is not None:
                _imshow_gray(ax, smap,
                             title=col_titles[col_idx] if row_idx == 0 else "")
            else:
                ax.axis("off")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Experiment 4 — Discriminativity map (Fig 6 of paper)
# Shows  scale(grad_class_A) - scale(grad_class_B)  on RdBu colormap
# ---------------------------------------------------------------------------
def save_discriminativity_map(original_img, map_a, map_b,
                               class_name_a, class_name_b,
                               method_label="SmoothGrad",
                               out_name="exp4_discriminativity.png"):
    """
    Args:
        original_img : PIL.Image
        map_a        : (H,W) float, saliency for class A, already in [0,1]
        map_b        : (H,W) float, saliency for class B, already in [0,1]
        class_name_a : str
        class_name_b : str
        method_label : label for method column header
    """
    _ensure_output_dir()

    # Difference map: scale to [-1, 1] as in paper Fig 6
    diff = map_a - map_b          # positive = class A, negative = class B

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    fig.suptitle(
        f"Experiment 4 — Discriminativity ({method_label})\n"
        f"Red = {class_name_a}  |  Blue = {class_name_b}",
        fontsize=11
    )

    # Col 0: original image
    axes[0].imshow(np.array(original_img.resize((224, 224))))
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    # Col 1: map for class A
    _imshow_gray(axes[1], map_a, title=f"Saliency\n({class_name_a})")

    # Col 2: map for class B
    _imshow_gray(axes[2], map_b, title=f"Saliency\n({class_name_b})")

    # Col 3: diverging difference map  [−1, 0, 1] → [blue, white, red]
    im = axes[3].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[3].set_title(f"Difference Map\n({class_name_a} − {class_name_b})", fontsize=9)
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Bonus: quick single-image preview (vanilla + smoothgrad side by side)
# ---------------------------------------------------------------------------
def save_single_comparison(pil_img, vanilla_map, smoothgrad_map, class_name,
                            img_name="image", out_name=None):
    _ensure_output_dir()
    if out_name is None:
        out_name = f"preview_{img_name}.png"

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(np.array(pil_img.resize((224, 224))))
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    _imshow_gray(axes[1], vanilla_map, title="Vanilla Gradient")
    _imshow_gray(axes[2], smoothgrad_map, title="SmoothGrad")

    fig.suptitle(f"Class: {class_name}", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
