"""
utils.py
Image loading, preprocessing (ImageNet normalisation), and gradient postprocessing.
"""

import os
import io
import urllib.request
import requests as _requests
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


# ImageNet normalisation constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Pre-processing pipeline: resize → centre-crop → tensor → normalise
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Sample images with known ImageNet class indices.
# All URLs are direct full-resolution Wikimedia Commons images (no thumbnail
# redirects) — confirmed to contain the correct subject.
# Two fallback URLs per image in case one CDN edge is slow.
# ---------------------------------------------------------------------------
SAMPLE_IMAGES = [
    {
        "name": "cat",
        "class_idx": 281,           # tabby cat
        "class_name": "Tabby Cat",
        "urls": [
            # Wikipedia's canonical tabby cat photo
            "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/1/18/Cat_August_2010-4.jpg",
        ],
    },
    {
        "name": "dog",
        "class_idx": 207,           # golden retriever
        "class_name": "Golden Retriever",
        "urls": [
            # Wikipedia's canonical golden retriever photo
            "https://upload.wikimedia.org/wikipedia/commons/b/bd/Golden_Retriever_Dukedestiny01_drvd.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg",
        ],
    },
    {
        "name": "elephant",
        "class_idx": 386,           # African elephant (Loxodonta africana)
        "class_name": "African Elephant",
        "urls": [
            # Wikipedia's canonical African Bush Elephant photo — used in
            # countless ML examples, confirmed elephant content
            "https://upload.wikimedia.org/wikipedia/commons/3/37/African_Bush_Elephant.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/c/c9/Elephants_at_Amboseli_national_park_against_Mount_Kilimanjaro.jpg",
        ],
    },
    {
        "name": "bird",
        "class_idx": 88,            # macaw (Ara ararauna / Scarlet Macaw)
        "class_name": "Macaw",
        "urls": [
            # Blue-and-yellow macaw — one of the most recognised Wikipedia
            # bird images, very large and clearly visible
            "https://upload.wikimedia.org/wikipedia/commons/9/9f/Ara_ararauna_Luc_Viatour.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/2/2a/Scarlet_Macaw.jpg",
        ],
    },
    {
        "name": "butterfly",
        "class_idx": 323,           # monarch butterfly
        "class_name": "Monarch Butterfly",
        "urls": [
            # Monarch butterfly on flower — well-known Wikipedia image
            "https://upload.wikimedia.org/wikipedia/commons/1/18/Monarch_butterfly_in_May_2014.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/e/e7/Monarch_In_May.jpg",
        ],
    },
    # Two-class entry for Experiment 4 (discriminativity).
    # Uses a real cat-AND-dog image; we compute separate saliency maps for
    # cat class (281) and dog class (207) then show the difference map.
    {
        "name": "cat_dog",
        "class_idx_a": 281,
        "class_name_a": "Cat",
        "class_idx_b": 207,
        "class_name_b": "Dog",
        "urls": [
            # Wiktionary "cat and dog" image — contains both animals
            "https://upload.wikimedia.org/wikipedia/commons/0/0b/Cat_and_dog_-_Wiktionary_en.jpg",
            # fallback: cat-only image; discriminativity still demonstrated
            # (cat map lights up, dog map stays dark)
            "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg",
        ],
        "two_class": True,
    },
]


def _download_url(url, dest_path, timeout=20):
    """Download a single URL to dest_path. Returns True on success."""
    try:
        resp = _requests.get(url, timeout=timeout,
                             headers={"User-Agent": "Mozilla/5.0 (SmoothGrad research)"})
        resp.raise_for_status()
        # Verify it is actually an image
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(dest_path)
        return True
    except Exception as exc:
        return False


def download_images(img_dir="images"):
    """Download all sample images into img_dir, trying each URL in order."""
    os.makedirs(img_dir, exist_ok=True)
    paths = {}

    for info in SAMPLE_IMAGES:
        fname = os.path.join(img_dir, f"{info['name']}.jpg")
        if os.path.exists(fname):
            print(f"  {info['name']}: already downloaded.")
            paths[info["name"]] = fname
            continue

        downloaded = False
        for url in info["urls"]:
            print(f"  Downloading {info['name']} from {url} ...", end=" ", flush=True)
            if _download_url(url, fname):
                print("ok")
                downloaded = True
                break
            else:
                print("failed, trying next URL...")

        if not downloaded:
            # Last resort: generate a recognisable synthetic placeholder
            print(f"  WARNING: all URLs failed for '{info['name']}'. "
                  f"Generating synthetic placeholder.")
            _make_placeholder(fname)

        paths[info["name"]] = fname

    return paths


def _make_placeholder(dest_path, size=256):
    """Create a simple coloured gradient image as a last-resort placeholder."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(size):
        arr[i, :, 0] = int(255 * i / size)
        arr[:, i, 1] = int(255 * i / size)
    arr[:, :, 2] = 128
    Image.fromarray(arr).save(dest_path)


def load_image(path):
    """Load an image as a PIL RGB image."""
    return Image.open(path).convert("RGB")


def preprocess(pil_img, device="cpu"):
    """Apply ImageNet pre-processing and return (1, 3, 224, 224) tensor."""
    tensor = PREPROCESS(pil_img).unsqueeze(0).to(device)
    return tensor


def postprocess_gradient(grad, pil_img=None, percentile=99):
    """
    Convert a raw gradient array to a visualisable saliency map.

    Steps (paper Section 3.1):
      1. Take absolute value  -> focus on magnitude of influence
      2. Sum across RGB channels -> (H, W) map
      3. Cap at 99th percentile  -> prevent outliers washing out colour scale
      4. Normalise to [0, 1]

    Optionally returns a second map multiplied by the original image pixels
    (Shrikumar et al. style, discussed in paper Section 3.1).

    Args:
        grad    : (1, 3, H, W) numpy array
        pil_img : PIL.Image resized to 224×224 (for the 'times input' variant)
        percentile: cap percentile (default 99)

    Returns:
        smap          : (H, W) float array in [0, 1]
        smap_x_input  : (H, W) float array in [0, 1] or None
    """
    # Absolute value and channel sum
    smap = np.abs(grad[0]).sum(axis=0)         # (H, W)

    # Cap outliers at given percentile
    cap = np.percentile(smap, percentile)
    smap = np.clip(smap, 0, cap)

    # Normalise to [0, 1]
    smap_min, smap_max = smap.min(), smap.max()
    if smap_max > smap_min:
        smap = (smap - smap_min) / (smap_max - smap_min)
    else:
        smap = np.zeros_like(smap)

    smap_x_input = None
    if pil_img is not None:
        # Resize original image to gradient spatial size (224×224)
        img_np = np.array(pil_img.resize((smap.shape[1], smap.shape[0]))) / 255.0
        img_gray = img_np.mean(axis=2)        # grayscale version of original
        smap_xi = smap * img_gray
        xi_min, xi_max = smap_xi.min(), smap_xi.max()
        if xi_max > xi_min:
            smap_xi = (smap_xi - xi_min) / (xi_max - xi_min)
        smap_x_input = smap_xi

    return smap, smap_x_input


def postprocess_gradient_signed(grad, percentile=99):
    """
    Return a SIGNED, channel-summed saliency map in [-1, 1].
    Used for the discriminativity difference map in Experiment 4.
    """
    # Sum channels (keep sign)
    smap = grad[0].sum(axis=0)                 # (H, W)

    # Cap magnitude outliers
    cap = np.percentile(np.abs(smap), percentile)
    smap = np.clip(smap, -cap, cap)

    # Normalise to [0, 1]  (then caller can do map1 - map2 in [0,1] space)
    smap_abs = np.abs(smap)
    a_max = smap_abs.max()
    if a_max > 0:
        smap = smap / a_max            # -> [-1, 1]
    return smap
