"""
config.py — the single source of truth for this project.

It holds TWO things:
  1) The EXACT model + preprocessing the Perturb validator uses (so our local copy
     of the world matches the subnet's copy of the world).
  2) The Perturb scoring constants (thresholds + weights) that the validator plugs
     into its score formula.

Everything downstream (model.py, flipper.py, verifier.py, demo.py) imports from here,
so if you change a number, the whole pipeline changes with it.
"""

import torch
from torchvision.models import EfficientNet_V2_L_Weights
from torchvision.transforms import functional as TF  # functional = call resize/crop directly

# ----------------------------------------------------------------------------------
# DEVICE: use a GPU if one exists, otherwise the CPU. Attacks are iterative, so a GPU
# is much faster — but everything still runs (slower) on CPU.
# ----------------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================================================================================
# 1) MODEL + PREPROCESSING  (must mirror perturbnet/model.py exactly)
# ==================================================================================
# The validator classifies images with EfficientNetV2-L using torchvision's official
# ImageNet weights. We load the SAME weights enum so our labels match theirs.
WEIGHTS = EfficientNet_V2_L_Weights.IMAGENET1K_V1

# weights.transforms() returns the *official* preprocessing object for these weights:
# it resizes, center-crops, scales pixels to [0,1], then normalizes with mean/std.
# We pull the individual parameters out of it so we can split preprocessing into:
#     (a) "geometry + scale to [0,1]"  -> the space we measure/attack in
#     (b) "normalize"                  -> applied INSIDE the model wrapper (model.py)
# Splitting matters because the validator measures L-infinity / RMSE in [0,1] PIXEL
# space, NOT in the normalized space the network actually consumes.
_TF = WEIGHTS.transforms()

# resize_size / crop_size come back as 1-element lists in modern torchvision; unwrap them.
RESIZE_SIZE = _TF.resize_size[0] if isinstance(_TF.resize_size, (list, tuple)) else _TF.resize_size
CROP_SIZE = _TF.crop_size[0] if isinstance(_TF.crop_size, (list, tuple)) else _TF.crop_size
MEAN = list(_TF.mean)            # per-channel mean used by normalize
STD = list(_TF.std)              # per-channel std used by normalize
INTERPOLATION = _TF.interpolation  # e.g. bicubic — keep identical to the official pipeline

# Pre-shaped tensors (C,1,1) so we can broadcast over a (C,H,W) image cheaply.
MEAN_T = torch.tensor(MEAN).view(3, 1, 1)
STD_T = torch.tensor(STD).view(3, 1, 1)

# ImageNet human-readable class names, straight from the weights metadata.
# predict() in model.py turns a class index into one of these strings.
CATEGORIES = WEIGHTS.meta["categories"]


def pil_to_unit_tensor(img):
    """
    PIL image -> float tensor in [0,1] pixel space, shape (3, CROP_SIZE, CROP_SIZE).

    This is steps (a) above: the EXACT geometry + [0,1] scaling of the official
    transform, but WITHOUT normalization. This is the space attacks perturb and the
    space the validator measures distortion in.
    """
    img = img.convert("RGB")                                   # force 3 channels (drop alpha/grayscale)
    img = TF.resize(img, RESIZE_SIZE, interpolation=INTERPOLATION)  # shrink shorter side to RESIZE_SIZE
    img = TF.center_crop(img, CROP_SIZE)                       # crop the center CROP_SIZE x CROP_SIZE
    return TF.to_tensor(img)                                   # HWC uint8 -> CHW float in [0,1]


def normalize(t):
    """Apply ImageNet mean/std normalization. Used INSIDE the model wrapper (model.py)."""
    return (t - MEAN_T.to(t.device)) / STD_T.to(t.device)


# ==================================================================================
# 2) PERTURB SCORING CONSTANTS
# ==================================================================================
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!  THESE NUMBERS ARE GUESSES / PLACEHOLDERS.                                    !!
# !!  REPLACE THEM WITH THE REAL VALUES FROM perturbnet/constants.py AND THE       !!
# !!  REAL MODEL/PREPROCESSING FROM perturbnet/model.py BEFORE TRUSTING ANY SCORE. !!
# !!  See README.md -> "Make it match the real subnet" for the exact grep to run.  !!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# --- Distortion budget (all in [0,1] PIXEL space) ---
EPSILON = 8 / 255          # GUESS: overall L-infinity budget (classic 8/255 ≈ 0.0314)
MIN_LINF_DELTA = 0.002     # GUESS: a perturbation TOO SMALL is rejected (anti free-lunch floor)
MAX_LINF_DELTA = 0.05      # GUESS: a perturbation TOO LARGE is rejected (visibility ceiling)

# --- Perceptual-similarity gates ---
MIN_SSIM = 0.90            # GUESS: structural similarity floor (1.0 = identical)
MIN_PSNR_DB = 30.0         # GUESS: peak signal-to-noise floor in decibels (higher = cleaner)

# --- Speed ---
TIMEOUT_SECONDS = 10.0     # GUESS: responses at/after this take a zero speed score

# --- Score weights (final = PERTURBATION_WEIGHT*perturb + SPEED_WEIGHT*speed) ---
PERTURBATION_WEIGHT = 0.7  # GUESS: how much the "how small/clean" part matters
SPEED_WEIGHT = 0.3         # GUESS: how much the "how fast" part matters

# --- Sub-weights inside the perturbation score (linf vs rmse component) ---
LINF_COMPONENT_WEIGHT = 0.5  # GUESS: weight on the L-infinity component
RMSE_COMPONENT_WEIGHT = 0.5  # GUESS: weight on the RMSE component
