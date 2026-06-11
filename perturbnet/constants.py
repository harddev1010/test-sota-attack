"""
constants.py — the REAL Perturb scoring constants, taken from the live subnet's
perturbnet/constants.py (defaults shown; in prod they are env-overridable).

These are no longer guesses — they are the values the validator actually uses.
"""

from __future__ import annotations

import os


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


MODEL_NAME = "EfficientNetV2-L"

# Hard-gate thresholds (measured in native-resolution [0,1] image space).
MIN_LINF_DELTA = _env_float("PERTURB_MIN_LINF_DELTA", 0.003)   # perturbation must be >= this
MAX_LINF_DELTA = _env_float("PERTURB_MAX_LINF_DELTA", 0.03)    # ...and <= min(epsilon, this)
MIN_SSIM = _env_float("PERTURB_MIN_SSIM", 0.98)               # structural similarity floor
MIN_PSNR_DB = _env_float("PERTURB_MIN_PSNR_DB", 38.0)         # peak signal-to-noise floor (dB)

# Perturbation sub-score component weights.
LINF_COMPONENT_WEIGHT = _env_float("PERTURB_LINF_COMPONENT_WEIGHT", 0.7)
RMSE_COMPONENT_WEIGHT = _env_float("PERTURB_RMSE_COMPONENT_WEIGHT", 0.3)

# Final score weights. NOTE: SPEED_WEIGHT defaults to 0 -> speed does NOT affect the score;
# only the per-challenge timeout (a hard cutoff) and processed_count eligibility care about it.
SPEED_WEIGHT = _env_float("PERTURB_SPEED_WEIGHT", 0.0)
PERTURBATION_WEIGHT = _env_float("PERTURB_PERTURBATION_WEIGHT", 1.0)

# Per-challenge timeout the validator allows (seconds). epsilon is sampled in [0.06, 0.20],
# so min(epsilon, MAX_LINF_DELTA) is effectively always MAX_LINF_DELTA = 0.03.
TIMEOUT_SECONDS = _env_float("PERTURB_TIMEOUT_SECONDS", 15.0)
