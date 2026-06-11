"""
verifier.py — a LOCAL replica of the Perturb (netuid 26) validator's scoring.

Given a clean image and an adversarial image (plus how long the attack took), it:
  1) runs the SAME hard gates the validator uses (any fail -> final score 0.0), and
  2) if all gates pass, computes the SAME sub-scores and the final weighted score.

This is how you tune: run an attack, read perturbation_score / speed_score, adjust.

IMPORTANT — measurement space:
  L-infinity and RMSE are measured in [0,1] PIXEL space (the resized/cropped image
  scaled to [0,1], BEFORE mean/std normalization). SSIM/PSNR are computed on the same
  [0,1] images. If perturbnet/model.py measures in a different space (e.g. 0-255 or the
  normalized space), change config.pil_to_unit_tensor / these calls to match.
"""

import argparse

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import config
from model import load_model, predict


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _load_unit_numpy(path):
    """Load an image through the official preprocessing -> HWC numpy array in [0,1]."""
    t = config.pil_to_unit_tensor(Image.open(path))   # (3,H,W) tensor in [0,1]
    return t.permute(1, 2, 0).cpu().numpy()           # -> (H,W,3) for skimage


def score(clean_path, adv_path, response_time):
    """
    Replicate the validator's scoring. Returns a dict with every gate, sub-score, and final.
    """
    model = load_model()

    # ---- Preprocess both images identically (same pipeline as the model sees) ----
    clean_np = _load_unit_numpy(clean_path)
    adv_np = _load_unit_numpy(adv_path)

    clean_t = config.pil_to_unit_tensor(Image.open(clean_path))
    adv_t = config.pil_to_unit_tensor(Image.open(adv_path))

    # ---- Distortion metrics (all in [0,1] pixel space) ----
    linf = float(np.abs(adv_np - clean_np).max())                 # worst-case per-pixel change
    rmse = float(np.sqrt(np.mean((adv_np - clean_np) ** 2)))      # typical per-pixel change
    ssim = float(structural_similarity(clean_np, adv_np, channel_axis=2, data_range=1.0))
    psnr = float(peak_signal_noise_ratio(clean_np, adv_np, data_range=1.0))

    # ---- Labels from the frozen classifier ----
    clean_label, clean_conf = predict(model, clean_t)
    adv_label, adv_conf = predict(model, adv_t)

    # Upper bound on the allowed L-infinity = the tighter of the two ceilings.
    linf_cap = min(config.EPSILON, config.MAX_LINF_DELTA)

    # ================= HARD GATES (any fail -> final = 0.0) =================
    gates = {
        "label_changed": adv_label != clean_label,
        "linf_above_floor": linf >= config.MIN_LINF_DELTA,
        "linf_below_cap": linf <= linf_cap,
        "ssim_ok": ssim >= config.MIN_SSIM,
        "psnr_ok": psnr >= config.MIN_PSNR_DB,
    }
    all_pass = all(gates.values())

    result = {
        "clean_label": clean_label, "clean_conf": clean_conf,
        "adv_label": adv_label, "adv_conf": adv_conf,
        "linf": linf, "rmse": rmse, "ssim": ssim, "psnr": psnr,
        "linf_cap": linf_cap, "response_time": response_time,
        "gates": gates, "all_gates_pass": all_pass,
    }

    if not all_pass:
        # A single failed gate zeroes the whole score — sub-scores are not computed.
        result.update({
            "linf_score": 0.0, "rmse_score": 0.0,
            "perturbation_score": 0.0, "speed_score": 0.0, "final": 0.0,
        })
        return result

    # ================= SUB-SCORES (gates all passed) =================
    span = linf_cap - config.MIN_LINF_DELTA                       # width of the "valid" L-inf band
    # How far up the allowed band did we land? 0 = right at the floor, 1 = at the cap.
    linf_ratio = _clamp((linf - config.MIN_LINF_DELTA) / span, 0, 1) if span > 0 else 0.0
    rmse_ratio = _clamp(rmse / linf_cap, 0, 1)

    # Squared so the reward drops off fast as the perturbation grows: smaller = much better.
    linf_score = (1 - linf_ratio) ** 2
    rmse_score = (1 - rmse_ratio) ** 2

    # Weighted average of the two distortion components (weights from config).
    wsum = config.LINF_COMPONENT_WEIGHT + config.RMSE_COMPONENT_WEIGHT
    perturbation_score = (config.LINF_COMPONENT_WEIGHT * linf_score
                          + config.RMSE_COMPONENT_WEIGHT * rmse_score) / wsum

    # Speed: full marks for instant, zero at/after the timeout, linear in between.
    speed_score = 1 - min(response_time / config.TIMEOUT_SECONDS, 1)

    # Final = how much the two halves are worth (weights from config).
    final = config.PERTURBATION_WEIGHT * perturbation_score + config.SPEED_WEIGHT * speed_score

    result.update({
        "linf_ratio": linf_ratio, "rmse_ratio": rmse_ratio,
        "linf_score": linf_score, "rmse_score": rmse_score,
        "perturbation_score": perturbation_score,
        "speed_score": speed_score, "final": final,
    })
    return result


def print_report(r):
    """Pretty-print the gate-by-gate + sub-score-by-sub-score breakdown."""
    g = r["gates"]
    print("\n================ PERTURB SCORE ================")
    print(f"clean label : {r['clean_label']}  (conf {r['clean_conf']:.3f})")
    print(f"adv   label : {r['adv_label']}  (conf {r['adv_conf']:.3f})")
    print("\n--- HARD GATES (any FAIL => final 0.0) ---")
    print(f"[{'PASS' if g['label_changed'] else 'FAIL'}] label changed        : "
          f"{r['clean_label']!r} -> {r['adv_label']!r}")
    print(f"[{'PASS' if g['linf_above_floor'] else 'FAIL'}] L-inf >= min_delta    : "
          f"{r['linf']:.5f} >= {config.MIN_LINF_DELTA:.5f}")
    print(f"[{'PASS' if g['linf_below_cap'] else 'FAIL'}] L-inf <= cap          : "
          f"{r['linf']:.5f} <= {r['linf_cap']:.5f}  (min(eps,max_delta))")
    print(f"[{'PASS' if g['ssim_ok'] else 'FAIL'}] SSIM >= min_ssim      : "
          f"{r['ssim']:.4f} >= {config.MIN_SSIM:.4f}")
    print(f"[{'PASS' if g['psnr_ok'] else 'FAIL'}] PSNR >= min_psnr_db   : "
          f"{r['psnr']:.2f} dB >= {config.MIN_PSNR_DB:.2f} dB")

    print("\n--- SUB-SCORES ---")
    print(f"linf_score          : {r['linf_score']:.4f}")
    print(f"rmse_score          : {r['rmse_score']:.4f}")
    print(f"perturbation_score  : {r['perturbation_score']:.4f}  "
          f"(weight {config.PERTURBATION_WEIGHT})")
    print(f"speed_score         : {r['speed_score']:.4f}  "
          f"(weight {config.SPEED_WEIGHT}; time {r['response_time']:.3f}s / {config.TIMEOUT_SECONDS}s)")

    verdict = "PASS" if r["all_gates_pass"] else "FAIL (a gate failed -> 0.0)"
    print(f"\nRESULT              : {verdict}")
    print(f"FINAL SCORE         : {r['final']:.4f}")
    print("==============================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="path to the clean image")
    ap.add_argument("--adv", required=True, help="path to the adversarial image")
    ap.add_argument("--time", type=float, required=True, help="attack wall-clock time in seconds")
    args = ap.parse_args()
    print_report(score(args.clean, args.adv, args.time))
