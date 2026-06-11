"""
demo.py — run BOTH engines on one sample image and print their validator-exact scores
side by side. This is the tuning loop: see the baseline (often 0.0, overshoots the cap)
vs the FMN engine (hugs the 0.003 floor).

    python demo.py                 # both engines
    python demo.py --engine fmn    # just the FMN engine
    python demo.py --engine base   # just the baseline
    python demo.py --steps 40      # FMN iteration count

The scoring (score_like_validator) mirrors neurons/validator.verify_and_score, so a PASS
here predicts a PASS on the real subnet.
"""

import argparse
import importlib.util
import os
import time

import torch

from perturbnet.model import load_efficientnet_v2_l
from flipper import load_clean_image, score_like_validator, target_index_of, warmup, perturb as perturb_fmn

# flipper-base.py has a hyphen (not import-friendly), so load it by file path.
_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flipper-base.py")
_spec = importlib.util.spec_from_file_location("flipper_base", _base_path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
perturb_base = _base.perturb


def _run(name, fn, model, clean, target_index, epsilon, min_delta, device, **kw):
    t0 = time.time()
    adv = fn(model, clean, target_index, epsilon, min_delta, device, **kw)
    attack_time = time.time() - t0                      # wall-clock the miner would spend
    report = score_like_validator(model, clean, adv, epsilon, min_delta, device)
    print(f"\n=== {name} ===")
    for k, v in report.items():
        print(f"{k:>18}: {v}")
    print(f"{'attack_time':>18}: {attack_time:.3f}s")   # informational only (speed isn't scored)
    return report


def main():
    ap = argparse.ArgumentParser(description="Compare baseline PGD vs FMN on the Perturb objective.")
    ap.add_argument("--engine", choices=["both", "fmn", "base"], default="both")
    ap.add_argument("--steps", type=int, default=30, help="FMN iteration count")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    model = load_efficientnet_v2_l(device)
    clean = load_clean_image(device)
    warmup(model, device, clean)                      # pay one-time CUDA/cuDNN init up front

    target_index = target_index_of(model, clean, device)  # model's clean prediction = class to flee
    epsilon, min_delta = 0.12, 0.003                  # representative challenge values

    if args.engine in ("both", "base"):
        _run("BASELINE (PGD)", perturb_base, model, clean, target_index, epsilon, min_delta, device)
    if args.engine in ("both", "fmn"):
        _run("FMN (minimal-norm, floor-hugging)", perturb_fmn, model, clean,
             target_index, epsilon, min_delta, device, steps=args.steps)


if __name__ == "__main__":
    main()
