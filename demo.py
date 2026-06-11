"""
demo.py — the default entry point and the whole tuning loop in one file.

    python demo.py                 # default attack (fmn) on a downloaded sample image
    python demo.py --attack pgd    # try a different attack
    python demo.py --attack fmn --hug-floor   # lift a too-small perturbation above the floor

It: (1) downloads ONE sample image if missing, (2) runs the chosen attack via
flipper.run_attack(), (3) scores the result with verifier.score() using the MEASURED
attack time, and (4) prints the combined report. Watch perturbation_score / speed_score
and adjust the attack or its knobs to push the FINAL score up.
"""

import argparse
import os
import urllib.request

from flipper import run_attack
from verifier import score, print_report

# A small, reliable ImageNet-style sample (a dog) from the official PyTorch hub assets.
SAMPLE_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
SAMPLE_PATH = "sample.jpg"


def ensure_sample():
    """Download the sample image once; reuse it on later runs."""
    if not os.path.exists(SAMPLE_PATH):
        print(f"downloading sample image -> {SAMPLE_PATH}")
        urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    else:
        print(f"using cached sample image: {SAMPLE_PATH}")
    return SAMPLE_PATH


def main():
    ap = argparse.ArgumentParser(description="Flip an EfficientNetV2-L label and score it like Perturb.")
    ap.add_argument("--attack", default="fmn", choices=["fmn", "ddn", "fab", "pgd", "cw"],
                    help="which SOTA attack to compose (default: fmn)")
    ap.add_argument("--hug-floor", action="store_true",
                    help="rescale tiny perturbations just above min_linf_delta")
    ap.add_argument("--steps", type=int, default=50, help="iteration count for iterative attacks")
    args = ap.parse_args()

    image = ensure_sample()

    # 1) ATTACK — flip the label and get back the saved adversarial PNG + timing/metrics.
    adv_path, info = run_attack(image, attack=args.attack,
                                hug_floor_on=args.hug_floor, steps=args.steps)

    # 2) SCORE — feed clean + adversarial + the measured attack time into the validator replica.
    report = score(image, adv_path, response_time=info["time"])
    print_report(report)

    # 3) One-line takeaway to guide the next tuning step.
    if report["final"] > 0:
        print(f">>> SUCCESS: final Perturb score = {report['final']:.4f}. "
              f"Lower distortion or faster attack -> higher score.")
    else:
        failed = [k for k, v in report["gates"].items() if not v]
        print(f">>> SCORE 0.0 — failed gate(s): {failed}. "
              f"If it's only 'linf_above_floor', re-run with --hug-floor.")


if __name__ == "__main__":
    main()
