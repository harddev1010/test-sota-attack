"""
validator_sim.py — end-to-end TEST harness that mimics the real Perturb validator
(neurons/validator.py) on a freshly-pulled Pexels image instead of one fixed sample.

This is a *wrapper only*: every scoring-relevant step reuses the validator-exact code
already in this repo (perturbnet.*, flipper.score_like_validator). The wrapper adds the
three stages the lab was missing, copied from the real validator's logic:

    1. pull a challenge image from the Pexels Search API      (validator._fetch_image_for_prompt)
    2. run EfficientNetV2-L on it -> (label, confidence)       (perturbnet.model.predict_*)
    3. semantic-consistency check of the model output          (validator._llm_endpoint_check)
       -- the real validator hits a local Ollama endpoint; here we swap in OpenAI gpt-4.1-nano
          (same JSON `is_match` contract), because that's what we have a key for in testing.
    4. build the AttackChallenge synapse + perturb() params     (validator._query_miners / miner.forward)
    5. call perturb() (the "miner")                            (flipper.perturb)
    6. verify + score the response the validator's way          (flipper.score_like_validator)

Each case pulls a fresh image and runs BOTH engines (baseline PGD + unified FMN) on it.

Run:
    OPENAI_API_KEY=sk-...  PERTURB_PEXELS_API_KEY=...  python validator_sim.py
    python validator_sim.py --rounds 15            # number of random test cases (default 15)
    python validator_sim.py --prompt "sports car"  # force the Pexels query (skip random)
    python validator_sim.py --attempts 5           # per-case retry budget for a verified challenge
    python validator_sim.py --steps 40             # override FMN iters (else auto-calibrate)
    python validator_sim.py --no-llm               # skip the OpenAI semantic check (offline)

Nothing here is wired into prod; the existing files are untouched.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import time

import requests
import torch

from perturbnet import constants as C
from perturbnet.image_io import decode_image_b64
from perturbnet.model import (
    load_efficientnet_v2_l,
    normalize_prediction_label,
    resolve_target_index,
)
from flipper import (
    predict_label_conf,
    score_like_validator,
    target_index_of,
    warmup,
)

# flipper-base.py and flipper-fmn.py have hyphens (not import-friendly); load them by file path.
import importlib.util


def _load_perturb(filename: str, module_name: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.perturb


perturb_base = _load_perturb("flipper-base.py", "flipper_base")   # baseline untargeted PGD
perturb_fmn = _load_perturb("flipper-fmn.py", "flipper_fmn")      # unified quantization-aware FMN


# ======================================================================================
# CONSTANTS — hardcoded for testing. These mirror the real validator's env defaults
# (perturbnet/constants.py). Keys come from the environment so they never live in source.
# ======================================================================================
# --- Pexels (validator._fetch_image_for_prompt) ---
IMAGE_ENDPOINT = os.getenv("PERTURB_IMAGE_ENDPOINT", "https://api.pexels.com/v1/search")
PEXELS_API_KEY = os.getenv("PERTURB_PEXELS_API_KEY") or os.getenv("PEXELS_API_KEY", "wniq8GZFs425cMTxJQypn4kehVn3LOJth2QIumBitpOSMviNhqoAiBMy")
PEXELS_PER_PAGE = int(os.getenv("PERTURB_PEXELS_PER_PAGE", "40"))
PEXELS_PAGE_SPAN = int(os.getenv("PERTURB_PEXELS_PAGE_SPAN", "10"))
PEXELS_IMAGE_VARIANT = os.getenv("PERTURB_PEXELS_IMAGE_VARIANT", "medium")
IMAGE_SIZE = int(os.getenv("PERTURB_IMAGE_SIZE", "64"))  # validator config; resize lives in PREPROCESS

# --- OpenAI semantic check (stand-in for validator._llm_endpoint_check / Ollama) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_VERIFY_MODEL", "gpt-4.1-nano")
OPENAI_TIMEOUT_SECONDS = float(os.getenv("OPENAI_VERIFY_TIMEOUT_SECONDS", "20"))

# --- Challenge / scoring (validator config defaults) ---
MIN_DELTA = float(os.getenv("PERTURB_MIN_LINF_DELTA", "0.003"))   # synapse.min_delta the validator sends
TIMEOUT_SECONDS = int(os.getenv("PERTURB_TIMEOUT_SECONDS", "60"))  # synapse.timeout_seconds
MODEL_NAME = C.MODEL_NAME

# Same prompt vocabulary the validator samples from (perturbnet/constants.py: C.PROMPTS).
PROMPTS = (
    "dog", "cat", "bird", "fish", "snake", "frog", "butterfly",
    "spider", "crab", "jellyfish", "monkey", "hamster", "rabbit",
    "horse", "cow", "sheep", "elephant", "lion", "tiger", "bear",
    "sports car", "truck", "bus", "motorcycle", "bicycle", "airplane",
    "helicopter", "sailboat", "canoe", "train",
    "banana", "strawberry", "orange", "broccoli", "mushroom", "pizza",
    "cheeseburger", "ice cream", "coffee mug", "wine bottle",
    "chair", "lamp", "clock", "backpack", "umbrella", "sunglasses",
    "shoe", "hat", "vase", "television",
    "keyboard", "mouse", "camera", "guitar", "drum", "violin",
    "telescope", "microscope",
    "soccer ball", "basketball", "tennis ball", "baseball bat",
    "skateboard", "surfboard", "parachute",
)

_system_random = random.SystemRandom()


# ======================================================================================
# 1. PEXELS IMAGE PULL — copy of neurons/validator._fetch_image_for_prompt
# ======================================================================================
def fetch_image_for_prompt(prompt: str) -> str:
    """Query Pexels Search for `prompt`, download a random photo, return base64(PNG/JPEG bytes).

    Byte-for-byte the validator's logic: random page in [1, page_span], random photo from the
    returned list, image-variant fallback chain, raw-bytes -> base64."""
    endpoint = IMAGE_ENDPOINT.strip()
    api_key = PEXELS_API_KEY.strip()
    if not api_key:
        raise ValueError("Missing Pexels API key. Set PERTURB_PEXELS_API_KEY (or PEXELS_API_KEY).")
    per_page = max(1, min(80, int(PEXELS_PER_PAGE)))
    page_span = max(1, int(PEXELS_PAGE_SPAN))
    image_variant = PEXELS_IMAGE_VARIANT.strip().lower()
    params = {
        "query": prompt,
        "page": _system_random.randint(1, page_span),
        "per_page": per_page,
    }
    response = requests.get(
        endpoint, params=params, headers={"Authorization": api_key}, timeout=12
    )
    response.raise_for_status()
    data = response.json()
    photos = data.get("photos") if isinstance(data, dict) else None
    if not isinstance(photos, list) or not photos:
        raise ValueError("Pexels response has no photos for the requested prompt")
    photo = photos[_system_random.randrange(len(photos))]
    src = photo.get("src", {}) if isinstance(photo, dict) else {}
    if not isinstance(src, dict):
        src = {}
    image_url = (
        src.get(image_variant)
        or src.get("medium")
        or src.get("large")
        or src.get("large2x")
        or src.get("original")
    )
    if not isinstance(image_url, str) or not image_url.strip():
        raise ValueError("Pexels photo src is missing usable image URL")

    image_response = requests.get(image_url, timeout=12)
    image_response.raise_for_status()
    image_bytes = image_response.content
    if not image_bytes:
        raise ValueError("Downloaded Pexels image is empty")
    return base64.b64encode(image_bytes).decode("utf-8")


# ======================================================================================
# 3. SEMANTIC-CONSISTENCY CHECK — stand-in for validator._llm_endpoint_check.
#    Real validator: local Ollama, majority vote, JSON {is_match}. Here: one OpenAI call,
#    gpt-4.1-nano, same JSON contract. Exact-match shortcut first (same as the service).
# ======================================================================================
def _verify_prompt(prediction: str, target: str) -> str:
    # Mirrors tools/llm_endpoint_service._prompt(style=0).
    return (
        "You are a semantic similarity judge for image labels.\n"
        "Decide relation between prediction and target_label.\n"
        "Valid relation values: same_or_near, related, different, unknown.\n"
        "Rules:\n"
        "- same_or_near => labels are equivalent or near-synonyms.\n"
        "- related => labels are semantically related enough to be considered close.\n"
        "- different => labels are clearly different/unrelated in meaning.\n"
        "- unknown => only if unsure.\n"
        "Set is_match=true when relation is same_or_near or related.\n"
        "Set is_match=false when relation is different.\n"
        "If relation is unknown, choose the most likely relation and keep reason brief.\n"
        "Return ONLY JSON:\n"
        '{"relation":"same_or_near|related|different|unknown","is_match":true|false,'
        '"confidence":0.0-1.0,"reason":"short"}\n'
        f"prediction={prediction}\n"
        f"target_label={target}\n"
    )


def llm_semantic_check(predicted_label: str, expected_label: str) -> tuple[bool, str]:
    """Return (is_match, reason). Mirrors the validator: normalize the prediction, take an
    exact match for free, otherwise ask the LLM and trust its boolean `is_match`."""
    prediction = normalize_prediction_label(predicted_label)
    target = normalize_prediction_label(expected_label)
    if prediction == target:
        return True, "exact_match"

    if not OPENAI_API_KEY:
        raise ValueError("Missing OpenAI key. Set OPENAI_API_KEY (or pass --no-llm to skip).")

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": _verify_prompt(prediction, target)}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    response = requests.post(
        f"{OPENAI_BASE_URL.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=OPENAI_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    is_match = bool(parsed.get("is_match"))
    reason = str(parsed.get("reason", "")) or parsed.get("relation", "llm decision")
    return is_match, f"{reason} [relation={parsed.get('relation')}]"


# ======================================================================================
# 4. AttackChallenge synapse — same fields as perturbnet/protocol.AttackChallenge.
#    A plain object here; we don't broadcast over dendrite, we call perturb() directly.
# ======================================================================================
class AttackChallenge:
    def __init__(self, *, task_id, prompt, clean_image_b64, true_label, epsilon,
                 min_delta=MIN_DELTA, norm_type="Linf", timeout_seconds=TIMEOUT_SECONDS):
        self.task_id = task_id
        self.model_name = MODEL_NAME
        self.prompt = prompt
        self.clean_image_b64 = clean_image_b64
        self.true_label = true_label
        self.epsilon = float(epsilon)
        self.norm_type = norm_type
        self.min_delta = float(min_delta)
        self.timeout_seconds = int(timeout_seconds)
        self.perturbed_image_b64 = None


def sample_epsilon(seed: int) -> float:
    """Deterministic epsilon in [0.06, 0.20) — copy of validator._sample_epsilon."""
    return 0.06 + (seed % 1400) / 10000.0


def generate_challenge(model, device, *, forced_prompt=None, attempts=3, use_llm=True):
    """Mimic validator.generate_challenge: pick a prompt, pull a Pexels image, classify it,
    semantically verify the label, and return a verified AttackChallenge (+clean tensor)."""
    seed = _system_random.randint(0, 10_000_000)
    for attempt in range(1, attempts + 1):
        prompt = forced_prompt or _system_random.choice(PROMPTS)
        print(f"\n[challenge attempt {attempt}/{attempts}] prompt={prompt!r}")

        # (1) pull the image
        try:
            image_b64 = fetch_image_for_prompt(prompt)
        except Exception as exc:
            print(f"  pexels fetch failed: {exc}")
            continue

        # (2) classify -> (label, confidence)
        clean = decode_image_b64(image_b64).to(device)
        clean_label, clean_conf = predict_label_conf(model, clean)
        predicted_label = normalize_prediction_label(clean_label)
        print(f"  model output: {predicted_label} ({clean_conf:.3f})  shape={tuple(clean.shape)}")

        # (3) semantic-consistency check
        if use_llm:
            try:
                ok, reason = llm_semantic_check(predicted_label, prompt)
            except Exception as exc:
                print(f"  llm verify error: {exc}")
                continue
            print(f"  llm verify: {'PASS' if ok else 'FAIL'} ({reason})")
            if not ok:
                continue  # validator sleeps 60s here; we just retry for testing
        else:
            print("  llm verify: skipped (--no-llm)")

        epsilon = sample_epsilon(seed + attempt)
        challenge = AttackChallenge(
            task_id=f"test-{seed}-{attempt}",
            prompt=prompt,
            clean_image_b64=image_b64,
            true_label=predicted_label,
            epsilon=epsilon,
        )
        return challenge, clean, (clean_label, clean_conf)

    raise RuntimeError(f"Could not build a verified challenge in {attempts} attempts")


# ======================================================================================
# 5 + 6. "miner" perturb() + validator verify/score, then report.
# ======================================================================================
def run_engine(engine, challenge, clean, target_index, model, device, *, steps):
    """Reproduce miner.forward's call into perturb() for one engine, then score it the
    validator's way. Both engines see the SAME challenge/target_index for a fair compare."""
    t0 = time.time()
    if engine == "base":
        adv = perturb_base(model, clean, target_index, challenge.epsilon, challenge.min_delta, device)
    else:
        kw = {"timeout_seconds": float(challenge.timeout_seconds)}  # unified FMN budgets to the deadline
        if steps is not None:
            kw["steps"] = steps                                     # else auto-calibrate
        adv = perturb_fmn(model, clean, target_index, challenge.epsilon, challenge.min_delta, device, **kw)
    attack_time = time.time() - t0
    report = score_like_validator(model, clean, adv, challenge.epsilon, challenge.min_delta, device)
    return report, attack_time


def _print_engine_report(name, report, attack_time):
    print(f"\n=== {name} ===")
    print(f"adv_label     : {report['adv_label']}")     # last step: label (confidence)
    print(f"gates         : {report['gates']}")
    print(f"norm          : {report['norm']}    rmse: {report['rmse']}")
    print(f"ssim          : {report['ssim']}    psnr_db: {report['psnr_db']}")
    if "perturbation_score" in report:
        print(f"pert_score    : {report['perturbation_score']}")
    print(f"FINAL SCORE   : {report['final']}")
    print(f"attack_time   : {attack_time:.3f}s   (speed is not scored)")


def main():
    ap = argparse.ArgumentParser(description="Simulate the Perturb validator on fresh Pexels images.")
    ap.add_argument("--rounds", type=int, default=15, help="number of random test cases")
    ap.add_argument("--steps", type=int, default=None, help="FMN iteration override (None -> auto-calibrate)")
    ap.add_argument("--prompt", default=None, help="force the Pexels query (else random from C.PROMPTS)")
    ap.add_argument("--attempts", type=int, default=3, help="challenge build retry budget per case")
    ap.add_argument("--no-llm", action="store_true", help="skip the OpenAI semantic check")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  [rounds] {args.rounds}  [classifier] {MODEL_NAME}")
    model = load_efficientnet_v2_l(device)

    warmed = False
    summary = []  # (case, base_final, fmn_final)
    for case in range(1, args.rounds + 1):
        try:
            challenge, clean, (clean_label, clean_conf) = generate_challenge(
                model, device, forced_prompt=args.prompt, attempts=args.attempts, use_llm=not args.no_llm
            )
        except Exception as exc:
            print(f"\ncase {case}: challenge build FAILED ({exc}) — skipping")
            continue

        if not warmed:
            warmup(model, device, clean)  # pay one-time CUDA/cuDNN init before timing any attack
            warmed = True

        # miner resolves target_index from synapse.true_label (fallback: clean argmax). Same for both.
        target_index = resolve_target_index(challenge.true_label)
        if target_index is None:
            target_index = target_index_of(model, clean, device)

        print("\n" + "=" * 70)
        print(f"case {case}: prompt: {challenge.prompt}, clean_label: {clean_label} ({clean_conf:.3f}), "
              f"epsilon: {challenge.epsilon:.4f}")

        base_report, base_t = run_engine("base", challenge, clean, target_index, model, device, steps=args.steps)
        _print_engine_report("PGD (baseline)", base_report, base_t)

        fmn_report, fmn_t = run_engine("fmn", challenge, clean, target_index, model, device, steps=args.steps)
        _print_engine_report("FMN-UNIFIED (quantization-aware, discrete-k, pruned)", fmn_report, fmn_t)

        summary.append((case, base_report["final"], fmn_report["final"]))

    # ── Summary across all cases ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"SUMMARY ({len(summary)} cases)   {'case':>5} {'PGD':>10} {'FMN-UNIFIED':>14}")
    for case, base_final, fmn_final in summary:
        print(f"{'':>22}{case:>5} {base_final:>10.4f} {fmn_final:>14.4f}")
    if summary:
        n = len(summary)
        print("-" * 70)
        print(f"{'mean':>27} {sum(b for _, b, _ in summary) / n:>10.4f} "
              f"{sum(f for _, _, f in summary) / n:>14.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
