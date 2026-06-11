"""
flipper.py  —  FMN-based perturbation engine for the Perturb subnet (netuid 26).

Drop-in replacement for the baseline PGD block in the real miner. Identical signature to
flipper-base.perturb, so deploying to prod = delete the PGD loop in miner.forward and:

    from flipper import perturb
    adv = perturb(self.model, clean, target_index, epsilon, min_delta, self.device)

Everything `perturb()` needs is imported from perturbnet.* (the same package prod has),
so the function copy-pastes with ZERO edits. The helpers below the engine
(load_clean_image / score_like_validator / warmup) are LAB-ONLY conveniences.

────────────────────────────────────────────────────────────────────────────────────
WHAT WE OPTIMISE FOR (from the real validator.verify_and_score)
────────────────────────────────────────────────────────────────────────────────────
Measured in the DECODED native-resolution [0,1] image:
    norm = max|adv-clean|,   rmse = sqrt(mean((adv-clean)^2))
Score (SPEED_WEIGHT=0, PERTURBATION_WEIGHT=1):
    perturbation_score = 0.7*(1-linf_ratio)^2 + 0.3*(1-rmse_ratio)^2
    linf_ratio = (norm-0.003)/(0.03-0.003)        # min(eps,0.03) is always 0.03
    rmse_ratio = rmse/0.03
…but ONLY if every gate passes: label changed, 0.003<=norm<=0.03, ssim>=0.98, psnr>=38.
Any failure -> 0.0. Speed is irrelevant to the score (only the 15s hard timeout matters).

So the score is maximised by flipping with `norm` as small as possible (hug the 0.003
floor) and tiny rmse. That is precisely what FMN (Fast Minimum-Norm) does — unlike PGD,
which burns a fixed, oversized budget and overshoots the 0.03 cap.

This engine:
  1) runs FMN (composed from foolbox) to get the minimal-L-inf flip DIRECTION, then
  2) does a PNG-AWARE line search along that direction for the SMALLEST perturbation that
     STILL flips after the validator's 8-bit PNG round-trip and passes every gate.
The validator scores a re-decoded PNG (pixels quantised to multiples of 1/255 ~ 0.00392),
so the smallest survivable norm is 1/255 — conveniently just above the 0.003 floor. We
hug exactly that.
"""

import time

import torch

import foolbox as fb  # SOTA attack library — we COMPOSE FMN from it, never reimplement it

from perturbnet import constants as C
from perturbnet.image_io import encode_image_b64, decode_image_b64
from perturbnet.model import logits_for_images, predict_index

# cuDNN autotuning: input H/W is constant within a run, so let cuDNN pick the fastest
# conv kernels (first call a touch slower, every later call faster).
torch.backends.cudnn.benchmark = True

_Q = 1.0 / 255.0  # one 8-bit quantisation step; the finest perturbation a PNG can carry


class _PreprocessedModel(torch.nn.Module):
    """Adapts the classifier so foolbox can feed it native-res [0,1] images.

    foolbox wants a module mapping the ATTACK space -> logits. Our attack space is the
    decoded native-resolution [0,1] image; logits_for_images() applies the exact
    resize+crop+normalize the validator uses, then the network. Wrapping it here means
    FMN's gradients flow back to native pixels — the very pixels the validator measures."""

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N,3,H,W) in [0,1]
        return logits_for_images(model=self.base, image_bchw=x)


def _png_roundtrip(image_chw: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return the image EXACTLY as the validator will see it: encode to PNG, decode back.
    Bakes in 8-bit quantisation so any check here matches scoring reality."""
    return decode_image_b64(encode_image_b64(image_chw)).to(device)


def perturb(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_index: int,
    epsilon: float,
    min_delta: float,
    device: torch.device,
    steps: int = 30,
    timeout_seconds: float = None,
    time_margin: float = 2.0,
) -> torch.Tensor:
    """Minimal-norm flip tuned to the validator's post-PNG scoring, with a hard time budget.

    Same first-6 args / return as flipper-base.perturb (drop-in). Extra optional args:
      timeout_seconds: the challenge timeout (defaults to C.TIMEOUT_SECONDS). The miner can
                       pass synapse.timeout_seconds here.
      time_margin:     finish this many seconds BEFORE the timeout so the encoded response is
                       safely back in time (the validator zeroes any late/missing reply).
    `steps` is the DESIRED FMN iteration count; it is automatically reduced if the hardware
    can't run that many within the budget."""
    t_start = time.time()
    timeout_seconds = float(C.TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds)
    deadline = t_start + max(1.0, timeout_seconds - time_margin)  # the moment we MUST be done by

    epsilon = float(epsilon)
    min_delta = float(min_delta)
    cap = min(epsilon, float(C.MAX_LINF_DELTA))   # score-eligible ceiling (0.03 in prod)
    floor = min_delta                             # score-eligible floor (0.003 in prod)
    clean = clean.to(device)

    fmodel = fb.PyTorchModel(_PreprocessedModel(model).to(device).eval(), bounds=(0.0, 1.0))

    # ---- size FMN steps to the time budget -------------------------------------------
    # One forward+backward ~= one FMN step. Probe that cost once on THIS image (resolution
    # varies per challenge), then fit as many steps as the budget allows, leaving room for
    # the cheap line search. This guarantees we don't overrun the deadline on slow CPUs.
    probe = time.time()
    xp = clean.unsqueeze(0).detach().requires_grad_(True)
    logits_for_images(model=model, image_bchw=xp).sum().backward()
    step_cost = max(time.time() - probe, 1e-3) * 1.3              # +30% safety margin
    line_search_reserve = 16 * 0.6 * step_cost                    # ~16 forward-only candidates
    fmn_budget = (deadline - time.time()) - line_search_reserve
    eff_steps = int(max(1, min(steps, fmn_budget // step_cost)))  # never fewer than 1 step

    # ---- 1) FMN: minimal-L-inf adversarial DIRECTION ---------------------------------
    attack = fb.attacks.LInfFMNAttack(steps=eff_steps)          # minimises the L-inf norm of the flip
    labels = torch.tensor([int(target_index)], device=device)  # the class to move AWAY from
    _, advs, _ = attack(fmodel, clean.unsqueeze(0), labels, epsilons=None)  # None -> minimal
    delta = advs.squeeze(0) - clean                            # perturbation direction FMN found
    base_linf = float(delta.abs().max().item())

    if base_linf < 1e-12:                                       # FMN produced nothing usable
        return clean.clone()                                   # validator will score 0.0

    # ---- 2) PNG-aware line search: smallest survivable flip --------------------------
    # Scale `delta` to target L-inf values from the floor up to the cap. For each, simulate
    # the exact PNG round-trip and accept the FIRST (=smallest, =highest score) one that
    # still flips and lands inside [floor, cap]. Stop early if we reach the deadline.
    start = max(floor, _Q)                                      # below 1/255 it rounds away to nothing
    best_fallback = (clean + delta * (cap / base_linf)).clamp(0.0, 1.0)  # strongest legal try

    target = start
    while target <= cap + 1e-9 and time.time() < deadline:     # <-- deadline guard
        t = min(target, cap)
        cand = (clean + delta * (t / base_linf)).clamp(0.0, 1.0)   # scale direction to L-inf=t
        seen = _png_roundtrip(cand, device)                       # what the validator decodes
        norm = float((seen - clean).abs().max().item())
        if floor <= norm <= cap and predict_index(model=model, image_chw=seen) != target_index:
            return cand          # smallest perturbation that survives PNG AND flips -> best score
        target += 0.5 * _Q       # half-a-level granularity: tight but cheap search

    return best_fallback         # deadline hit or nothing in-band flipped: best effort


# ======================================================================================
# LAB-ONLY helpers (not needed in prod). Shared with flipper-base.py's self-test.
# score_like_validator mirrors neurons/validator.verify_and_score so a local PASS == a
# real PASS.
# ======================================================================================
def warmup(model: torch.nn.Module, device: torch.device, example_chw: torch.Tensor) -> None:
    """Call ONCE at startup: a forward+backward to pay CUDA/cuDNN init up front, so the
    first real challenge isn't slowed by lazy kernel autotuning."""
    x = example_chw.detach().clone().to(device).unsqueeze(0).requires_grad_(True)
    logits_for_images(model=model, image_bchw=x).sum().backward()


_SAMPLE_URL = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
_SAMPLE_PATH = "sample.jpg"


def load_clean_image(device: torch.device) -> torch.Tensor:
    """Download a sample image once and decode it the SAME way the validator does."""
    import base64
    import os
    import urllib.request

    if not os.path.exists(_SAMPLE_PATH):
        print(f"downloading sample image -> {_SAMPLE_PATH}")
        urllib.request.urlretrieve(_SAMPLE_URL, _SAMPLE_PATH)
    with open(_SAMPLE_PATH, "rb") as handle:
        b64 = base64.b64encode(handle.read()).decode("utf-8")
    return decode_image_b64(b64).to(device)


def target_index_of(model, clean, device) -> int:
    """The 'original' class to flee = the model's clean prediction (what the validator uses)."""
    return predict_index(model=model, image_chw=clean.to(device))


def predict_label_conf(model, image_chw) -> tuple:
    """Return (label_string, confidence) from EfficientNetV2-L for a CHW [0,1] image.

    confidence = softmax probability of the top-1 class. The validator only uses the label
    (argmax) for gating; confidence is reported here purely so you can SEE how decisively the
    label flipped."""
    from perturbnet.model import LABELS

    with torch.no_grad():
        logits = logits_for_images(model=model, image_bchw=image_chw.unsqueeze(0))
        probs = logits.softmax(dim=1)
        conf, idx = probs.max(dim=1)
    i = int(idx.item())
    label = LABELS[i] if 0 <= i < len(LABELS) else str(i)
    return label, float(conf.item())


def _compute_ssim(x_clean: torch.Tensor, x_adv: torch.Tensor, kernel_size: int = 11) -> float:
    """Copy of validator._compute_ssim (avg-pool SSIM, c1=0.01^2, c2=0.03^2)."""
    import torch.nn.functional as F

    padding = kernel_size // 2
    x, y = x_clean.unsqueeze(0), x_adv.unsqueeze(0)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x = F.avg_pool2d(x, kernel_size, 1, padding)
    mu_y = F.avg_pool2d(y, kernel_size, 1, padding)
    sigma_x = F.avg_pool2d(x * x, kernel_size, 1, padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size, 1, padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size, 1, padding) - mu_x * mu_y
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float((num / (den + 1e-12)).mean().item())


def _compute_psnr_db(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    """Copy of validator._compute_psnr_db (data_range=1.0; 99.0 if identical)."""
    import math

    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def score_like_validator(model, clean, adv, epsilon, min_delta, device) -> dict:
    """Reproduce neurons/validator.verify_and_score exactly, on the PNG-decoded adv."""
    seen = _png_roundtrip(adv, device)            # validator scores the decoded PNG, not your tensor
    cap = min(float(epsilon), float(C.MAX_LINF_DELTA))
    floor = float(min_delta)
    original_index = target_index_of(model, clean, device)

    # EfficientNetV2-L label + confidence, before and after the attack (informational).
    clean_label, clean_conf = predict_label_conf(model, clean)
    adv_label, adv_conf = predict_label_conf(model, seen)

    norm = float((seen - clean).abs().max().item())
    rmse = float(torch.sqrt(torch.mean((seen - clean) ** 2)).item())
    ssim = _compute_ssim(clean, seen)
    psnr = _compute_psnr_db(clean, seen)
    flipped = predict_index(model=model, image_chw=seen) != original_index

    gates = {  # hard gates, in the validator's order; any False -> final 0.0
        "label_changed": flipped,
        "norm>=floor": norm >= floor,
        "norm<=cap": norm <= cap,
        "ssim>=0.98": ssim >= float(C.MIN_SSIM),
        "psnr>=38": psnr >= float(C.MIN_PSNR_DB),
    }
    base = {"clean_label": f"{clean_label} ({clean_conf:.3f})",
            "adv_label": f"{adv_label} ({adv_conf:.3f})",
            "gates": gates, "norm": round(norm, 6), "rmse": round(rmse, 6),
            "ssim": round(ssim, 5), "psnr_db": round(psnr, 3)}
    if not all(gates.values()):
        return {**base, "perturbation_score": 0.0, "final": 0.0}

    linf_ratio = min(max((norm - floor) / max(1e-12, cap - floor), 0.0), 1.0)
    rmse_ratio = min(max(rmse / max(1e-12, cap), 0.0), 1.0)
    linf_score = (1.0 - linf_ratio) ** 2
    rmse_score = (1.0 - rmse_ratio) ** 2
    lw, rw = float(C.LINF_COMPONENT_WEIGHT), float(C.RMSE_COMPONENT_WEIGHT)
    perturbation_score = (lw * linf_score + rw * rmse_score) / (lw + rw)
    final = float(C.PERTURBATION_WEIGHT) * perturbation_score  # SPEED_WEIGHT=0 -> speed ignored
    return {**base, "linf_score": round(linf_score, 4), "rmse_score": round(rmse_score, 4),
            "perturbation_score": round(perturbation_score, 4), "final": round(final, 4)}


if __name__ == "__main__":
    from perturbnet.model import load_efficientnet_v2_l

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    model = load_efficientnet_v2_l(device)
    clean = load_clean_image(device)
    warmup(model, device, clean)                  # pay one-time CUDA/cuDNN init up front
    target_index = target_index_of(model, clean, device)

    epsilon, min_delta = 0.12, 0.003              # representative challenge values
    t0 = time.time()
    adv = perturb(model, clean, target_index, epsilon, min_delta, device)
    attack_time = time.time() - t0

    report = score_like_validator(model, clean, adv, epsilon, min_delta, device)
    print("\n=== FMN (minimal-norm, floor-hugging) ===")
    for k, v in report.items():
        print(f"{k:>18}: {v}")
    safe = attack_time < (C.TIMEOUT_SECONDS - 1.0)
    print(f"{'attack_time':>18}: {attack_time:.3f}s")
    print(f"{'within_timeout':>18}: {'PASS' if safe else 'FAIL'} (cutoff {C.TIMEOUT_SECONDS:.0f}s)")
