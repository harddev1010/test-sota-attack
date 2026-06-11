"""
flipper-fmn.py  —  Quantization-aware FMN-L-inf perturbation engine (the "unified" design).

Drop-in replacement for the baseline PGD block in the real miner — IDENTICAL signature to
flipper-base.perturb / flipper.perturb, so deploying to prod = delete the PGD loop in
miner.forward and:

    from flipper_fmn import perturb            # (file is flipper-fmn.py; import via path like demo.py)
    adv = perturb(self.model, clean, target_index, epsilon, min_delta, self.device,
                  timeout_seconds=float(synapse.timeout_seconds))

Everything `perturb()` needs is imported from perturbnet.* (the same package prod has), so the
engine copy-pastes with ZERO edits. The helpers ABOVE the engine (load_clean_image /
score_like_validator / warmup / SSIM / PSNR) are LAB-ONLY conveniences.

PER THE USER'S LAYOUT REQUEST: the lab/shared helpers come FIRST; the actual perturbation
algorithm (every `_*` attack helper plus the public `perturb`) is grouped at the very END,
right before the __main__ self-test.

────────────────────────────────────────────────────────────────────────────────────────
WHY THIS ENGINE (vs flipper.py's single-direction FMN)
────────────────────────────────────────────────────────────────────────────────────────
The validator scores a re-decoded PNG, so every final per-pixel delta is an integer multiple
of 1/255 (~0.00392). The real problem is therefore DISCRETE, not continuous:

    valid final L-inf  ≈  k / 255,    with    k_min = ceil(min_delta*255)  ..  k_max = floor(cap*255)
    (defaults: min_delta=0.003 -> k_min=1 ; cap=0.03 -> k_max=7)

linf_score = (1 - (Linf-min_delta)/(cap-min_delta))^2 collapses fast per level
(k=1 -> 0.93, k=2 -> 0.67, k=3 -> 0.46 ...), so the dominant objective is FIND THE SMALLEST k
THAT FLIPS. RMSE (weight 0.3) is a tiebreak among miners who tie on k — EXCEPT at high k where
a *dense* perturbation can drop PSNR below 38 dB and fail gate 9 outright, so sparsity there is
mandatory, not cosmetic.

LIVE vs DEFENSIVE constants (the rule that keeps us robust across validators):
  * TRANSMITTED in the synapse -> read live, never hardcode:  min_delta, epsilon, timeout_seconds.
  * NOT transmitted (validator-local, env-overridable)     :  max_linf_delta, min_ssim, min_psnr.
    We use the C.* defaults for the score MATH but never AIM near the cap — aiming for the
    smallest flipping k is safe under any plausible ceiling (a lower cap only ever hurts a
    miner that hugged 0.03).

Pipeline (all candidates judged on the PNG-decoded image, never the float tensor):
  0. calibrate per-step + per-PNG cost; build wall-clock deadline = start + timeout - reserve.
  1. cheap signed-gradient probes at k_min..k_min+2     (easy images finish here).
  2. FMN locate (foolbox) -> minimal direction; ascend k and verify each post-PNG -> smallest k.
  3. PGD fallback (ascending k) only if 1+2 never flipped — the reliability net vs a 0.0 round.
  4. polish: saliency PRUNE the won candidate to cut RMSE / rescue PSNR, L-inf held at k.
  5. always return best_valid (smallest verified k, then lowest RMSE); never the last candidate.

Gradients route through the model's resize+center-crop, so sign(grad)=0 on cropped-out / dead
pixels -> the signed steps never waste L-inf or RMSE there (a free crop mask). We still verify
the FULL gate set (label, band, SSIM, PSNR) on the decoded PNG before trusting any candidate.
"""

from __future__ import annotations  # so `float | None` hints parse on Python < 3.10 too

import math
import time

import torch

import foolbox as fb  # SOTA attack lib — we COMPOSE FMN from it, never reimplement it

from perturbnet import constants as C
from perturbnet.image_io import encode_image_b64, decode_image_b64
from perturbnet.model import logits_for_images, predict_index

# cuDNN autotuning: input H/W is constant within a run, so let cuDNN pick the fastest kernels.
torch.backends.cudnn.benchmark = True

_Q = 1.0 / 255.0  # one 8-bit quantisation step; the finest perturbation a PNG can carry


# ==========================================================================================
# LAB / SHARED helpers (used by the self-test AND by the engine's post-PNG verification).
# _png_roundtrip / _compute_ssim / _compute_psnr_db are shared with the engine below;
# everything else here is LAB-ONLY. score_like_validator mirrors validator.verify_and_score
# so a local PASS == a real PASS.
# ==========================================================================================
def _png_roundtrip(image_chw: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return the image EXACTLY as the validator will see it: encode to PNG, decode back.
    Bakes in 8-bit quantisation so any check here matches scoring reality."""
    return decode_image_b64(encode_image_b64(image_chw)).to(device)


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
    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def warmup(model: torch.nn.Module, device: torch.device, example_chw: torch.Tensor) -> None:
    """Call ONCE at startup: a forward+backward to pay CUDA/cuDNN init up front, so the first
    real challenge isn't slowed by lazy kernel autotuning."""
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
    """Return (label_string, confidence) from EfficientNetV2-L for a CHW [0,1] image."""
    from perturbnet.model import LABELS

    with torch.no_grad():
        logits = logits_for_images(model=model, image_bchw=image_chw.unsqueeze(0))
        probs = logits.softmax(dim=1)
        conf, idx = probs.max(dim=1)
    i = int(idx.item())
    label = LABELS[i] if 0 <= i < len(LABELS) else str(i)
    return label, float(conf.item())


def score_like_validator(model, clean, adv, epsilon, min_delta, device) -> dict:
    """Reproduce neurons/validator.verify_and_score exactly, on the PNG-decoded adv."""
    seen = _png_roundtrip(adv, device)            # validator scores the decoded PNG, not your tensor
    cap = min(float(epsilon), float(C.MAX_LINF_DELTA))
    floor = float(min_delta)
    original_index = target_index_of(model, clean, device)

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
            "k_level": round(norm * 255.0, 3),
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


# ==========================================================================================
# PERTURBATION ALGORITHM  (the engine — everything from here to __main__).
# Public entry point is `perturb`, defined LAST; the `_*` helpers above it are its building
# blocks. This is the block that copy-pastes into the real miner.
# ==========================================================================================
class _PreprocessedModel(torch.nn.Module):
    """Adapts the classifier so foolbox can feed it native-res [0,1] images. logits_for_images
    applies the validator's exact resize+crop+normalize, so FMN's gradients flow back to the
    very native pixels the validator measures."""

    def __init__(self, base: torch.nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N,3,H,W) in [0,1]
        return logits_for_images(model=self.base, image_bchw=x)


def _margin_and_grad(model, x_chw, target_index):
    """Untargeted CW-style margin and its gradient wrt the native pixels.

        margin = logit[true] - max_{j != true} logit[j]
    margin < 0  <=>  the image is adversarial (true class no longer the argmax). We DESCEND
    margin. Returns (margin_value, grad) with grad the same shape as x_chw. sign(grad)==0 on
    cropped-out / dead pixels, so signed steps naturally avoid wasting budget there."""
    x = x_chw.detach().clone().requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=x.unsqueeze(0))[0]
    true_logit = logits[target_index]
    others = logits.clone()
    others[target_index] = float("-inf")
    margin = true_logit - others.max()
    grad = torch.autograd.grad(margin, x)[0]
    return float(margin.item()), grad.detach()


def _signed_step(clean, grad, radius):
    """Steepest-descent L-inf step of size `radius`: move every pixel against sign(grad).
    Dead-gradient pixels (sign 0) are left untouched -> the perturbation stays as sparse as the
    model's sensitivity allows."""
    return (clean - radius * grad.sign()).clamp(0.0, 1.0)


def _scale_dir(clean, direction, base_linf, target_linf):
    """Scale a perturbation DIRECTION so its L-inf == target_linf, add to clean, clamp."""
    return (clean + direction * (target_linf / base_linf)).clamp(0.0, 1.0)


def _fmn_direction(model, clean, target_index, device, steps):
    """FMN (minimal-L-inf) perturbation DIRECTION = raw adversarial minus clean, or None on
    failure. NB: take attack()'s FIRST return (raw_advs); the SECOND with epsilons=None comes
    back as the CLEAN image — taking it was the classic 'FMN did nothing' bug."""
    try:
        fmodel = fb.PyTorchModel(_PreprocessedModel(model).to(device).eval(), bounds=(0.0, 1.0))
        attack = fb.attacks.LInfFMNAttack(steps=int(steps))
        labels = torch.tensor([int(target_index)], device=device)  # class to move AWAY from
        raw_advs, _, _ = attack(fmodel, clean.unsqueeze(0), labels, epsilons=None)
        return raw_advs.squeeze(0) - clean
    except Exception:
        return None  # foolbox hiccup -> caller falls back to probes / PGD (reliability first)


def _evaluate(model, clean, cand_chw, target_index, device, floor, cap):
    """Judge a candidate the way the validator will: PNG round-trip, then measure on the decoded
    image. `valid` means ALL hard gates pass (label flipped, floor<=Linf<=cap, SSIM, PSNR).
    SSIM/PSNR are only computed when the cheap checks already pass (they can fail at high k)."""
    seen = _png_roundtrip(cand_chw, device)
    diff = seen - clean
    linf = float(diff.abs().max().item())
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    in_band = floor <= linf <= cap
    flipped = predict_index(model=model, image_chw=seen) != target_index
    valid, ssim, psnr = False, None, None
    if in_band and flipped:
        ssim = _compute_ssim(clean, seen)
        psnr = _compute_psnr_db(clean, seen)
        valid = ssim >= float(C.MIN_SSIM) and psnr >= float(C.MIN_PSNR_DB)
    return {"cand": cand_chw, "linf": linf, "rmse": rmse, "flipped": flipped,
            "valid": valid, "ssim": ssim, "psnr": psnr}


def perturb(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_index: int,
    epsilon: float,
    min_delta: float,
    device: torch.device,
    timeout_seconds: float = 15.0,
    reserve_seconds: float | None = None,
    start_time: float | None = None,
    steps: int | None = None,
) -> torch.Tensor:
    """Quantization-aware minimum-L-inf flip, verified in PNG space, time-budgeted to the deadline.

    Drop-in: the first six args match flipper-base.perturb / flipper.perturb. The extras are
    optional and SHOULD be wired in prod:
        timeout_seconds = float(synapse.timeout_seconds)   # live deadline (do NOT hardcode 15)
        start_time      = the time the miner began handling this challenge (accounts for decode)
        reserve_seconds = headroom for encode + network RTT + final verify (defaults from calib)
        steps           = override FMN iterations (None -> auto from calibration)

    Objective order: (1) flip at all, (2) smallest k=Linf*255, (3) lowest RMSE, all gates passing
    on the re-decoded PNG. Always returns a decodable in-range image — best_valid if we flipped,
    else the strongest legal best-effort (which scores 0 but never errors a gate)."""
    t_start = start_time if start_time is not None else time.time()
    clean = clean.to(device).clamp(0.0, 1.0)

    # ---- live floor / defensive cap, mapped to discrete PNG levels -----------------------
    # We work ONLY in exact k/255 radii. A continuous radius like the 0.03 cap is dangerous:
    # 0.03*255 = 7.65, so a peak pixel stepped by 0.03 ROUNDS to 8/255 (0.0314) and busts the
    # cap. A radius of exactly k/255 adds an integer number of levels -> the peak lands on
    # exactly k, never k+1. So every radius below is k*_Q, never `cap` or `floor` directly.
    floor = float(min_delta)
    cap = min(float(epsilon), float(C.MAX_LINF_DELTA))   # cap from defaults; we never AIM near it
    if floor > cap:                                      # pathological config guard
        floor = cap
    k_min = max(1, int(math.ceil(floor * 255.0 - 1e-6)))         # first PNG level >= floor
    k_max = max(k_min, int(math.floor(cap * 255.0 + 1e-6)))      # last PNG level <= cap
    levels = list(range(k_min, k_max + 1))

    # ---- seed gradient (mandatory) doubles as our t_step calibration --------------------
    # One fwd+bwd we need anyway; timing it avoids burning extra forwards just to calibrate
    # (on CPU a wasted forward is seconds). t_png starts as a guess and self-corrects on the
    # first real PNG verify inside consider().
    g_t0 = time.time()
    _, grad0 = _margin_and_grad(model, clean, target_index)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_step = max(1e-4, time.time() - g_t0)
    t_png = 0.6 * t_step                                 # forward-only ~ 0.6 of a fwd+bwd

    if reserve_seconds is None:
        reserve_seconds = 0.7                            # encode + RTT cushion; per-op gating does the rest
    deadline = t_start + max(0.05, float(timeout_seconds) - float(reserve_seconds))

    def time_left():
        return deadline - time.time()

    best = None  # the smallest-(linf,rmse) VALID candidate seen so far -> what we return

    def consider(cand_chw):
        """PNG-verify a candidate; promote to `best` if valid and strictly smaller (linf, then
        rmse). Also refreshes t_png with the measured round-trip+verify cost so later phases
        budget against reality, not the initial guess."""
        nonlocal best, t_png
        c0 = time.time()
        res = _evaluate(model, clean, cand_chw, target_index, device, floor, cap)
        t_png = max(1e-4, time.time() - c0)
        if res["valid"] and (best is None or (res["linf"], res["rmse"]) < (best["linf"], best["rmse"])):
            best = res
        return res

    # =========================== PHASE 1 — cheap signed-gradient probes ===================
    # Ascend k from the floor; first flip is the smallest probe k. Many easy images stop here.
    for k in levels[:3]:
        if time_left() <= 1.3 * t_png:
            break
        if consider(_signed_step(clean, grad0, k * _Q))["valid"]:
            break  # ascending => lowest k a probe can reach; no point testing higher

    # =========================== PHASE 2 — FMN locate + discrete k verify =================
    # foolbox FMN gives a minimal-norm DIRECTION; we ascend k along it and verify each post-PNG,
    # reading off the smallest surviving level. `affordable` = how many FMN steps fit in ~55% of
    # the remaining budget; skip entirely if fewer than 4 fit (no minimum that could overrun).
    won_k_min = best is not None and best["linf"] <= k_min * _Q + 1e-9
    affordable = int(0.55 * time_left() / t_step)
    if not won_k_min and affordable >= 4:
        fmn_steps = int(steps) if steps else min(60, affordable)
        direction = _fmn_direction(model, clean, target_index, device, fmn_steps)
        if direction is not None:
            base_linf = float(direction.abs().max().item())
            if base_linf > 1e-12:
                for k in levels:
                    if time_left() <= 1.3 * t_png:
                        break
                    if best is not None and k * _Q >= best["linf"] - 1e-12:
                        break  # can't beat what we already hold
                    if consider(_scale_dir(clean, direction, base_linf, k * _Q))["valid"]:
                        break

    # =========================== PHASE 3 — PGD fallback (reliability net) =================
    # Only if NOTHING has flipped yet. Ascend k; inside each radius take signed steps and pay a
    # PNG verify ONLY when the float margin says we're adversarial -> few round-trips. A flip at
    # any k beats the 0.0 of returning a non-adversarial image.
    if best is None:
        delta = torch.zeros_like(clean)
        for k in levels:
            r = k * _Q
            alpha = max(_Q, r / 4.0)
            delta = delta.clamp(-r, r)                    # warm-start from the previous (smaller) k
            local = 0
            while time_left() > t_step + 1.3 * t_png and local < 12:
                x = (clean + delta).clamp(0.0, 1.0)
                margin, grad = _margin_and_grad(model, x, target_index)
                if margin < 0.0 and consider(x)["valid"]:
                    break
                delta = ((clean + delta - alpha * grad.sign()).clamp(0.0, 1.0) - clean).clamp(-r, r)
                local += 1
            if best is not None:
                break

    # =========================== PHASE 4 — RMSE polish via saliency pruning ===============
    # L-inf (the 0.7 term) is already minimal; now shrink RMSE (and rescue PSNR at high k) by
    # zeroing the least-influential perturbed pixels while the flip survives. L-inf is held
    # because we only ever REMOVE perturbation, never grow the peak.
    if best is not None and time_left() > t_step + 1.3 * t_png:
        base_cand = best["cand"].clone()
        delta = base_cand - clean
        _, grad_p = _margin_and_grad(model, base_cand, target_index)
        importance = (grad_p * delta).abs().view(-1)     # low |grad*delta| => safe to drop
        frac = 0.5
        while time_left() > 1.3 * t_png and frac >= 1.0 / 64.0:
            nz = torch.nonzero(delta.view(-1).abs() > 0, as_tuple=False).squeeze(1)
            if nz.numel() == 0:
                break
            order = nz[torch.argsort(importance[nz])]    # ascending influence
            n_remove = int(order.numel() * frac)
            if n_remove == 0:
                frac *= 0.5
                continue
            trial = delta.clone()
            trial.view(-1)[order[:n_remove]] = 0.0
            if consider((clean + trial).clamp(0.0, 1.0))["valid"]:
                delta = trial                            # accept the pruning; keep shrinking
            else:
                frac *= 0.5                              # too aggressive -> halve the bite and retry

    # =========================== RETURN ==================================================
    if best is not None:
        return best["cand"].detach().clamp(0.0, 1.0)
    # Never flipped: return the strongest IN-BAND legal step (exact k_max level, so it can't round
    # past the cap). Same score as baseline failure (0.0 — likely fails SSIM/PSNR at high k), but
    # a clean, in-range reply keeps the round well-formed instead of busting gate 6.
    return _signed_step(clean, grad0, k_max * _Q).detach()


# ==========================================================================================
# Self-test: load model + sample image, run the engine, print the validator-EXACT score.
#     python flipper-fmn.py                  # default representative challenge
# (demo.py compares engines; this is the standalone smoke test, mirroring flipper.py's main.)
# ==========================================================================================
if __name__ == "__main__":
    from perturbnet.model import load_efficientnet_v2_l

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    model = load_efficientnet_v2_l(device)
    clean = load_clean_image(device)
    warmup(model, device, clean)                      # pay one-time CUDA/cuDNN init up front
    target_index = target_index_of(model, clean, device)

    epsilon, min_delta = 0.12, 0.003                  # representative challenge values
    t0 = time.time()
    adv = perturb(model, clean, target_index, epsilon, min_delta, device, timeout_seconds=15.0)
    attack_time = time.time() - t0

    report = score_like_validator(model, clean, adv, epsilon, min_delta, device)
    print("\n=== FMN-UNIFIED (quantization-aware, discrete-k, pruned) ===")
    for k, v in report.items():
        print(f"{k:>18}: {v}")
    print(f"{'attack_time':>18}: {attack_time:.3f}s")
