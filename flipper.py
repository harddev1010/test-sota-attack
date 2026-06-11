"""
flipper.py — COMPOSE state-of-the-art adversarial attacks (we never reinvent them).

Goal: take a clean image, nudge its pixels by a tiny amount, and make EfficientNetV2-L
predict a DIFFERENT label (untargeted: any change counts).

We offer 5 SOTA attacks via two libraries:
  fmn  (foolbox)      : minimal-norm L-infinity attack — finds the SMALLEST L-inf change that flips
  ddn  (foolbox)      : minimal-norm L2 attack         — finds the SMALLEST L2 change that flips
  fab  (torchattacks) : Fast Adaptive Boundary         — walks to the nearest decision boundary
  pgd  (torchattacks) : Projected Gradient Descent     — strong fixed-budget L-inf baseline
  cw   (torchattacks) : Carlini-Wagner                 — optimization attack, very low distortion

"Minimal-norm" attacks (fmn/ddn/fab/cw) try to be as INVISIBLE as possible, which is
exactly what the Perturb score rewards. PGD is included as a loud, reliable baseline.
"""

import argparse
import time
from datetime import datetime  # for human-readable wall-clock timestamps in the logs

import torch
import numpy as np
from PIL import Image
import torchattacks
import foolbox as fb

import config
from model import load_model, predict, predict_index


def _stamp():
    """Current wall-clock time as MM:SS.mmm (minute:second:millisecond) for logs."""
    now = datetime.now()
    # %M=minute, %S=second; the last 3 digits of %f (microseconds) give milliseconds.
    return now.strftime("%M:%S") + f".{now.microsecond // 1000:03d}"


# ----------------------------------------------------------------------------------
# Distortion measurements — in [0,1] PIXEL space, the same space the validator uses.
# ----------------------------------------------------------------------------------
def linf_norm(clean, adv):
    """L-infinity = the single largest per-pixel change (worst-case visibility)."""
    return (adv - clean).abs().max().item()


def rmse(clean, adv):
    """Root-mean-square error = the typical per-pixel change (overall noisiness)."""
    return torch.sqrt(((adv - clean) ** 2).mean()).item()


# ----------------------------------------------------------------------------------
# OPTIONAL helper (OFF by default).
# Pure minimal-norm attacks aim for the SMALLEST possible change — which can land BELOW
# the validator's MIN_LINF_DELTA floor and fail the gate ("too small to count").
# This rescales the perturbation up so its L-inf sits just above that floor.
# Enable it (run_attack(..., hug_floor_on=True)) ONLY if your score report says the
# "L-inf >= min_delta" gate FAILED for being too small.
# ----------------------------------------------------------------------------------
def hug_floor(delta, min_linf_delta):
    cur = delta.abs().max().item()              # current largest change
    if cur < min_linf_delta and cur > 0:        # only act if we're under the floor
        scale = (min_linf_delta * 1.05) / cur   # +5% so PNG rounding won't dip us back under
        delta = delta * scale
    return delta


def _save_triptych(clean, adv, out_path):
    """Save a side-by-side PNG: [ original | perturbation x-amplified | adversarial ]."""
    delta = (adv - clean)                                    # the raw perturbation
    # Stretch the perturbation to [0,1] so we can actually SEE it (it's normally tiny).
    amp = delta - delta.min()
    amp = amp / (amp.max() + 1e-12)

    def to_pil(t):
        arr = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # CHW->HWC, [0,255]
        return Image.fromarray(arr)

    panels = [to_pil(clean), to_pil(amp), to_pil(adv)]
    w, h = panels[0].size
    canvas = Image.new("RGB", (w * 3, h))                    # blank wide strip for 3 panels
    for i, p in enumerate(panels):
        canvas.paste(p, (i * w, 0))
    canvas.save(out_path)


def run_attack(image_path, attack="fmn", hug_floor_on=False, steps=50, eps=None, **knobs):
    """
    Run one attack on one image.

    Returns (adv_path, info_dict). demo.py calls this, then scores adv_path.
      image_path   : path to the clean input image (any format Pillow reads)
      attack       : one of {fmn, ddn, fab, pgd, cw}
      hug_floor_on : rescale perturbation above MIN_LINF_DELTA (see hug_floor above)
      steps        : iteration count for the iterative attacks (more = stronger/slower)
      eps          : L-infinity budget; defaults to config.EPSILON
    """
    eps = config.EPSILON if eps is None else eps

    # Tell the user which compute device is actually being used. If this says "cpu" on a
    # machine with a GPU, torch can't see CUDA (e.g. a CPU-only torch wheel) -> expect it slow.
    print(f"[device] using {config.DEVICE.upper()}"
          + (f" ({torch.cuda.get_device_name(0)})" if config.DEVICE == "cuda" else ""))

    model = load_model()

    # Preprocess EXACTLY like the validator: -> [0,1] tensor at the model's input size.
    clean = config.pil_to_unit_tensor(Image.open(image_path)).to(config.DEVICE)
    clean_b = clean.unsqueeze(0)                              # add batch dim -> (1,3,H,W)

    # Untargeted attacks need the CURRENT (clean) label to push away from.
    orig_label, orig_conf = predict(model, clean)
    y = torch.tensor([predict_index(model, clean)], device=config.DEVICE)

    print(f"[{_stamp()}] starting perturbing.. (attack={attack}, steps={steps}, device={config.DEVICE})")
    t0 = time.time()                                         # start the wall clock

    if attack in ("fmn", "ddn"):
        # ---- foolbox minimal-norm attacks ----
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))       # tell foolbox inputs live in [0,1]
        if attack == "fmn":
            atk = fb.attacks.LInfFMNAttack(steps=steps)      # minimize the L-infinity norm
        else:
            atk = fb.attacks.DDNAttack(steps=steps)          # minimize the L2 norm
        # epsilons=None => return the MINIMAL adversarial example (no fixed budget).
        _, adv_b, _ = atk(fmodel, clean_b, y, epsilons=None)

    elif attack in ("fab", "pgd", "cw"):
        if attack == "fab":
            # FAB: geometrically walks to the NEAREST decision boundary (minimal-norm).
            atk = torchattacks.FAB(model, norm="Linf", eps=eps, steps=steps, n_classes=1000)
        elif attack == "pgd":
            # PGD: strongest FIXED-budget L-inf attack; alpha = per-step size.
            atk = torchattacks.PGD(model, eps=eps, alpha=eps / 4, steps=steps)
        else:
            # CW: optimization attack trading off "flip it" vs "stay tiny" (very clean).
            atk = torchattacks.CW(model, c=1.0, steps=max(steps, 100))
        adv_b = atk(clean_b, y)                               # torchattacks works in [0,1] directly
    else:
        raise ValueError(f"unknown attack {attack!r}; choose fmn|ddn|fab|pgd|cw")

    # Optional: lift a too-small perturbation above the validator's floor.
    if hug_floor_on:
        delta = hug_floor(adv_b - clean_b, config.MIN_LINF_DELTA)
        adv_b = (clean_b + delta).clamp(0, 1)                # stay inside valid pixel range

    elapsed = time.time() - t0                                # wall-clock attack time
    print(f"[{_stamp()}] done perturbing.. (took {elapsed:.3f}s)")
    adv = adv_b.squeeze(0).detach()                          # drop batch dim

    # Re-classify the adversarial image.
    new_label, new_conf = predict(model, adv)
    flipped = new_label != orig_label

    # Save the adversarial image as PNG (the validator scores a saved file, so do we).
    adv_path = str(image_path).rsplit(".", 1)[0] + f"_adv_{attack}.png"
    arr = (adv.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(adv_path)

    # Save the explanatory three-panel figure next to it.
    triptych_path = str(image_path).rsplit(".", 1)[0] + f"_triptych_{attack}.png"
    _save_triptych(clean, adv, triptych_path)

    info = {
        "attack": attack,
        "orig_label": orig_label, "orig_conf": orig_conf,
        "new_label": new_label, "new_conf": new_conf,
        "flipped": flipped,
        "linf": linf_norm(clean, adv),
        "rmse": rmse(clean, adv),
        "time": elapsed,
        "adv_path": adv_path,
        "triptych_path": triptych_path,
    }

    # Human-readable summary so you FEEL the speed/quality trade-off.
    print("\n================= ATTACK =================")
    print(f"attack            : {attack}")
    print(f"original label    : {orig_label}  (conf {orig_conf:.3f})")
    print(f"new label         : {new_label}  (conf {new_conf:.3f})")
    print(f"FLIPPED?          : {'YES' if flipped else 'NO'}")
    print(f"final L-infinity  : {info['linf']:.5f}   (in [0,1] pixel space)")
    print(f"final RMSE        : {info['rmse']:.5f}   (in [0,1] pixel space)")
    print(f"wall-clock time   : {elapsed:.3f} s")
    print(f"adversarial PNG   : {adv_path}")
    print(f"triptych PNG      : {triptych_path}")
    print("==========================================\n")

    return adv_path, info


if __name__ == "__main__":
    # Allow running the attack stage on its own:  python flipper.py --image x.jpg --attack fmn
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to the clean input image")
    ap.add_argument("--attack", default="fmn", choices=["fmn", "ddn", "fab", "pgd", "cw"])
    ap.add_argument("--hug-floor", action="store_true", help="rescale tiny perturbations above min_delta")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    run_attack(args.image, attack=args.attack, hug_floor_on=args.hug_floor, steps=args.steps)
