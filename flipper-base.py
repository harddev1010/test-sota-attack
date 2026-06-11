"""
flipper-base.py  —  the BASELINE perturbation engine, extracted verbatim from the real
subnet miner (neurons/miner.py): small-step untargeted PGD.

WHY THIS FILE EXISTS
--------------------
This is the *reference* engine — the exact algorithm the stock miner ships with, lifted
out of `PerturbMiner.forward` into a single reusable `perturb(...)` so you can run/score
it in isolation and compare it head-to-head with the FMN engine in flipper.py (which has
the identical call signature).

DROP-IN CONTRACT (shared with flipper.py)
-----------------------------------------
    adv = perturb(model, clean, target_index, epsilon, min_delta, device)
matches the variables already present in miner.forward:
    model        = self.model                                (EfficientNetV2-L)
    clean        = decode_image_b64(...).to(self.device)     (CHW float [0,1], NATIVE res)
    target_index = resolve_target_index(synapse.true_label)  (the ORIGINAL class to flee)
    epsilon      = float(synapse.epsilon)
    min_delta    = float(getattr(synapse, "min_delta", 0.002))
    device       = self.device
Returns: adv, a CHW float tensor in [0,1] at the SAME resolution as `clean`.

KNOWN WEAKNESS (the bar flipper.py beats)
-----------------------------------------
The validator only scores perturbations with L-inf <= min(epsilon, 0.03)=0.03, but this
engine clamps to ±epsilon (up to 0.2) and steps by epsilon/4 (up to 0.05). So it often
overshoots 0.03 -> validator reason="above_max_delta" -> score 0.0. It also keeps the
LARGEST-delta step as "best", which is the opposite of what scoring rewards.
"""

import torch
import torch.nn.functional as F

from perturbnet.model import logits_for_images, predict_index


def perturb(
    model: torch.nn.Module,
    clean: torch.Tensor,
    target_index: int,
    epsilon: float,
    min_delta: float,
    device: torch.device,
    steps: int = 10,
) -> torch.Tensor:
    """Untargeted PGD, byte-for-byte the same as the stock miner's inline loop."""
    epsilon = float(epsilon)
    min_delta = float(min_delta)

    # step_size: how far we move each iteration; at least one 8-bit level (1/255).
    step_size = max(epsilon / 4.0, 1.0 / 255.0)

    adv = clean.clone().detach()      # start from the clean image
    best = adv.clone()                # best-so-far image we return
    best_delta = 0.0                  # largest L-inf seen so far (baseline's odd choice)
    final_pred = target_index

    for _ in range(steps):
        adv.requires_grad_(True)                                   # track gradients on pixels
        logits = logits_for_images(model=model, image_bchw=adv.unsqueeze(0))  # resize+normalize inside
        # Cross-entropy vs the ORIGINAL class; ascending it pushes the image AWAY from it.
        loss = F.cross_entropy(logits, torch.tensor([target_index], device=device))
        grad = torch.autograd.grad(loss, adv)[0]                   # d(loss)/d(pixels)
        adv = adv.detach() + step_size * grad.sign()               # FGSM-style signed step
        # Project back into the ±epsilon L-inf ball and valid pixel range.
        adv = torch.max(torch.min(adv, clean + epsilon), clean - epsilon).clamp(0.0, 1.0)

        pred = predict_index(model=model, image_chw=adv)           # current top-1 class
        final_pred = pred
        delta = float((adv - clean).abs().max().item())            # current L-infinity
        if delta > best_delta:                                     # baseline keeps the BIGGEST delta
            best = adv.clone()
            best_delta = delta
        if pred != target_index and delta >= min_delta:            # flipped AND above floor -> stop
            best = adv.clone()
            break

    _ = final_pred  # parity with the miner's log line; unused here
    return best


# --------------------------------------------------------------------------------------
# Self-test: load model + a sample image, run the baseline, print the validator-EXACT
# score.  Run:  python flipper-base.py
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    from perturbnet.model import load_efficientnet_v2_l, predict_index as _pi
    # flipper.py owns the shared image loader + validator-exact scorer.
    from flipper import load_clean_image, score_like_validator, target_index_of

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_efficientnet_v2_l(device)
    clean = load_clean_image(device)
    target_index = target_index_of(model, clean, device)  # model's own clean prediction

    epsilon, min_delta = 0.12, 0.003                       # representative challenge values
    adv = perturb(model, clean, target_index, epsilon, min_delta, device)

    report = score_like_validator(model, clean, adv, epsilon, min_delta, device)
    print("\n=== BASELINE (PGD) ===")
    for k, v in report.items():
        print(f"{k:>18}: {v}")
