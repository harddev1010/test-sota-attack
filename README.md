# sota-attack

A local lab for the **Perturb subnet (netuid 26)**: flip an EfficientNetV2-L label with a
**minimal-norm white-box attack (FMN)** and score it **exactly like the real validator** —
no bittensor, no networking, no chain.

It mirrors the live subnet repo so the engine code is **copy-paste-ready for prod**: the
`perturbnet/` package here is a trimmed mirror of the subnet's `perturbnet/`, and both
attack engines import the same names the real miner does.

## What's inside

| File | Role |
|------|------|
| [perturbnet/](perturbnet/) | Mirror of the subnet's package: `model.py`, `image_io.py`, `constants.py` (the **real** thresholds/weights) |
| [flipper-base.py](flipper-base.py) | **Baseline** engine — the miner's stock PGD, lifted verbatim into `perturb(...)` |
| [flipper.py](flipper.py) | **FMN** engine — minimal-norm, floor-hugging, PNG-aware. Same `perturb(...)` signature |
| [demo.py](demo.py) | Runs both engines on a sample image, prints validator-exact scores side by side |

## Install & run

```bash
pip install -r requirements.txt
python demo.py
```

`python demo.py` downloads one sample image, runs both engines, and prints a full
gate-by-gate + score report for each. Flags: `--engine {both,fmn,base}`, `--steps N`.

You can also run either engine's self-test directly:

```bash
python flipper.py          # FMN only
python flipper-base.py     # baseline only
```

## How scoring works (the real objective)

Measured in the **decoded native-resolution `[0,1]` image** (what `decode_image_b64`
returns — NOT the 480px model space):

```
norm = max|adv-clean| ,  rmse = sqrt(mean((adv-clean)^2))
```

Hard gates (any fail → **0.0**): label changed · `0.003 ≤ norm ≤ 0.03` · `ssim ≥ 0.98` ·
`psnr ≥ 38`. If they pass:

```
perturbation_score = 0.7·(1-linf_ratio)² + 0.3·(1-rmse_ratio)²
final = 1·perturbation_score + 0·speed_score      # SPEED_WEIGHT = 0
```

**Key consequences** (read straight from the subnet's `validator.py` / `constants.py`):

- **Speed doesn't score.** Only the 15s hard timeout and `processed_count` eligibility
  care about speed. So optimize purely for a small, clean perturbation.
- **`min(epsilon, 0.03)` is always 0.03** (epsilon is sampled in `[0.06, 0.20]`).
- **The optimum is to hug the floor.** `linf_score` is maximized as `norm → 0.003`.
- **PNG quantization sets the real floor.** The validator scores a re-decoded PNG, so
  pixels snap to multiples of `1/255 ≈ 0.00392`. The smallest *survivable* `norm` is
  `1/255`, which is just above 0.003 → `linf_score ≈ 0.93`.

## Why FMN beats the baseline

The baseline PGD clamps to `±epsilon` (up to 0.2) and steps by `epsilon/4` — so it
routinely **overshoots the 0.03 cap → scores 0.0**. The FMN engine instead:

1. runs **FMN** (via foolbox) to find the *minimal-L∞* flip direction, then
2. does a **PNG-aware line search**: scales that direction to the smallest `norm` that
   *still flips after the 8-bit PNG round-trip* and passes every gate.

`score_like_validator()` in [flipper.py](flipper.py) reproduces the validator's own
avg-pool SSIM, PSNR, gate order, and formula — so a local PASS predicts a real PASS.

## Deploying to prod (the real miner)

The engine is a literal drop-in. In the real `neurons/miner.py`, replace the inline PGD
loop in `forward(...)` with:

```python
from flipper import perturb
adv = perturb(self.model, clean, target_index, epsilon, min_delta, self.device)
```

Then add `foolbox` to the miner's `requirements.txt`. Optionally call `warmup(...)` once
in the miner's `__init__` so the first challenge isn't slowed by cold CUDA init. Because
`perturb()` only imports from `perturbnet.*` (which prod already has), it copies over with
no edits.

## Tuning loop

1. `python demo.py --steps N` — read each engine's `perturbation_score` and which gates passed.
2. If FMN's `final` is 0.0, check the failed gate: usually it means the minimal flip needs
   `norm > 0.03` (image is hard to flip subtly) — raise `--steps`, or accept it's a hard image.
3. The achievable ceiling per image ≈ `linf_score 0.93` (norm at `1/255`) with `rmse` near 0.
