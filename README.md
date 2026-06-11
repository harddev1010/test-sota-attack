# sota-attack

Flip an **EfficientNetV2-L** ImageNet label with **state-of-the-art adversarial attacks**
(composed from `foolbox` / `torchattacks` — never reinvented), then score the result
**exactly like the Bittensor "Perturb" subnet (netuid 26) validator**.

**What/why.** Perturb is a subnet where miners must change a classifier's prediction with a
perturbation that is as *small and clean* as possible, as *fast* as possible. The best tools
for that are **white-box minimal-norm attacks** (FMN, DDN, FAB, C&W): instead of spending a
fixed noise budget like PGD, they search for the *smallest* change that flips the label —
which is precisely what the Perturb score rewards. This repo lets you feel those attacks and
tune them against the *real objective* with one command.

## Install & run

```bash
pip install -r requirements.txt
python demo.py
```

`python demo.py` downloads one sample image, runs the default attack (`fmn`), saves the
adversarial PNG + a three-panel figure, and prints a full **PASS/FAIL + score** report.
(First run also downloads the EfficientNetV2-L weights, a few hundred MB.)

### Attack options

```bash
python demo.py --attack fmn   # foolbox  : minimal L-infinity (default)
python demo.py --attack ddn   # foolbox  : minimal L2
python demo.py --attack fab   # torchattacks: nearest decision boundary (minimal-norm)
python demo.py --attack pgd   # torchattacks: fixed-budget L-inf baseline (loud but reliable)
python demo.py --attack cw    # torchattacks: Carlini-Wagner optimization (very clean)
```

Extra knobs: `--steps N` (more = stronger/slower), `--hug-floor` (see below).

You can also run the stages alone:

```bash
python flipper.py --image sample.jpg --attack cw
python verifier.py --clean sample.jpg --adv sample_adv_cw.png --time 1.5
```

## The tuning loop

1. **Run an attack** — `python demo.py --attack <name> --steps <n>`.
2. **Read the report** — look at `perturbation_score` (how small/clean) and `speed_score`
   (how fast), plus which hard gates passed.
3. **Adjust:**
   - Score is `0.0` and only the **`linf_above_floor`** gate failed? The attack was *too
     subtle*. Re-run with `--hug-floor` (rescales the perturbation just above `MIN_LINF_DELTA`).
   - `perturbation_score` low? Switch to a cleaner attack (`cw`, `fmn`, `fab`) or fewer steps.
   - `speed_score` low? Use fewer `--steps` or a faster attack (`fmn`/`pgd`) — feel the trade-off.
4. Repeat and push the **FINAL** score up.

## Make it match the real subnet

The numbers in [config.py](config.py) under the LOUD warning block are **placeholders**.
Before you trust a score, replace them with the subnet's real values:

1. **Model + preprocessing** — copy the exact model construction and the exact
   resize/crop/normalize from `perturbnet/model.py` into [config.py](config.py)
   (`WEIGHTS`, `pil_to_unit_tensor`, `normalize`) and mirror its label logic in
   [model.py](model.py)'s `predict()`.
2. **Scoring constants** — paste the real values from `perturbnet/constants.py`. Find them with:

   ```bash
   grep -rn "EPSILON\|MIN_LINF_DELTA\|MAX_LINF_DELTA\|MIN_SSIM\|MIN_PSNR\|TIMEOUT\|PERTURBATION_WEIGHT\|SPEED_WEIGHT\|COMPONENT_WEIGHT" perturbnet/
   ```

3. **Measurement space** — confirm whether the validator measures L-infinity / RMSE / SSIM /
   PSNR in `[0,1]` pixel space (what this repo assumes) or `0-255` or the normalized space,
   and adjust [verifier.py](verifier.py) / [config.py](config.py) accordingly.

## Files

| File | Role |
|------|------|
| [config.py](config.py) | Single source of truth: model, preprocessing, all Perturb thresholds + weights |
| [model.py](model.py) | Loads the frozen EfficientNetV2-L, `predict()` + label mapping |
| [flipper.py](flipper.py) | Composes the SOTA attacks via foolbox / torchattacks |
| [verifier.py](verifier.py) | Local replica of the validator's gates + score |
| [demo.py](demo.py) | End-to-end: download → attack → save → score → report |
