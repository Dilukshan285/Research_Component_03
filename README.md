# Research_Component_03

**UEF-Net — Uncertainty-aware Ordinal Ejection-Fraction Network**
Four-class left-ventricular ejection fraction (EF) severity grading from 2D echocardiography, under extreme class imbalance.

---

## Problem

Ejection fraction is the primary measure of cardiac pumping function. Most published work on the EchoNet-Dynamic dataset reports either **EF regression** (MAE/R²) or **binary** reduced-vs-normal classification. Clinically actionable **four-class severity grading** is far less studied — precisely because the minority and middle classes are hard under an ~11:1 imbalance.

| Class | EF range | Share of dataset |
|---|---|---|
| Severe | < 30 | ~6% |
| Moderate | 30 – 40 | ~7% |
| Mild | 40 – 55 | ~18% |
| Normal | ≥ 55 | ~69% |

This project targets that gap: strong EF regression **and** honest per-class severity recall.

---

## Results

Evaluated on the **untouched official EchoNet-Dynamic test split (1,277 videos)**. All decision rules are frozen on validation data — **no test-set tuning**.

### Two-seed ensemble (best)

| Metric | Value |
|---|---|
| **MAE** | **3.994 EF points** |
| **R²** | **0.817** |
| RMSE | 5.237 |
| Overall accuracy | **74.9%** (957 / 1,277) |
| Balanced accuracy | 74.3% |
| Macro-F1 | 0.697 |
| Min-class recall | 0.711 |

For reference, the EchoNet-Dynamic benchmark (Ouyang et al., *Nature* 2020) reports EF MAE ≈ 4.05.

### Per-class recall

| Class | Correct / Total | Recall |
|---|---|---|
| Severe (<30) | 59 / 83 | 0.711 |
| Moderate (30–40) | 60 / 77 | **0.779** |
| Mild (40–55) | 174 / 241 | 0.722 |
| Normal (≥55) | 664 / 876 | **0.758** |

### Clinical safety

- **99.6%** of predictions fall **within one severity class** (only 5 / 1,277 off by more than one)
- **Zero catastrophic errors** — no Severe case predicted Normal, no Normal case predicted Severe

### Single model (uefnet_v3)

MAE 4.138 · R² 0.804 · overall 72.1% · balanced 73.0% · min-recall 0.687

---

## Methodological contributions

| # | Component | Description | Measured effect |
|---|---|---|---|
| 1 | **Measurement-uncertainty soft labels** | EF is itself a noisy measurement (inter-observer σ ≈ 4 EF pts). Ordinal targets use `s_k = 1 − Φ((t_k − EF)/σ)` instead of hard labels, giving boundary cases honest soft supervision. | Core ordinal supervision |
| 2 | **Ordered-CORAL head** | A single severity score compared against monotonically ordered learned cut-points, so `P(y>0) ≥ P(y>1) ≥ …` holds **by construction** rather than by post-hoc repair. | Rank-consistent output |
| 3 | **Cycle-aware sampling + motion channel** | Clips are constrained to contain the ED→ES contraction that defines EF; a temporal-difference channel supplies an explicit wall-motion cue. | Physically grounded features |
| 4 | **Heteroscedastic uncertainty + pairwise rank loss** | The network predicts its own variance and is penalised for mis-ordering clinically distinguishable EF pairs. | Enables conformal intervals |
| 5 | **Harmonized cross-dataset co-training** | CAMUS (mean EF 44 vs EchoNet 56) adds 1,000 low/mid-EF clips. Intensity is affine-matched to EchoNet so the network cannot exploit scanner brightness as a minority-class shortcut. | Moderate recall 0.53 → 0.62 (validation, matched epochs) |
| 6 | **Logit adjustment** | Class-head logits shifted by `τ·log P(class)` during training (Menon et al., 2021) to widen the margin for squeezed middle classes. | Lifts middle-band recall |
| 7 | **Tail-decompression calibration** | Variance expansion counteracts regression-to-the-mean that pushes true-Severe cases into Moderate; thresholds fit on the **full** validation split for tiny-class stability. | Severe recall 0.590 → 0.687 |
| 8 | **Seed ensembling** | Per-video EF and class distributions averaged across independently trained models; one strategy selected on validation and frozen. | MAE 4.138 → 3.994 |

---

## Architecture

```
Input clip (2 × 32 × 112 × 112)          grayscale + temporal-difference motion
        │
   R(2+1)D-18 backbone                   Kinetics-400 pretrained, stem adapted to 2 channels
        │
   ┌────┴─────┬──────────────┬────────────────┐
   │          │              │                │
 EF regr.  Ordered-CORAL   Softmax class    log-variance
 (MAE)     (ordinal)       (auxiliary)      (uncertainty)
```

31.3 M parameters. Trained with AMP + gradient accumulation on a single 8 GB GPU.

**Loss** = LDS-weighted Huber (regression) + soft-CORAL BCE with uncertainty targets (ordinal) + soft cross-entropy (class) + Gaussian NLL (uncertainty) + pairwise rank + dual-head consistency.

**Imbalance handling** = deferred re-weighting (natural sampling → class-balanced at epoch 15) + class-balanced sampler + effective-number weights + logit adjustment.

---

## Repository layout

```
Dilukshan/
├── preprocessing/          decode → verify → cache → manifest pipeline
│   ├── build_camus.py      CAMUS (NIfTI) → EchoNet-format converter + harmonizer
│   ├── utils/sampling.py   cycle-aware clip sampling + motion channel
│   └── artifacts/          statistics, audit + verification reports, visualizations
└── training/
    ├── config.py           single source of truth for all hyperparameters
    ├── models/uef_net.py   backbone + four heads
    ├── losses/losses.py    multi-task objective
    ├── engine/
    │   ├── trainer.py      training loop, EMA, resume, final calibration
    │   ├── calibrate.py    strategy fitting, thresholds, conformal, temperature
    │   ├── evaluate.py     multi-clip TTA inference
    │   └── metrics.py      per-class metrics with Wilson confidence intervals
    ├── run_train.py        training entry point
    ├── run_eval.py         frozen-strategy test evaluation
    └── run_ensemble.py     multi-seed ensemble evaluation
```

---

## Usage

### 1. Data

**Not included in this repository.** EchoNet-Dynamic and CAMUS are distributed under Research Data Use Agreements that prohibit redistribution. Obtain them from the official sources:

- **EchoNet-Dynamic** — <https://echonet.github.io/dynamic/>
- **CAMUS** — <https://www.creatis.insa-lyon.fr/Challenge/camus/>

### 2. Preprocessing

```bash
cd Dilukshan/preprocessing
python run_all.py                 # decode, verify, cache, build manifest
python build_camus.py             # optional: convert + harmonize CAMUS
```

### 3. Training

```bash
cd Dilukshan/training
python run_train.py --v2 \
    --extra-manifest artifacts/camus_manifest.csv \
    --logit-adjustment-tau 0.5 \
    --run-name uefnet_v3 --epochs 45 --patience 18
```

Interrupted runs resume with full state (weights, optimizer, EMA, RNG):

```bash
python run_train.py --resume --run-name uefnet_v3
```

### 4. Evaluation

```bash
python run_eval.py --run-name uefnet_v3 --n-tta 10
python run_ensemble.py --runs uefnet_v3 uefnet_v3b --n-tta 10
```

---

## Evaluation protocol

Reported numbers are intended to be trustworthy rather than favourable:

- Validation is split into **tuning** and **calibration** partitions; decision rules are fitted only on validation data
- The test split is evaluated **once** with a frozen strategy — no thresholds, blends, or scaling are tuned on test
- Both an **operational** rule (balanced) and a **clinical reference** (raw regression at the published 30/40/55 boundaries) are reported
- Per-class metrics carry **Wilson confidence intervals**; EF predictions carry **split-conformal** intervals
- Multi-clip test-time augmentation uses **label-free** sampling (no privileged ED/ES tracings at inference)

---

## Limitations

Stated explicitly, since they bound what the results can claim:

1. **Not all classes reach 75% recall.** Min-class recall is bounded above by balanced accuracy (a minimum cannot exceed a mean), which is 0.743 here. Post-hoc calibration can redistribute recall between classes but cannot raise the mean.
2. **Boundary error is largely irreducible.** 36.7% of test patients lie within ±4 EF points of a class boundary — within one MAE — and 343 of those sit at the EF = 55 threshold alone. Reported expert inter-observer variability is ~4–5 EF points, comparable to the model's error.
3. **Small extreme-tail test set.** Severe contains 83 test cases, giving a 95% confidence interval of roughly ±10% on its recall.
4. **Ablations are partially controlled.** The cross-dataset co-training effect is measured across runs at matched epochs rather than as a single-variable ablation; the calibration and ensembling effects are controlled comparisons on identical weights.
5. **Single-cohort evaluation.** CAMUS is used for training only; no external test cohort is held out.

---

## Citations

If you use this work, please cite the underlying datasets:

```bibtex
@article{ouyang2020video,
  title   = {Video-based AI for beat-to-beat assessment of cardiac function},
  author  = {Ouyang, David and He, Bryan and Ghorbani, Amirata and others},
  journal = {Nature},
  volume  = {580},
  pages   = {252--256},
  year    = {2020}
}

@article{leclerc2019deep,
  title   = {Deep Learning for Segmentation using an Open Large-Scale Dataset
             in 2D Echocardiography},
  author  = {Leclerc, Sarah and Smistad, Erik and Pedrosa, Joao and others},
  journal = {IEEE Transactions on Medical Imaging},
  volume  = {38},
  number  = {9},
  pages   = {2198--2210},
  year    = {2019}
}
```

Methods referenced: CORAL ordinal regression (Cao et al., 2020), deferred re-weighting (Cao et al., 2019), effective-number class weighting (Cui et al., 2019), label distribution smoothing (Yang et al., 2021), logit adjustment (Menon et al., 2021).

---

## Status

Research project (PP2). Results above are from the two-seed ensemble; a third seed is in progress.
