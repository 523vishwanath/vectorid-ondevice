# VectorCam — Field-Deployable Mosquito Species Classifier

A phone-based classifier that identifies three mosquito species — *Aedes aegypti*, *Anopheles gambiae*, and the invasive *Anopheles stephensi* — from a single photo, built for malaria vector surveillance in Kenya.

**The bottom line:** on real Kenya field data the model reaches **macro-F1 ≈ 0.65**, which isn't good enough to deploy this quarter — but I show the gap is *missing field data*, not the model, and that a small amount of field data lifts performance by **+0.31**. Full reasoning in **[WRITEUP.md](WRITEUP.md)**.

**Author:** Vishwanath Ninganolla · [GitHub](https://github.com/523vishwanath) · [LinkedIn](https://linkedin.com/in/vishwanathninganolla)

---

## Start here

| If you want… | Go to |
|---|---|
| The recommendation and reasoning (read first) | **[WRITEUP.md](WRITEUP.md)** |
| How a photo flows from capture to prediction | **[system_diagram.svg](system_diagram.svg)** |
| To reproduce the numbers | [Reproducing the results](#reproducing-the-results) below |
| To try the live on-device app | Open `index.html` on GitHub Pages (offline-capable PWA) |

---

## Headline results

| Evaluation | Macro-F1 | What it tells you |
|---|---|---|
| Validation (insectary, held-out specimens) | 0.99 | Learns the lab easily |
| Test A — unseen devices (drop 0623) | 0.92 | Handles new phones |
| **Test B — Kenya field** | **0.65** | **The honest deployment number** |
| Kenya, after adding ~230 field specimens | 0.96 | The fix: +0.31 from field data |

The 0.65 is a model that has *never seen Kenya*, tested on all of Kenya — the true cold-start deployment number. The 0.96 comes from adding half of Kenya to training and testing on the held-out half; it's the evidence that field data closes the gap.

---

## The dataset

Four data drops, provided as-is, assembled by different people on different devices across several months.

| Drop | Source | Images | Specimens | Notes |
|---|---|---|---|---|
| 0610 | Insectary | 1,686 | 71 | Lab-reared, controlled |
| 0618 | Insectary | 2,618 | 97 | Lab-reared, controlled |
| 0623 | Insectary | 442 | 62 | **Two new phone models** (device-generalization test) |
| kenya_01 | **Field (Kenya)** | 753 | 461 | Field-caught (deployment test) |

After cleaning: **5,479 images / 691 specimens.**

**The condition split that drives everything.** Breaking the images down by specimen condition (fresh vs dried) exposes the core problem — for every species, training and Kenya use opposite conditions:

| Species | Insectary (train) | Kenya (test) |
|---|---|---|
| *Ae. aegypti* | 1767 fresh / 0 dried | 0 fresh / 146 dried |
| *An. gambiae* | 666 fresh / 972 dried | 326 fresh / 0 dried |
| *An. stephensi* | 1321 fresh / 0 dried | 0 fresh / 266 dried |

The model is always tested on the condition it saw *least*. There are **zero dried stephensi in training**, and every Kenya stephensi is dried — which is why stephensi is the weakest class.

**Data cleaning decisions.** Ten specimens carried images labelled as more than one species. Six (in 0618, at ~29:1 image ratios) were stray-frame mislabels, corrected by majority vote. Four (in 0623) were genuine ID collisions where a specimen ID was reused across sessions — dropped, since they couldn't be safely resolved. Unlabelled images were dropped, not guessed. Result: zero species-spanning specimens.

---

## How the evaluation is set up

- **Split:** train/validate on insectary (0610, 0618); test on unseen device (0623) and Kenya field. Insectary → Kenya *is* the deployment scenario.
- **Grouping:** every split is grouped by `SpecimenID`, never by image — each insectary specimen has ~20 near-identical photos, so image-level splitting would leak duplicates and inflate the score. A hard assertion fails the run on any specimen leak.
- **Metric:** macro-F1 (Kenya is ~2/3 gambiae; accuracy would reward a lazy majority-guesser).
- **Cropping:** U2Net segmentation removes the tray/label so the model attends to the mosquito, not the background.

---

## What I found (error analysis)

**The gap is data-bound, not model-bound — verified six ways.** Six architectures (MobileNetV3, EfficientNet-B0 ×2 resolutions, EfficientViT-B0/B3, ConvNeXt-nano), 1.5M–46M params, all land on Kenya between 0.64 and 0.76. The largest model did *worse*. A pure CNN didn't beat the hybrid. The ceiling is the data.

**Shortcuts, found and ruled out.** Grad-CAM showed the model attending to the tray background — a shortcut. Segmentation removed it, cross-device accuracy improved (0.89 → 0.92), Grad-CAM confirmed the model now looks at the mosquito — but Kenya *didn't recover*. So background was a real shortcut but not the field-gap cause. A fresh-vs-dried "condition shortcut" was separately tested and ruled out.

**Root cause:** the condition reversal above, plus field image-quality degradation — Kenya specimens are often squished, broken, or blurry with wings (where the *Anopheles* differences live) obscured.

**The fix, proven:** adding ~230 Kenya specimens to training lifts held-out Kenya macro-F1 from 0.648 to 0.955 (+0.31), stephensi from 0.49 to 0.95, repeated across three seeds. The 0.955 is optimistic (train/test Kenya share conditions), so the **+0.31 lift** is the reported finding, not the absolute.

Figures in [`results/`](results/): `kenya_field_data_effect.png` (the lift, 3 seeds), `cm_baseline.png` / `cm_augmented.png` (confusion matrices showing the Anopheles confusion and its resolution), `gradcam_baseline.png` (model attends to the mosquito).

---

## Training setup

- **Hardware:** NVIDIA RTX 5090 (RunPod), plus Google Colab for data prep and export.
- **Backbone:** EfficientViT-B0 (timm), 256×256 input, ImageNet-pretrained.
- **Training:** class-weighted cross-entropy, AdamW, cosine schedule, early stopping on validation macro-F1, augmentations tuned for fine-grained ID (aspect-preserving, hue held at 0 to preserve leg banding).
- **Experiment tracking:** Comet ML (project `vectorcam-classifier`) logs every run's metrics, per-class scores, hyperparameters, and model artifacts. Set `COMET_API_KEY` to enable; runs are also saved to `runs/*.log` as a backup.
- **Preprocessing:** U2Net segmentation crop applied offline (notebook `05c`); the phone would use a lightweight u2netp or the app's capture framing at inference.

---

## The live app (VectorID)

A working on-device demo — the classifier runs **entirely in the phone browser**, no server, offline after first load.

**How it works:**
1. Capture or upload one or more specimen photos.
2. Each image is preprocessed in-browser (resize + normalize, matching training) and run through the model via **ONNX Runtime Web**.
3. The predicted species, confidence, and per-class probabilities are shown, alongside the exact image the model saw.

**Honest safeguards built in:**
- **Confidence + margin rejection:** because the model has no "not a mosquito" class, a prediction is declined ("No confident match") when the top probability is low *or* there's no clear margin over second place — so selfies and non-mosquitoes aren't force-classified.
- **Expert-review flag:** borderline predictions are flagged rather than reported as certain.
- **True offline:** a service worker (`sw.js`) caches the app and model; it's installable as a PWA and runs with no connection.

**Files:** `index.html` (app), `vectorcam.onnx` (exported model, ~8.5 MB), `sw.js` + `manifest.json` + `icon-*.png` (PWA). Deployed via GitHub Pages.

Known limitation: the demo resizes the full image rather than segmenting; production would run u2netp on-device so the inference crop matches the segmented training crop. Stated plainly because it matters for field accuracy.

---

## Repository structure

```
├── WRITEUP.md                      # go/no-go write-up — read first
├── system_diagram.svg              # capture → prediction pipeline
├── README.md                       # this file
│
├── scripts/
│   ├── data_prep.py                # clean labels, resolve contamination, build master table
│   ├── train.py                    # train + evaluate one model (insectary → Kenya)
│   ├── train_with_kenya.py         # field-data lift experiment; saves both checkpoints + splits
│   ├── prove_shortcut.py           # fresh-vs-dried shortcut test
│   └── requirements.txt            # environment
│
├── notebooks/
│   ├── 01_data_prep.ipynb          # data assembly + EDA (with outputs)
│   ├── 05c_segment_from_originals.ipynb   # U2Net segmentation crop
│   ├── 07_final_metrics_gradcam.ipynb     # final metrics, confusion matrices, Grad-CAM (both models)
│   └── 10_onnx_export.ipynb        # export model for the on-device app
│
├── runs/
│   ├── efficientvit_b0_baseline_insectary.pt   # insectary-only model (Kenya 0.65)
│   ├── efficientvit_b0_augmented_kenya.pt      # insectary + Kenya model (Kenya 0.96)
│   ├── kenya_test_split.csv / kenya_train_split.csv   # exact split for reproduction
│   └── augmented_run.log
│
├── results/
│   ├── kenya_field_data_effect.png # the +0.31 lift (3 seeds)
│   ├── cm_baseline.png / cm_augmented.png   # confusion matrices
│   └── gradcam_baseline.png        # model attends to the mosquito
│
└── index.html, vectorcam.onnx, sw.js, manifest.json, icon-*.png   # the live on-device app
```

---

## Reproducing the results

**1. Environment**
```bash
pip install -r scripts/requirements.txt
```

**2. Data preparation** — cleans labels, resolves the 10 contaminated specimens, builds the master table
```bash
python scripts/data_prep.py
```

**3. Train + evaluate** the insectary → Kenya model
```bash
python scripts/train.py --model efficientvit_b0 --master master_seg.csv --images data_seg
```
Prints validation, Test A (device), and Test B (Kenya) macro-F1 with per-class breakdowns.

**4. Reproduce the field-data lift** — trains baseline + augmented, saves both checkpoints and the exact split
```bash
python scripts/train_with_kenya.py --model efficientvit_b0 --master master_seg.csv --images data_seg --kenya-frac 0.5
```

**5. Metrics, confusion matrices, Grad-CAM** — open `notebooks/07_final_metrics_gradcam.ipynb`.

The provided checkpoints in `runs/` let you reproduce every reported number directly, and the split CSVs let you verify the exact train/test partition. Everything is grouped by `SpecimenID` with leakage assertions.

*Note: the raw image drops are JHU's data and aren't committed here; point the scripts at the provided data to regenerate.*

---

## Recommendation, in one line

**No-go this quarter** (field macro-F1 ≈ 0.65; the two *Anopheles* aren't separable yet), **with a proven path to go**: targeted field-data collection, specified at the end of [WRITEUP.md](WRITEUP.md).
