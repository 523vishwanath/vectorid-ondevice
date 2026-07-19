# VectorCam Mosquito Classifier — Field Deployment Assessment

Can we build a pipeline that classifies field specimens reliably enough to deploy in Kenya, and is it ready this quarter? **Short answer: no-go this quarter, with a proven path to go.** Field macro-F1 is ~0.65; adding field training data lifts it by +0.31. Full reasoning in [`WRITEUP.md`](WRITEUP.md).

**Author:** Vishwanath Ninganolla · [GitHub](https://github.com/523vishwanath) · [LinkedIn](https://linkedin.com/in/vishwanathninganolla)

---

## The deliverables

| Deliverable | Where |
|---|---|
| **Write-up** (go/no-go + the number) | [`WRITEUP.md`](WRITEUP.md) |
| **System diagram** (capture → prediction) | [`system_diagram.svg`](system_diagram.svg) |
| **Reproducible pipeline** | `scripts/` + `notebooks/` (see below) |
| **What next / data ask** | end of [`WRITEUP.md`](WRITEUP.md) |
| **Live on-device demo**  | [github.com/523vishwanath/vectorid-ondevice](https://523vishwanath.github.io/vectorid-ondevice/) |

---

## Headline result

| Evaluation | Macro-F1 |
|---|---|
| Validation (insectary, held-out specimens) | 0.99 |
| Test A — unseen devices (drop 0623) | 0.92 |
| **Test B — Kenya field (the number that counts)** | **0.65** |
| Kenya, after adding ~230 field specimens to training | 0.96 (+0.31) |

The 0.65 is the honest insectary→field deployment number. The +0.31 lift from adding field data is the evidence behind the data-collection recommendation.

---

## Repository structure

```
├── WRITEUP.md                         # go/no-go write-up (read this first)
├── system_diagram.svg                 # capture-to-prediction pipeline
├── requirements.txt                   # environment
├── scripts/
│   ├── data_prep.py                   # clean + build the master label table
│   ├── train.py                       # train + evaluate one model (insectary -> Kenya)
│   ├── train_with_kenya.py            # field-data lift experiment (saves both checkpoints)
│   └── prove_shortcut.py              # fresh-vs-dried shortcut test
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 05c_segment_from_originals.ipynb   # U2Net segmentation crop
│   ├── 07_final_metrics_gradcam.ipynb     # metrics + Grad-CAM (both models)
│   └── 10_onnx_export.ipynb               # export model for on-device demo
├── runs/
│   ├── efficientvit_b0_baseline_insectary.pt   # insectary-only model (Kenya 0.65)
│   ├── efficientvit_b0_augmented_kenya.pt      # insectary+Kenya model (Kenya 0.96)
│   ├── kenya_test_split.csv / kenya_train_split.csv  # exact split, for reproduction
│   └── augmented_run.log
└── results/
    ├── kenya_field_data_effect.png    # the +0.31 lift, 3 seeds
    └── gradcam_grids.png              # model attends to the mosquito
```

---

## Reproduce the results

**1. Environment**
```bash
pip install -r requirements.txt
```

**2. Data prep** (clean labels, resolve contamination, build master table)
```bash
python scripts/data_prep.py
```

**3. Train + evaluate** the insectary→Kenya model
```bash
python train.py --model efficientvit_b0 --master master_seg.csv --images data_seg
```
Prints validation, Test A (device), and Test B (Kenya) macro-F1 with per-class breakdowns.

**4. Reproduce the field-data lift** (trains baseline + augmented, saves both checkpoints)
```bash
python train_with_kenya.py --model efficientvit_b0 --master master_seg.csv --images data_seg --kenya-frac 0.5
```

Everything is grouped by `SpecimenID` with hard leakage assertions, so the split is honest and repeatable.

---

## Key decisions (the short version)

- **Evaluation split:** insectary → Kenya, because that *is* the deployment. A pooled random split scores ~0.95 but answers the wrong question.
- **Metric:** macro-F1 on Kenya field data (Kenya is 2/3 gambiae; accuracy would reward a lazy majority-guesser).
- **No leakage:** grouped by `SpecimenID`; ~20 images per insectary specimen would otherwise leak across splits.
- **Data cleaning:** 10 species-spanning specimens resolved (6 relabelled by majority, 4 dropped as ID collisions).
- **Root cause of the field gap:** a per-species condition reversal (each species tested on the condition it saw least) plus field image-quality degradation — not the model. Confirmed by six architectures tying, shortcuts ruled out, and a +0.31 lift from field data.

Full reasoning and the honest calibration of every number are in [`WRITEUP.md`](WRITEUP.md).

---

## Edge / deployment

Recommended model: **EfficientViT-B0** — 8.5 MB, ~6 ms/image, runs fully offline on a low-end phone. A working browser demo (ONNX Runtime Web, on-device, offline after first load) is at [vectorid-ondevice](https://github.com/523vishwanath/vectorid-ondevice), with low-confidence predictions flagged for expert review.
