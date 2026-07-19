# VectorCam Mosquito Classifier — Field Evaluation and Go/No-Go

**Vishwanath Ninganolla** · Assistant Research Engineer assessment · July 2026

---

## Recommendation: NO-GO this quarter

**The number I stake this on: macro-F1 ≈ 0.65 on held-out Kenya field specimens.**

A classifier trained on the insectary data reaches ~0.99 in the lab but drops to **0.648 macro-F1** on real Kenya field images. The failure lands exactly where the program cannot afford it: the model cannot reliably tell the two *Anopheles* species apart (*An. gambiae* F1 0.56, *An. stephensi* F1 0.49). Since the program's primary objective is detecting invasive *An. stephensi* among established *An. gambiae*, a model that identifies the invasive target barely better than chance is not deployable.

This is a **no-go with a proven path to go.** Below I show the gap is caused by *missing field training data*, not by the model — and that adding even a small amount of field data lifts field macro-F1 by **+0.31**, taking all three species above 0.95. The fix is a targeted collection effort, specified at the end.

---

## How I set up the evaluation, and why the number is honest

**The trap I deliberately avoided.** The easy way to post a high number is to pool everything (insectary + Kenya) and split it randomly. That scores ~0.95 — but it puts field data in the training set and makes the test set specimens the model has effectively already seen. It answers "can it re-recognize familiar specimens?", not "will it work in Kenya?". That is the high number I could not defend, so I did not use it.

**The split I used — insectary → Kenya.** Train and validate on insectary data (drops 0610, 0618); then test on data the model never saw, in two stages that mirror deployment:

| Test set | What it probes | Macro-F1 |
|---|---|---|
| **Test A — 0623 (new phones)** | generalization to unseen devices | **0.92** |
| **Test B — Kenya (field)** | deployment reality | **0.65** |

Insectary → Kenya *is* the deployment: a lab-trained model meeting field data for the first time. That is why the 0.65, not the 0.95, is the number I report.

**Leakage control.** Splits are grouped by `SpecimenID`, never by image. Each insectary specimen is photographed ~20 times; splitting by image would scatter near-duplicates across train and test and inflate the score. A hard assertion fails the run if any specimen lands in two splits.

**Metric — macro-F1, not accuracy.** Kenya is ~two-thirds *An. gambiae*, so a model that always answers "gambiae" would score well on accuracy while being useless for surveillance. Macro-F1 weights all three species equally and penalizes both false positives and false negatives, which is what catching an invasive species demands.

**Data judgment before modelling.** I interrogated the four drops before trusting them:
- **Label contamination:** ten specimens carried images labelled as more than one species. Six (in 0618, at ~29:1 image ratios) were stray-frame mislabels — corrected by majority vote. Four (in 0623) were genuine ID collisions where a specimen ID was reused across sessions — dropped, because they could not be safely resolved.
- **Unlabelled images** were dropped, not guessed.
- **Result:** zero species-spanning specimens remain. The cleaned set is 5,479 images / 691 specimens.

---

## The three things that most shaped my confidence

**1. The gap is data-bound, not model-bound — I tested this six ways.**
Six architectures from 1.5M to 46M parameters (MobileNetV3, EfficientNet-B0 at two resolutions, EfficientViT-B0/B3, ConvNeXt-nano) all land on Kenya in the same 0.64–0.76 band. A *larger* model (EfficientViT-B3, 46M) did *worse* on Kenya, not better — it overfit the lab. A pure CNN did not beat the hybrid. When 30× more capacity and four architecture families all tie, the ceiling is the data, not the model.

**2. I ruled out the shortcuts before blaming the data.**
Grad-CAM on the insectary-only model showed it keying on the *tray background*, not the mosquito — a classic shortcut. I removed the tray with U2Net segmentation and retrained. Cross-device generalization improved (0.89 → 0.92) and Grad-CAM confirmed the model now attends to the specimen — but **Kenya did not recover (~0.65).** So the background was a real shortcut, but not the cause of the field gap. I separately tested and ruled out a fresh-vs-dried "condition shortcut." The gap is intrinsic to the field specimens themselves.

**3. The root cause: a condition reversal, compounded by field image quality.**
Breaking the data down by specimen condition exposes the core problem. For **every** species, the condition seen in training is the *opposite* of the condition seen in Kenya:

| Species | Insectary (train) | Kenya (test) |
|---|---|---|
| *Ae. aegypti* | ~100% fresh | ~100% dried |
| *An. stephensi* | ~100% fresh | ~100% dried |
| *An. gambiae* | mostly dried | ~100% fresh |

The model is always tested on the condition it saw *least* for each species. This is why *An. stephensi* is the worst class — the training set contains **zero dried stephensi**, and every Kenya stephensi is dried. Inspecting the images adds a second factor: Kenya specimens are frequently squished, broken, or blurry, with wings obscured — and the wings carry the fine morphology that separates the two *Anopheles*. So the field gap is condition reversal (major), plus genuine field image-quality degradation. Both are data-coverage problems, and both are fixable by collecting the right images.

---

## The evidence that data closes the gap

The decisive experiment: split Kenya's specimens in half (grouped by `SpecimenID`, stratified by species, no leakage), add one half to insectary training, and test on the **other, held-out half**.

| Training set | Held-out Kenya macro-F1 |
|---|---|
| Insectary only (baseline) | **0.648** |
| Insectary + ~230 Kenya specimens | **0.955** |
| **Lift** | **+0.307** |

Per class, adding field data takes *An. stephensi* from **0.49 → 0.95** and *An. gambiae* from **0.56 → 0.96** — exactly the confusion that blocks deployment. Repeated across three independent random splits, the lift held every time (+0.20, +0.29, +0.27).

**Honest calibration.** The 0.955 is *optimistic* — the added Kenya data and the held-out Kenya data come from the same collection sessions, so they share conditions a fresh deployment would not. I therefore report the **+0.31 lift** as the robust finding (field data helps, and a lot), and I do *not* present 0.955 as a production number. The lift is the evidence for the collection ask; the absolute would need more diverse field data to pin down.

---

## Recommended model and edge feasibility

**EfficientViT-B0**, trained on segmented crops: **8.5 MB, ~6 ms/image**, small and fast enough for a low-end phone. MobileNetV3 (6 MB) is a fallback if the target hardware handles pure-CNN operations better than attention layers. The classifier runs fully offline. Preprocessing (locating and cropping the specimen) must also run on-device and must match the training crop — for a low-connectivity phone that means a lightweight on-device segmenter (u2netp, ~5 MB) or aligning with VectorCam's existing capture framing. A working on-device browser demo (ONNX Runtime Web, no server, offline after first load) is included, with low-confidence predictions flagged for expert review.

---

## What I would do next

**On the modelling side.**
1. Add a lightweight on-device segmenter (u2netp) so inference crops match the segmented training crops exactly, removing the train/inference mismatch.
2. Add a confidence threshold that routes uncertain specimens to human review instead of forcing a guess — the current model has no "not a mosquito" class and no way to abstain.
3. Once field data arrives, train the production model on a condition-stratified split so it sees every species in both fresh and dried states — but evaluate it on a *held-out future field batch*, so the reported number stays honest.

**What to ask the Kenya team to collect (~2,000 images).** Spend them on the gaps, not evenly:
- **~35% fresh field *An. stephensi*** — the invasive target; **zero fresh stephensi exist today.**
- **~40% fresh field *An. gambiae*** — the dominant field species and the hardest fresh case.
- **~15% *Ae. aegypti* in the missing condition (fresh field)** — Aedes looks easy now *only* because all current Aedes is one condition; the other condition will fail the same way stephensi does today. Collect it before it becomes a field surprise.
- **~10% dried controls** across species.

Plus two capture-quality requirements, because image quality is itself part of the gap: photograph the **same specimen on multiple phones** (so the model learns the mosquito, not the camera), take **more angles per specimen** (Kenya currently averages ~1.6 images vs ~20 in the lab), and minimize squishing and damage so wings stay visible. Specimens too damaged or blurry to read should be flagged for expert review rather than force-classified.

---

*Reproducibility: the repository contains the full pipeline (data preparation → training → evaluation), both model checkpoints (insectary-only and insectary+Kenya), the exact Kenya split CSVs, and the run logs — so every number above can be regenerated on your machine.*
