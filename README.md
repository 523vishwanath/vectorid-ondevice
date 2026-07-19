# VectorID — On-Device Mosquito Species Classifier

Real-time mosquito species identification that runs **entirely in the browser**, on
the phone, with **no internet needed after first load**. Built for the VectorCam
low-connectivity field target.

**[Try it live »](https://YOUR_USERNAME.github.io/YOUR_REPO/)**  ← update after enabling Pages

Classifies three species: *Aedes aegypti*, *Anopheles gambiae*, *Anopheles stephensi* —
with the primary goal of flagging the invasive *An. stephensi* among the established
*An. gambiae*.

## What makes this version different

This is the on-device successor to my earlier [VectorID](https://github.com/523vishwanath/VectorCam_Mosquito_classifier),
which ran inference through a cloud API. VectorCam's real target is low-end phones with
poor connectivity, so this version runs the model **on the device itself** via
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) — the phone downloads the
~8.5 MB model once, then classifies offline.

| | Old VectorID | This version |
|---|---|---|
| Inference | Cloud API (needs internet each time) | On-device (offline after first load) |
| Model | EfficientNet-B0, 7 species + sex | EfficientViT-B0, 3 target species |
| Preprocessing | Server-side | In-browser (matches training) |

## How it works

1. Take a photo (camera) or choose one from the gallery.
2. The image is preprocessed in-browser — resize to 291, center-crop to 256, normalize —
   to match the training pipeline.
3. `vectorcam.onnx` runs in the browser via ONNX Runtime Web.
4. The predicted species, confidence, and per-class probabilities are shown. Low-confidence
   results (<75%) are flagged for expert review.

## Honest limitations

- **Not field-ready.** Field macro-F1 is ~0.65. The two *Anopheles* (gambiae vs
  stephensi) are frequently confused — which is the program's core task. See the main
  assessment write-up for the full go/no-go analysis.
- **Center-crop, not segmentation.** The model was trained on U2Net-segmented crops.
  This demo uses a center-crop for simplicity; users should frame the specimen centrally.
  Production would run a lightweight on-device segmenter (u2netp, ~5 MB) so inference
  matches the segmented training crop exactly.
- **No "not a mosquito" class.** Any image is forced into one of the three species; the
  confidence floor is a rough guard, not true out-of-distribution rejection.

## Files

- `index.html` — the complete app (UI + preprocessing + inference, single file)
- `vectorcam.onnx` — the exported EfficientViT-B0 classifier (~8.5 MB)
- `labels.json` — class label order

## Run locally

```bash
python -m http.server 8000
# open http://localhost:8000
```

(Must be served over http — ONNX Runtime Web won't load from a file:// path.)

## Author

**Vishwanath Ninganolla** — [GitHub](https://github.com/523vishwanath) · [LinkedIn](https://linkedin.com/in/vishwanathninganolla)

Built as part of the Johns Hopkins VectorCam technical assessment.
