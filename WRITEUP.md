# VectorCam Mosquito Classifier — Field Evaluation and Go/No-Go

**Vishwanath Ninganolla** · Assistant Research Engineer assessment · July 2026

---

## My call: not this quarter.

**The number I'd stake it on: macro-F1 ≈ 0.65 on Kenya field specimens the model has never seen.**

A classifier trained on the insectary data looks excellent in the lab — about 0.99. But that number evaporates on real Kenya field images, where it lands at **0.648**. And it fails in the worst possible place: it can't reliably tell the two *Anopheles* apart (*An. gambiae* 0.56, *An. stephensi* 0.49). Since the entire point of the program is spotting invasive *An. stephensi* hiding among *An. gambiae*, a model that calls that distinction barely better than a coin flip isn't something I'd deploy.

So it's a no-go — but a confident one, because I know *why*, and I know the fix. The gap isn't the model. It's missing field data. When I add even a small amount of Kenya data to training, field performance jumps by **+0.31**, and all three species clear 0.95. The path to "go" is a specific, targeted collection effort, laid out at the end.

---

## How I measured it, and why I trust the number

**The tempting shortcut I didn't take.** The easy way to post a big number is to throw all the data — insectary and Kenya — into one pile and split it randomly. Do that and you get ~0.95. But it's a lie: it drops Kenya specimens into training and then tests on specimens the model has effectively already met. It answers "can you re-recognize something familiar?" when the real question is "will you work in Kenya?" I threw that split out.

**The split I actually used.** Train on insectary (0610, 0618). Then test on data the model has never touched, in two steps:

| Test | What it really asks | Macro-F1 |
|---|---|---|
| **0623 — new phones** | does it survive an unseen device? | **0.92** |
| **Kenya — field** | does it survive the field? | **0.65** |

Insectary → Kenya isn't an arbitrary choice. It *is* the deployment: a lab-trained model meeting the field cold. That's why I report 0.65 and not 0.95 — one is the truth about deployment, the other is a story I told myself with leaked data.

**Keeping it honest under the hood.** Every split is grouped by `SpecimenID`, never by image. Each insectary specimen shows up in ~20 photos; split by image and near-identical shots leak across train and test, quietly inflating the score. A hard assertion kills the run if a single specimen lands in two splits.

**Why macro-F1, not accuracy.** Kenya is about two-thirds *An. gambiae*. A lazy model that just yells "gambiae!" every time would score well on plain accuracy while being useless. Macro-F1 treats all three species equally and punishes both kinds of error — which is exactly the pressure you want when the whole job is catching a rare invasive.

**I cleaned the data before I trusted it.** Ten specimens had images labelled as more than one species. Six (in 0618, at ~29:1 ratios) were obvious stray-frame mistakes — I fixed those by majority vote. Four (in 0623) were genuine ID collisions, the same ID reused across sessions — I dropped those, because guessing would've been worse than losing them. Unlabelled images got dropped, not invented. What's left has zero species-spanning specimens: 5,479 images, 691 specimens.

---

## The three things that decided my confidence

**1. It's the data, not the model — and I checked that six ways.** I ran six architectures from 1.5M to 46M parameters (MobileNetV3, EfficientNet-B0 at two resolutions, EfficientViT-B0/B3, ConvNeXt-nano). Every one lands on Kenya between 0.64 and 0.76. The *biggest* model did *worse*, not better. A pure CNN didn't beat the hybrid. When 30× more capacity and four different architecture families all pile up at the same wall, the wall is the data. No architecture is going to think its way past missing information.

**2. I ruled out the cheap explanations before blaming the data.** Grad-CAM caught the model red-handed early on — it was looking at the *tray*, not the mosquito. Classic shortcut. So I cut the tray out with U2Net segmentation and retrained. Cross-device generalization improved (0.89 → 0.92), and Grad-CAM confirmed the model was now actually looking at the specimen. But Kenya *still* didn't budge — stuck at ~0.65. That was the tell: the background was a real shortcut, but it wasn't what was breaking field performance. (I separately tested and killed a "fresh vs dried" shortcut theory too.) The gap lives in the field specimens themselves.

**3. The root cause is a condition reversal — and it's almost poetic how clean it is.** Break the data down by specimen condition and the problem jumps out. For *every single species*, the condition it trained on is the opposite of the condition it meets in Kenya:

| Species | Trained on | Tested on (Kenya) |
|---|---|---|
| *Ae. aegypti* | ~100% fresh | ~100% dried |
| *An. stephensi* | ~100% fresh | ~100% dried |
| *An. gambiae* | mostly dried | ~100% fresh |

The model is always graded on the version of each species it saw *least*. That's exactly why *An. stephensi* is the worst class — there are **zero dried stephensi in training**, and every Kenya stephensi is dried. Looking at the images adds the second half of the story: the Kenya specimens are often squished, broken, or blurry, with the wings — where the *Anopheles* differences actually live — obscured or gone. So the field gap is condition reversal, plus real image-quality degradation. Both are coverage problems. Both are fixable by collecting the right images.

---

## The proof that data closes it

Here's the experiment that turns "I think it's the data" into "it's the data." Split Kenya in half by specimen (grouped, stratified, no leakage). Add one half to insectary training. Test on the other half — specimens the model never saw.

| Trained on | Held-out Kenya macro-F1 |
|---|---|
| Insectary only | **0.648** |
| Insectary + ~230 Kenya specimens | **0.955** |
| **Lift** | **+0.31** |

*An. stephensi* goes from **0.49 to 0.95**. *An. gambiae* from **0.56 to 0.96**. That's the exact confusion blocking deployment, resolved by showing the model the field conditions it was missing. I ran it three times on different random splits; the lift held every time (+0.20, +0.29, +0.27).

**Where I keep myself honest:** that 0.955 is optimistic. The Kenya half I trained on and the Kenya half I tested on come from the same collection sessions, so they share conditions a genuinely new deployment wouldn't. So I don't sell 0.955 as a production number — I sell the **+0.31 lift**, which is the robust, repeatable finding. Field data helps, a lot. The exact deployable number needs more diverse field data to pin down — which is precisely what the collection ask is for.

---

## The model I'd ship, and whether it fits a phone

**EfficientViT-B0**, trained on segmented crops: **8.5 MB, ~6 ms/image**. Small and fast enough for the low-end phones field teams already carry. MobileNetV3 (6 MB) is my fallback if the target hardware handles plain CNN ops better than attention. The classifier runs fully offline. The one catch: the crop step has to run on-device too and has to match the training crop — for a low-connectivity phone that means a lightweight on-device segmenter (u2netp, ~5 MB) or leaning on VectorCam's existing capture framing.

I built a working on-device demo to prove the edge story isn't hypothetical: a browser app running the model via ONNX Runtime Web, fully offline after first load (installable as a PWA), with low-confidence and low-margin predictions declined rather than forced — because the model has no "not a mosquito" class, and pretending otherwise in the field would be dangerous.

---

## What I'd do next

**On the modelling side:**
1. Put a lightweight segmenter (u2netp) on-device so the inference crop matches the training crop — closing the train/inference gap the demo currently papers over.
2. Add a real abstention path: a confidence-and-margin gate (in the demo already) now, and a proper "not a mosquito" class once there's negative data to train it on.
3. When field data arrives, train the production model on a condition-stratified split so it sees every species fresh *and* dried — but grade it on a **held-out future field batch**, so the number stays honest.

**What I'd ask the Kenya team to collect (~2,000 images).** Spend them on the holes, not evenly:
- **~35% fresh field *An. stephensi*** — the invasive target, and we have **zero fresh stephensi** right now.
- **~40% fresh field *An. gambiae*** — the dominant field species and the hardest fresh case.
- **~15% *Ae. aegypti* in its missing condition (fresh field)** — Aedes only *looks* easy today because every Aedes we have is one condition. The other condition will fail exactly like stephensi does now. Collect it before it bites us in production.
- **~10% dried controls** across species.

And two capture-quality asks, because image quality is half the gap: shoot the **same specimen on multiple phones** (so the model learns the mosquito, not the camera), take **more angles per specimen** (Kenya averages ~1.6 shots vs ~20 in the lab), and handle specimens gently enough that the wings survive. Anything too damaged or blurry to read should go to a human, not get a forced guess.

---

*Everything here reproduces: the repo has the full pipeline (prep → train → evaluate), both checkpoints (insectary-only and insectary+Kenya), the exact Kenya split files, and the run logs. Detail on the dataset, training setup, experiment tracking, and the app is in the README.*
