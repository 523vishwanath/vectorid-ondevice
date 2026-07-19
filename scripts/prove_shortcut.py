#!/usr/bin/env python3
"""
VectorCam — prove_shortcut.py
=============================
Turns the "model learned condition, not species" claim from an inference into a
demonstrated fact.

Idea:
  In the INSECTARY, An. gambiae was photographed BOTH fresh and dried.
  (Kenya can't test this — there all gambiae are fresh.)
  So we ask: does the model classify DRIED gambiae well but FRESH gambiae badly?
  If yes -> the species is identical, only condition changed, so the model is
  keying on CONDITION, not morphology. That is the shortcut, proven.

  A true-morphology model would score fresh and dried gambiae about equally.

Also prints, for every species, accuracy split by condition — so you can see the
whole condition effect at once.

Usage:
  python prove_shortcut.py --ckpt runs/efficientvit_b0_vectorcam.pt
"""

import argparse, os
import numpy as np, pandas as pd, torch, timm
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

CLASSES = ["aedes_aegypti", "anopheles_gambiae", "anopheles_stephensi"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}


class DS(Dataset):
    def __init__(self, df, tf): self.df = df.reset_index(drop=True); self.tf = tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(r["path"]).convert("RGB")), r["y"], i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/efficientvit_b0_vectorcam.pt")
    ap.add_argument("--master", default="master_cropped.csv")
    ap.add_argument("--images", default="data_cropped")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device)
    name, img = ck["model"], ck.get("img", 256)
    model = timm.create_model(name, pretrained=False, num_classes=3)
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    print(f"loaded {name} @ {img}px\n")

    m = pd.read_csv(args.master)
    m = m[m["cropped_path"].notna()].reset_index(drop=True)
    def resolve(r):
        p = r["cropped_path"]
        if isinstance(p, str) and os.path.exists(p): return p
        return os.path.join(args.images, f"image_{int(r['ImageID'])}.jpg")
    m["path"] = m.apply(resolve, axis=1)
    m = m[m["path"].map(os.path.exists)].reset_index(drop=True)
    m["y"] = m["species"].map(CLS2IDX)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    resize = int(img * 1.14)
    tf = T.Compose([T.Resize(resize), T.CenterCrop(img), T.ToTensor(), T.Normalize(mean, std)])

    def predict_df(df):
        dl = DataLoader(DS(df, tf), batch_size=args.batch, num_workers=4)
        preds = np.zeros(len(df), dtype=int)
        with torch.no_grad():
            for x, y, idx in dl:
                with torch.amp.autocast("cuda"):
                    out = model(x.to(device))
                preds[idx.numpy()] = out.argmax(1).cpu().numpy()
        d = df.copy(); d["pred"] = preds; d["correct"] = d["pred"] == d["y"]
        return d

    # --- normalise condition labels ---
    m["cond"] = m["SessionSpecimenCondition"].astype(str).str.lower()
    def bucket(c):
        if "fresh" in c: return "fresh"
        if "des" in c or "dry" in c or "dri" in c: return "dried"
        return "other"
    m["cond"] = m["cond"].map(bucket)

    # ================= THE KEY TEST: insectary gambiae, fresh vs dried =============
    ins = predict_df(m[m["source"] == "insectary"])
    gamb = ins[ins["species"] == "anopheles_gambiae"]

    print("=" * 60)
    print("KEY TEST — INSECTARY An. gambiae, split by condition")
    print("(same species; only fresh/dried differs)")
    print("=" * 60)
    tab = gamb.groupby("cond")["correct"].agg(["mean", "size"])
    print(tab, "\n")
    if {"fresh", "dried"}.issubset(set(tab.index)):
        f, d = tab.loc["fresh", "mean"], tab.loc["dried", "mean"]
        print(f"fresh gambiae acc = {f:.3f}   dried gambiae acc = {d:.3f}")
        print(f"gap = {d - f:+.3f}")
        print("\nINTERPRETATION:")
        if d - f > 0.15:
            print("  Large gap -> model is keying on CONDITION, not species.")
            print("  Same mosquito species scores very differently by fresh/dried.")
            print("  => shortcut learning is DEMONSTRATED, not just suspected.")
        else:
            print("  Small gap -> condition is not the main driver for gambiae.")

    # where do the misclassified fresh gambiae GO?
    fg = gamb[gamb["cond"] == "fresh"]
    if len(fg):
        print("\nFresh insectary gambiae -> predicted as:")
        print(fg["pred"].map({i: c for c, i in CLS2IDX.items()}).value_counts())

    # ================= full picture: every species x condition ==================
    print("\n" + "=" * 60)
    print("FULL PICTURE — insectary accuracy by species x condition")
    print("=" * 60)
    piv = ins.groupby(["species", "cond"])["correct"].agg(["mean", "size"])
    print(piv)


if __name__ == "__main__":
    main()
