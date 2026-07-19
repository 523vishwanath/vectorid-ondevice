#!/usr/bin/env python3
"""
VectorCam — train_with_kenya.py
================================
Diagnostic experiment: does adding a little FIELD data lift FIELD performance?

Method (no specimen leakage anywhere):
  1. Split Kenya's specimens by SpecimenID into kenya_train / kenya_test.
     (grouped + stratified by species so both halves have all 3 classes)
  2. Baseline run   : train on insectary only        -> test on kenya_test
  3. Augmented run  : train on insectary + kenya_train -> test on SAME kenya_test
  4. Report both Kenya-test macro-F1 side by side. If augmented > baseline,
     that is evidence the incoming 2,000 field images will help.

Both runs use the IDENTICAL kenya_test set, or the comparison is meaningless.

Usage:
  python train_with_kenya.py --model efficientvit_b0
  python train_with_kenya.py --model efficientvit_b0 --kenya-frac 0.5 --epochs 25

--kenya-frac = fraction of Kenya SPECIMENS put into training (rest = test).
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

CLASSES = ["aedes_aegypti", "anopheles_gambiae", "anopheles_stephensi"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}


def make_transforms(img):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    resize = int(img * 1.14)
    train_tf = T.Compose([
        T.Resize(resize),
        T.RandomResizedCrop(img, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(25),
        T.ColorJitter(0.2, 0.2, 0.05, 0.0),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.Resize(resize), T.CenterCrop(img),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    return train_tf, eval_tf


class BugDS(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True); self.tf = tf
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(r["path"]).convert("RGB")), r["y"]


def load_master(master_csv, images_dir):
    m = pd.read_csv(master_csv)
    m = m[m["cropped_path"].notna()].reset_index(drop=True)
    def resolve(r):
        p = r["cropped_path"]
        if isinstance(p, str) and os.path.exists(p): return p
        return os.path.join(images_dir, f"image_{int(r['ImageID'])}.jpg")
    m["path"] = m.apply(resolve, axis=1)
    m = m[m["path"].map(os.path.exists)].reset_index(drop=True)
    m["y"] = m["species"].map(CLS2IDX)
    return m


def split_kenya(kenya, frac, seed=42):
    """Split Kenya SPECIMENS into train/test, grouped + stratified, no leakage."""
    spec = kenya.drop_duplicates("SpecimenID")[["SpecimenID", "y"]]
    # StratifiedGroupKFold with n_splits chosen so test ~= (1-frac)
    n_splits = max(2, round(1 / (1 - frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, te_idx = next(sgkf.split(spec, spec["y"], groups=spec["SpecimenID"]))
    train_specs = set(spec.iloc[tr_idx]["SpecimenID"])
    test_specs = set(spec.iloc[te_idx]["SpecimenID"])
    assert not (train_specs & test_specs), "kenya leakage!"
    k_train = kenya[kenya.SpecimenID.isin(train_specs)].reset_index(drop=True)
    k_test = kenya[kenya.SpecimenID.isin(test_specs)].reset_index(drop=True)
    return k_train, k_test


def build_insectary_trainval(m, seed=42):
    pool = m[m["drop"].isin(["0610", "0618"])].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    spec = pool.drop_duplicates("SpecimenID")
    tr, va = next(sgkf.split(spec, spec["y"], groups=spec["SpecimenID"]))
    val_specs = set(spec.iloc[va]["SpecimenID"])
    pool["split"] = pool["SpecimenID"].apply(lambda s: "val" if s in val_specs else "train")
    return (pool[pool.split == "train"].reset_index(drop=True),
            pool[pool.split == "val"].reset_index(drop=True))


def train_model(train_df, val_df, model_name, img, batch, epochs, lr, wd, workers, device):
    train_tf, eval_tf = make_transforms(img)
    dl = lambda d, tf, sh: DataLoader(BugDS(d, tf), batch_size=batch, shuffle=sh,
                                      num_workers=workers, pin_memory=True)
    dl_train, dl_val = dl(train_df, train_tf, True), dl(val_df, eval_tf, False)

    model = timm.create_model(model_name, pretrained=True, num_classes=3).to(device)
    counts = train_df["y"].value_counts().sort_index().reindex(range(3), fill_value=1).values
    w = np.sqrt(counts.sum() / counts); w = w / w.mean()
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    def ev(loader):
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for x, y in loader:
                with torch.amp.autocast("cuda"): out = model(x.to(device))
                ps += out.argmax(1).cpu().tolist(); ys += y.tolist()
        return np.array(ys), np.array(ps)

    best_f1, best_state, bad = 0, None, 0
    for ep in range(epochs):
        model.train()
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"): loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        yv, pv = ev(dl_val); f1 = f1_score(yv, pv, average="macro")
        print(f"  ep{ep+1:2d} val_macroF1={f1:.4f}")
        if f1 > best_f1: best_f1, best_state, bad = f1, deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= 6: print("  early stop"); break
    model.load_state_dict(best_state)
    return model, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientvit_b0")
    ap.add_argument("--master", default="master_cropped.csv")
    ap.add_argument("--images", default="data_cropped")
    ap.add_argument("--kenya-frac", type=float, default=0.5,
                    help="fraction of Kenya specimens used for TRAINING")
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs", help="dir to save checkpoints + splits")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| model:", args.model, "| kenya_frac:", args.kenya_frac)

    m = load_master(args.master, args.images)
    ins_train, ins_val = build_insectary_trainval(m, args.seed)
    kenya = m[m["source"] == "field"].reset_index(drop=True)
    k_train, k_test = split_kenya(kenya, args.kenya_frac, args.seed)

    print(f"\ninsectary train: {len(ins_train)} imgs / {ins_train.SpecimenID.nunique()} spec")
    print(f"kenya_train: {len(k_train)} imgs / {k_train.SpecimenID.nunique()} spec")
    print(f"kenya_test : {len(k_test)} imgs / {k_test.SpecimenID.nunique()} spec")
    print("kenya_test species:", dict(k_test['species'].value_counts()))
    # leak checks
    assert not (set(k_train.SpecimenID) & set(k_test.SpecimenID)), "KENYA LEAK"
    print("no kenya specimen leakage  ✓\n")

    def report(model, ev, name):
        y, p = ev(DataLoader(BugDS(k_test, make_transforms(args.img)[1]),
                             batch_size=args.batch, num_workers=args.workers))
        f1 = f1_score(y, p, average="macro")
        print(f"\n### {name}  ->  kenya_test macro-F1 = {f1:.4f}")
        print(classification_report(y, p, target_names=CLASSES, digits=3, zero_division=0))
        return f1

    # ---- BASELINE: insectary only ----
    print("=" * 55)
    print("BASELINE: train on INSECTARY ONLY")
    print("=" * 55)
    m1, ev1 = train_model(ins_train, ins_val, args.model, args.img, args.batch,
                          args.epochs, args.lr, args.wd, args.workers, device)
    f1_base = report(m1, ev1, "BASELINE (insectary only)")

    # ---- AUGMENTED: insectary + kenya_train ----
    print("\n" + "=" * 55)
    print("AUGMENTED: train on INSECTARY + KENYA_TRAIN")
    print("=" * 55)
    aug_train = pd.concat([ins_train, k_train], ignore_index=True)
    m2, ev2 = train_model(aug_train, ins_val, args.model, args.img, args.batch,
                          args.epochs, args.lr, args.wd, args.workers, device)
    f1_aug = report(m2, ev2, "AUGMENTED (insectary + kenya_train)")

    print("\n" + "=" * 55)
    print("RESULT")
    print("=" * 55)
    print(f"kenya_test macro-F1  baseline (insectary only)      : {f1_base:.4f}")
    print(f"kenya_test macro-F1  augmented (+ kenya_train)       : {f1_aug:.4f}")
    print(f"lift from adding {k_train.SpecimenID.nunique()} kenya specimens : {f1_aug - f1_base:+.4f}")
    print("\nIf lift is positive, field data helps -> supports the 2,000-image ask.")

    # ---- SAVE both checkpoints + the exact kenya_test split (for Grad-CAM / reuse) ----
    os.makedirs(args.out, exist_ok=True)
    meta = {"model": args.model, "img": args.img, "classes": CLASSES}
    torch.save({**meta, "state_dict": m1.state_dict(),
                "trained_on": "insectary_only", "kenya_test_f1": float(f1_base)},
               os.path.join(args.out, f"{args.model}_baseline_insectary.pt"))
    torch.save({**meta, "state_dict": m2.state_dict(),
                "trained_on": "insectary+kenya_train", "kenya_test_f1": float(f1_aug)},
               os.path.join(args.out, f"{args.model}_augmented_kenya.pt"))
    # save which ImageIDs were in kenya_test so Grad-CAM uses the held-out set consistently
    k_test[["ImageID", "species", "SpecimenID"]].to_csv(
        os.path.join(args.out, "kenya_test_split.csv"), index=False)
    k_train[["ImageID", "species", "SpecimenID"]].to_csv(
        os.path.join(args.out, "kenya_train_split.csv"), index=False)
    print(f"\nsaved checkpoints + splits to {args.out}/:")
    print(f"  {args.model}_baseline_insectary.pt   (insectary only)")
    print(f"  {args.model}_augmented_kenya.pt       (insectary + kenya_train)")
    print(f"  kenya_test_split.csv / kenya_train_split.csv")


if __name__ == "__main__":
    main()
