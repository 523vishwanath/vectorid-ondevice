#!/usr/bin/env python3
"""
VectorCam — train.py  (VSCode / RunPod version of the Colab notebook)
=====================================================================
Trains a lightweight species classifier and evaluates on TWO held-out sets:
  - Test A: 0623 (unseen phone models)
  - Test B: kenya_01 (field) -> the go/no-go number

Evaluation design (unchanged from Colab):
  train+val = 0610 + 0618, split by StratifiedGroupKFold on SpecimenID
  (no specimen in both train and val -> no shortcut learning)

Usage
-----
  python train.py --model efficientnet_b0
  python train.py --model mobilenetv3_small_100 --epochs 25
  python train.py --model efficientvit_b0 --img 256 --batch 24

Expects on disk (defaults assume current folder):
  --master  master_cropped.csv    (has cropped_path, species, drop, source, SpecimenID)
  --images  data_cropped/         (cropped jpgs; only needed if paths are relative)

Comet: set COMET_API_KEY env var (or pass --no-comet to skip).
"""

import argparse
import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score,
)
import matplotlib
matplotlib.use("Agg")  # no display on a remote pod
import matplotlib.pyplot as plt

CLASSES = ["aedes_aegypti", "anopheles_gambiae", "anopheles_stephensi"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}


# ----------------------------------------------------------------------
def build_splits(master_csv, images_dir):
    m = pd.read_csv(master_csv)
    m = m[m["cropped_path"].notna()].reset_index(drop=True)

    # Resolve image paths: if cropped_path is absolute and exists use it,
    # else rebuild from images_dir + image_<ImageID>.jpg
    def resolve(row):
        p = row["cropped_path"]
        if isinstance(p, str) and os.path.exists(p):
            return p
        cand = os.path.join(images_dir, f"image_{int(row['ImageID'])}.jpg")
        return cand
    m["path"] = m.apply(resolve, axis=1)
    missing = (~m["path"].map(os.path.exists)).sum()
    if missing:
        print(f"[warn] {missing} images not found on disk; dropping them")
        m = m[m["path"].map(os.path.exists)].reset_index(drop=True)

    m["y"] = m["species"].map(CLS2IDX)

    pool = m[m["drop"].isin(["0610", "0618"])].reset_index(drop=True)
    test_dev = m[m["drop"] == "0623"].reset_index(drop=True)
    test_field = m[m["source"] == "field"].reset_index(drop=True)

    # grouped, stratified train/val split on SpecimenID
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    spec = pool.drop_duplicates("SpecimenID")
    tr_idx, va_idx = next(sgkf.split(spec, spec["y"], groups=spec["SpecimenID"]))
    val_specimens = set(spec.iloc[va_idx]["SpecimenID"])
    pool["split"] = pool["SpecimenID"].apply(lambda s: "val" if s in val_specimens else "train")

    train_df = pool[pool.split == "train"].reset_index(drop=True)
    val_df = pool[pool.split == "val"].reset_index(drop=True)

    assert len(set(train_df.SpecimenID) & set(val_df.SpecimenID)) == 0, "LEAK!"
    print(f"pool: {len(pool)} imgs / {pool.SpecimenID.nunique()} specimens")
    print(f"train: {len(train_df)} / {train_df.SpecimenID.nunique()} spec")
    print(f"val:   {len(val_df)} / {val_df.SpecimenID.nunique()} spec")
    print(f"test_dev (0623): {len(test_dev)} / {test_dev.SpecimenID.nunique()}")
    print(f"test_field (kenya): {len(test_field)} / {test_field.SpecimenID.nunique()}")
    print("NO specimen leakage between train and val  ✓")
    return train_df, val_df, test_dev, test_field


# ----------------------------------------------------------------------
class BugDS(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True)
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tf(img), r["y"]


def make_transforms(img):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    resize = int(img * 1.14)
    train_tf = T.Compose([
        T.Resize(resize),
        T.RandomResizedCrop(img, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(25),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05, hue=0.0),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.Resize(resize), T.CenterCrop(img),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    return train_tf, eval_tf


# ----------------------------------------------------------------------
def full_metrics(ys, ps, prefix=""):
    d = {f"{prefix}acc": accuracy_score(ys, ps),
         f"{prefix}macro_f1": f1_score(ys, ps, average="macro"),
         f"{prefix}weighted_f1": f1_score(ys, ps, average="weighted")}
    p, r, _, _ = precision_recall_fscore_support(ys, ps, average="macro", zero_division=0)
    d[f"{prefix}macro_precision"] = p
    d[f"{prefix}macro_recall"] = r
    pc, rc, fc, _ = precision_recall_fscore_support(
        ys, ps, labels=range(len(CLASSES)), zero_division=0)
    for i, c in enumerate(CLASSES):
        d[f"{prefix}{c}_f1"] = fc[i]
        d[f"{prefix}{c}_recall"] = rc[i]
        d[f"{prefix}{c}_precision"] = pc[i]
    return d


def evaluate(model, loader, crit, device, compute_loss=False):
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); yt = y.to(device)
            with torch.amp.autocast("cuda"):
                out = model(x)
                if compute_loss:
                    loss_sum += crit(out, yt).item() * len(x); n += len(x)
            ps += out.argmax(1).cpu().tolist(); ys += y.tolist()
    ys, ps = np.array(ys), np.array(ps)
    return ys, ps, (loss_sum / n if compute_loss else None)


def save_cm(y, p, name, out_png):
    cm = confusion_matrix(y, p, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(4.5, 4)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(CLASSES, fontsize=7)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", fontsize=9)
    ax.set_title(name); ax.set_ylabel("true"); ax.set_xlabel("pred")
    plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close()


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--master", default="master_cropped.csv")
    ap.add_argument("--images", default="data_cropped")
    ap.add_argument("--img", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--no-comet", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, "| timm", timm.__version__, "| model", args.model)

    # ---- data ----
    train_df, val_df, test_dev, test_field = build_splits(args.master, args.images)
    train_tf, eval_tf = make_transforms(args.img)
    dl = lambda d, tf, sh: DataLoader(
        BugDS(d, tf), batch_size=args.batch, shuffle=sh,
        num_workers=args.workers, pin_memory=True)
    dl_train = dl(train_df, train_tf, True)
    dl_val = dl(val_df, eval_tf, False)
    dl_dev = dl(test_dev, eval_tf, False)
    dl_field = dl(test_field, eval_tf, False)

    # ---- model ----
    model = timm.create_model(args.model, pretrained=True, num_classes=len(CLASSES)).to(device)
    counts = train_df["y"].value_counts().sort_index().values
    w = np.sqrt(counts.sum() / counts); w = w / w.mean()
    class_w = torch.tensor(w, dtype=torch.float32, device=device)
    print("class weights:", dict(zip(CLASSES, w.round(3))))
    crit = nn.CrossEntropyLoss(weight=class_w)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    # ---- comet (optional) ----
    exp = None
    if not args.no_comet and os.environ.get("COMET_API_KEY"):
        import comet_ml
        exp = comet_ml.Experiment(project_name="vectorcam-classifier",
                                  auto_metric_logging=False)
        exp.set_name(args.model)
        exp.log_parameters(vars(args))

    # ---- train ----
    best_f1, best_state, bad = 0, None, 0
    for ep in range(args.epochs):
        model.train(); tot = 0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(x)
        sched.step()
        tr_loss = tot / len(train_df)
        yv, pv, vloss = evaluate(model, dl_val, crit, device, compute_loss=True)
        mv = full_metrics(yv, pv, "val_")
        if exp:
            exp.log_metrics({**mv, "train_loss": tr_loss, "val_loss": vloss,
                             "lr": opt.param_groups[0]["lr"]}, epoch=ep + 1)
        print(f"ep{ep+1:2d} tr_loss={tr_loss:.3f} val_loss={vloss:.3f} "
              f"val_acc={mv['val_acc']:.3f} val_macroF1={mv['val_macro_f1']:.4f}")
        if mv["val_macro_f1"] > best_f1:
            best_f1, best_state, bad = mv["val_macro_f1"], deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break
    model.load_state_dict(best_state)
    print(f"\nBest val macro-F1: {best_f1:.4f}")

    # ---- size / latency ----
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = n_params * 4 / 1e6
    model.eval(); dummy = torch.randn(1, 3, args.img, args.img).to(device)
    for _ in range(5):
        with torch.no_grad(), torch.amp.autocast("cuda"): model(dummy)
    t = time.time()
    for _ in range(50):
        with torch.no_grad(), torch.amp.autocast("cuda"): model(dummy)
    lat_ms = (time.time() - t) / 50 * 1000
    print(f"{args.model}: {n_params/1e6:.2f}M params, ~{size_mb:.1f}MB, {lat_ms:.1f}ms/img (GPU)")
    if exp:
        exp.log_metrics({"params_M": n_params/1e6, "size_MB": size_mb, "latency_ms_gpu": lat_ms})

    # ---- evaluation on all three sets ----
    def report(name, loader, prefix):
        y, p, _ = evaluate(model, loader, crit, device)
        f1 = f1_score(y, p, average="macro")
        print(f"\n===== {name} =====")
        print(f"macro-F1: {f1:.4f} | acc: {(y==p).mean():.4f}")
        print(classification_report(y, p, target_names=CLASSES, digits=3, zero_division=0))
        if exp:
            exp.log_metrics(full_metrics(y, p, prefix))
        save_cm(y, p, f"{name} (mF1={f1:.3f})",
                os.path.join(args.out, f"cm_{args.model}_{prefix.strip('_')}.png"))
        return f1, y, p

    f1_val, _, _ = report("VAL (insectary, grouped)", dl_val, "val_")
    f1_dev, _, _ = report("TEST A — 0623 unseen DEVICE", dl_dev, "dev_")
    f1_field, yf, pf = report("TEST B — KENYA field (HEADLINE)", dl_field, "kenya_")

    print("\n" + "=" * 50)
    print(f"Val (insectary):      {f1_val:.3f}")
    print(f"Test A (new device):  {f1_dev:.3f}")
    print(f"Test B (KENYA field): {f1_field:.3f}   <-- go/no-go number")

    # ---- field error slices ----
    tf = test_field.copy(); tf["pred"] = pf; tf["correct"] = (tf["y"].values == pf)
    print("\nKenya accuracy by specimen condition:")
    print(tf.groupby("SessionSpecimenCondition")["correct"].agg(["mean", "size"]))
    print("\nKenya accuracy by device:")
    print(tf.groupby("DeviceModel")["correct"].agg(["mean", "size"]))
    print("\nKenya accuracy by TRUE species:")
    print(tf.groupby("species")["correct"].agg(["mean", "size"]))

    # ---- save model ----
    ckpt = os.path.join(args.out, f"{args.model}_vectorcam.pt")
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES,
                "model": args.model, "img": args.img}, ckpt)
    print(f"\nsaved {ckpt}")
    if exp:
        exp.log_model("model", ckpt); exp.end()


if __name__ == "__main__":
    main()
