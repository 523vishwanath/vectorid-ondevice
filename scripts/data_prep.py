#!/usr/bin/env python3
"""
VectorCam — data_prep.py
========================
Builds ONE clean master table from the four raw drops.

What it does
------------
1. Reads the four drop CSVs.
2. Tags every row with `drop` (0610/0618/0623/kenya) and `source`
   (insectary vs field) and `country`.
3. Finds each row's actual image file on disk (handles the split
   sub-folders like "0618 2", "0618 3", ... automatically).
4. Inner-joins CSV rows to files: any row without a real file, and any
   file without a label row, is dropped. This removes the 135 known
   unusable items (104 unlabeled 0623, 8 unlabeled kenya, 23 kenya rows
   with a blank filename) with no manual work.
5. Normalises the species label to a clean 3-class target.
6. Writes master.csv + prints a full inventory so you can eyeball it.

Runs the same in a Colab cell or as a VSCode module:
    python data_prep.py --root /path/to/VectorCam --out master.csv
In Colab after unzipping into /content/data:
    !python data_prep.py --root /content/data --out /content/master.csv
"""

import argparse
import glob
import os
import re
import sys
import pandas as pd

# ----------------------------------------------------------------------
# Config: how each drop maps to its CSV and its image folder(s).
# The image folders include the Google-Drive split parts. Missing folders
# are skipped silently, so this works whether or not you've merged them.
# ----------------------------------------------------------------------
DROPS = {
    "0610": {
        "csv": ["0610/0610_specimens_modeling_master.csv"],
        "img_dirs": ["0610/downloaded_images", "0610 2/downloaded_images"],
        "source": "insectary",
    },
    "0618": {
        "csv": ["0618/0618_specimens_modeling_master.csv"],
        "img_dirs": [
            "0618/downloaded_images", "0618 2/downloaded_images",
            "0618 3/downloaded_images", "0618 4/downloaded_images",
        ],
        "source": "insectary",
    },
    "0623": {
        "csv": ["0623/0623_specimens_modeling_master.csv"],
        "img_dirs": ["0623/downloaded_images"],
        "source": "insectary",
    },
    "kenya": {
        "csv": ["kenya_01/kenya_specimens_modeling_master.csv"],
        "img_dirs": ["kenya_01/downloaded_images", "kenya_01 2/downloaded_images"],
        "source": "field",
    },
}

# Map the three target classes to clean canonical names.
SPECIES_CANON = {
    "anopheles_gambiae": "anopheles_gambiae",
    "anopheles_stephensi": "anopheles_stephensi",
    "aedes_aegypti": "aedes_aegypti",
}

IMG_RE = re.compile(r"image_(\d+)_[0-9a-f]+\.jpg$", re.IGNORECASE)


def _first_existing(root, candidates):
    """Return the first path in `candidates` that exists under root, else None."""
    for c in candidates:
        p = os.path.join(root, c)
        if os.path.exists(p):
            return p
    return None


def build_disk_index(root, img_dirs):
    """Map ImageID -> absolute file path for every image found on disk."""
    index = {}
    for d in img_dirs:
        folder = os.path.join(root, d)
        if not os.path.isdir(folder):
            continue
        for p in glob.glob(os.path.join(folder, "*.jpg")):
            m = IMG_RE.search(os.path.basename(p))
            if m:
                index[int(m.group(1))] = os.path.abspath(p)
    return index


def load_drop(root, drop, cfg):
    """Load one drop's CSV, attach metadata, resolve image paths, inner-join."""
    csv_path = _first_existing(root, cfg["csv"])
    if csv_path is None:
        print(f"  [!] {drop}: CSV not found (looked for {cfg['csv']}) — skipping")
        return None

    df = pd.read_csv(csv_path)
    n_raw = len(df)

    # Tag metadata
    df["drop"] = drop
    df["source"] = cfg["source"]              # insectary | field
    df["ImageID"] = df["ImageID"].astype(int)

    # Resolve each row to a file on disk
    disk = build_disk_index(root, cfg["img_dirs"])
    df["image_path"] = df["ImageID"].map(disk)

    # Count what we're about to drop, and why, for the report
    no_fname = df["DownloadedFilename"].isna().sum()
    no_file = df["image_path"].isna().sum()
    extra_on_disk = len(set(disk) - set(df["ImageID"]))

    # INNER JOIN: keep only rows that have a real file on disk
    df = df[df["image_path"].notna()].copy()

    # Normalise species; drop anything not one of the 3 targets (safety)
    df["species"] = df["SpeciesLabel"].str.strip().str.lower().map(SPECIES_CANON)
    bad_species = df["species"].isna().sum()
    df = df[df["species"].notna()].copy()

    print(f"  {drop:6s} | csv_rows={n_raw:5d} | blank_filename={no_fname:3d} "
          f"| rows_without_file={no_file:3d} | unlabeled_files_on_disk={extra_on_disk:3d} "
          f"| bad_species={bad_species:2d} | KEPT={len(df):5d}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="VectorCam folder (contains 0610, 0618, ...)")
    ap.add_argument("--out", default="master.csv", help="output CSV path")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    print(f"Root: {root}\n")
    print("Per-drop load (inner-join CSV<->disk):")

    frames = []
    for drop, cfg in DROPS.items():
        f = load_drop(root, drop, cfg)
        if f is not None:
            frames.append(f)

    if not frames:
        print("\nNo data loaded. Check --root points at the VectorCam folder.")
        sys.exit(1)

    master = pd.concat(frames, ignore_index=True)

    # Keep a tidy column set (plus a few useful raw fields for analysis).
    keep = [
        "drop", "source", "ProgramCountry",
        "SpecimenID", "ImageID", "species",
        "DeviceModel", "PhoneModel", "SessionID",
        "SessionSpecimenCondition", "SessionCollectorName",
        "image_path",
    ]
    keep = [c for c in keep if c in master.columns]
    master = master[keep].rename(columns={"ProgramCountry": "country"})

    # ---- Inventory report ----
    print("\n================ MASTER INVENTORY ================")
    print(f"Total usable images : {len(master)}")
    print(f"Unique specimens    : {master['SpecimenID'].nunique()}")
    print("\nBy drop:")
    print(master.groupby("drop").agg(
        images=("ImageID", "size"),
        specimens=("SpecimenID", "nunique"),
    ))
    print("\nBy source:")
    print(master.groupby("source").agg(
        images=("ImageID", "size"),
        specimens=("SpecimenID", "nunique"),
    ))
    print("\nSpecies (all):")
    print(master["species"].value_counts())
    print("\nSpecies x source (images):")
    print(pd.crosstab(master["source"], master["species"]))
    print("\nImages per specimen (describe):")
    print(master.groupby("SpecimenID").size().describe())

    # Safety: no specimen should span multiple species
    span = master.groupby("SpecimenID")["species"].nunique()
    print(f"\nSpecimens spanning >1 species (must be 0): {(span > 1).sum()}")

    master.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}  ({len(master)} rows)")


if __name__ == "__main__":
    main()
