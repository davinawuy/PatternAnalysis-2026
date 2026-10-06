# Data audit for 2D segmentation datasets such as the OASIS brain slices.
#
# Makes no assumptions about the layout and reports what is actually on disk:
#   * folder tree with file counts, extensions and sample filenames
#   * for every leaf folder: image shapes, dtypes, intensity range and, for mask
#     folders, the full set of unique label values
#   * slice and patient counts per folder (patient ID taken from the filename by regex)
#   * image/mask pairing per split (every image has a mask with a matching filename)
#   * the patient overlap check between splits
#
# A folder counts as a mask folder if its name contains 'seg', 'mask' or 'label',
# and its split is the first of train/val/validate/test that appears in its name.
# Check both rules against the printed tree before trusting the counts.
#
# The output informs the loader in dataset.py and the feasibility review.
# PNG/JPEG files are read with Pillow. Nifti files (.nii, .nii.gz) need nibabel,
# following the loading approach in Appendix B of the COMP3710 assignment spec.
#
# Usage:
#     python data_audit.py --root /home/groups/comp3710/OASIS

import argparse
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".nii", ".nii.gz")
SPLITS = ("train", "validate", "val", "test")  # 'validate' before 'val' so the longer name wins
MASK_HINTS = ("seg", "mask", "label")


def is_image(path):
    return path.name.lower().endswith(IMAGE_EXTS)


def load_array(path):
    """Read a 2D/3D image file into a NumPy array with its stored dtype."""
    if path.name.lower().endswith((".nii", ".nii.gz")):
        import nibabel as nib  # only needed for Nifti, so imported lazily
        return np.asarray(nib.load(str(path)).dataobj)
    with Image.open(path) as img:
        return np.array(img)


def pairing_key(name, pattern):
    """Filename from the patient ID onward, so 'case_001_slice_0.png' pairs with 'seg_001_slice_0.png'."""
    match = pattern.search(name)
    return name[match.start():].lower() if match else name.lower()


def print_tree(root, max_depth, n_samples):
    print(f"=== Folder tree under {root} (depth <= {max_depth}) ===")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        exts = Counter("".join(Path(f).suffixes[-2:]).lower() for f in filenames)
        print(f"{'  ' * depth}{rel.as_posix() or '.'}/  dirs={len(dirnames)} files={len(filenames)} exts={dict(exts)}")
        if filenames:
            print(f"{'  ' * depth}  e.g. {sorted(filenames)[:n_samples]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="dataset root directory")
    parser.add_argument("--patient-regex", default=r"(\d+)",
                        help="regex whose first group is the patient ID (default: first run of digits). "
                             "Verify it against the sample filenames in the tree output.")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--tree-samples", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=0,
                        help="read at most this many files per folder (0 = all; OASIS PNGs are small)")
    args = parser.parse_args()

    root = args.root.resolve()
    print_tree(root, args.max_depth, args.tree_samples)

    files_by_dir = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.is_file() and is_image(path):
            files_by_dir[path.parent].append(path)

    pattern = re.compile(args.patient_regex)
    patients_by_split = defaultdict(set)
    names_by_split = defaultdict(lambda: {"image": set(), "mask": set()})

    print("\n=== Per-folder contents ===")
    for folder, paths in sorted(files_by_dir.items()):
        name = folder.name.lower()
        split = next((s for s in SPLITS if s in name), "?")
        kind = "mask" if any(h in name for h in MASK_HINTS) else "image"
        print(f"\n[{folder.relative_to(root).as_posix()}] split={split} kind={kind} files={len(paths)}")

        shapes, dtypes, values = Counter(), Counter(), set()
        lo, hi = float("inf"), float("-inf")
        for path in paths[:args.max_files or None]:
            arr = load_array(path)
            shapes[arr.shape] += 1
            dtypes[str(arr.dtype)] += 1
            lo, hi = min(lo, arr.min()), max(hi, arr.max())
            # Mask label values must be known exactly to build the value -> class index map.
            if kind == "mask":
                values.update(np.unique(arr).tolist())
        print(f"  shapes={dict(shapes)}  dtypes={dict(dtypes)}  min={lo} max={hi}")
        if len(shapes) > 1:
            print("  WARNING: shapes vary inside this folder")
        if kind == "mask":
            print(f"  unique mask values ({len(values)}): {sorted(values)[:50]}")

        slices_per_patient = Counter()
        unmatched = []
        for path in paths:
            match = pattern.search(path.name)
            if match:
                slices_per_patient[match.group(1)] += 1
            else:
                unmatched.append(path.name)
        counts = list(slices_per_patient.values()) or [0]
        print(f"  patients={len(slices_per_patient)} slices/patient min={min(counts)} "
              f"median={statistics.median(counts)} max={max(counts)} e.g. {sorted(slices_per_patient)[:3]}")
        if unmatched:
            print(f"  WARNING: {len(unmatched)} filenames did not match the patient regex, e.g. {unmatched[:3]}")

        patients_by_split[split].update(slices_per_patient)
        names_by_split[split][kind].update(pairing_key(p.name, pattern) for p in paths)

    print("\n=== Image/mask pairing per split (filename from the patient ID onward) ===")
    for split, names in sorted(names_by_split.items()):
        images, masks = names["image"], names["mask"]
        print(f"{split}: images={len(images)} masks={len(masks)} paired={len(images & masks)} "
              f"image-only={sorted(images - masks)[:3]} mask-only={sorted(masks - images)[:3]}")

    print("\n=== Patient overlap between splits ===")
    split_names = sorted(patients_by_split)
    leak = False
    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            shared = patients_by_split[a] & patients_by_split[b]
            leak |= bool(shared)
            print(f"{a} & {b}: {len(shared)} shared patients {sorted(shared)[:10]}")
    print(f"OVERLAP CHECK: {'FAIL' if leak else 'PASS'}")


if __name__ == "__main__":
    main()
