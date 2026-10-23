# Data audit for segmentation datasets such as the OASIS brain slices and the
# HipMRI pelvic slices and volumes.
#
# Makes no assumptions about the layout and reports what is actually on disk:
#   * folder tree with file counts, extensions and sample filenames
#   * filename patterns (digit runs replaced by '#'), to see how patient, scan
#     and slice numbers are encoded before trusting any regex
#   * for every leaf folder: image shapes, dtypes, intensity range and, for mask
#     folders, every label value with its share of all pixels (class imbalance)
#   * for Nifti files: voxel spacing and orientation (axis codes such as RAS),
#     which decide how 3D volumes can be resampled and which axis is which
#   * slice and patient counts per folder (patient ID taken from the filename by
#     regex), and scans per patient when --scan-regex is given
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
#     python data_audit.py --root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
#         --scan-regex "(\d+_week_\d+)"
#     python data_audit.py --root /home/groups/comp3710/HipMRI_Study_open --skip keras_slices_data

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


def nifti_geometry(path):
    """(voxel spacing rounded to 0.01 mm, orientation axis codes) from a Nifti header, without reading the voxels."""
    import nibabel as nib
    image = nib.load(str(path))
    return tuple(round(float(z), 2) for z in image.header.get_zooms()), "".join(nib.aff2axcodes(image.affine))


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
    parser = argparse.ArgumentParser(description="Audit a segmentation dataset on disk.")
    parser.add_argument("--root", type=Path, required=True, help="dataset root directory")
    parser.add_argument("--patient-regex", default=r"(\d+)",
                        help="regex whose first group is the patient ID (default: first run of digits). "
                             "Verify it against the sample filenames in the tree output.")
    parser.add_argument("--scan-regex",
                        help="regex whose first group identifies one scan, e.g. patient and week, to count "
                             "scans per patient. Verify it against the filename patterns first.")
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--tree-samples", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=0,
                        help="read at most this many files per folder (0 = all; OASIS PNGs are small)")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="folder names to leave out of the per-folder audit (still shown in the tree)")
    args = parser.parse_args()

    root = args.root.resolve()
    print_tree(root, args.max_depth, args.tree_samples)

    files_by_dir = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if path.is_file() and is_image(path) and not set(args.skip) & set(path.relative_to(root).parts):
            files_by_dir[path.parent].append(path)

    pattern = re.compile(args.patient_regex)
    scan_pattern = re.compile(args.scan_regex) if args.scan_regex else None
    patients_by_split = defaultdict(set)
    names_by_split = defaultdict(lambda: {"image": set(), "mask": set()})

    print("\n=== Per-folder contents ===")
    for folder, paths in sorted(files_by_dir.items()):
        name = folder.name.lower()
        split = next((s for s in SPLITS if s in name), "?")
        kind = "mask" if any(h in name for h in MASK_HINTS) else "image"
        print(f"\n[{folder.relative_to(root).as_posix()}] split={split} kind={kind} files={len(paths)}")

        templates = Counter(re.sub(r"\d+", "#", p.name) for p in paths)
        print(f"  filename patterns: {dict(templates.most_common(5))}")

        shapes, dtypes, values, geometry = Counter(), Counter(), Counter(), Counter()
        lo, hi = float("inf"), float("-inf")
        for path in paths[:args.max_files or None]:
            arr = load_array(path)
            if path.name.lower().endswith((".nii", ".nii.gz")):
                geometry[nifti_geometry(path)] += 1
            shapes[arr.shape] += 1
            dtypes[str(arr.dtype)] += 1
            lo, hi = min(lo, arr.min()), max(hi, arr.max())
            # Mask label values must be known exactly to build the value -> class index map.
            if kind == "mask":
                labels, counts = np.unique(arr, return_counts=True)
                values.update(dict(zip(labels.tolist(), counts.tolist())))
        print(f"  shapes ({len(shapes)} distinct, most common first)={dict(shapes.most_common(8))}")
        print(f"  dtypes={dict(dtypes)}  min={lo} max={hi}")
        if geometry:
            print(f"  (spacing mm, orientation) ({len(geometry)} distinct)={dict(geometry.most_common(8))}")
        if len(shapes) > 1:
            print("  WARNING: shapes vary inside this folder")
        if kind == "mask":
            total = sum(values.values())
            shares = {v: f"{n / total:.4%}" for v, n in sorted(values.items())[:50]}
            print(f"  unique mask values ({len(values)}) with pixel share: {shares}")

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
        if scan_pattern:
            scans = {m.group(1) for p in paths if (m := scan_pattern.search(p.name))}
            scans_per_patient = Counter(pattern.search(s).group(1) for s in scans if pattern.search(s))
            per = list(scans_per_patient.values()) or [0]
            print(f"  scans={len(scans)} scans/patient min={min(per)} median={statistics.median(per)} "
                  f"max={max(per)} e.g. {sorted(scans)[:3]}")

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
