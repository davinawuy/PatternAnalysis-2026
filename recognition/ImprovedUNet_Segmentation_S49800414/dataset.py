# Data loading for 2D segmentation of OASIS brain slices and HipMRI pelvic slices.
#
# OASIS: train / validate / test folders of 256x256 8-bit PNG slices, already
# split by patient, with masks storing four tissue classes as the grey levels
# 0, 85, 170 and 255. Images are scaled to [0, 1].
#
# HipMRI: the same three-way patient-level split of Nifti slices
# (case_PPP_week_W_slice_S.nii.gz), 60 slices per weekly scan, with masks
# labelled 0-5. Every slice is 256x128 except one training scan at 256x144, which
# is centre-cropped to 256x128. MRI intensities have no fixed scale between
# scans, so each slice is standardised to zero mean and unit variance.
#
# Layouts, shapes and label values were confirmed by data_audit.py before this
# loader was written. Masks are returned as class indices (H, W), which is what
# cross-entropy expects; one_hot() gives the (C, H, W) encoding for Dice.
#
# Based on the data pipeline guidelines in the COMP3710 assignment spec
# (Section 2.2: patient-level splitting, intensity normalisation, one-hot
# masks). The Nifti loading is adapted from load_data_2D in Appendix B of the
# spec, reading one file at a time instead of pre-allocating arrays sized from
# the first file, which would break on the 256x144 scan.
#
# Self-check (run on a compute node):
#     python dataset.py --dataset oasis
#     python dataset.py --dataset hipmri

import itertools
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SPLITS = ("train", "validate", "test")
HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------- OASIS

ROOT = Path(os.environ.get("DATA_ROOT", "/home/groups/comp3710/OASIS"))
MASK_VALUES = (0, 85, 170, 255)
CLASS_NAMES = ("Background", "CSF", "Grey matter", "White matter")
NUM_CLASSES = len(CLASS_NAMES)

# Grey level -> class index. Every other value maps to 255 so a stray value is
# caught instead of silently becoming a valid class.
_LOOKUP = np.full(256, 255, dtype=np.uint8)
_LOOKUP[list(MASK_VALUES)] = np.arange(NUM_CLASSES)


def load_image(path):
    """8-bit greyscale PNG -> float32 tensor (1, H, W) in [0, 1]."""
    array = np.asarray(Image.open(path), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def load_mask(path):
    """Mask PNG -> int64 tensor (H, W) of class indices 0..3."""
    labels = _LOOKUP[np.asarray(Image.open(path))]
    if (labels == 255).any():
        raise ValueError(f"Unexpected mask values in {path}")
    return torch.from_numpy(labels.astype(np.int64))


def slice_key(path):
    """'case_001_slice_5.nii.png' or 'seg_001_slice_5.nii.png' -> (1, 5)."""
    match = re.fullmatch(r"(?:case|seg)_(\d+)_slice_(\d+)\.nii\.png", path.name)
    if match is None:
        raise ValueError(f"Unexpected filename: {path.name}")
    return int(match[1]), int(match[2])


# ------------------------------------------------------------------ HipMRI

HIPMRI_ROOT = Path(os.environ.get("HIPMRI_ROOT", "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"))
# Label order follows the spec's description of the dataset ("body outline,
# bone, bladder, rectum, and prostate"). The overlay saved by
# python dataset.py --dataset hipmri agrees: 5 is a round gland at the centre of
# the pelvis, 3 the larger fluid-filled organ against it (1.95% of pixels against
# 0.43% for 5) and 2 the femoral heads. 4 does not appear in those slices and is
# named by elimination.
HIPMRI_CLASS_NAMES = ("Background", "Body", "Bone", "Bladder", "Rectum", "Prostate")
HIPMRI_SHAPE = (256, 128)


def load_nifti_slice(path):
    """One 2D Nifti slice -> float64 array (H, W), dropping a stray third axis as Appendix B does."""
    array = nib.load(str(path)).get_fdata(caching="unchanged")
    return array[:, :, 0] if array.ndim == 3 else array


def centre_crop(array):
    """Crop the last axis to HIPMRI_SHAPE[1] columns, keeping the centre (only the 256x144 scan needs it)."""
    extra = array.shape[-1] - HIPMRI_SHAPE[1]
    if extra < 0 or array.shape[-2] != HIPMRI_SHAPE[0]:
        raise ValueError(f"Cannot crop shape {array.shape} to {HIPMRI_SHAPE}")
    return array[..., extra // 2: extra // 2 + HIPMRI_SHAPE[1]]


def load_hipmri_image(path):
    """Nifti slice -> float32 tensor (1, 256, 128), standardised to zero mean and unit variance."""
    array = centre_crop(load_nifti_slice(path))
    # Per-slice standardisation (the normImage option of Appendix B) uses no
    # statistics from any split, so nothing from validation or test can leak in.
    array = (array - array.mean()) / (array.std() + 1e-8)
    return torch.from_numpy(array.astype(np.float32)).unsqueeze(0)


def load_hipmri_mask(path):
    """Nifti mask slice -> int64 tensor (256, 128) of class indices 0..5."""
    full = load_nifti_slice(path)
    labels = np.rint(full).astype(np.int64)
    if not np.array_equal(labels, full) or labels.min() < 0 or labels.max() >= len(HIPMRI_CLASS_NAMES):
        raise ValueError(f"Unexpected mask values in {path}")
    cropped = centre_crop(labels)
    # Cropping must never cut off the small organs being segmented: the three
    # smallest classes (3-5) may not appear in the removed columns.
    if (labels >= 3).sum() != (cropped >= 3).sum():
        raise ValueError(f"Centre crop would remove organ pixels in {path}")
    return torch.from_numpy(np.ascontiguousarray(cropped))


def hipmri_key(path):
    """'case_004_week_1_slice_5.nii.gz' or 'seg_...' -> (patient 4, week 1, slice 5)."""
    match = re.fullmatch(r"(?:case|seg)_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz", path.name)
    if match is None:
        raise ValueError(f"Unexpected filename: {path.name}")
    return int(match[1]), int(match[2]), int(match[3])


# ----------------------------------------------------------------- shared

def one_hot(masks, num_classes=NUM_CLASSES):
    """Batched class indices (B, *S) -> float one-hot (B, C, *S), for 2D or 3D masks."""
    return F.one_hot(masks, num_classes).movedim(-1, 1).float()


def split_pairs(root=ROOT, image_dir="keras_png_slices_{}", mask_dir="keras_png_slices_seg_{}",
                key=slice_key, pattern="*.png"):
    """
    {split: [(image_path, mask_path), ...]} for each split.

    Args:
        root: dataset folder.
        image_dir, mask_dir: folder name templates, formatted with the split name.
        key: filename -> tuple whose first element is the patient number.
        pattern: glob for the slice files.

    Images and masks are paired by key rather than by sorted position, so one
    missing file cannot shift every later pair. Raises if any patient appears in
    two splits: slices, or other weekly scans, of a patient seen in training
    would leak into evaluation.
    """
    pairs, patients = {}, {}
    for split in SPLITS:
        images = {key(p): p for p in (root / image_dir.format(split)).glob(pattern)}
        masks = {key(p): p for p in (root / mask_dir.format(split)).glob(pattern)}
        if not images or images.keys() != masks.keys():
            raise ValueError(f"{split}: images and masks under {root} do not pair up")
        pairs[split] = [(images[k], masks[k]) for k in sorted(images)]
        patients[split] = {k[0] for k in images}
    for a, b in itertools.combinations(SPLITS, 2):
        shared = patients[a] & patients[b]
        if shared:
            raise ValueError(f"Patients in both {a} and {b}: {sorted(shared)[:5]}")
    return pairs


def hipmri_split_pairs(root=HIPMRI_ROOT):
    """split_pairs for the HipMRI Nifti slices, keyed by (patient, week, slice)."""
    return split_pairs(root, "keras_slices_{}", "keras_slices_seg_{}", hipmri_key, "*.nii.gz")


class OASISDataset(Dataset):
    """
    (image (1, H, W) float, mask (H, W) int64) pairs from OASIS PNG slices.

    Args:
        pairs: [(image_path, mask_path), ...] from split_pairs.
        augment: random left-right flip (training split only).
    """

    load_image = staticmethod(load_image)
    load_mask = staticmethod(load_mask)
    # Tensor axis that runs left-right, for the flip augmentation.
    flip_dim = -1

    def __init__(self, pairs, augment=False):
        self.pairs, self.augment = list(pairs), augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image, mask = self.load_image(image_path), self.load_mask(mask_path)
        # Flip image and mask together; left-right is the safe axis because
        # axial slices are roughly symmetric. A flip moves pixels without
        # interpolating, so mask labels stay exact. Flip is the only augmentation
        # for now; small rotations or scaling can follow if validation Dice
        # plateaus below training Dice.
        if self.augment and self.flip_dim is not None and torch.rand(()) < 0.5:
            image, mask = image.flip(self.flip_dim), mask.flip(self.flip_dim)
        return image, mask


class HipMRIDataset(OASISDataset):
    """(image (1, 256, 128) float, mask (256, 128) int64) pairs from HipMRI Nifti slices."""

    load_image = staticmethod(load_hipmri_image)
    load_mask = staticmethod(load_hipmri_mask)
    # Rows run left-right (the femoral heads sit at the top and bottom of each
    # slice in the label overlay). Flipping columns instead would swap front and
    # back, putting the rectum in front of the bladder.
    flip_dim = -2


# key: filename -> (patient, ..., slice); everything but the slice identifies one scan.
DATASETS = {
    "oasis": {"root": ROOT, "classes": CLASS_NAMES, "pairs": split_pairs, "dataset": OASISDataset,
              "key": slice_key},
    "hipmri": {"root": HIPMRI_ROOT, "classes": HIPMRI_CLASS_NAMES, "pairs": hipmri_split_pairs,
               "dataset": HipMRIDataset, "key": hipmri_key},
}


def get_loaders(batch_size, workers=4, root=None, limit=0, dataset="oasis"):
    """
    {split: DataLoader} for one dataset in DATASETS.

    Only train is shuffled and augmented. root=None uses the dataset's default
    folder; limit>0 keeps N slices per split for debugging.
    """
    spec = DATASETS[dataset]
    pairs = spec["pairs"](root or spec["root"])
    return {
        split: DataLoader(
            spec["dataset"](pairs[split][:limit or None], augment=split == "train"),
            batch_size=batch_size,
            shuffle=split == "train",
            drop_last=split == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        )
        for split in SPLITS
    }


def check_hipmri_labels(pairs, path):
    """
    Load every HipMRI slice once (validating labels and the crop), print where
    each class sits on average, and save overlays of the slices with the most
    prostate so the label order and orientation can be checked by eye.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    names = HIPMRI_CLASS_NAMES
    rows, cols = np.mgrid[:HIPMRI_SHAPE[0], :HIPMRI_SHAPE[1]]
    centroid_sums, pixel_counts, prostate = np.zeros((len(names), 2)), np.zeros(len(names)), []
    for split in SPLITS:
        for image_path, mask_path in pairs[split]:
            mask = load_hipmri_mask(mask_path).numpy()
            load_hipmri_image(image_path)
            for c in range(len(names)):
                inside = mask == c
                centroid_sums[c] += rows[inside].sum(), cols[inside].sum()
                pixel_counts[c] += inside.sum()
            prostate.append(((mask == 5).sum(), hipmri_key(image_path)[0], image_path, mask_path))
    print("mean (row, column) position of each class over all slices:")
    for c, name in enumerate(names):
        row, col = centroid_sums[c] / max(pixel_counts[c], 1)
        print(f"  {c} {name:10s} row {row:6.1f}  column {col:6.1f}  share {pixel_counts[c] / pixel_counts.sum():.4%}")

    chosen, seen = [], set()
    for count, patient, image_path, mask_path in sorted(prostate, key=lambda item: -item[0]):
        if patient not in seen:
            chosen.append((image_path, mask_path))
            seen.add(patient)
        if len(chosen) == 4:
            break
    colours = ListedColormap(["black", "tab:gray", "white", "tab:blue", "tab:brown", "tab:red"])
    fig, axes = plt.subplots(2, len(chosen), figsize=(3.2 * len(chosen), 9), squeeze=False)
    for column, (image_path, mask_path) in enumerate(chosen):
        axes[0, column].imshow(load_hipmri_image(image_path)[0], cmap="gray")
        axes[0, column].set_title(image_path.name, fontsize=7)
        axes[1, column].imshow(load_hipmri_mask(mask_path), cmap=colours, vmin=0, vmax=5, interpolation="nearest")
    for ax in axes.flat:
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours(i)) for i in range(len(names))]
    fig.legend(handles, [f"{i} {n}" for i, n in enumerate(names)], loc="lower center", ncol=3)
    fig.suptitle("HipMRI slices with the most label-5 (prostate) pixels")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"label overlay saved to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-check a dataset loader.")
    parser.add_argument("--dataset", choices=DATASETS, default="oasis")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    name, root = args.dataset, args.root
    spec = DATASETS[name]
    num_classes = len(spec["classes"])

    pairs = spec["pairs"](root or spec["root"])
    key = spec["key"]
    for split in SPLITS:
        keys = [key(image) for image, _ in pairs[split]]
        print(f"{split}: {len(keys)} slices, {len({k[0] for k in keys})} patients, {len({k[:-1] for k in keys})} scans")

    images, masks = next(iter(get_loaders(batch_size=16, workers=0, root=root, dataset=name)["train"]))
    assert images.dtype == torch.float32 and images.shape[:2] == (16, 1), images.shape
    assert masks.dtype == torch.int64 and masks.shape == (16, *images.shape[2:]), masks.shape
    # The networks downsample four times, so every spatial size must divide by 16.
    assert all(size % 16 == 0 for size in images.shape[2:]), images.shape
    assert 0 <= masks.min() and masks.max() < num_classes
    encoded = one_hot(masks, num_classes)
    assert encoded.shape == (16, num_classes, *images.shape[2:]) and torch.all(encoded.sum(1) == 1)

    # Augmentation must flip image and mask together or not at all.
    plain, augmented = spec["dataset"](pairs["train"][:1]), spec["dataset"](pairs["train"][:1], augment=True)
    image, mask = plain[0]
    dim = spec["dataset"].flip_dim
    for _ in range(20):
        aug_image, aug_mask = augmented[0]
        flipped = not torch.equal(aug_image, image)
        assert dim is not None or not flipped
        assert torch.equal(aug_image, image.flip(dim) if flipped else image)
        assert torch.equal(aug_mask, mask.flip(dim) if flipped else mask)

    share = torch.bincount(masks.flatten(), minlength=num_classes) / masks.numel()
    print("class share in one batch:", {n: round(s.item(), 4) for n, s in zip(spec["classes"], share)})
    if name == "hipmri":
        check_hipmri_labels(pairs, HERE / "outputs" / "hipmri_labels.png")
    print("dataset self-check passed")
