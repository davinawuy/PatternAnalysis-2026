# Data loading for segmentation of OASIS brain slices, HipMRI pelvic slices and HipMRI pelvic volumes.
#
# OASIS: train / validate / test folders of 256x256 8-bit PNG slices, already
# split by patient, with masks storing four tissue classes as the grey levels
# 0, 85, 170 and 255. Images are scaled to [0, 1].
#
# HipMRI 2D: the same three-way patient-level split of Nifti slices
# (case_PPP_week_W_slice_S.nii.gz), 60 coronal slices per weekly scan (indices
# 98-157 of the volume's front-back axis, rows left-right, columns foot to head),
# with masks labelled 0-5. Every slice is 256x128 except one training scan at 256x144, which
# is centre-cropped to 256x128. MRI intensities have no fixed scale between
# scans, so each slice is standardised to zero mean and unit variance.
#
# HipMRI 3D: the 211 Nifti volumes the slices come from (256x256x128, stored in
# LPS orientation: left-right, front-back, foot-head), with the same labels. Each
# patient keeps its 2D split, so 2D and 3D models are tested on the same scans.
# Volumes are standardised as a whole and downsampled by an integer factor:
# images by averaging, masks by the majority label of each block (exact ties
# broken at random, so no class gains voxels).
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
#     python dataset.py --dataset hipmri3d

import hashlib
import itertools
import os
from collections import Counter, defaultdict
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


# --------------------------------------------------------------- HipMRI 3D

HIPMRI3D_ROOT = Path(os.environ.get("HIPMRI3D_ROOT", "/home/groups/comp3710/HipMRI_Study_open"))


def hipmri3d_key(path):
    """'B006_Week1_LFOV.nii.gz' or 'B006_Week1_SEMANTIC.nii.gz' -> (patient 6, week 1)."""
    match = re.fullmatch(r"[A-Za-z]+(\d+)_Week(\d+)_(?:LFOV|SEMANTIC)\.nii\.gz", path.name)
    if match is None:
        raise ValueError(f"Unexpected filename: {path.name}")
    return int(match[1]), int(match[2])


def load_volume(path):
    """Nifti volume -> array (256, 256, 128) in stored (LPS) axis order, centre-cropping the foot-head axis."""
    return centre_crop(np.asarray(nib.load(str(path)).dataobj))


def load_hipmri3d_image(path, factor=2):
    """Nifti volume -> float32 tensor (1, 256/f, 256/f, 128/f), standardised over the whole volume."""
    array = load_volume(path).astype(np.float32)
    array = (array - array.mean()) / (array.std() + 1e-8)
    image = torch.from_numpy(array).unsqueeze(0)
    return F.avg_pool3d(image, factor) if factor > 1 else image


def load_hipmri3d_mask(path, factor=2, ties="random"):
    """
    Nifti mask volume -> int64 tensor (256/f, 256/f, 128/f) of class indices 0..5.

    ties: how a block split equally between classes is labelled. "random" picks
        one of the tied classes with a fixed seed, so no class is favoured;
        "smaller" gives it to the higher label (the smaller organ), which inflates
        the prostate by about 3.6% and is kept to reproduce the first 3D runs.
    """
    full = np.asarray(nib.load(str(path)).dataobj)
    if full.min() < 0 or full.max() >= len(HIPMRI_CLASS_NAMES):
        raise ValueError(f"Unexpected mask values in {path}")
    cropped = centre_crop(full)
    if (full >= 3).sum() != (cropped >= 3).sum():
        raise ValueError(f"Centre crop would remove organ pixels in {path}")
    mask = torch.from_numpy(cropped.astype(np.int64))
    if factor == 1:
        return mask
    # Majority label of each factor^3 block. Nearest-neighbour sampling would keep
    # one voxel per block and could drop thin organ edges.
    shares = torch.stack([F.avg_pool3d((mask == c).float()[None], factor)[0] for c in range(len(HIPMRI_CLASS_NAMES))])
    if ties == "smaller":
        return len(HIPMRI_CLASS_NAMES) - 1 - shares.flip(0).argmax(0)
    if ties != "random":
        raise ValueError(f"Unknown tie rule {ties!r}")
    # Shares are multiples of 1/factor^3, so noise below half of that only
    # reorders classes with exactly equal shares.
    noise = torch.rand(shares.shape, generator=torch.Generator().manual_seed(0)) / (2 * factor ** 3)
    return (shares + noise).argmax(0)


def hipmri3d_split_pairs(root=HIPMRI3D_ROOT, slices_root=None):
    """
    {split: [(image_path, mask_path), ...]} for the HipMRI volumes.

    Each scan gets the split its 2D slices are in (slices_root, default
    HIPMRI_ROOT), so the patient-level split and its overlap check carry over.
    Raises if a scan exists only as volumes or only as slices.
    """
    scan_split = {hipmri_key(image)[:2]: split
                  for split, pairs in hipmri_split_pairs(slices_root or HIPMRI_ROOT).items() for image, _ in pairs}
    images = {hipmri3d_key(p): p for p in (root / "semantic_MRs").glob("*.nii.gz")}
    masks = {hipmri3d_key(p): p for p in (root / "semantic_labels_only").glob("*.nii.gz")}
    if not images or images.keys() != masks.keys():
        raise ValueError(f"Volumes and masks under {root} do not pair up")
    if images.keys() != scan_split.keys():
        raise ValueError(f"Volumes and 2D slices cover different scans: {sorted(images.keys() ^ scan_split.keys())[:5]}")
    pairs = {split: [] for split in SPLITS}
    for key in sorted(images):
        pairs[scan_split[key]].append((images[key], masks[key]))
    return pairs


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
    # The slices are coronal: rows run left-right and columns foot to head
    # (check_hipmri_2d_3d). Flipping columns instead would put the bladder below
    # the prostate.
    flip_dim = -2


class HipMRI3DDataset(OASISDataset):
    """
    (image (1, X, Y, Z) float, mask (X, Y, Z) int64) HipMRI volumes.

    Args:
        pairs: [(image_path, mask_path), ...] from hipmri3d_split_pairs.
        augment: random left-right flip (training split only).
        factor: integer downsampling factor along every axis (1 keeps 256x256x128).
        ties: tie rule for mask downsampling, see load_hipmri3d_mask.
        cache: folder for preprocessed volumes, or None to preprocess on every read.
            Decompressing and downsampling a volume takes far longer than a
            training step, so without a cache data loading dominates each epoch.
    """

    # Axis 0 of the stored volume runs left-right (the L of LPS).
    flip_dim = -3

    def __init__(self, pairs, augment=False, factor=2, ties="random", cache=None):
        super().__init__(pairs, augment)
        self.factor, self.ties, self.cache = factor, ties, cache

    def cached(self, path, load, tag):
        """load(path), saved under self.cache after the first call; tag names the preprocessing."""
        if self.cache is None:
            return load(path)
        # The name hashes the full source path, so different data folders never share entries.
        # Note: cache entries are not invalidated when the preprocessing code changes; delete the cache folder then.
        digest = hashlib.md5(str(Path(path).resolve()).encode()).hexdigest()[:12]
        file = Path(self.cache) / f"{digest}_{tag}_{Path(path).name}.pt"
        if file.exists():
            tensor = torch.load(file)
            return tensor.long() if tensor.dtype == torch.uint8 else tensor
        tensor = load(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        # Masks are stored as uint8 (labels 0-5) to save space. Writing to a temporary
        # file first means a reader in another worker never sees a half-written entry.
        temporary = file.with_name(f"{file.name}.{os.getpid()}.tmp")
        torch.save(tensor.to(torch.uint8) if tensor.dtype == torch.int64 else tensor, temporary)
        os.replace(temporary, file)
        return tensor

    def load_image(self, path):
        return self.cached(path, lambda p: load_hipmri3d_image(p, self.factor), f"x{self.factor}")

    def load_mask(self, path):
        return self.cached(path, lambda p: load_hipmri3d_mask(p, self.factor, self.ties), f"x{self.factor}_{self.ties}")


# key: filename -> (patient, ..., slice); everything but the slice identifies one
# scan. Volumes have no slice number, so their key is the scan itself.
DATASETS = {
    "oasis": {"root": ROOT, "classes": CLASS_NAMES, "pairs": split_pairs, "dataset": OASISDataset,
              "key": slice_key},
    "hipmri": {"root": HIPMRI_ROOT, "classes": HIPMRI_CLASS_NAMES, "pairs": hipmri_split_pairs,
               "dataset": HipMRIDataset, "key": hipmri_key},
    "hipmri3d": {"root": HIPMRI3D_ROOT, "classes": HIPMRI_CLASS_NAMES, "pairs": hipmri3d_split_pairs,
                 "dataset": HipMRI3DDataset, "key": hipmri3d_key},
}


def get_loaders(batch_size, workers=4, root=None, limit=0, dataset="oasis", **dataset_args):
    """
    {split: DataLoader} for one dataset in DATASETS.

    Only train is shuffled and augmented. root=None uses the dataset's default
    folder; limit>0 keeps N slices (or volumes) per split for debugging.
    dataset_args go to the dataset class, e.g. factor=2 for hipmri3d.
    """
    spec = DATASETS[dataset]
    pairs = spec["pairs"](root or spec["root"])
    return {
        split: DataLoader(
            spec["dataset"](pairs[split][:limit or None], augment=split == "train", **dataset_args),
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


def check_hipmri_2d_3d(pairs, split="test"):
    """
    Match every 2D slice of the split's scans to its 3D volume voxel for voxel, to
    find which volume axis the slices were cut along, any flips, and which volume
    slices they are. Prints one line per scan and returns the (axis, flips) counts.
    """
    slices = defaultdict(list)
    for image_path, mask_path in hipmri_split_pairs()[split]:
        patient, week, number = hipmri_key(image_path)
        slices[patient, week].append((number, image_path, mask_path))
    found = Counter()
    for image_path, mask_path in pairs[split]:
        scan = hipmri3d_key(mask_path)
        volume = np.asarray(nib.load(str(mask_path)).dataobj)
        image = np.asarray(nib.load(str(image_path)).dataobj)
        matched, image_error = [], 0.0
        for number, slice_image, slice_mask in sorted(slices[scan]):
            target = np.rint(load_nifti_slice(slice_mask))
            for axis, flips in itertools.product(range(3), ((), (0,), (1,), (0, 1))):
                candidate = np.flip(target, flips) if flips else target
                sections = np.moveaxis(volume, axis, 0)
                if sections.shape[1:] != candidate.shape:
                    continue
                hits = np.flatnonzero((sections == candidate).all((1, 2)))
                if len(hits):
                    found[axis, flips] += 1
                    index = hits[0]
                    matched.append((number, index, len(hits)))
                    section = np.moveaxis(image, axis, 0)[index]
                    source = load_nifti_slice(slice_image)
                    image_error = max(image_error, np.abs(section - (np.flip(source, flips) if flips else source)).max())
                    break
            else:
                found["unmatched"] += 1
        offsets = sorted({int(index - number) for number, index, _ in matched})
        indices = [index for _, index, _ in matched]
        print(f"  patient {scan[0]} week {scan[1]}: {len(matched)}/{len(slices[scan])} slices matched, "
              f"volume indices {min(indices, default='-')}-{max(indices, default='-')}, "
              f"index minus slice number {offsets[:3]}, ambiguous {sum(n > 1 for *_, n in matched)}, "
              f"max image difference {image_error:.4g}")
    print(f"{split} slices by (volume axis, flips): {dict(found)}")
    return found


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
    volumes = name == "hipmri3d"
    for split in SPLITS:
        keys = [key(image) for image, _ in pairs[split]]
        scans = set(keys) if volumes else {k[:-1] for k in keys}
        print(f"{split}: {len(keys)} {'volumes' if volumes else 'slices'}, {len({k[0] for k in keys})} patients, "
              f"{len(scans)} scans")

    batch = 2 if volumes else 16
    images, masks = next(iter(get_loaders(batch_size=batch, workers=0, root=root, dataset=name)["train"]))
    print("batch shapes", tuple(images.shape), tuple(masks.shape))
    assert images.dtype == torch.float32 and images.shape[:2] == (batch, 1), images.shape
    assert masks.dtype == torch.int64 and masks.shape == (batch, *images.shape[2:]), masks.shape
    # The networks downsample four times, so every spatial size must divide by 16.
    assert all(size % 16 == 0 for size in images.shape[2:]), images.shape
    assert 0 <= masks.min() and masks.max() < num_classes
    encoded = one_hot(masks, num_classes)
    assert encoded.shape == (batch, num_classes, *images.shape[2:]) and torch.all(encoded.sum(1) == 1)

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
    if volumes:
        # Downsampling must keep the small organs: compare voxel counts with full resolution.
        mask_path = pairs["train"][0][1]
        full_counts = torch.bincount(load_hipmri3d_mask(mask_path, 1).flatten(), minlength=num_classes)
        for ties in ("random", "smaller"):
            small = load_hipmri3d_mask(mask_path, 2, ties)
            change = 8 * torch.bincount(small.flatten(), minlength=num_classes) / full_counts - 1
            print(f"voxel count change when downsampling by 2, ties {ties}:",
                  {n: f"{c.item():+.2%}" for n, c in zip(spec["classes"], change)})
        found = check_hipmri_2d_3d(pairs)
        assert "unmatched" not in found, "some 2D slices match no volume section"
    print("dataset self-check passed")
