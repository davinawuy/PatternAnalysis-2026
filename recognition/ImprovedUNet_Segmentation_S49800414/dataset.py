# OASIS brain MRI data loading for 2D tissue segmentation.
#
# The dataset ships already split by patient into train / validate / test folders
# of 256x256 8-bit PNG slices, with a matching mask folder per split. Masks store
# the four tissue classes as the grey levels 0, 85, 170 and 255. Layout, shapes
# and label values were confirmed by data_audit.py before this loader was written.
#
# Images are scaled to [0, 1]. Masks are returned as class indices (H, W), which
# is what cross-entropy expects; one_hot() gives the (C, H, W) encoding for Dice.
#
# Based on the data pipeline guidelines in the COMP3710 assignment spec
# (Section 2.2: patient-level splitting, intensity normalisation, one-hot masks).
#
# Self-check (run on a compute node):
#     python dataset.py

import itertools
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(os.environ.get("DATA_ROOT", "/home/groups/comp3710/OASIS"))
SPLITS = ("train", "validate", "test")

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


def one_hot(mask):
    """Class indices (..., H, W) -> float one-hot (..., C, H, W)."""
    return F.one_hot(mask, NUM_CLASSES).movedim(-1, -3).float()


def slice_key(path):
    """'case_001_slice_5.nii.png' or 'seg_001_slice_5.nii.png' -> (1, 5)."""
    match = re.fullmatch(r"(?:case|seg)_(\d+)_slice_(\d+)\.nii\.png", path.name)
    if match is None:
        raise ValueError(f"Unexpected filename: {path.name}")
    return int(match[1]), int(match[2])


def split_pairs(root=ROOT):
    """
    {split: [(image_path, mask_path), ...]} for each split.

    Images and masks are paired by (patient, slice) rather than by sorted
    position, so one missing file cannot shift every later pair. Raises if any
    patient appears in two splits, since slices of a seen brain would leak into
    evaluation.
    """
    pairs, patients = {}, {}
    for split in SPLITS:
        images = {slice_key(p): p for p in (root / f"keras_png_slices_{split}").glob("*.png")}
        masks = {slice_key(p): p for p in (root / f"keras_png_slices_seg_{split}").glob("*.png")}
        if not images or images.keys() != masks.keys():
            raise ValueError(f"{split}: images and masks under {root} do not pair up")
        pairs[split] = [(images[k], masks[k]) for k in sorted(images)]
        patients[split] = {patient for patient, _ in images}
    for a, b in itertools.combinations(SPLITS, 2):
        shared = patients[a] & patients[b]
        if shared:
            raise ValueError(f"Patients in both {a} and {b}: {sorted(shared)[:5]}")
    return pairs


class OASISDataset(Dataset):
    """(image (1, H, W) float, mask (H, W) int64) pairs."""

    def __init__(self, pairs, augment=False):
        self.pairs, self.augment = list(pairs), augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image, mask = load_image(image_path), load_mask(mask_path)
        # Flip image and mask together; left-right is the safe axis because
        # axial brain slices are roughly symmetric. A flip moves pixels without
        # interpolating, so mask labels stay exact. Flip is the only augmentation
        # for now; small rotations or scaling can follow if validation Dice
        # plateaus below training Dice.
        if self.augment and torch.rand(()) < 0.5:
            image, mask = image.flip(-1), mask.flip(-1)
        return image, mask


def get_loaders(batch_size, workers=4, root=ROOT, limit=0):
    """{split: DataLoader}. Only train is shuffled and augmented. limit>0 keeps N slices per split for debugging."""
    pairs = split_pairs(root)
    return {
        split: DataLoader(
            OASISDataset(pairs[split][:limit or None], augment=split == "train"),
            batch_size=batch_size,
            shuffle=split == "train",
            drop_last=split == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        )
        for split in SPLITS
    }


if __name__ == "__main__":
    pairs = split_pairs()
    for split in SPLITS:
        print(f"{split}: {len(pairs[split])} slices, {len({slice_key(i)[0] for i, _ in pairs[split]})} patients")

    images, masks = next(iter(get_loaders(batch_size=16, workers=0)["train"]))
    assert images.shape == (16, 1, 256, 256) and images.dtype == torch.float32, images.shape
    assert masks.shape == (16, 256, 256) and masks.dtype == torch.int64, masks.shape
    assert 0 <= images.min() and images.max() <= 1
    assert 0 <= masks.min() and masks.max() < NUM_CLASSES
    encoded = one_hot(masks)
    assert encoded.shape == (16, NUM_CLASSES, 256, 256) and torch.all(encoded.sum(1) == 1)

    # Augmentation must flip image and mask together or not at all.
    plain, augmented = OASISDataset(pairs["train"][:1]), OASISDataset(pairs["train"][:1], augment=True)
    image, mask = plain[0]
    for _ in range(20):
        aug_image, aug_mask = augmented[0]
        flipped = not torch.equal(aug_image, image)
        assert torch.equal(aug_image, image.flip(-1) if flipped else image)
        assert torch.equal(aug_mask, mask.flip(-1) if flipped else mask)

    share = torch.bincount(masks.flatten(), minlength=NUM_CLASSES) / masks.numel()
    print("class share in one batch:", {n: round(s.item(), 3) for n, s in zip(CLASS_NAMES, share)})
    print("dataset self-check passed")
