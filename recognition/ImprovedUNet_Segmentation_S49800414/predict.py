# Test-set evaluation and visualisation for a trained segmentation model.
#
# Loads best.pt from train.py (which records whether it was trained on OASIS or
# HipMRI), runs inference one case at a time and reports what the COMP3710
# assignment spec (Section 2.2) asks for. A case is one scan: one patient for
# OASIS, one patient's weekly scan (60 slices) for HipMRI.
#   * per-class Dice and IoU for every test case, their averages and spread
#   * a pixel confusion matrix (ground truth rows, prediction columns)
#   * resource profile: parameters, peak inference VRAM, per-case latency, and
#     training time and VRAM read back from the training run
#   * a connectivity check for the open research dilemma: high global Dice can
#     hide broken or fragmented structures, so connected components per class
#     are counted in the prediction and the ground truth
#   * multi-panel figures of the worst slices: input, ground truth, prediction
#     and a false positive / false negative overlay
# For HipMRI, the failure modes the spec names for prostate radiotherapy:
#   * prostate Dice from apex to base (axial levels, i.e. columns of the coronal
#     slices), where global Dice hides local failure
#   * leakage: prostate predicted on rectum or bladder pixels and the reverse
#   * prostate predicted on slices without any prostate, and slices missed entirely
# For HipMRI 3D checkpoints, every test volume is predicted at the training
# resolution and scored against the full-resolution masks. With --compare, a
# 2D checkpoint is run on the same scans and both are scored on the coronal slab
# the 2D slices come from, so 2D and 3D are compared on identical voxels. The
# acceptance criteria of the feasibility review (AC1-AC5) are checked, and the
# worst volumes are drawn at their worst apex and base levels with prostate
# false positive / false negative overlays.
#
# NumPy and SciPy are used here for metrics and plotting only; the model itself
# is pure PyTorch (modules.py).
#
# Usage:
#     python predict.py --checkpoint outputs/improved/best.pt
#     python predict.py --checkpoint outputs/hipmri_improved/best.pt
#     python predict.py --checkpoint outputs/hipmri3d_improved/best.pt --compare outputs/hipmri_improved/best.pt

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib

# Headless backend: Slurm jobs have no display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from scipy import ndimage

from dataset import DATASETS, hipmri3d_key, load_hipmri3d_image, load_hipmri3d_mask, load_volume
from train import MODELS, TARGETS, columns, dice_counts

# One fixed colour per class index so every figure reads the same way.
COLOURS = ["black", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
# 8-connectivity: pixels touching diagonally count as connected, so a thin
# strand running diagonally is not counted as broken. A component split
# therefore means a real gap in the prediction.
CONNECTIVITY = np.ones((3, 3), dtype=int)


def load_model(path, device):
    """
    Rebuild a trained model from a train.py checkpoint.

    Returns:
        (model in eval mode on device, checkpoint dict with "settings" and "epoch").
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    settings = checkpoint["settings"]
    # Checkpoints from before the --dataset option are all OASIS.
    settings.setdefault("dataset", "oasis")
    num_classes = len(DATASETS[settings["dataset"]]["classes"])
    model = MODELS[settings["model"]](num_classes=num_classes, dims=3 if settings["dataset"] == "hipmri3d" else 2)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def case_name(case):
    """(patient,) or (patient, week) -> 'patient 4' or 'patient 4 week 1'."""
    return " ".join(f"{label} {value}" for label, value in zip(("patient", "week"), case))


def synchronise(device):
    """Wait for queued GPU work, so wall-clock timings measure the computation."""
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def infer(model, images, device):
    """(B, 1, H, W) images on device -> (B, C, H, W) float32 logits, with the same mixed precision as training."""
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return model(images).float()


def evaluate(model, pairs, device, spec):
    """
    Predict every case (scan) in a split.

    Args:
        pairs: [(image_path, mask_path), ...] from the dataset's split function.
        spec: the dataset's entry in dataset.DATASETS.
    Returns:
        {case tuple: dict with "slices" (N,), "images" (N, H, W) float, "masks"
        and "preds" (N, H, W) uint8, "intersection" and "total" (N, C) Dice
        counts, "seconds" inference time for the whole case}.
    """
    cases = defaultdict(list)
    for pair in pairs:
        cases[spec["key"](pair[0])[:-1]].append(pair)

    results, warmed_up = {}, False
    for case, case_pairs in sorted(cases.items()):
        dataset = spec["dataset"](case_pairs)
        images, masks = map(torch.stack, zip(*(dataset[i] for i in range(len(dataset)))))
        images_on_device = images.to(device)
        if not warmed_up:
            # The first call pays one-off costs (cuDNN kernel selection, memory
            # allocation) that would otherwise inflate the first case's latency.
            infer(model, images_on_device, device)
            warmed_up = True
        synchronise(device)
        start = time.perf_counter()
        logits = infer(model, images_on_device, device)
        synchronise(device)
        seconds = time.perf_counter() - start

        logits = logits.cpu()
        intersection, total = dice_counts(logits, masks)
        results[case] = {
            "slices": np.array([spec["key"](image)[-1] for image, _ in case_pairs]),
            "images": images[:, 0].numpy(),
            "masks": masks.numpy().astype(np.uint8),
            "preds": logits.argmax(1).numpy().astype(np.uint8),
            "intersection": intersection.numpy(),
            "total": total.numpy(),
            "seconds": seconds,
        }
    return results


def dice_iou(intersection, total):
    """
    Dice = 2|P and G| / (|P| + |G|) and IoU = |P and G| / |P or G| from counts.

    Both are undefined when a class is absent from prediction and ground truth
    alike, so those entries are NaN rather than a misleading 0 or 1.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = np.where(total > 0, 2 * intersection / total, np.nan)
        iou = np.where(total > 0, intersection / (total - intersection), np.nan)
    return dice, iou


def confusion(results, num_classes):
    """(C, C) pixel counts pooled over the split; rows are ground truth, columns prediction."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for case in results.values():
        pairs = case["masks"].ravel().astype(np.int64) * num_classes + case["preds"].ravel()
        matrix += np.bincount(pairs, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return matrix


def connectivity(results, names):
    """
    Connected components per non-background class, prediction against ground truth.

    Returns:
        {class name: mean components per slice in the ground truth and the
        prediction, and the share of slices where the prediction has more
        components (fragmented) or fewer (merged or missing) than the truth}.
    """
    counts = defaultdict(list)
    for case in results.values():
        for mask, pred in zip(case["masks"], case["preds"]):
            for c in range(1, len(names)):
                truth = ndimage.label(mask == c, CONNECTIVITY)[1]
                predicted = ndimage.label(pred == c, CONNECTIVITY)[1]
                counts[c].append((truth, predicted))
    summary = {}
    for c, pairs in counts.items():
        truth, predicted = np.array(pairs).T
        summary[names[c]] = {
            "truth_components_per_slice": truth.mean(),
            "predicted_components_per_slice": predicted.mean(),
            "slices_more_components": (predicted > truth).mean(),
            "slices_fewer_components": (predicted < truth).mean(),
        }
    return summary


def gland_position(masks, prostate, bladder):
    """
    Position of every axial level along the prostate, from apex (0) to base (1).

    The foot-head axis is the last axis of masks (the columns of the coronal
    HipMRI slices, the last volume axis in 3D). The levels with prostate in the
    ground truth are placed from apex to base. The base is the end nearer the
    bladder, which lies above the prostate, so the direction comes from the
    labels rather than from an assumed axis order.

    Returns:
        (position per level, boolean per level: prostate present), or None if
        the case has no prostate or no bladder.
    """
    prostate_voxels = (masks == prostate).sum((0, 1))
    bladder_voxels = (masks == bladder).sum((0, 1))
    present = prostate_voxels > 0
    if not present.any() or not bladder_voxels.any():
        return None
    levels = np.arange(len(present))
    first, last = levels[present].min(), levels[present].max()
    position = (levels - first) / max(last - first, 1)
    if np.average(levels, weights=bladder_voxels) < np.average(levels, weights=prostate_voxels):
        position = 1 - position
    return position, present


def prostate_by_position(results, prostate, bladder):
    """
    Prostate Dice along the gland (see gland_position), for the spec's apex and
    base failure mode. Cases without any bladder are skipped.

    Returns:
        (per-level rows [(position 0..1, level Dice, case name)],
         {"apex" | "mid" | "base": Dice pooled over that third of the gland}).
    """
    rows, thirds = [], {name: np.zeros(2) for name in ("apex", "mid", "base")}
    for case_id, case in results.items():
        located = gland_position(case["masks"], prostate, bladder)
        if located is None:
            continue
        position, present = located
        truth, predicted = case["masks"] == prostate, case["preds"] == prostate
        intersection = (truth & predicted).sum((0, 1))
        total = truth.sum((0, 1)) + predicted.sum((0, 1))
        for z in np.flatnonzero(present):
            rows.append((position[z], 2 * intersection[z] / total[z], case_name(case_id)))
            thirds[("apex", "mid", "base")[min(int(position[z] * 3), 2)]] += intersection[z], total[z]
    return rows, {name: 2 * i / t if t else float("nan") for name, (i, t) in thirds.items()}


def prostate_slices(results, prostate):
    """
    Slice-level detection errors on the (coronal) input slices: prostate
    predicted on slices whose ground truth has none (false-positive slices), and
    prostate slices with no prostate predicted at all (missed slices). Slice
    Dice is undefined or zero on these, so they need counting separately.
    """
    without, with_prostate, missed, false_positive_pixels = 0, 0, 0, []
    for case in results.values():
        truth = (case["masks"] == prostate).any((1, 2))
        predicted = (case["preds"] == prostate).sum((1, 2))
        without += (~truth).sum()
        with_prostate += truth.sum()
        missed += (truth & (predicted == 0)).sum()
        false_positive_pixels += [n for n in predicted[~truth] if n > 0]
    return {
        "slices_without_prostate": int(without),
        "false_positive_slices": len(false_positive_pixels),
        "false_positive_slice_share": len(false_positive_pixels) / max(without, 1),
        "false_positive_pixels_median": float(np.median(false_positive_pixels)) if false_positive_pixels else 0.0,
        "slices_with_prostate": int(with_prostate),
        "missed_slices": int(missed),
        "missed_slice_share": missed / max(with_prostate, 1),
    }


def worst_slices(results, count):
    """
    The count lowest-scoring slices by mean Dice over non-background classes, at
    most one per case so the figure shows different scans.

    Returns:
        [(mean Dice, case, index within the case), ...], worst first.
    """
    ranked = []
    for case_id, case in results.items():
        dice, _ = dice_iou(case["intersection"], case["total"])
        for index, row in enumerate(dice[:, 1:]):
            # Slices with no foreground in either map have no defined score.
            if not np.isnan(row).all():
                ranked.append((np.nanmean(row), case_id, index))
    ranked.sort()
    chosen, seen = [], set()
    for score, case_id, index in ranked:
        if case_id not in seen:
            chosen.append((score, case_id, index))
            seen.add(case_id)
        if len(chosen) == count:
            break
    return chosen


def plot_worst(results, chosen, names, path):
    """
    One row per slice: input, ground truth, prediction, and an overlay of false
    positives (red) and false negatives (blue) for that slice's worst class.
    """
    colours = ListedColormap(COLOURS[:len(names)])
    height, width = results[chosen[0][1]]["masks"].shape[1:]
    # Row height follows the slice shape: square OASIS slices, tall 256x128 HipMRI slices.
    fig, axes = plt.subplots(len(chosen), 4, figsize=(13, 3.3 * height / width * len(chosen)), squeeze=False)
    for row, (score, case_id, index) in zip(axes, chosen):
        case = results[case_id]
        image, mask, pred = case["images"][index], case["masks"][index], case["preds"][index]
        dice, _ = dice_iou(case["intersection"][index], case["total"][index])
        worst = 1 + int(np.nanargmin(dice[1:]))
        # Rescale to [0, 1] for the RGB overlay; HipMRI slices are standardised, not bounded.
        grey = (image - image.min()) / (image.max() - image.min() + 1e-8)
        overlay = np.repeat(grey[..., None], 3, axis=-1)
        overlay[(pred == worst) & (mask != worst)] = (1, 0, 0)
        overlay[(mask == worst) & (pred != worst)] = (0, 0.4, 1)
        dice_text = " ".join("-" if np.isnan(d) else f"{d:.2f}" for d in dice)
        panels = [
            (grey, "gray", 1, f"{case_name(case_id)}, slice {case['slices'][index]}"),
            (mask, colours, len(names) - 1, "Ground truth"),
            (pred, colours, len(names) - 1, f"Prediction, Dice {dice_text}"),
            (overlay, None, 1, f"{names[worst]}: FP red, FN blue"),
        ]
        for ax, (values, cmap, vmax, title) in zip(row, panels):
            # Nearest-neighbour display so label maps are not blurred into
            # colours that belong to no class.
            ax.imshow(values, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours(i)) for i in range(len(names))]
    fig.legend(handles, names, loc="lower center", ncol=len(names))
    fig.suptitle(f"Worst test slices by mean foreground Dice (Dice order: {', '.join(names)})")
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_distribution(per_case_dice, names, target, path):
    """Box plot of per-case Dice for each class, with every case drawn as a point."""
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 2, 4))
    # Classes absent from a case have NaN Dice; leave them out of that box.
    ax.boxplot([d[~np.isnan(d)] for d in per_case_dice.T], showfliers=False)
    ax.set_xticks(range(1, len(names) + 1), names)
    for c in range(len(names)):
        jitter = np.random.default_rng(c).uniform(-0.08, 0.08, len(per_case_dice))
        ax.scatter(c + 1 + jitter, per_case_dice[:, c], s=12, alpha=0.7)
    ax.axhline(target, color="grey", linewidth=0.8)  # the task's Dice target
    ax.set(title=f"Per-case test Dice ({len(per_case_dice)} cases)", ylabel="Dice")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_position(rows, thirds, target, path):
    """Prostate Dice of each axial level against position from apex to base, with the pooled Dice of each third."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in sorted({case for _, _, case in rows}):
        points = np.array([(p, d) for p, d, case in rows if case == name])
        ax.scatter(*points.T, s=10, alpha=0.6, label=name)
    for i, (name, dice) in enumerate(thirds.items()):
        ax.hlines(dice, i / 3, (i + 1) / 3, color="black", linewidth=2)
        ax.text((i + 0.5) / 3, 0.02, f"{name} {dice:.2f}", ha="center")
    ax.axhline(target, color="grey", linewidth=0.8)
    ax.set(title="Prostate Dice per axial level, apex (0) to base (1)", xlabel="Position in gland", ylabel="Dice",
           ylim=(0, 1.02))
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------- HipMRI 3D

# The 2D HipMRI slices are the coronal sections 98-157 of each volume's
# front-back axis (verified voxel for voxel by dataset.check_hipmri_2d_3d), so
# 2D and 3D predictions can be compared on exactly these voxels.
SLAB = slice(98, 158)
# 26-connectivity: voxels touching at a face, edge or corner count as connected.
CONNECTIVITY_3D = np.ones((3, 3, 3), dtype=int)
# Acceptance limits from the approved feasibility review (AC3-AC5).
MAX_LEAKAGE, MAX_TRAINING_VRAM_GB, MAX_TRAINING_MINUTES, MAX_INFERENCE_SECONDS = 0.01, 10, 60, 1


def predict_volume(model, image, shape, device):
    """
    Downsampled image (1, x, y, z) -> uint8 labels at full resolution (shape).

    Logits are upsampled trilinearly before the argmax, so the 3D model is scored
    against the original masks rather than the majority-vote downsampled ones,
    whose organ sizes depend on the tie rule (about +3% with ties to the smaller organ).
    """
    logits = infer(model, image[None].to(device), device)
    logits = F.interpolate(logits, size=shape, mode="trilinear", align_corners=False)
    return logits.argmax(1)[0].to(torch.uint8).cpu().numpy()


def predict_slab(model, volume, device):
    """
    2D model predictions for the coronal slab a volume's 2D slices were cut from.

    Args:
        volume: (256, 256, 128) raw image volume in stored (LPS) order.
    Returns:
        uint8 labels (256, 60, 128) in volume order.
    """
    # The 2D files hold volume[:, 98 + s, :] flipped left-right; rebuild them and
    # standardise each slice exactly as dataset.load_hipmri_image does.
    sections = np.flip(volume[:, SLAB, :], 0).transpose(1, 0, 2).astype(np.float64)
    sections = (sections - sections.mean((1, 2), keepdims=True)) / (sections.std((1, 2), keepdims=True) + 1e-8)
    images = torch.from_numpy(sections.astype(np.float32))[:, None]
    labels = infer(model, images.to(device), device).argmax(1).to(torch.uint8).cpu().numpy()
    return np.ascontiguousarray(np.flip(labels.transpose(1, 0, 2), 0))


def summarise_volumes(results, names):
    """
    Pooled Dice and IoU per class, prostate Dice by third of the gland, leakage,
    and per-case prostate Dice and connected components, for a dict of
    {case: {"masks": labels, "preds": labels}} arrays whose last axis is foot-head.
    """
    prostate, bladder, rectum = (names.index(n) for n in ("Prostate", "Bladder", "Rectum"))
    matrix = confusion(results, len(names))
    dice, iou = dice_iou(np.diag(matrix), matrix.sum(0) + matrix.sum(1))
    rows, thirds = prostate_by_position(results, prostate, bladder)
    per_case = {}
    for case, arrays in results.items():
        truth, predicted = arrays["masks"] == prostate, arrays["preds"] == prostate
        per_case[case_name(case)] = {
            "prostate_dice": 2 * (truth & predicted).sum() / max(truth.sum() + predicted.sum(), 1),
            "prostate_components": int(ndimage.label(predicted, CONNECTIVITY_3D)[1]),
        }
    summary = {
        "dice": dict(zip(names, dice)),
        "iou": dict(zip(names, iou)),
        "prostate_dice_by_position": thirds,
        "predicted_prostate_on_rectum": matrix[rectum, prostate] / max(matrix[:, prostate].sum(), 1),
        "predicted_prostate_on_bladder": matrix[bladder, prostate] / max(matrix[:, prostate].sum(), 1),
        "per_case": per_case,
    }
    return summary, rows


def plot_position_2d_3d(rows_3d, thirds_3d, rows_2d, thirds_2d, target, path):
    """Prostate Dice of each axial level from apex to base on the shared slab, 3D against 2D."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, rows, thirds, colour in (("3D", rows_3d, thirds_3d, "tab:blue"), ("2D", rows_2d, thirds_2d, "tab:orange")):
        if rows:
            ax.scatter(*np.array([(p, d) for p, d, _ in rows]).T, s=8, alpha=0.5, color=colour, label=label)
        for i, dice in enumerate(thirds.values()):
            ax.hlines(dice, i / 3, (i + 1) / 3, color=colour, linewidth=2)
    ax.axhline(target, color="grey", linewidth=0.8)
    ax.set(title="Prostate Dice per axial level on the shared coronal slab, apex (0) to base (1)",
           xlabel="Position in gland", ylabel="Dice", ylim=(0, 1.02))
    ax.title.set_fontsize(9)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def prostate_errors(grey, truth, predicted):
    """RGB overlay of prostate false positives (red) and false negatives (blue) on a [0, 1] image."""
    overlay = np.repeat(grey[..., None], 3, axis=-1)
    overlay[predicted & ~truth] = (1, 0, 0)
    overlay[truth & ~predicted] = (0, 0.4, 1)
    return overlay


def plot_worst_volumes(cases, results, slab_2d, images, names, path, crop=96):
    """
    Two rows per volume, at the apex and base failures: in the third of the gland
    nearest each end (see gland_position), the axial level with the lowest 3D
    prostate Dice. Each row shows the image, ground truth, 3D prediction and the
    3D model's prostate false positives (red) and false negatives (blue), plus
    the same overlay for the 2D model with --compare. Panels are cropped to
    crop x crop voxels around the prostate so boundary errors are visible.
    """
    colours = ListedColormap(COLOURS[:len(names)])
    prostate, bladder = names.index("Prostate"), names.index("Bladder")
    rows = []
    for case in cases:
        truth, predicted = results[case]["masks"] == prostate, results[case]["preds"] == prostate
        dice = 2 * (truth & predicted).sum((0, 1)) / np.maximum(truth.sum((0, 1)) + predicted.sum((0, 1)), 1)
        located = gland_position(results[case]["masks"], prostate, bladder)
        if located is None:
            rows.append((case, truth.shape[2] // 2, "centre"))
            continue
        position, present = located
        # Same thirds as prostate_by_position.
        for end, third in (("apex", position < 1 / 3), ("base", position >= 2 / 3)):
            levels = np.flatnonzero(present & third)
            if len(levels):
                rows.append((case, int(levels[np.argmin(dice[levels])]), end))

    columns_ = 5 if slab_2d else 4
    fig, axes = plt.subplots(len(rows), columns_, figsize=(2.6 * columns_, 2.9 * len(rows)), squeeze=False)
    for row, (case, z, end) in zip(axes, rows):
        mask, pred = results[case]["masks"], results[case]["preds"]
        # Crop window centred on the prostate in plane, kept inside the volume.
        centre = np.argwhere((mask == prostate).any(2)).mean(0).astype(int) if (mask == prostate).any() \
            else np.array(mask.shape[:2]) // 2
        x0, y0 = (int(np.clip(c - crop // 2, 0, size - crop)) for c, size in zip(centre, mask.shape[:2]))
        window = (slice(x0, x0 + crop), slice(y0, y0 + crop), z)
        image = images[case][window].astype(np.float64)
        grey = (image - image.min()) / (image.max() - image.min() + 1e-8)
        truth, predicted = mask[window] == prostate, pred[window] == prostate
        level_dice = 2 * (truth & predicted).sum() / max(truth.sum() + predicted.sum(), 1)
        panels = [(grey, "gray", 1, f"{case_name(case)}, {end}, axial level {z}"),
                  (mask[window], colours, len(names) - 1, "Ground truth"),
                  (pred[window], colours, len(names) - 1, "3D prediction"),
                  (prostate_errors(grey, truth, predicted), None, 1, f"3D prostate Dice {level_dice:.2f}: FP red, FN blue")]
        if slab_2d:
            # Outside the slab the 2D model never predicted anything; count it as background.
            band = np.zeros_like(mask[:, :, z])
            band[:, SLAB] = slab_2d[case]["preds"][:, :, z]
            predicted_2d = band[window[:2]] == prostate
            dice_2d = 2 * (truth & predicted_2d).sum() / max(truth.sum() + predicted_2d.sum(), 1)
            panels.append((prostate_errors(grey, truth, predicted_2d), None, 1, f"2D prostate Dice {dice_2d:.2f}: FP red, FN blue"))
        for ax, (values, cmap, vmax, title) in zip(row, panels):
            # Radiological view: front of the body at the top, patient's right on the left.
            ax.imshow(values.transpose(1, 0, *range(2, values.ndim)), cmap=cmap, vmin=0, vmax=vmax,
                      interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours(i)) for i in range(len(names))]
    fig.legend(handles, names, loc="lower center", ncol=len(names), fontsize=8)
    fig.suptitle("Test volumes with the lowest full-resolution prostate Dice: worst apex and base levels", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate_volumes(args, model, checkpoint, device, out):
    """
    Score a 3D HipMRI checkpoint at full resolution on every test volume, and,
    with --compare, a 2D checkpoint on the same scans' coronal slab. Writes
    metrics.json (including the acceptance criteria), per_case.csv and figures.
    """
    settings = checkpoint["settings"]
    spec = DATASETS["hipmri3d"]
    names = spec["classes"]
    factor = settings.get("downsample", 2)
    pairs = spec["pairs"](args.data_root or spec["root"])[args.split][:args.limit or None]
    model_2d = load_model(args.compare, device)[0] if args.compare else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    full, slab_3d, slab_2d, images, seconds = {}, {}, {}, {}, []
    for image_path, mask_path in pairs:
        case = hipmri3d_key(mask_path)
        mask = load_hipmri3d_mask(mask_path, 1).numpy().astype(np.uint8)
        image = load_hipmri3d_image(image_path, factor)
        if not seconds:
            predict_volume(model, image, mask.shape, device)  # warm-up, as in evaluate()
        synchronise(device)
        start = time.perf_counter()
        pred = predict_volume(model, image, mask.shape, device)
        synchronise(device)
        seconds.append(time.perf_counter() - start)
        full[case] = {"masks": mask, "preds": pred}
        slab_3d[case] = {"masks": mask[:, SLAB], "preds": pred[:, SLAB]}
        volume = load_volume(image_path)
        images[case] = volume
        if model_2d is not None:
            slab_2d[case] = {"masks": mask[:, SLAB], "preds": predict_slab(model_2d, volume, device)}

    summary_full, _ = summarise_volumes(full, names)
    summary_3d, rows_3d = summarise_volumes(slab_3d, names)
    summary_2d, rows_2d = summarise_volumes(slab_2d, names) if slab_2d else (None, [])

    run = args.checkpoint.parent
    history = list(csv.DictReader((run / "history.csv").open(newline=""))) if (run / "history.csv").exists() else []
    training = json.loads((run / "test.json").read_text()) if (run / "test.json").exists() else {}
    profile = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "input_size": list(image.shape[1:]),
        "inference_peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
        "inference_seconds_per_volume": float(np.mean(seconds)),
        "training_minutes": sum(float(row["seconds"]) for row in history) / 60 if history else None,
        "training_peak_vram_gb": training.get("peak_vram_gb"),
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
    }

    foreground = names[1:]
    single = sum(c["prostate_components"] == 1 for c in summary_full["per_case"].values())
    acceptance = {
        "AC1_dice_every_foreground_label": {
            "lowest": min(summary_full["dice"][n] for n in foreground),
            "pass": all(summary_full["dice"][n] >= TARGETS["hipmri3d"] for n in foreground)},
        "AC3_leakage": {
            "rectum": summary_full["predicted_prostate_on_rectum"], "bladder": summary_full["predicted_prostate_on_bladder"],
            "pass": bool(max(summary_full["predicted_prostate_on_rectum"], summary_full["predicted_prostate_on_bladder"]) < MAX_LEAKAGE)},
        # "At least 8 of 9" in the review: every volume but one.
        "AC4_single_prostate_component": {"volumes": single, "of": len(full), "pass": single >= len(full) - 1},
        "AC5_resources": {
            "training_peak_vram_gb": profile["training_peak_vram_gb"], "training_minutes": profile["training_minutes"],
            "inference_seconds_per_volume": profile["inference_seconds_per_volume"],
            "pass": (profile["training_peak_vram_gb"] or 0) <= MAX_TRAINING_VRAM_GB
                    and (profile["training_minutes"] or 0) <= MAX_TRAINING_MINUTES
                    and profile["inference_seconds_per_volume"] <= MAX_INFERENCE_SECONDS},
    }
    if summary_2d:
        third_3d, third_2d = summary_3d["prostate_dice_by_position"], summary_2d["prostate_dice_by_position"]
        acceptance["AC2_apex_base_vs_2d"] = {
            "3d": third_3d, "2d": third_2d,
            "pass": bool(third_3d["apex"] >= third_2d["apex"] and third_3d["base"] >= third_2d["base"])}

    metrics = {
        "model": settings["model"], "dataset": "hipmri3d", "best_epoch": checkpoint["epoch"], "split": args.split,
        "compare_2d": str(args.compare) if args.compare else None, "cases": len(full),
        "full_volume": summary_full, "slab_3d": summary_3d, "slab_2d": summary_2d,
        "acceptance": acceptance, "profile": profile,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    with (out / "per_case.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["patient", "week", "prostate_dice_full", "prostate_components", "prostate_dice_slab_3d",
                         "prostate_dice_slab_2d", "latency_ms"])
        for case, latency in zip(full, seconds):
            name = case_name(case)
            writer.writerow([*case, round(summary_full["per_case"][name]["prostate_dice"], 5),
                             summary_full["per_case"][name]["prostate_components"],
                             round(summary_3d["per_case"][name]["prostate_dice"], 5),
                             round(summary_2d["per_case"][name]["prostate_dice"], 5) if summary_2d else "",
                             round(latency * 1000, 2)])

    plot_position_2d_3d(rows_3d, summary_3d["prostate_dice_by_position"], rows_2d,
                        summary_2d["prostate_dice_by_position"] if summary_2d else {}, TARGETS["hipmri3d"],
                        out / "prostate_position.png")
    worst = sorted(full, key=lambda c: summary_full["per_case"][case_name(c)]["prostate_dice"])[:args.worst]
    plot_worst_volumes(worst, full, slab_2d, images, names, out / "worst_cases.png")

    print(f"{settings['model']} 3D (epoch {checkpoint['epoch']}) on hipmri3d {args.split}: {len(full)} volumes, "
          f"input {profile['input_size']}, scored at full resolution")
    print(f"{'class':10s} {'Dice full':>9s} {'IoU full':>9s} {'Dice slab 3D':>13s} {'Dice slab 2D':>13s}")
    for n in names:
        print(f"{n:10s} {summary_full['dice'][n]:9.4f} {summary_full['iou'][n]:9.4f} {summary_3d['dice'][n]:13.4f} "
              + (f"{summary_2d['dice'][n]:13.4f}" if summary_2d else f"{'-':>13s}"))
    for key, value in acceptance.items():
        print(f"{key}: {'PASS' if value['pass'] else 'FAIL'}  {json.dumps({k: v for k, v in value.items() if k != 'pass'}, default=float)}")
    print("profile", json.dumps(profile, default=float))
    print(f"saved to {out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt written by train.py")
    parser.add_argument("--data-root", type=Path, help="dataset folder (default: the dataset's Rangpur folder)")
    parser.add_argument("--split", choices=("validate", "test"), default="test")
    parser.add_argument("--out", type=Path, help="output folder (default: eval/ next to the checkpoint)")
    parser.add_argument("--worst", type=int, default=5, help="number of worst slices (volumes for hipmri3d) to plot")
    parser.add_argument("--limit", type=int, default=0, help="use only N slices or volumes (smoke tests)")
    parser.add_argument("--compare", type=Path,
                        help="hipmri3d only: a HipMRI 2D best.pt to score on the same scans' coronal slab")
    return parser.parse_args()


def main():
    args = parse_args()
    out = args.out or args.checkpoint.parent / "eval"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model(args.checkpoint, device)
    name, dataset = checkpoint["settings"]["model"], checkpoint["settings"]["dataset"]
    if dataset == "hipmri3d":
        evaluate_volumes(args, model, checkpoint, device, out)
        return
    spec = DATASETS[dataset]
    names, target = spec["classes"], TARGETS[dataset]
    pairs = spec["pairs"](args.data_root or spec["root"])[args.split][:args.limit or None]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    results = evaluate(model, pairs, device, spec)

    cases = sorted(results)
    per_case = [dice_iou(results[c]["intersection"].sum(0), results[c]["total"].sum(0)) for c in cases]
    per_case_dice = np.array([dice for dice, _ in per_case])
    per_case_iou = np.array([iou for _, iou in per_case])
    pooled_dice, pooled_iou = dice_iou(
        sum(results[c]["intersection"].sum(0) for c in cases), sum(results[c]["total"].sum(0) for c in cases))

    with (out / "per_case.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        names_csv = columns(names)
        writer.writerow([*("patient", "week")[:len(cases[0])], *(f"dice_{c}" for c in names_csv),
                         *(f"iou_{c}" for c in names_csv), "latency_ms"])
        for case, dice, iou in zip(cases, per_case_dice, per_case_iou):
            writer.writerow([*case, *dice.round(5), *iou.round(5), round(results[case]["seconds"] * 1000, 2)])

    latencies = np.array([results[c]["seconds"] for c in cases])
    slices_per_case = np.mean([len(results[c]["slices"]) for c in cases])
    profile = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "inference_peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
        "latency_per_case_ms": latencies.mean() * 1000,
        "latency_per_slice_ms": latencies.mean() * 1000 / slices_per_case,
        "slices_per_case": slices_per_case,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
    }
    run = args.checkpoint.parent
    if (run / "history.csv").exists():
        with (run / "history.csv").open(newline="") as file:
            seconds = [float(row["seconds"]) for row in csv.DictReader(file)]
        profile["training_epochs"], profile["training_minutes"] = len(seconds), sum(seconds) / 60
    if (run / "test.json").exists():
        profile["training_peak_vram_gb"] = json.loads((run / "test.json").read_text()).get("peak_vram_gb")

    matrix = confusion(results, len(names))
    summary = {
        "model": name,
        "dataset": dataset,
        "best_epoch": checkpoint["epoch"],
        "split": args.split,
        "cases": len(cases),
        "dice_pooled": dict(zip(names, pooled_dice)),
        "iou_pooled": dict(zip(names, pooled_iou)),
        "dice_per_case_mean": dict(zip(names, np.nanmean(per_case_dice, 0))),
        "dice_per_case_std": dict(zip(names, np.nanstd(per_case_dice, 0))),
        "dice_per_case_min": dict(zip(names, np.nanmin(per_case_dice, 0))),
        "iou_per_case_mean": dict(zip(names, np.nanmean(per_case_iou, 0))),
        "confusion_pixels": {"rows_truth_columns_prediction": list(names), "counts": matrix.tolist()},
        "connectivity": connectivity(results, names),
        "profile": profile,
    }
    if dataset == "hipmri":
        prostate, bladder, rectum = (names.index(n) for n in ("Prostate", "Bladder", "Rectum"))
        rows, thirds = prostate_by_position(results, prostate, bladder)
        summary["prostate_dice_by_position"] = thirds
        # Leakage across the thin organ interfaces, as shares of the prostate pixels involved.
        summary["prostate_leakage"] = {
            "predicted_prostate_on_rectum": matrix[rectum, prostate] / matrix[:, prostate].sum(),
            "predicted_prostate_on_bladder": matrix[bladder, prostate] / matrix[:, prostate].sum(),
            "true_prostate_predicted_rectum": matrix[prostate, rectum] / matrix[prostate].sum(),
            "true_prostate_predicted_bladder": matrix[prostate, bladder] / matrix[prostate].sum(),
        }
        summary["prostate_slices"] = prostate_slices(results, prostate)
        plot_position(rows, thirds, target, out / "prostate_position.png")
    chosen = worst_slices(results, args.worst)
    summary["worst_slices"] = [{"case": case_name(c), "slice": int(results[c]["slices"][i]), "mean_foreground_dice": s}
                               for s, c, i in chosen]
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    plot_distribution(per_case_dice, names, target, out / "dice_distribution.png")
    plot_worst(results, chosen, names, out / "worst_cases.png")

    print(f"{name} (epoch {checkpoint['epoch']}) on {dataset} {args.split}: {len(cases)} cases")
    print(f"{'class':14s} {'Dice':>7s} {'IoU':>7s} {'case mean':>10s} {'case std':>9s} {'case min':>9s}")
    for c, class_name in enumerate(names):
        print(f"{class_name:14s} {pooled_dice[c]:7.4f} {pooled_iou[c]:7.4f} {np.nanmean(per_case_dice[:, c]):10.4f} "
              f"{np.nanstd(per_case_dice[:, c]):9.4f} {np.nanmin(per_case_dice[:, c]):9.4f}")
    print("connectivity (components per slice, truth -> prediction):")
    for class_name, stats in summary["connectivity"].items():
        print(f"  {class_name:14s} {stats['truth_components_per_slice']:.2f} -> "
              f"{stats['predicted_components_per_slice']:.2f}  fragmented on "
              f"{stats['slices_more_components']:.0%} of slices, merged/missing on {stats['slices_fewer_components']:.0%}")
    for key in ("prostate_dice_by_position", "prostate_leakage", "prostate_slices"):
        if key in summary:
            print(key, json.dumps(summary[key], default=float))
    print("profile", json.dumps(profile, default=float))
    print("worst slices", json.dumps(summary["worst_slices"], default=float))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
