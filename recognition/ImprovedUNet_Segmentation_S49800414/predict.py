# Test-set evaluation and visualisation for a trained OASIS segmentation model.
#
# Loads best.pt from train.py, runs inference one patient (case) at a time and
# reports what the COMP3710 assignment spec (Section 2.2) asks for:
#   * per-class Dice and IoU for every test case, their averages and spread
#   * resource profile: parameters, peak inference VRAM, per-case latency, and
#     training time and VRAM read back from the training run
#   * a connectivity check for the open research dilemma: high global Dice can
#     hide broken or fragmented thin cortical structures, so connected components
#     per tissue class are counted in the prediction and the ground truth
#   * multi-panel figures of the worst slices: input, ground truth, prediction
#     and a false positive / false negative overlay
#
# NumPy and SciPy are used here for metrics and plotting only; the model itself
# is pure PyTorch (modules.py).
#
# Usage:
#     python predict.py --checkpoint outputs/improved/best.pt

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
from matplotlib.colors import ListedColormap
from scipy import ndimage

from dataset import CLASS_NAMES, NUM_CLASSES, ROOT, OASISDataset, slice_key, split_pairs
from train import COLUMNS, MODELS, dice_counts

# One fixed colour per class so every figure reads the same way.
CLASS_COLOURS = ListedColormap(["black", "tab:blue", "tab:orange", "tab:green"])
# 8-connectivity: pixels touching diagonally count as connected, so a thin
# cortical strand running diagonally is not counted as broken. A component
# split therefore means a real gap in the prediction.
CONNECTIVITY = np.ones((3, 3), dtype=int)


def load_model(path, device):
    """
    Rebuild a trained model from a train.py checkpoint.

    Returns:
        (model in eval mode on device, checkpoint dict with "settings" and "epoch").
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = MODELS[checkpoint["settings"]["model"]](dims=2)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def synchronise(device):
    """Wait for queued GPU work, so wall-clock timings measure the computation."""
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def infer(model, images, device):
    """(B, 1, H, W) images on device -> (B, C, H, W) float32 logits, with the same mixed precision as training."""
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return model(images).float()


def evaluate(model, pairs, device):
    """
    Predict every case (patient) in a split.

    Args:
        pairs: [(image_path, mask_path), ...] from dataset.split_pairs.
    Returns:
        {patient: dict with "slices" (N,), "images" (N, H, W) float, "masks" and
        "preds" (N, H, W) uint8, "intersection" and "total" (N, C) Dice counts,
        "seconds" inference time for the whole case}.
    """
    cases = defaultdict(list)
    for pair in pairs:
        cases[slice_key(pair[0])[0]].append(pair)

    results, warmed_up = {}, False
    for patient, case_pairs in sorted(cases.items()):
        dataset = OASISDataset(case_pairs)
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
        results[patient] = {
            "slices": np.array([slice_key(image)[1] for image, _ in case_pairs]),
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


def connectivity(results):
    """
    Connected components per tissue class, prediction against ground truth.

    Returns:
        {class name: mean components per slice in the ground truth and the
        prediction, and the share of slices where the prediction has more
        components (fragmented) or fewer (merged or missing) than the truth}.
    """
    counts = defaultdict(list)
    for case in results.values():
        for mask, pred in zip(case["masks"], case["preds"]):
            for c in range(1, NUM_CLASSES):
                truth = ndimage.label(mask == c, CONNECTIVITY)[1]
                predicted = ndimage.label(pred == c, CONNECTIVITY)[1]
                counts[c].append((truth, predicted))
    summary = {}
    for c, pairs in counts.items():
        truth, predicted = np.array(pairs).T
        summary[CLASS_NAMES[c]] = {
            "truth_components_per_slice": truth.mean(),
            "predicted_components_per_slice": predicted.mean(),
            "slices_more_components": (predicted > truth).mean(),
            "slices_fewer_components": (predicted < truth).mean(),
        }
    return summary


def worst_slices(results, count):
    """
    The count lowest-scoring slices by mean tissue Dice (background excluded),
    at most one per patient so the figure shows different brains.

    Returns:
        [(mean tissue Dice, patient, index within the case), ...], worst first.
    """
    ranked = []
    for patient, case in results.items():
        dice, _ = dice_iou(case["intersection"], case["total"])
        tissue = dice[:, 1:]
        for index, row in enumerate(tissue):
            # Slices with no tissue in either map have no defined score.
            if not np.isnan(row).all():
                ranked.append((np.nanmean(row), patient, index))
    ranked.sort()
    chosen, seen = [], set()
    for score, patient, index in ranked:
        if patient not in seen:
            chosen.append((score, patient, index))
            seen.add(patient)
        if len(chosen) == count:
            break
    return chosen


def plot_worst(results, chosen, path):
    """
    One row per slice: input, ground truth, prediction, and an overlay of false
    positives (red) and false negatives (blue) for that slice's worst tissue class.
    """
    fig, axes = plt.subplots(len(chosen), 4, figsize=(13, 3.3 * len(chosen)), squeeze=False)
    for row, (score, patient, index) in zip(axes, chosen):
        case = results[patient]
        image, mask, pred = case["images"][index], case["masks"][index], case["preds"][index]
        dice, _ = dice_iou(case["intersection"][index], case["total"][index])
        worst = 1 + int(np.nanargmin(dice[1:]))
        overlay = np.repeat(image[..., None], 3, axis=-1)
        overlay[(pred == worst) & (mask != worst)] = (1, 0, 0)
        overlay[(mask == worst) & (pred != worst)] = (0, 0.4, 1)
        dice_text = " ".join("-" if np.isnan(d) else f"{d:.3f}" for d in dice)
        panels = [
            (image, "gray", 1, f"Patient {patient:03d}, slice {case['slices'][index]}"),
            (mask, CLASS_COLOURS, NUM_CLASSES - 1, "Ground truth"),
            (pred, CLASS_COLOURS, NUM_CLASSES - 1, f"Prediction, Dice {dice_text}"),
            (overlay, None, 1, f"{CLASS_NAMES[worst]}: FP red, FN blue"),
        ]
        for ax, (values, cmap, vmax, title) in zip(row, panels):
            # Nearest-neighbour display so label maps are not blurred into
            # colours that belong to no class.
            ax.imshow(values, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOURS(i)) for i in range(NUM_CLASSES)]
    fig.legend(handles, CLASS_NAMES, loc="lower center", ncol=NUM_CLASSES)
    fig.suptitle(f"Worst test slices by mean tissue Dice (Dice order: {', '.join(CLASS_NAMES)})")
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_distribution(per_case_dice, path):
    """Box plot of per-case Dice for each class, with every case drawn as a point."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([per_case_dice[:, c] for c in range(NUM_CLASSES)], showfliers=False)
    ax.set_xticks(range(1, NUM_CLASSES + 1), CLASS_NAMES)
    for c in range(NUM_CLASSES):
        jitter = np.random.default_rng(c).uniform(-0.08, 0.08, len(per_case_dice))
        ax.scatter(c + 1 + jitter, per_case_dice[:, c], s=12, alpha=0.7)
    ax.axhline(0.9, color="grey", linewidth=0.8)  # the task's target on every label
    ax.set(title=f"Per-case test Dice ({len(per_case_dice)} cases)", ylabel="Dice")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained OASIS segmentation model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt written by train.py")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--split", choices=("validate", "test"), default="test")
    parser.add_argument("--out", type=Path, help="output folder (default: eval/ next to the checkpoint)")
    parser.add_argument("--worst", type=int, default=5, help="number of worst slices to plot")
    parser.add_argument("--limit", type=int, default=0, help="use only N slices (smoke tests)")
    return parser.parse_args()


def main():
    args = parse_args()
    out = args.out or args.checkpoint.parent / "eval"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, checkpoint = load_model(args.checkpoint, device)
    name = checkpoint["settings"]["model"]
    pairs = split_pairs(args.data_root)[args.split][:args.limit or None]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    results = evaluate(model, pairs, device)

    patients = sorted(results)
    per_case = [dice_iou(results[p]["intersection"].sum(0), results[p]["total"].sum(0)) for p in patients]
    per_case_dice = np.array([dice for dice, _ in per_case])
    per_case_iou = np.array([iou for _, iou in per_case])
    pooled_dice, pooled_iou = dice_iou(
        sum(results[p]["intersection"].sum(0) for p in patients), sum(results[p]["total"].sum(0) for p in patients))

    with (out / "per_case.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["patient", *(f"dice_{c}" for c in COLUMNS), *(f"iou_{c}" for c in COLUMNS), "latency_ms"])
        for patient, dice, iou in zip(patients, per_case_dice, per_case_iou):
            writer.writerow([patient, *dice.round(5), *iou.round(5), round(results[patient]["seconds"] * 1000, 2)])

    latencies = np.array([results[p]["seconds"] for p in patients])
    slices_per_case = np.mean([len(results[p]["slices"]) for p in patients])
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

    summary = {
        "model": name,
        "best_epoch": checkpoint["epoch"],
        "split": args.split,
        "cases": len(patients),
        "dice_pooled": dict(zip(CLASS_NAMES, pooled_dice)),
        "iou_pooled": dict(zip(CLASS_NAMES, pooled_iou)),
        "dice_per_case_mean": dict(zip(CLASS_NAMES, np.nanmean(per_case_dice, 0))),
        "dice_per_case_std": dict(zip(CLASS_NAMES, np.nanstd(per_case_dice, 0))),
        "dice_per_case_min": dict(zip(CLASS_NAMES, np.nanmin(per_case_dice, 0))),
        "iou_per_case_mean": dict(zip(CLASS_NAMES, np.nanmean(per_case_iou, 0))),
        "connectivity": connectivity(results),
        "profile": profile,
    }
    chosen = worst_slices(results, args.worst)
    summary["worst_slices"] = [{"patient": p, "slice": int(results[p]["slices"][i]), "mean_tissue_dice": s}
                               for s, p, i in chosen]
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    plot_distribution(per_case_dice, out / "dice_distribution.png")
    plot_worst(results, chosen, out / "worst_cases.png")

    print(f"{name} (epoch {checkpoint['epoch']}) on {args.split}: {len(patients)} cases")
    print(f"{'class':14s} {'Dice':>7s} {'IoU':>7s} {'case mean':>10s} {'case std':>9s} {'case min':>9s}")
    for c, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name:14s} {pooled_dice[c]:7.4f} {pooled_iou[c]:7.4f} {np.nanmean(per_case_dice[:, c]):10.4f} "
              f"{np.nanstd(per_case_dice[:, c]):9.4f} {np.nanmin(per_case_dice[:, c]):9.4f}")
    print("connectivity (components per slice, truth -> prediction):")
    for class_name, stats in summary["connectivity"].items():
        print(f"  {class_name:14s} {stats['truth_components_per_slice']:.2f} -> "
              f"{stats['predicted_components_per_slice']:.2f}  fragmented on "
              f"{stats['slices_more_components']:.0%} of slices, merged/missing on {stats['slices_fewer_components']:.0%}")
    print("profile", json.dumps(profile, default=float))
    print("worst slices", json.dumps(summary["worst_slices"], default=float))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
