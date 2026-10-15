# Test-set evaluation and visualisation for a trained 2D segmentation model.
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
#
# NumPy and SciPy are used here for metrics and plotting only; the model itself
# is pure PyTorch (modules.py).
#
# Usage:
#     python predict.py --checkpoint outputs/improved/best.pt
#     python predict.py --checkpoint outputs/hipmri_improved/best.pt

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

from dataset import DATASETS
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
    model = MODELS[settings["model"]](num_classes=num_classes, dims=2)
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


def prostate_by_position(results, prostate, bladder):
    """
    Prostate Dice along the gland, for the spec's apex and base failure mode.

    The HipMRI slices are coronal, so the foot-head axis is the last axis of the
    stacked case (N slices, rows, columns): each column index is one axial
    level. The levels with prostate in the ground truth are placed from apex (0)
    to base (1). The base is the end nearer the bladder, which lies above the
    prostate, so the direction comes from the labels rather than from an assumed
    axis order. Cases without any bladder are skipped.

    Returns:
        (per-level rows [(position 0..1, level Dice, case name)],
         {"apex" | "mid" | "base": Dice pooled over that third of the gland}).
    """
    rows, thirds = [], {name: np.zeros(2) for name in ("apex", "mid", "base")}
    for case_id, case in results.items():
        truth, predicted = case["masks"] == prostate, case["preds"] == prostate
        prostate_voxels = truth.sum((0, 1))
        bladder_voxels = (case["masks"] == bladder).sum((0, 1))
        present = prostate_voxels > 0
        if not present.any() or not bladder_voxels.any():
            continue
        levels = np.arange(len(present))
        first, last = levels[present].min(), levels[present].max()
        position = (levels - first) / max(last - first, 1)
        if np.average(levels, weights=bladder_voxels) < np.average(levels, weights=prostate_voxels):
            position = 1 - position
        intersection = (truth & predicted).sum((0, 1))
        total = prostate_voxels + predicted.sum((0, 1))
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained 2D segmentation model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="best.pt written by train.py")
    parser.add_argument("--data-root", type=Path, help="dataset folder (default: the dataset's Rangpur folder)")
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
    name, dataset = checkpoint["settings"]["model"], checkpoint["settings"]["dataset"]
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
