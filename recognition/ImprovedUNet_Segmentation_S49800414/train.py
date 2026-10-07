# Training, validation and testing for the OASIS segmentation models.
#
# Trains the baseline U-Net or the Improved U-Net (selected with --model) with
# cross-entropy plus multi-class soft Dice loss, keeps the checkpoint with the
# best mean validation Dice, plots the curves every epoch and finally scores the
# best checkpoint on the test split.
#
# Optimiser settings follow Isensee et al., "Brain Tumor Segmentation and
# Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge", 2018:
# Adam, initial learning rate 5e-4 decayed by 0.985 per epoch, L2 weight decay
# 1e-5, and a Dice-based loss for class imbalance.
#
# Usage:
#     python train.py --model improved
#     python train.py --model baseline --epochs 2 --limit 64   # smoke test

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

# Headless backend: Slurm jobs have no display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import CLASS_NAMES, ROOT, get_loaders, one_hot
from modules import ImprovedUNet, UNet

MODELS = {"baseline": UNet, "improved": ImprovedUNet}
HERE = Path(__file__).resolve().parent
COLUMNS = [name.lower().replace(" ", "_") for name in CLASS_NAMES]


def dice_loss(logits, masks, smooth=1.0):
    """
    Multi-class soft Dice loss.

    Args:
        logits: (B, C, *S) raw network outputs.
        masks: (B, *S) int64 class indices.
        smooth: added to numerator and denominator so a class absent from the
            batch gives Dice 1 instead of 0/0.
    Returns:
        Scalar tensor, 1 minus the soft Dice averaged over classes.
    """
    # Softmax probabilities instead of a hard argmax keep the loss differentiable.
    # Computed in float32 because bfloat16 loses precision when summing over a whole batch.
    probs = logits.float().softmax(1)
    target = one_hot(masks)
    sum_dims = (0, *range(2, probs.ndim))
    intersection = (probs * target).sum(sum_dims)
    total = probs.sum(sum_dims) + target.sum(sum_dims)
    # Averaging the per-class Dice gives every class equal weight however few
    # pixels it covers, so CSF (about 6% of pixels) counts as much as background
    # (about 72%). Cross-entropy alone would be dominated by the background.
    return 1 - ((2 * intersection + smooth) / (total + smooth)).mean()


def dice_counts(logits, masks):
    """
    Ingredients of the hard (argmax) Dice score for each sample and class.

    Args:
        logits: (B, C, *S) raw network outputs.
        masks: (B, *S) int64 class indices.
    Returns:
        (intersection, total), each (B, C): |P and G| and |P| + |G|. Returning the
        parts lets callers pool them over a whole split before dividing.
    """
    pred, target = one_hot(logits.argmax(1)), one_hot(masks)
    sum_dims = tuple(range(2, pred.ndim))
    return (pred * target).sum(sum_dims), (pred + target).sum(sum_dims)


def run_epoch(model, loader, device, optimiser=None):
    """
    One pass over a loader; trains when an optimiser is given, otherwise evaluates.

    Returns:
        (mean loss per sample, per-class Dice tensor (C,) pooled over the whole split).
    """
    training = optimiser is not None
    model.train(training)
    loss_sum, samples, intersection, total = 0.0, 0, 0, 0
    with torch.set_grad_enabled(training):
        for images, masks in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            # Mixed precision: bfloat16 matmuls are much faster on the A100 and,
            # unlike float16, keep float32's exponent range, so no gradient
            # scaler is needed.
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(images)
            loss = F.cross_entropy(logits.float(), masks) + dice_loss(logits, masks)
            if training:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
            loss_sum += loss.item() * len(images)
            samples += len(images)
            batch_intersection, batch_total = dice_counts(logits.detach(), masks)
            intersection = intersection + batch_intersection.sum(0)
            total = total + batch_total.sum(0)
    return loss_sum / samples, (2 * intersection / total.clamp(min=1)).cpu()


def plot_curves(history, path):
    """Save loss curves and per-class validation Dice curves to one image."""
    epochs = [row["epoch"] for row in history]
    fig, (loss_ax, dice_ax) = plt.subplots(1, 2, figsize=(11, 4))
    for key in ("train_loss", "val_loss"):
        loss_ax.plot(epochs, [row[key] for row in history], label=key)
    loss_ax.set(title="Loss (cross-entropy + Dice)", xlabel="Epoch")
    for column, name in zip(COLUMNS, CLASS_NAMES):
        dice_ax.plot(epochs, [row[f"val_dice_{column}"] for row in history], label=name)
    dice_ax.plot(epochs, [row["val_dice_mean"] for row in history], "k--", label="Mean")
    dice_ax.axhline(0.9, color="grey", linewidth=0.8)  # the task's target on every label
    dice_ax.set(title="Validation Dice", xlabel="Epoch")
    for ax in (loss_ax, dice_ax):
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an OASIS segmentation model.")
    parser.add_argument("--model", choices=MODELS, default="improved")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, help="output folder (default outputs/<model>)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-decay", type=float, default=0.985, help="learning rate multiplier per epoch")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="use only N slices per split (smoke tests)")
    return parser.parse_args()


def main():
    args = parse_args()
    out = args.out or HERE / "outputs" / args.model
    out.mkdir(parents=True, exist_ok=True)

    # Seeds weight initialisation, shuffling and augmentation (DataLoader workers
    # derive their seeds from this). cuDNN may still pick non-deterministic
    # kernels, so GPU runs repeat closely but not bit for bit.
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}, args {vars(args)}", flush=True)

    loaders = get_loaders(args.batch_size, args.workers, args.data_root, args.limit)
    model = MODELS[args.model](dims=2).to(device)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, args.lr_decay)

    # Plain types only, so the checkpoint loads with weights_only=True.
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    history, best_dice = [], -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        lr = optimiser.param_groups[0]["lr"]
        train_loss, train_dice = run_epoch(model, loaders["train"], device, optimiser)
        val_loss, val_dice = run_epoch(model, loaders["validate"], device)
        scheduler.step()

        row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": val_loss,
               "train_dice_mean": train_dice.mean().item(),
               **{f"val_dice_{c}": d.item() for c, d in zip(COLUMNS, val_dice)},
               "val_dice_mean": val_dice.mean().item(), "seconds": time.time() - start}
        history.append(row)
        with (out / "history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(history)
        plot_curves(history, out / "curves.png")

        improved = row["val_dice_mean"] > best_dice
        if improved:
            best_dice = row["val_dice_mean"]
            torch.save({"model": model.state_dict(), "settings": settings, "epoch": epoch, "val_dice": val_dice},
                       out / "best.pt")
        print(f"epoch {epoch:3d}  loss {train_loss:.4f}/{val_loss:.4f}  val Dice "
              + " ".join(f"{d:.4f}" for d in val_dice) + f"  mean {row['val_dice_mean']:.4f}"
              + ("  *" if improved else "") + f"  {row['seconds']:.0f}s", flush=True)

    # Score the best checkpoint, not the last epoch, on the held-out test split.
    checkpoint = torch.load(out / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_dice = run_epoch(model, loaders["test"], device)
    result = {"best_epoch": checkpoint["epoch"], "test_loss": test_loss,
              "test_dice": dict(zip(CLASS_NAMES, test_dice.tolist())), "test_dice_mean": test_dice.mean().item()}
    if device.type == "cuda":
        result["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    (out / "test.json").write_text(json.dumps(result, indent=2))
    print("test", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
