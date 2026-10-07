# Segmentation networks for 2D slices and 3D volumes.
#
#   UNet          plain U-Net, the baseline. Ronneberger et al., "U-Net:
#                 Convolutional Networks for Biomedical Image Segmentation",
#                 MICCAI 2015.
#   ImprovedUNet  Improved U-Net following Isensee et al., "Brain Tumor
#                 Segmentation and Radiomics Survival Prediction: Contribution
#                 to the BRATS 2017 Challenge", 2018: residual context modules,
#                 strided-conv downsampling, localization modules, instance norm,
#                 leaky ReLU and deep supervision.
#
# The spatial dimension is a constructor argument (dims=2 or 3) that selects the
# matching convolution, normalisation, dropout and pooling layers, so the 3D
# networks of Cicek et al., "3D U-Net: Learning Dense Volumetric Segmentation
# from Sparse Annotation", MICCAI 2016, are the same code with dims=3.
#
# Both models take (B, C_in, *spatial) and return (B, num_classes, *spatial)
# logits; softmax is left to the loss. Every spatial size must be divisible by
# 16. Pure PyTorch, as the spec requires for this file.
#
# Self-check:
#     python modules.py

import torch
from torch import nn
from torch.nn import functional as F


def layer(name, dims):
    """
    Return the torch.nn class for a layer type in the given spatial dimension.

    Args:
        name: layer family without the suffix, e.g. "Conv", "InstanceNorm", "Dropout".
        dims: 2 for slices (H, W) or 3 for volumes (D, H, W).
    Returns:
        The class, e.g. layer("Conv", 3) is nn.Conv3d.
    """
    if dims not in (2, 3):
        raise ValueError(f"dims must be 2 or 3, got {dims}")
    return getattr(nn, f"{name}{dims}d")


# ---------------------------------------------------------------- baseline

def double_conv(cin, cout, dims):
    """
    Two (3x3 conv, BatchNorm, ReLU) layers, the repeating unit of a U-Net.

    Args:
        cin, cout: input and output channels.
        dims: spatial dimension, 2 or 3.
    Returns:
        nn.Sequential mapping (B, cin, *S) to (B, cout, *S).
    """
    Conv, Norm = layer("Conv", dims), layer("BatchNorm", dims)
    # bias=False because the BatchNorm that follows supplies its own shift.
    return nn.Sequential(
        Conv(cin, cout, 3, padding=1, bias=False), Norm(cout), nn.ReLU(inplace=True),
        Conv(cout, cout, 3, padding=1, bias=False), Norm(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """
    Baseline U-Net: four max-pool levels down to 1/16 resolution, transposed-conv
    upsampling and concatenated skip connections.

    Args:
        in_channels: image channels (1 for MRI).
        num_classes: output classes, one logit channel each.
        base: channels at the first level; doubled at each level below.
        dims: spatial dimension, 2 or 3.

    forward: (B, in_channels, *S) -> (B, num_classes, *S) logits.
    """

    def __init__(self, in_channels=1, num_classes=4, base=32, dims=2):
        super().__init__()
        widths = [base * 2 ** i for i in range(4)]
        UpConv = layer("ConvTranspose", dims)
        self.down = nn.ModuleList(double_conv(cin, cout, dims) for cin, cout in zip([in_channels] + widths, widths))
        self.pool = layer("MaxPool", dims)(2)
        self.bottleneck = double_conv(widths[-1], widths[-1] * 2, dims)
        self.up = nn.ModuleList(UpConv(w * 2, w, 2, stride=2) for w in reversed(widths))
        # w * 2 inputs: the skip is concatenated onto the upsampled features.
        self.refine = nn.ModuleList(double_conv(w * 2, w, dims) for w in reversed(widths))
        self.head = layer("Conv", dims)(base, num_classes, 1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            # Keep full-resolution features before pooling: the bottleneck knows
            # which tissue is present but has lost where its edges are, and the
            # skips restore that detail in the decoder.
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, refine, skip in zip(self.up, self.refine, reversed(skips)):
            x = refine(torch.cat([skip, up(x)], dim=1))
        return self.head(x)


# ---------------------------------------------------------- improved U-Net

def conv_norm_act(cin, cout, dims, kernel=3, stride=1):
    """
    Conv, InstanceNorm, LeakyReLU(0.01), the basic block of the Improved U-Net.

    Instance norm normalises each sample on its own, so it stays stable with the
    small batches that 3D volumes force.

    Args:
        cin, cout: input and output channels.
        dims: spatial dimension, 2 or 3.
        kernel: convolution size (3, or 1 for channel mixing).
        stride: 2 halves every spatial size, replacing max-pooling.
    Returns:
        nn.Sequential mapping (B, cin, *S) to (B, cout, *S / stride).
    """
    return nn.Sequential(
        layer("Conv", dims)(cin, cout, kernel, stride, padding=kernel // 2, bias=False),
        layer("InstanceNorm", dims)(cout, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class ContextModule(nn.Module):
    """
    Pre-activation residual block: two (norm, leaky ReLU, 3x3 conv) with dropout
    between them, added back to the input.

    Args:
        channels: input and output channels.
        dims: spatial dimension, 2 or 3.
        dropout: channel dropout probability (0.3 in the paper).

    forward: (B, channels, *S) -> (B, channels, *S).
    """

    def __init__(self, channels, dims, dropout=0.3):
        super().__init__()
        Conv, Norm = layer("Conv", dims), layer("InstanceNorm", dims)
        self.body = nn.Sequential(
            Norm(channels, affine=True), nn.LeakyReLU(0.01),
            Conv(channels, channels, 3, padding=1),
            layer("Dropout", dims)(dropout),
            Norm(channels, affine=True), nn.LeakyReLU(0.01),
            Conv(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        # The residual path lets each level learn a correction to its input,
        # which keeps gradients flowing through the deep encoder.
        return x + self.body(x)


def upsampling_module(cin, cout, dims):
    """
    Nearest-neighbour x2 upsample then a 3x3 conv, as in the paper (cheaper than a transposed conv).

    Returns:
        nn.Sequential mapping (B, cin, *S) to (B, cout, *2S).
    """
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), conv_norm_act(cin, cout, dims))


def localization_module(channels, dims):
    """
    3x3 conv on the concatenated skip and upsampled features, then a 1x1 conv halving the channels.

    Returns:
        nn.Sequential mapping (B, channels, *S) to (B, channels // 2, *S).
    """
    return nn.Sequential(conv_norm_act(channels, channels, dims), conv_norm_act(channels, channels // 2, dims, kernel=1))


class ImprovedUNet(nn.Module):
    """
    Improved U-Net with five levels of widths base * (1, 2, 4, 8, 16).

    Encoder level: 3x3 conv (stride 2 below the first level) then a context
    module. Decoder level: upsampling module, concatenate the skip, localization
    module. The finest level uses a plain 3x3 conv in place of localization.

    Args:
        in_channels: image channels (1 for MRI).
        num_classes: output classes, one logit channel each.
        base: channels at the first level (16 in the paper).
        dropout: context-module dropout probability.
        dims: spatial dimension, 2 or 3.

    forward: (B, in_channels, *S) -> (B, num_classes, *S) logits.
    """

    def __init__(self, in_channels=1, num_classes=4, base=16, dropout=0.3, dims=2):
        super().__init__()
        w = [base * 2 ** i for i in range(5)]
        self.encode = nn.ModuleList(
            nn.Sequential(conv_norm_act(cin, cout, dims, stride=1 if i == 0 else 2), ContextModule(cout, dims, dropout))
            for i, (cin, cout) in enumerate(zip([in_channels] + w, w))
        )
        # Decoder from the bottom up: 16b->8b, 8b->4b, 4b->2b, 2b->b.
        self.upsample = nn.ModuleList(upsampling_module(w[i], w[i - 1], dims) for i in range(4, 0, -1))
        # After concatenating the skip each level has w[i] channels again; localization halves them.
        self.localize = nn.ModuleList(localization_module(w[i], dims) for i in range(4, 1, -1))
        self.final = conv_norm_act(w[1], w[1], dims)
        self.heads = nn.ModuleList(layer("Conv", dims)(c, num_classes, 1) for c in (w[2], w[1], w[1]))

    def forward(self, x):
        skips = []
        for level in self.encode:
            x = level(x)
            skips.append(x)
        x = skips.pop()

        # Deep supervision: the three finest decoder levels each predict a
        # segmentation, and each coarser prediction is upsampled and added to the
        # next. The coarse levels then receive gradient straight from the loss
        # rather than only through the layers above them, which speeds up and
        # stabilises training.
        segmentation = None
        for i, up in enumerate(self.upsample):
            x = torch.cat([skips.pop(), up(x)], dim=1)
            x = self.localize[i](x) if i < len(self.localize) else self.final(x)
            if i == 0:
                continue  # the deepest decoder level has no segmentation head
            head = self.heads[i - 1](x)
            if segmentation is not None:
                head = head + F.interpolate(segmentation, scale_factor=2, mode="nearest")
            segmentation = head
        return segmentation


if __name__ == "__main__":
    torch.manual_seed(0)
    # 2D: one OASIS-sized slice batch. 3D: a small volume, divisible by 16 on every axis.
    inputs = {2: torch.randn(2, 1, 256, 256), 3: torch.randn(1, 1, 32, 64, 64)}
    for dims, images in inputs.items():
        for model in (UNet(dims=dims), ImprovedUNet(dims=dims)):
            logits = model(images)
            assert logits.shape == (images.shape[0], 4, *images.shape[2:]), logits.shape
            logits.mean().backward()
            unused = [n for n, p in model.named_parameters() if p.grad is None]
            assert not unused, f"parameters without gradient: {unused}"
            params = sum(p.numel() for p in model.parameters())
            print(f"{dims}D {type(model).__name__:12s} {params / 1e6:6.2f}M parameters  "
                  f"{tuple(images.shape)} -> {tuple(logits.shape)}")

        # Dropout must be off in eval mode, so inference is deterministic.
        model = ImprovedUNet(dims=dims).eval()
        with torch.no_grad():
            assert torch.equal(model(images), model(images))

    try:
        UNet(dims=1)
    except ValueError:
        pass
    else:
        raise AssertionError("dims=1 should be rejected")
    print("modules self-check passed")
