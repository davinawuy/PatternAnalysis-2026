# Segmentation networks for 2D OASIS brain slices.
#
#   UNet          plain 2D U-Net, the baseline. Ronneberger et al., "U-Net:
#                 Convolutional Networks for Biomedical Image Segmentation",
#                 MICCAI 2015.
#   ImprovedUNet  Improved U-Net following Isensee et al., "Brain Tumor
#                 Segmentation and Radiomics Survival Prediction: Contribution
#                 to the BRATS 2017 Challenge", 2018: residual context modules,
#                 strided-conv downsampling, localization modules, instance norm,
#                 leaky ReLU and deep supervision.
#
# Both take (B, 1, H, W) images and return (B, num_classes, H, W) logits; softmax
# is left to the loss. H and W must be divisible by 16. Pure PyTorch, as the spec
# requires for this file.
#
# Self-check:
#     python modules.py

import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------- baseline

def double_conv(cin, cout):
    """Two 3x3 conv -> BatchNorm -> ReLU, the repeating unit of a U-Net."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Four max-pool levels down to 1/16 resolution, transposed-conv upsampling, concatenated skips."""

    def __init__(self, in_channels=1, num_classes=4, base=32):
        super().__init__()
        widths = [base * 2 ** i for i in range(4)]
        self.down = nn.ModuleList(double_conv(cin, cout) for cin, cout in zip([in_channels] + widths, widths))
        self.bottleneck = double_conv(widths[-1], widths[-1] * 2)
        self.up = nn.ModuleList(nn.ConvTranspose2d(w * 2, w, 2, stride=2) for w in reversed(widths))
        # w * 2 inputs: the skip is concatenated onto the upsampled features.
        self.refine = nn.ModuleList(double_conv(w * 2, w) for w in reversed(widths))
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, refine, skip in zip(self.up, self.refine, reversed(skips)):
            x = refine(torch.cat([skip, up(x)], dim=1))
        return self.head(x)


# ---------------------------------------------------------- improved U-Net

def conv_norm_act(cin, cout, kernel=3, stride=1):
    """Conv -> InstanceNorm -> LeakyReLU(0.01). Instance norm because the paper trains with small batches."""
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride, padding=kernel // 2, bias=False),
        nn.InstanceNorm2d(cout, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class ContextModule(nn.Module):
    """Pre-activation residual block: two (norm, leaky ReLU, 3x3 conv) with dropout between, added to the input."""

    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.body = nn.Sequential(
            nn.InstanceNorm2d(channels, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.Dropout2d(dropout),
            nn.InstanceNorm2d(channels, affine=True), nn.LeakyReLU(0.01),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.body(x)


def upsampling_module(cin, cout):
    """Nearest-neighbour x2 upsample then a 3x3 conv halving the channels."""
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), conv_norm_act(cin, cout))


def localization_module(channels):
    """3x3 conv on the concatenated features, then a 1x1 conv halving the channels."""
    return nn.Sequential(conv_norm_act(channels, channels), conv_norm_act(channels, channels // 2, kernel=1))


class ImprovedUNet(nn.Module):
    """
    Five levels with widths base * (1, 2, 4, 8, 16).

    Encoder level: 3x3 conv (stride 2 below the first level) followed by a
    residual context module. Decoder level: upsampling module, concatenate the
    skip, localization module. Deep supervision: 1x1 segmentation heads on the
    three finest decoder levels are upsampled and summed into the output, so
    coarse levels get gradient directly.
    """

    def __init__(self, in_channels=1, num_classes=4, base=16, dropout=0.3):
        super().__init__()
        w = [base * 2 ** i for i in range(5)]
        self.encode = nn.ModuleList(
            nn.Sequential(conv_norm_act(cin, cout, stride=1 if i == 0 else 2), ContextModule(cout, dropout))
            for i, (cin, cout) in enumerate(zip([in_channels] + w, w))
        )
        # Decoder from the bottom up: 16b->8b, 8b->4b, 4b->2b, 2b->b.
        self.upsample = nn.ModuleList(upsampling_module(w[i], w[i - 1]) for i in range(4, 0, -1))
        # After concatenation each level has w[i] channels again; localization halves them.
        self.localize = nn.ModuleList(localization_module(w[i]) for i in range(4, 1, -1))
        # The finest level uses a plain 3x3 conv instead of a localization module, as in the paper.
        self.final = conv_norm_act(w[1], w[1])
        self.heads = nn.ModuleList(nn.Conv2d(c, num_classes, 1) for c in (w[2], w[1], w[1]))

    def forward(self, x):
        skips = []
        for level in self.encode:
            x = level(x)
            skips.append(x)
        x = skips.pop()

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
    images = torch.randn(2, 1, 256, 256)
    for model in (UNet(), ImprovedUNet()):
        logits = model(images)
        assert logits.shape == (2, 4, 256, 256), logits.shape
        logits.mean().backward()
        unused = [n for n, p in model.named_parameters() if p.grad is None]
        assert not unused, f"parameters without gradient: {unused}"
        params = sum(p.numel() for p in model.parameters())
        print(f"{type(model).__name__}: {params / 1e6:.2f}M parameters, output {tuple(logits.shape)}")

    # Dropout must be off in eval mode, so inference is deterministic.
    model = ImprovedUNet().eval()
    with torch.no_grad():
        assert torch.equal(model(images), model(images))
    print("modules self-check passed")
