"""ResNet-18 adapted for CIFAR-scale images (32×32).

Standard ImageNet ResNet-18 uses a 7×7 Conv + MaxPool stem, which is
too aggressive for 32×32 inputs.  This variant replaces the stem with a
single 3×3 Conv (stride 1, padding 1) followed by no pooling, matching
the common CIFAR-ResNet convention used in continual-learning benchmarks.

The classification head is exposed as ``model.classifier`` (a single
``nn.Linear`` layer) so that scope-parsing utilities in ``svd_agent.py``
can locate it by name.

Usage::

    model = resnet18_cifar(n_classes=10)   # single head
"""

from typing import Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


# ---------------------------------------------------------------------------
# ResNet-18 (CIFAR variant)
# ---------------------------------------------------------------------------

class ResNet18CIFAR(nn.Module):
    """ResNet-18 with a CIFAR-friendly stem (3×3 conv, no max-pool).

    Attributes:
        layer1 – layer4:  Four residual stages (as in standard ResNet-18).
        classifier:       Final ``nn.Linear`` head; exposed by name so that
                          ``_get_classifier_params()`` in svd_agent.py can
                          find it directly.
    """

    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        # CIFAR stem: small 3×3 conv, no max-pool
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64,  64,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, n_classes)

        self._init_weights()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_layer(
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = []
        downsample: Optional[nn.Module] = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers.append(_BasicBlock(in_channels, out_channels, stride, downsample))
        for _ in range(1, n_blocks):
            layers.append(_BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def resnet18_cifar(n_classes: int = 10) -> ResNet18CIFAR:
    """Return a CIFAR-adapted ResNet-18 with ``n_classes`` outputs."""
    return ResNet18CIFAR(n_classes=n_classes)
