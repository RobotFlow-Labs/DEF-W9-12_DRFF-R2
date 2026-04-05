from __future__ import annotations

from typing import Any


def build_model(num_classes: int, arch: str = "smallcnn", pretrained: bool = False) -> Any:
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for training/evaluation") from exc

    class SmallCNN(nn.Module):
        def __init__(self, classes: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(64, classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    if arch.startswith("efficientnet"):
        try:
            from torchvision import models
        except ModuleNotFoundError:
            return SmallCNN(num_classes)

        if arch == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
        else:
            return SmallCNN(num_classes)

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    return SmallCNN(num_classes)
