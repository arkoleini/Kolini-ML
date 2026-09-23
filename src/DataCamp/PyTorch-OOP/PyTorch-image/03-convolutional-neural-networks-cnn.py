import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.activation1 = nn.ELU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.activation2 = nn.ELU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[tuple[str, tuple[int, ...]]]]:
        shapes: list[tuple[str, tuple[int, ...]]] = [("input", tuple(x.shape))]

        x = self.conv1(x)
        shapes.append(("conv1", tuple(x.shape)))

        x = self.activation1(x)
        shapes.append(("elu1", tuple(x.shape)))

        x = self.pool1(x)
        shapes.append(("pool1", tuple(x.shape)))

        x = self.conv2(x)
        shapes.append(("conv2", tuple(x.shape)))

        x = self.activation2(x)
        shapes.append(("elu2", tuple(x.shape)))

        x = self.pool2(x)
        shapes.append(("pool2", tuple(x.shape)))

        x = self.flatten(x)
        shapes.append(("flatten", tuple(x.shape)))
        return x, shapes


if __name__ == "__main__":
    model = CNNFeatureExtractor()
    sample_batch = torch.randn(4, 3, 128, 128)
    feature_vector, feature_shapes = model(sample_batch)

    for stage_name, shape in feature_shapes:
        print(f"{stage_name:>7}: {shape}")

    print(f"Final feature vector width: {feature_vector.shape[1]}")