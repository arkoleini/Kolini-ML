from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def load_one_batch(batch_size: int = 16) -> tuple[DataLoader, list[str]]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=True,
        download=False,
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset.classes


if __name__ == "__main__":
    dataloader, classes = load_one_batch()
    images, labels = next(iter(dataloader))
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    model = ImageClassifier(num_classes=len(classes)).to(DEVICE)
    logits = model(images)

    print(f"Input batch shape: {tuple(images.shape)}")
    print(f"Output logits shape: {tuple(logits.shape)}")
    print(f"Predicted class indices: {torch.argmax(logits, dim=1)[:5].tolist()}")