from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_loaders(batch_size: int = 64) -> tuple[DataLoader, DataLoader, int]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=True,
        download=False,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=False,
        download=False,
        transform=transform,
    )

    train_subset = Subset(train_dataset, range(512))
    test_subset = Subset(test_dataset, range(256))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, len(train_dataset.classes)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def evaluate(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    return correct / total


if __name__ == "__main__":
    train_loader, test_loader, num_classes = build_loaders()
    model = SmallCNN(num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    average_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    accuracy = evaluate(model, test_loader)

    print(f"Training loss after one epoch: {average_loss:.4f}")
    print(f"Validation accuracy on the evaluation subset: {accuracy:.2%}")