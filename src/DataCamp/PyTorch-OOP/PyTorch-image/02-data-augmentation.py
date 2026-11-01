from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


train_transforms = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ]
)


def load_dataset() -> datasets.CIFAR10:
    return datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=True,
        download=False,
        transform=train_transforms,
    )


if __name__ == "__main__":
    torch.manual_seed(7)
    dataset = load_dataset()
    label_index = 3

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    for axis in axes:
        image, label = dataset[label_index]
        axis.imshow(image.permute(1, 2, 0))
        axis.set_title(dataset.classes[label])
        axis.axis("off")

    fig.suptitle("Repeated augmentation of the same CIFAR-10 sample")
    plt.tight_layout()
    plt.show()