from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def build_transform(image_size: int = 128) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


def load_dataset() -> datasets.CIFAR10:
    return datasets.CIFAR10(
        root=str(DATA_ROOT),
        train=True,
        download=False,
        transform=build_transform(),
    )


def tensor_to_image(image: torch.Tensor) -> torch.Tensor:
    return image.squeeze(0).permute(1, 2, 0)


if __name__ == "__main__":
    dataset = load_dataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    images, labels = next(iter(dataloader))
    class_name = dataset.classes[labels.item()]

    print(f"Dataset size: {len(dataset)}")
    print(f"Batch shape [B, C, H, W]: {tuple(images.shape)}")
    print(f"Class label: {labels.item()} ({class_name})")

    display_image = tensor_to_image(images)
    plt.imshow(display_image)
    plt.title(f"CIFAR-10 sample: {class_name}")
    plt.axis("off")
    plt.show()