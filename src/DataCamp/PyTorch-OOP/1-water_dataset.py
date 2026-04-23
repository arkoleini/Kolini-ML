import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


DATASET_PATH = (
    Path(__file__).resolve().parent
    / "datasets"
    / "Water_potability_dataset"
    / "water_potability_dataset.csv"
)


class WaterDataset(Dataset):
    def __init__(self, csv_path=DATASET_PATH):
        super().__init__()
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Water potability dataset not found: {csv_path}")
        df = pd.read_csv(csv_path)
        self.data = df.to_numpy()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        features = self.data[idx, :-1]
        label = self.data[idx, -1]
        return features, label


if __name__ == "__main__":
    try:
        dataset = WaterDataset()
    except FileNotFoundError as error:
        print(error)
        raise SystemExit(1)

    print(f"Dataset loaded from: {DATASET_PATH}")
    print(f"Dataset size: {len(dataset)}")

    features, label = dataset[0]
    print(f"First sample - features: {features}, label: {label}")

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for batch_features, batch_labels in loader:
        print(f"Batch - features shape: {batch_features.shape}, labels shape: {batch_labels.shape}")
        break
