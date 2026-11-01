import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path


DATASET_PATH = (
    Path(__file__).resolve().parent
    / "datasets"
    / "Water_potability_dataset"
    / "water_potability_dataset.csv"
)


class WaterDataset(Dataset):
    def __init__(self, csv_path):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Water potability dataset not found: {csv_path}")

        df = pd.read_csv(csv_path)
        self.data = df.values.astype("float32")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx, :-1]
        label = self.data[idx, -1]
        return torch.tensor(features), torch.tensor(label)


dataset = WaterDataset(DATASET_PATH)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

dataloader_test = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(9, 16)
        # Normalize first hidden representation per batch.
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 8)
        # Normalize second hidden representation per batch.
        self.bn2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 1)

        init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        init.zeros_(self.fc1.bias)

        init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")
        init.zeros_(self.fc2.bias)

        init.kaiming_uniform_(self.fc3.weight, nonlinearity="sigmoid")
        init.zeros_(self.fc3.bias)

    def forward(self, x):
        # First block: linear -> batch norm -> nonlinearity.
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.elu(x)

        # Second block mirrors the first block.
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.elu(x)

        x = nn.functional.sigmoid(self.fc3(x))
        return x


model = Net()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(5):
    model.train()

    for features, labels in dataloader:
        labels = labels.view(-1, 1)
        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for features, labels in dataloader_test:
        labels = labels.view(-1, 1)
        outputs = model(features)
        preds = (outputs >= 0.5).float()

        correct += (preds == labels).sum().item()
        total += labels.numel()

accuracy = correct / total
print(f"Accuracy: {accuracy}")