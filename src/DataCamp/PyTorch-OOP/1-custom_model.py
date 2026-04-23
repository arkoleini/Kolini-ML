# custom_model.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path


DATASET_PATH = (
    Path(__file__).resolve().parent
    / "datasets"
    / "Water_potability_dataset"
    / "water_potability_dataset.csv"
)

# -----------------------------
# Custom Dataset Class
# -----------------------------
class WaterDataset(Dataset):
    def __init__(self, csv_path):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Water potability dataset not found: {csv_path}")

        # Load CSV file
        df = pd.read_csv(csv_path)

        # Convert to numpy array
        self.data = df.values.astype('float32')

    def __len__(self):
        # Number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Extract features and label
        features = self.data[idx, :-1]
        label = self.data[idx, -1]

        return torch.tensor(features), torch.tensor(label)


# -----------------------------
# Load Dataset + DataLoader
# -----------------------------
dataset = WaterDataset(DATASET_PATH)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

dataloader_test = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False
)


# -----------------------------
# Custom Neural Network Class
# -----------------------------
class Net(nn.Module):
    def __init__(self):
        # Initialize parent class (VERY IMPORTANT)
        super().__init__()

        # Define layers explicitly
        self.fc1 = nn.Linear(9, 16)   # first layer
        self.fc2 = nn.Linear(16, 8)   # second layer
        self.fc3 = nn.Linear(8, 1)    # output layer

    def forward(self, x):
        """
        Defines how data flows through the network
        This is the core computation logic
        """

        # Pass through first layer + activation
        x = torch.relu(self.fc1(x))

        # Pass through second layer + activation
        x = torch.relu(self.fc2(x))

        # Final layer + sigmoid for probability
        x = torch.sigmoid(self.fc3(x))

        return x


# Instantiate model
model = Net()


# -----------------------------
# Loss and Optimizer
# -----------------------------
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(5):
    model.train()

    for features, labels in dataloader:
        labels = labels.view(-1, 1)

        # Forward pass (calls forward() internally)
        outputs = model(features)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# -----------------------------
# Model Evaluation
# -----------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for features, labels in dataloader_test:
        labels = labels.view(-1, 1)

        # Model outputs probabilities because the final layer uses sigmoid.
        outputs = model(features)
        preds = (outputs >= 0.5).float()

        correct += (preds == labels).sum().item()
        total += labels.numel()

accuracy = correct / total
print(f"Accuracy: {accuracy}")
