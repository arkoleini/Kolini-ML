# sequential_model.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# -----------------------------
# Custom Dataset Class
# -----------------------------
class WaterDataset(Dataset):
    def __init__(self, csv_path):
        # Load CSV into pandas DataFrame
        df = pd.read_csv(csv_path)

        # Convert to numpy array (float32 for PyTorch)
        self.data = df.values.astype('float32')

    def __len__(self):
        # Return total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Split features and label
        features = self.data[idx, :-1]   # all columns except last
        label = self.data[idx, -1]       # last column

        # Convert to tensors
        return torch.tensor(features), torch.tensor(label)


# -----------------------------
# Load Dataset + DataLoader
# -----------------------------
dataset = WaterDataset("water_train.csv")

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)


# -----------------------------
# Model using nn.Sequential
# -----------------------------
# nn.Sequential automatically connects layers in order
# No need to define forward() manually
model = nn.Sequential(
    nn.Linear(9, 16),   # input layer: 9 features → 16 neurons
    nn.ReLU(),          # activation function
    nn.Linear(16, 8),   # hidden layer
    nn.ReLU(),
    nn.Linear(8, 1),    # output layer (1 value)
    nn.Sigmoid()        # converts output to probability (0–1)
)


# -----------------------------
# Loss and Optimizer
# -----------------------------
criterion = nn.BCELoss()  # binary classification loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(5):
    for features, labels in dataloader:
        labels = labels.view(-1, 1)  # reshape to match output

        # Forward pass (automatic through Sequential)
        outputs = model(features)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")