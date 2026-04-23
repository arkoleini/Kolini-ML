import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------
# 1. Prepare dummy dataset
# -------------------------------
X_train = torch.randn(100, 10)
y_train = torch.randint(0, 3, (100,))  # 3-class classification

X_val = torch.randn(30, 10)
y_val = torch.randint(0, 3, (30,))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16)

# -------------------------------
# 2. Define model, loss, optimizer, accuracy metric
# -------------------------------
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 3)   # 3-class output
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
train_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)
val_acc_metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)

# -------------------------------
# 3. Training & Validation Loop
# -------------------------------
num_epochs = 5

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    train_loss = 0.0
    for features, labels in train_loader:
        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc_metric.update(outputs, labels)

    train_loss /= len(train_loader)
    train_accuracy = train_acc_metric.compute()
    train_acc_metric.reset()

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc_metric.update(outputs, labels)

    val_loss /= len(val_loader)
    val_accuracy = val_acc_metric.compute()
    val_acc_metric.reset()

    print(f"Epoch {epoch+1} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")