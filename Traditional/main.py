import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from NPCdataset import NpcDataset
from NPCnetwork import NpcNetwork, train, validate, test

device = "cuda:0" if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

feature_cols = ["cpus_allocated", "eligible_time", "job_id", "num_cores_req",
                             "num_nodes_req", "nodes_allocated",
                             "mem_req", "priority", "start_time",
                             "submit_time", "time_limit", "user_id"]

train_dataframe = pd.read_parquet("training_dataset.parquet")
validation_dataframe = pd.read_parquet("validation_dataset.parquet")
test_dataframe = pd.read_parquet("test_dataset.parquet")

# -----  Datasets -----
train_dataset = NpcDataset(
    train_dataframe,
    feature_cols=feature_cols,
    target_col="node_power_consumption",
    transform=None, target_transform=None
)

# Transform and target_transform extraction
train_transform = train_dataset.transform
train_target_transform = train_dataset.target_transform

train_target_mean = train_target_transform.mean.item()
train_target_std = train_target_transform.std.item()

# We pass the same standardization parameters to validation and test datasets
val_dataset = NpcDataset(
    validation_dataframe,
    feature_cols=feature_cols,
    target_col="node_power_consumption",
    transform=train_transform,
    target_transform=train_target_transform
)

test_dataset = NpcDataset(
    test_dataframe,
    feature_cols=feature_cols,
    target_col="node_power_consumption",
    transform=train_transform,
    target_transform=train_target_transform
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ----------- Network definition -----------
model = NpcNetwork().to(device)

# Cost function and Optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)

# Simulation values
epochs = 3

# Plot arrays
train_losses = []
val_losses = []

# --------- 1.Training and 2.Validation ---------
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loss = train(train_loader, model, loss_fn, optimizer, device)
    val_loss = validate(val_loader, model, loss_fn, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

# Plots
epochs_range = range(1, epochs + 1)
plt.figure()
plt.plot(epochs_range, train_losses)
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.grid(True)
plt.title("Training Average MSE")

plt.figure()
plt.plot(epochs_range, val_losses)
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.grid(True)
plt.title("Validation Average MSE")
print("\n\n---------------------------")

# -------------------- 3. Test --------------------
test(test_loader, model,
     loss_fn, device,
     target_mean=train_target_mean,
     target_std=train_target_std
)

print("Done!")