import torch
from torch import nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
import matplotlib.pyplot as plt

class NpcNetwork(nn.Module):
    """Node Power Consumption Network class"""
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

# -----------------------------
# TRAINING
# -----------------------------
def train(train_dataloader, net, loss_fn, optimizer, device):
    """Train the network on the provided dataloader."""

    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    total_loss = 0
    net.train()
    for batch, (job_features, label) in enumerate(train_dataloader):
        job_features, label = job_features.to(device), label.to(device)

        # Compute prediction error
        prediction = net(job_features).squeeze(1)
        loss = loss_fn(prediction, label)
        total_loss += loss.item()

        # Backprogation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(job_features)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    avg_training_loss = total_loss / num_batches
    print(f"Training Avg Loss: {avg_training_loss:>5f}")
    return avg_training_loss

# -----------------------------------
# VALIDATION
# ----------------------------------
def validate(val_dataloader, net, loss_fn, device):
    """Evaluate the network on the provided dataloader."""

    num_batches = len(val_dataloader)
    val_loss = 0
    net.eval()
    with torch.no_grad():
        print("\nComputing validation...")
        for job_features, label in val_dataloader:
            job_features, label = job_features.to(device), label.to(device)
            prediction = net(job_features).squeeze(1)
            val_loss += loss_fn(prediction, label).item()

    avg_val_loss = val_loss / num_batches
    print(f"Avg Validation Error: {avg_val_loss:>5f} \n")
    return avg_val_loss

# -----------------------------------
# TEST
# -----------------------------------
def plot_metrics(ae: list[float], ape: list[float], se: list[float]):
    """Plots significant performance threshold metrics."""

    # Absolute Error histogram
    plt.figure()
    plt.hist(ae, bins=50, range=(0, 100))
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequence")
    plt.title("Absolute Error Distribution (Watt)")
    plt.xlim(0, 100)
    plt.grid(True)

    # Squared Error histogram
    plt.figure()
    plt.hist(se, bins=50, range=(0, 100))
    plt.xlabel("Squared Error")
    plt.ylabel("Frequence")
    plt.title("Squared Error Distribution (Watt)")
    plt.xlim(0, 100)
    plt.grid(True)

    # Absolute Percentage Error histogram
    plt.figure()
    plt.hist(ape, bins=15, range=(0, 15))
    plt.xlabel("Absolute Percentage Error")
    plt.ylabel("Frequence")
    plt.title("Absolute Percentage Error Distribution (%)")
    plt.xlim(0, 15)
    plt.grid(True)

    plt.show()

def test(test_loader, net, loss_fn, device, target_mean, target_std):
    """Test the network on the provided dataloader."""

    num_batches = len(test_loader)
    test_loss = 0
    all_predictions = []
    all_targets = []
    net.eval()
    with torch.no_grad():
        print("\nTesting...")
        for job_features, label in test_loader:
            job_features, label = job_features.to(device), label.to(device)
            prediction = net(job_features).squeeze(1)

            test_loss += loss_fn(prediction, label).item()
            all_predictions.append(prediction)     # 'dim_batch' number of predictions
            all_targets.append(label)              # 'dim_batch' number of targets
    test_loss /= num_batches                       # Avg MSE

    # Union in one single vector
    all_predictions = torch.cat(all_predictions).to("cpu").numpy()
    all_targets = torch.cat(all_targets).to("cpu").numpy()

    # 1. De-standardization
    log_predictions = all_predictions * target_std + target_mean
    log_targets = all_targets * target_std + target_mean

    # 2. Exponential to return in the initial base
    all_predictions = np.expm1(log_predictions)
    all_targets = np.expm1(log_targets)

    # Mean test metrics
    mae_test = mean_absolute_error(all_targets, all_predictions)
    mape_test = mean_absolute_percentage_error(all_targets, all_predictions) * 100
    rmse_test = root_mean_squared_error(all_targets, all_predictions)

    print(f"TEST METRICS: \n"
          f"\tAvg loss (MSE): {test_loss:>8f} \n"
          f"\tMean Absolute Error (MAE): {mae_test:>6f} \n"
          f"\tMean Absolute Percentage Error (MAPE): {mape_test} \n"
          f"\tRoot Mean Squared Error (RMSE): {rmse_test}\n")

    # Thresholds metrics
    ae_list = []
    se_list = []
    ape_list = []
    ae_lt_100 = []
    se_lt_100 = []
    ape_lt_15 = []

    # Populate the threshold metrics
    for prediction, target in zip(all_predictions, all_targets):
        ae = abs(prediction - target)
        se = (prediction - target)**2
        ape = abs((prediction - target)/target) * 100

        if ae < 100:
            ae_lt_100.append(ae)
        if se < 100:
            se_lt_100.append(se)
        if ape < 15:
            ape_lt_15.append(ape)

        ae_list.append(ae)
        se_list.append(se)
        ape_list.append(ape)

    print(f"TEST THRESHOLDS: \n"
          f"Absolute Error (AE) < 100W:  [{len(ae_lt_100)} / {len(ae_list)}] ({(len(ae_lt_100) / len(ae_list) * 100):>3f})\n"
          f"Absolute Percentage Error (APE) < 15%:  [{len(ape_lt_15)} / {len(ape_list)}] ({(len(ape_lt_15) / len(ape_list) * 100):>3f})\n"
          f"Squared Error (SE) < 100W^2:  [{len(se_lt_100)} / {len(se_list)}] ({(len(se_lt_100) / len(se_list) * 100):>3f})\n"
    )

    # Plot histograms
    plot_metrics(ae_lt_100, ape_lt_15, se_lt_100)