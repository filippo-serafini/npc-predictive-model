from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from flwr.app import ArrayRecord, MetricRecord
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from my_app.NPCdataset import NpcDataset, Standardize
from my_app.NPCnetwork import NpcNetwork

FEATURE_COLS = ["cpus_allocated", "eligible_time", "job_id", "num_cores_req",
                             "num_nodes_req", "nodes_allocated",
                             "mem_req", "priority", "start_time",
                             "submit_time", "time_limit", "user_id"]

def partition_dataset(dataset, num_clients: int, batch_size: int, seed: int = 42):
    """
    Divide a dataset into 'num_clients' partitions (IID).
    Args:
        dataset: dataset to divide.
        num_clients (int): number of clients or partitions.
        batch_size (int): dataLoader's batch size.
        seed (int): seed for reproducibility.
    Returns:
        dict: {client_id: dataloader}
    """
    g = torch.Generator().manual_seed(seed)

    # Compute partition's dimensions
    partitions_lengths = [len(dataset) // num_clients] * num_clients
    for i in range(len(dataset) % num_clients):
        partitions_lengths[i] += 1

    # Dataset splitting
    partitions = random_split(dataset, partitions_lengths, generator=g)

    client_loaders = {}
    for client_id, subset in enumerate(partitions):
        # DataLoader
        data_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders[client_id] = data_loader

    return client_loaders

global_train_dataset = None # Cache dataset
global_val_dataset = None # Cache dataset

train_transform = None
train_target_transform = None

def load_data(partition_id: int, num_partitions: int):
    """Load partition PM100-FDATA dataloaders."""
    
    # Only initialize `NPC_Dataset`s once
    global global_train_dataset
    global global_val_dataset
    global train_transform
    global train_target_transform

    if global_train_dataset or global_val_dataset is None:
        # Initial data upload
        train_dataframe = pd.read_parquet("training_dataset.parquet")
        val_dataframe = pd.read_parquet("validation_dataset.parquet")
        
        global_train_dataset = NpcDataset(
            train_dataframe,
            feature_cols=FEATURE_COLS,
            target_col="node_power_consumption",
            transform=None,
            target_transform=None
        )
        train_target_transform = global_train_dataset.target_transform
        train_transform = global_train_dataset.transform

        global_val_dataset = NpcDataset(
            val_dataframe,
            feature_cols=FEATURE_COLS,
            target_col="node_power_consumption",
            transform=train_transform,
            target_transform=train_target_transform
        )

    # Divide data on each client (num_partitions)
    client_train_dataloaders = partition_dataset(
        dataset=global_train_dataset,
        batch_size=128,
        num_clients=num_partitions,
    )
    client_val_dataloaders = partition_dataset(
        dataset=global_val_dataset,
        batch_size=128,
        num_clients=num_partitions
    )

    # Retrieve client's specific dataloaders
    train_loader = client_train_dataloaders[partition_id]
    val_loader = client_val_dataloaders[partition_id]

    return train_loader, val_loader

def initialize_transforms() -> Tuple[Standardize, Standardize]:
    """Helper function to initialize transforms for validation and test."""

    global train_transform
    global train_target_transform

    if train_target_transform is None:
        # Load training_dataset to compute standardization parameters
        train_dataframe = pd.read_parquet("training_dataset.parquet")
        global_train_dataset = NpcDataset(
            train_dataframe,
            feature_cols=FEATURE_COLS,
            target_col="node_power_consumption",
            transform=None,
            target_transform=None
        )
        train_target_transform = global_train_dataset.target_transform
        train_transform = global_train_dataset.transform

    return train_transform, train_target_transform

def train(net: NpcNetwork, trainloader: DataLoader, local_epochs: int, lr: float, device):
    """Train the model on the training loader."""

    net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    num_batches = len(trainloader)
    training_loss = 0.0
    net.train()
    for _ in range(local_epochs): 
        # Iterate over local_epochs
        for batch, (job_features, label) in enumerate(trainloader): 
            # Iterate over batches
            job_features, label = job_features.to(device), label.to(device)

            # Compute prediction error
            prediction = net(job_features).squeeze(1)
            loss = loss_fn(prediction, label)
            training_loss += loss.item()

            # Backprogation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return training_loss / num_batches


def validate(net: NpcNetwork, testloader: DataLoader, device):
    """Evaluate the model on the client side."""
    net.to(device)
    loss_fn = nn.MSELoss()

    num_batches = len(testloader)
    test_loss = 0
    net.eval()
    with torch.no_grad():
        for job_features, label in testloader:
            job_features, label = job_features.to(device), label.to(device)
            prediction = net(job_features).squeeze(1)
            test_loss += loss_fn(prediction, label).item()
    avg_train_loss = test_loss / num_batches

    return avg_train_loss

server_val_dataset = None # Cache server validation dataset

def central_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate the model on the server side."""

    train_transform, train_target_transform = initialize_transforms()
    
    # Load the model and initialize it with the received weights
    model = NpcNetwork()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the dataset
    global server_val_dataset

    if server_val_dataset is None:
        server_val_df = pd.read_parquet("server_validation_dataset.parquet")
        server_val_dataset = NpcDataset(
            server_val_df,
            feature_cols=FEATURE_COLS,
            target_col="node_power_consumption",
            transform=train_transform,
            target_transform=train_target_transform
        )

    server_dataloader = DataLoader(server_val_dataset, batch_size=128)
    loss = validate(model, server_dataloader, device)
    return MetricRecord({"loss": loss})   

# -----------------------------------
# TEST
# -----------------------------------
def test_final_model(net, transform, target_transform):
    """Test the final model on the test dataset."""

    # Load the test dataset
    test_dataframe = pd.read_parquet("test_dataset.parquet")
    test_dataset = NpcDataset(
        test_dataframe,
        feature_cols=FEATURE_COLS,
        target_col="node_power_consumption",
        transform=transform,
        target_transform=target_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=128)

    loss_fn = nn.MSELoss()
    num_batches = len(test_loader)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_loss = 0
    all_predictions = []
    all_targets = []
    target_std = target_transform.std.item()
    target_mean = target_transform.mean.item()
   
    net.eval()
    with torch.no_grad():
        print("\nTesting...")
        for job_features, label in test_loader:
            job_features, label = job_features.to(device), label.to(device)
            prediction = net(job_features).squeeze(1)

            test_loss += loss_fn(prediction, label).item()
            all_predictions.append(prediction)     
            all_targets.append(label)              
    test_loss /= num_batches                     

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