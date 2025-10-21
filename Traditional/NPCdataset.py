import torch
import numpy as np
from pandas import DataFrame
from torch.utils.data import Dataset


class Standardize:
    """Standardize data before training."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = torch.where(std == 0, torch.ones_like(std), std)

    def __call__(self, x):
        return (x - self.mean) / self.std

class NpcDataset(Dataset):
    """Node Power Consumption dataset class"""

    def __init__(self, df_data: DataFrame, feature_cols: list[str], target_col: str, transform, target_transform):
        self.features = torch.tensor(
              df_data[feature_cols].to_numpy(),
              dtype=torch.float32
            )
        self.targets = torch.tensor(
            # Log transformation
            np.log1p(df_data[target_col].to_numpy()),
            dtype=torch.float32
            )

        # Features transform
        if transform is None:
          self.feat_mean = self.features.mean(dim=0)
          self.feat_std = self.features.std(dim=0)
          self.transform = Standardize(self.feat_mean, self.feat_std)
        else:
          self.transform = transform

        # Target transform
        if target_transform is None:
          self.target_mean = self.targets.mean()
          self.target_std = self.targets.std()
          self.target_transform = Standardize(self.target_mean, self.target_std)
        else:
          self.target_transform = target_transform

    def __len__(self):
      return len(self.targets)

    def __getitem__(self, idx):
        target = self.targets[idx]
        features = self.features[idx]

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            target = self.target_transform(target)

        return features, target