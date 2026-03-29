"""Dataset loading and split construction for the synthetic experiments."""

import pickle
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from synthetic_router_settings import CLUSTER_NUM, REPO_ROOT


@dataclass
class SettingData:
    setting: int
    train_data: torch.Tensor
    train_labels: torch.Tensor
    test_data: torch.Tensor
    test_labels: torch.Tensor
    centers: torch.Tensor
    features: torch.Tensor
    train_cluster_idx: List[List[int]]
    test_cluster_idx: List[List[int]]
    train_cluster_labels: torch.Tensor
    test_cluster_labels: torch.Tensor


def build_cluster_labels(num_samples: int, cluster_idx: Sequence[Sequence[int]]) -> torch.Tensor:
    labels = torch.empty(num_samples, dtype=torch.long)
    for cluster_id, indices in enumerate(cluster_idx):
        labels[torch.tensor(indices, dtype=torch.long)] = cluster_id
    return labels


def load_setting(setting: int) -> SettingData:
    setting_dir = REPO_ROOT / f"synthetic_data_s{setting}"

    train_data = torch.load(setting_dir / "train_data.pt", map_location="cpu")
    train_labels = torch.load(setting_dir / "train_labels.pt", map_location="cpu").long()
    test_data = torch.load(setting_dir / "test_data.pt", map_location="cpu")
    test_labels = torch.load(setting_dir / "test_labels.pt", map_location="cpu").long()
    centers = torch.load(setting_dir / "centers.pt", map_location="cpu")
    features = torch.load(setting_dir / "features.pt", map_location="cpu")

    with open(setting_dir / "train_cluster", "rb") as fp:
        train_cluster_idx = pickle.load(fp)
    with open(setting_dir / "test_cluster", "rb") as fp:
        test_cluster_idx = pickle.load(fp)

    return SettingData(
        setting=setting,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        centers=centers,
        features=features,
        train_cluster_idx=train_cluster_idx,
        test_cluster_idx=test_cluster_idx,
        train_cluster_labels=build_cluster_labels(train_data.shape[0], train_cluster_idx),
        test_cluster_labels=build_cluster_labels(test_data.shape[0], test_cluster_idx),
    )


def build_stratify_labels(setting_data: SettingData) -> np.ndarray:
    return (setting_data.train_labels.numpy() * CLUSTER_NUM + setting_data.train_cluster_labels.numpy()).astype(np.int64)


def build_cv_splits(setting_data: SettingData, cv_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    stratify_labels = build_stratify_labels(setting_data)
    _, counts = np.unique(stratify_labels, return_counts=True)
    min_count = int(counts.min())
    if min_count < cv_folds:
        raise ValueError(
            f"Cannot build {cv_folds} stratified folds for setting s{setting_data.setting}; "
            f"smallest bucket has only {min_count} samples."
        )

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    dummy_x = np.zeros_like(stratify_labels)
    return list(splitter.split(dummy_x, stratify_labels))


def tensor_subset(tensor: torch.Tensor, indices: Optional[Sequence[int]], device: torch.device) -> torch.Tensor:
    if indices is None:
        return tensor.to(device)
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    return tensor.index_select(0, index_tensor).to(device)
