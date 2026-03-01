from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class KnapsackInstance:
    weights: np.ndarray  # shape (n_items,)
    values: np.ndarray  # shape (n_items,)
    capacity: int
    n_items: int
    optimal_value: float | None = None
    optimal_solution: np.ndarray | None = None  # binary mask


class KnapsackGenerator:
    def __init__(
        self,
        min_items: int = 10,
        max_items: int = 50,
        min_weight: int = 1,
        max_weight: int = 50,
        min_value: int = 1,
        max_value: int = 100,
        capacity_ratio_mean: float = 0.5,
    ):
        self.min_items = min_items
        self.max_items = max_items
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_value = min_value
        self.max_value = max_value
        self.capacity_ratio_mean = capacity_ratio_mean

    def generate(self, n_instances: int = 1, fixed_n_items: int = None) -> list[KnapsackInstance]:
        instances = []
        for _ in range(n_instances):
            n = fixed_n_items if fixed_n_items is not None else np.random.randint(self.min_items, self.max_items + 1)
            weights = np.random.randint(self.min_weight, self.max_weight + 1, size=n)
            values = np.random.randint(self.min_value, self.max_value + 1, size=n)

            # Capacity is a fraction of total weight to make it interesting
            total_weight = weights.sum()
            # Randomize capacity ratio slightly around mean
            ratio = np.clip(np.random.normal(self.capacity_ratio_mean, 0.1), 0.1, 0.9)
            capacity = int(total_weight * ratio)

            instances.append(
                KnapsackInstance(
                    weights=weights,
                    values=values,
                    capacity=max(1, capacity),  # Ensure at least 1
                    n_items=n,
                )
            )
        return instances


class KnapsackDataset(Dataset):
    def __init__(self, instances: list[KnapsackInstance], normalize: bool = True):
        self.instances = instances
        self.normalize = normalize

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        instance = self.instances[idx]

        # Features: [weight, value] per item
        # We can also add normalized versions

        w = torch.FloatTensor(instance.weights)
        v = torch.FloatTensor(instance.values)
        c = float(instance.capacity)

        if self.normalize:
            # Normalize weights by capacity
            w_norm = w / c
            # Normalize values (e.g. by max value or just scale down)
            v_norm = v / v.max() if v.max() > 0 else v

            # Feature vector: [w_norm, v_norm]
            # Shape: (n_items, 2)
            features = torch.stack([w_norm, v_norm], dim=1)
        else:
            features = torch.stack([w, v], dim=1)

        # Target (if pre-solved)
        label = (
            torch.FloatTensor(instance.optimal_solution) if instance.optimal_solution is not None else torch.tensor([])
        )

        return {
            "features": features,
            "capacity": c,  # Normalized capacity is essentially 1.0 if we divide weights by it
            "label": label,
            "weights": w,
            "values": v,
            "n_items": instance.n_items,
        }


def collate_fn(batch):
    """Custom collate function to handle variable number of items.
    Pads sequences to the maximum length in the batch.
    """
    # Sort by length for potentially better packing (optional, skipping for simplicity)

    max_len = max(item["n_items"] for item in batch)

    features_padded = []
    labels_padded = []
    masks = []  # 1 for real items, 0 for padding

    weights_padded = []
    values_padded = []
    capacities = []

    for item in batch:
        n = item["n_items"]
        pad_len = max_len - n

        # Pad features with 0
        f = item["features"]
        f_pad = torch.nn.functional.pad(f, (0, 0, 0, pad_len), value=0)
        features_padded.append(f_pad)

        # Create mask
        mask = torch.cat([torch.ones(n), torch.zeros(pad_len)])
        masks.append(mask)

        # Pad labels (if they exist)
        if item["label"].numel() > 0:
            l = item["label"]
            l_pad = torch.nn.functional.pad(l, (0, pad_len), value=0)
            labels_padded.append(l_pad)

        # Keep original weights/values for validatoin
        weights_padded.append(torch.nn.functional.pad(item["weights"], (0, pad_len), value=0))
        values_padded.append(torch.nn.functional.pad(item["values"], (0, pad_len), value=0))
        capacities.append(item["capacity"])

    return {
        "features": torch.stack(features_padded),  # (B, max_N, 2)
        "mask": torch.stack(masks).bool(),  # (B, max_N)
        "labels": torch.stack(labels_padded) if labels_padded else None,
        "weights": torch.stack(weights_padded),
        "values": torch.stack(values_padded),
        "capacities": torch.FloatTensor(capacities),
    }
