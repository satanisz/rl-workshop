from abc import ABC, abstractmethod

import numpy as np
from dataset import KnapsackInstance


class Solver(ABC):
    @abstractmethod
    def solve(self, instance: KnapsackInstance) -> tuple[float, np.ndarray]:
        """Returns:
        optimal_value: float
        solution: np.ndarray (binary array of shape (n_items,))
        """
        pass


class GreedySolver(Solver):
    def solve(self, instance: KnapsackInstance) -> tuple[float, np.ndarray]:
        n = instance.n_items
        weights = instance.weights
        values = instance.values
        capacity = instance.capacity

        # Calculate ratios
        # Avoid division by zero
        ratios = values / (weights + 1e-9)

        # Sort indices by ratio descending
        sorted_indices = np.argsort(ratios)[::-1]

        current_weight = 0
        current_value = 0
        solution = np.zeros(n, dtype=int)

        for idx in sorted_indices:
            if current_weight + weights[idx] <= capacity:
                solution[idx] = 1
                current_weight += weights[idx]
                current_value += values[idx]

        return float(current_value), solution


class DynamicProgrammingSolver(Solver):
    def solve(self, instance: KnapsackInstance) -> tuple[float, np.ndarray]:
        n = instance.n_items
        weights = instance.weights
        values = instance.values
        capacity = instance.capacity

        # DP table: dp[i][w] = max value with first i items and capacity w
        # Optimziation: We can use 1D array if we iterate backwards,
        # but to reconstruct solution, 2D (or keeping track of choices) is easier.
        # For simplicity and N=50/W=Large, standard 2D might be memory heavy if W is huge.
        # However, for our generator parameters (W ~ 50*50*0.5 ~ 1250), it's very small.

        dp = np.zeros((n + 1, capacity + 1), dtype=int)

        for i in range(1, n + 1):
            w = weights[i - 1]
            v = values[i - 1]
            for j in range(capacity + 1):
                if w <= j:
                    dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - w] + v)
                else:
                    dp[i][j] = dp[i - 1][j]

        optimal_value = dp[n][capacity]

        # Reconstruct solution
        solution = np.zeros(n, dtype=int)
        w_curr = capacity
        for i in range(n, 0, -1):
            if dp[i][w_curr] != dp[i - 1][w_curr]:
                # Item i-1 was included
                solution[i - 1] = 1
                w_curr -= weights[i - 1]

        return float(optimal_value), solution


class NeuralGreedySolver(Solver):
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.model.eval()

    def solve(self, instance: KnapsackInstance) -> tuple[float, np.ndarray]:
        # Prepare input
        import torch
        from dataset import KnapsackDataset, collate_fn

        # We need to minimally wrap the instance to use the model
        dataset = KnapsackDataset([instance])
        batch = collate_fn([dataset[0]])

        features = batch["features"].to(self.device)
        mask = batch["mask"].to(self.device)

        with torch.no_grad():
            # Get probabilities with low temperature (sharpening) NOT needed for sorting,
            # but standard output is fine.
            probs = self.model(features, mask=mask, temperature=1.0)
            probs = probs.squeeze(0).cpu().numpy()  # (N,)

        # Neural Greedy: Sort by Probability Descending
        # Only consider valid items (mask is 1)
        n = instance.n_items
        weights = instance.weights
        values = instance.values
        capacity = instance.capacity

        # Indices of items
        indices = np.arange(n)

        # Sort indices by probability descending
        # We use valid items only (though masking in generation handles this, let's be safe)
        valid_indices = indices  # All are potentially valid candidates

        sorted_indices = valid_indices[np.argsort(probs[valid_indices])[::-1]]

        current_weight = 0
        current_value = 0
        solution = np.zeros(n, dtype=int)

        for idx in sorted_indices:
            if current_weight + weights[idx] <= capacity:
                solution[idx] = 1
                current_weight += weights[idx]
                current_value += values[idx]

        return float(current_value), solution
