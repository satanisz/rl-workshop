import random
from dataclasses import dataclass, field

import numpy as np

type Action = str | int
type Reward = float


@dataclass(slots=True)
class ThompsonSamplingAgent:
    """A Thompson Sampling bandit algorithm.

    Attributes:
        action_alphas: Internal state estimating the alpha parameter (successes) of each action.
        action_betas: Internal state estimating the beta parameter (failures) of each action.
    """

    action_alphas: dict[Action, float] = field(default_factory=dict)
    action_betas: dict[Action, float] = field(default_factory=dict)
    _rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def select_action(self, available_actions: list[Action]) -> Action:
        """Selects an action using the Thompson Sampling algorithm.

        Samples from a Beta distribution for each available action and
        picks the action with the highest sampled value.

        Args:
            available_actions: A list of actions that can be taken.

        Returns:
            The selected action.
        """
        best_action = available_actions[0]
        max_sample = -1.0

        for action in available_actions:
            alpha = self.action_alphas.get(action, 1.0)
            beta = self.action_betas.get(action, 1.0)

            sample = float(self._rng.beta(alpha, beta))

            if sample > max_sample:
                max_sample = sample
                best_action = action

        return best_action

    def update(self, action: Action, reward: Reward) -> None:
        """Update the posterior distribution parameters of an action based on the received reward.

        Args:
            action: The action that was taken.
            reward: The reward received after taking the action (typically 1 for success, 0 for failure).
        """
        alpha = self.action_alphas.get(action, 1.0)
        beta = self.action_betas.get(action, 1.0)

        if reward == 1.0:
            self.action_alphas[action] = alpha + 1.0
            if action not in self.action_betas:
                self.action_betas[action] = beta
        else:
            self.action_betas[action] = beta + 1.0
            if action not in self.action_alphas:
                self.action_alphas[action] = alpha


class BernoulliArm:
    """A Bernoulli arm that returns a reward of 1 with a fixed probability."""

    def __init__(self, probability: float) -> None:
        """Initialize the Bernoulli arm with a fixed probability of success."""
        self.probability = probability

    def pull(self) -> float:
        """Pull the arm and return a reward."""
        return 1.0 if random.random() < self.probability else 0.0  # noqa: S311


def run_ts_simulation(n_steps: int = 1000) -> None:
    """Run a simulation of the Thompson Sampling algorithm."""
    arms = [BernoulliArm(p) for p in [0.1, 0.3, 0.8, 0.2]]
    agent = ThompsonSamplingAgent()

    for _ in range(n_steps):
        action = agent.select_action(list(range(len(arms))))
        reward = arms[int(action)].pull()
        agent.update(action, reward)

    print(f"True probabilities: {[arm.probability for arm in arms]}")
    estimates = {}
    action_counts = {}
    for a in range(len(arms)):
        alpha = agent.action_alphas.get(a, 1.0)
        beta = agent.action_betas.get(a, 1.0)
        estimates[a] = round(alpha / (alpha + beta), 3)
        action_counts[a] = int(alpha + beta - 2.0)

    print(f"Estimated values (alpha / (alpha + beta)): {estimates}")
    print(f"Number of selections: {action_counts}")


if __name__ == "__main__":
    run_ts_simulation(100000)
