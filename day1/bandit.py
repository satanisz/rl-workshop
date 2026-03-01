from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BanditEnvironment(Protocol):
    """Protocol defining the environment interface (analogous to Gymnasium)."""

    def step(self, action: int) -> float: ...
    @property
    def n_arms(self) -> int: ...


@dataclass(slots=True)
class ThompsonSamplingAgent:
    n_arms: int
    alphas: np.ndarray = field(init=False)
    betas: np.ndarray = field(init=False)
    numbers_of_selections: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.alphas = np.ones(self.n_arms)
        self.betas = np.ones(self.n_arms)
        self.numbers_of_selections = np.zeros(self.n_arms)

    def select_action(self) -> int:
        """Key TS step: sampling from distributions and selecting the maximum."""
        samples = np.random.beta(self.alphas, self.betas)
        chosen = int(np.argmax(samples))
        self.numbers_of_selections[chosen] += 1
        return chosen

    def update(self, action: int, reward: float) -> None:
        """Bayesian update of posterior parameters."""
        if reward == 1:
            self.alphas[action] += 1
        else:
            self.betas[action] += 1


class BernoulliEnv:
    def __init__(self, probabilities: list[float]):
        self.probs = np.array(probabilities)
        self.n_arms = len(probabilities)

    def step(self, action: int) -> float:
        return 1.0 if np.random.random() < self.probs[action] else 0.0


def run_mab_simulation(n_steps: int = 1000):
    true_probs = [0.1, 0.3, 0.8, 0.2]
    env = BernoulliEnv(true_probs)
    agent = ThompsonSamplingAgent(n_arms=env.n_arms)

    for _ in range(n_steps):
        action = agent.select_action()
        reward = env.step(action)
        agent.update(action, reward)

    print(f"True probabilities: {true_probs}")

    estimates = (agent.alphas / (agent.alphas + agent.betas)).round(2).tolist()
    print(f"Estimates (alpha / (alpha + beta)): {estimates}")
    print(f"Number of selections: {agent.numbers_of_selections}")


if __name__ == "__main__":
    run_mab_simulation(100000)
