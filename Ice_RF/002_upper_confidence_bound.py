import random
import math
from dataclasses import dataclass, field

type Action = str | int
type Reward = float

@dataclass(slots=True)
class UCBBandit:
    """An Upper Confidence Bound (UCB) bandit algorithm.

    Attributes:
        exploration_param: The exploration parameter (c) controlling the degree of exploration.
        action_values: Internal state estimating the value of each action.
        action_counts: Internal state tracking the number of times each action was chosen.
        total_steps: The total number of steps taken.
    """

    exploration_param: float = 1.41

    action_values: dict[Action, float] = field(default_factory=dict)
    action_counts: dict[Action, int] = field(default_factory=dict)
    total_steps: int = 0

    def select_action(self, available_actions: list[Action]) -> Action:
        """Selects an action using the UCB algorithm.

        Initialization rule: ensures each action is tested at least once before
        using the Upper Confidence Bound formula to select the best action.

        Args:
            available_actions: A list of actions that can be taken.

        Returns:
            The selected action.
        """
        self.total_steps += 1

        for action in available_actions:
            if self.action_counts.get(action, 0) == 0:
                return action

        return max(
            available_actions,
            key=lambda a: self.action_values[a]
            + self.exploration_param * math.sqrt(math.log(self.total_steps) / self.action_counts[a]),
        )

    def update(self, action: Action, reward: Reward) -> None:
        """Update the value of an action based on the received reward.

        Notice the absence of 'state' and 'next_state' parameters. The bandit
        optimizes only for the immediate payout (reward) for a given arm (action).
        The update uses an incremental moving average (step-by-step value update).

        Args:
            action: The action that was taken.
            reward: The reward received after taking the action.
        """
        count = self.action_counts.get(action, 0) + 1
        self.action_counts[action] = count

        current_value = self.action_values.get(action, 0.0)
        self.action_values[action] = current_value + (1 / count) * (reward - current_value)


class BernoulliArm:
    """A Bernoulli arm that returns a reward of 1 with a fixed probability."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def pull(self) -> float:
        """Pull the arm and return a reward."""
        return 1.0 if random.random() < self.probability else 0.0


def run_ucb_simulation(n_steps: int = 1000) -> None:
    """Run a simulation of the UCB algorithm."""
    arms = [BernoulliArm(p) for p in [0.1, 0.3, 0.8, 0.2]]
    agent = UCBBandit()

    for _ in range(n_steps):
        action = agent.select_action(list(range(len(arms))))
        reward = arms[action].pull()
        agent.update(action, reward)

    print(f"True probabilities: {[arm.probability for arm in arms]}")
    print(f"Estimated values: {agent.action_values}")
    print(f"Number of selections: {agent.action_counts}")


if __name__ == "__main__":
    run_ucb_simulation(100000)
