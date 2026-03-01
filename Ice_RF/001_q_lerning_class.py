import random
from dataclasses import dataclass, field

type State = str | int
type Action = str
type Reward = float


@dataclass(slots=True)
class QLearningAgent:
    """A Q-Learning Agent representing knowledge without building a transition model.

    Attributes:
        learning_rate: The learning rate (alpha) for the Q-learning updates.
        discount_factor: The discount factor (gamma) for future rewards.
        exploration_rate: The probability (epsilon) of choosing a random action for exploration.
        q_table: A dictionary mapping (state, action) pairs to their Q-values.
    """

    learning_rate: float = 0.1
    discount_factor: float = 0.99
    exploration_rate: float = 0.1
    q_table: dict[tuple[State, Action], float] = field(default_factory=dict)

    def get_q_value(self, state: State, action: Action) -> float:
        """Get the Q-value for a state-action pair."""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state: State, available_actions: list[Action]) -> Action:
        """Choose an action given the current state and available actions.

        Uses an epsilon-greedy strategy for environment exploration, otherwise
        selects the action maximizing the expected reward (exploitation).

        Args:
            state: The current state of the environment.
            available_actions: A list of possible actions to choose from.

        Returns:
            The selected action.
        """
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)

        return max(available_actions, key=lambda a: self.get_q_value(state, a), default=available_actions[0])

    def update(
        self, state: State, action: Action, reward: Reward, next_state: State, next_actions: list[Action]
    ) -> None:
        """Perform a Temporal Difference (TD) update on the Q-table.

        Notice the lack of any probability distributions P(s'|s, a).
        The agent updates knowledge based on the collected sample (s, a, r, s').
        This utilizes the Bellman equation in a model-free approach.

        Args:
            state: The state in which the action was taken.
            action: The action that was taken.
            reward: The reward received after taking the action.
            next_state: The new state reached after taking the action.
            next_actions: A list of available actions in the new state.
        """
        current_q = self.get_q_value(state, action)

        max_next_q = max((self.get_q_value(next_state, a) for a in next_actions), default=0.0)

        td_target = reward + self.discount_factor * max_next_q

        self.q_table[(state, action)] = current_q + self.learning_rate * (td_target - current_q)
