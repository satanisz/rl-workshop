from enum import IntEnum
import numpy as np
from typing import Annotated, Any

class Action(IntEnum):
    RIGHT = 0
    LEFT = 1


class SimplesEnv:
    def __init__(self, size: int = 10, verbose=False) -> None:
        self.state = 0
        self.endstate = size - 1
        self.action_space: tuple[Action, ...] = (Action.RIGHT, Action.LEFT)
        self.size: int = size
        self.done: bool = False
        self.verbose: bool = verbose

    def play_interactive(self) -> None:
        # // TEAM_005: Renamed run() to play_interactive() to separate debugging from training loops
        self.reset()
        while not self.done:
            action_input: str = input(" Action (0: Right, 1: Left): ")
            try:
                action: int = int(action_input)
            except ValueError:
                print("Invalid input: Please enter an integer (0-1).")
                continue
            self.step(action)
        print(" You Win!")

    def step(self, action: int) -> tuple[int, float, bool]:
        # // TEAM_005: Standardizing return signature (next_state, reward, done)
        if self.colision_detection(action):
            if action == Action.RIGHT:
                self.state += 1
            else:
                self.state -= 1
        self.render()
        
        self.done = True if self.state >= self.endstate else False
        reward: float = 1.0 if self.done else 0.0
        
        return self.state, reward, self.done

    def colision_detection(self, action: int) -> bool:
        state_proposed = self.state + 1 if Action.RIGHT == action else self.state - 1
        if state_proposed < 0:
            return False
        elif state_proposed > self.endstate:
            return False
        return True

    def reset(self) -> int:
        self.state = 0
        self.done = False
        return self.state

    def render(self) -> None:
        observation_space: list[int] = [0 for _ in range(self.size)]
        observation_space[self.state] = 1
        print(f"\r{observation_space}", end="")

    def close(self) -> None:
        pass


class QLearningAgent:
    def __init__(
        self, 
        state_size: int, 
        action_size: int, 
        learning_rate: float = 0.1, 
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01
    ) -> None:
        # // TEAM_006: Initializing strictly typed Q-table
        self.q_table: Annotated[np.ndarray, "State, Action"] = np.zeros((state_size, action_size))
        self.lr: float = learning_rate
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.epsilon_decay: float = epsilon_decay
        self.min_epsilon: float = min_epsilon
        self.action_size: int = action_size

    def choose_action(self, state: int) -> int:
        # // TEAM_006: Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.action_size))
        
        # Avoid arbitrary tie-breaking issues by selecting randomly among max actions
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        return int(np.random.choice(best_actions))

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        # // TEAM_006: Bellman Equation Q-value update
        best_next_action_val = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_action_val
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error
        
    def decay_epsilon(self) -> None:
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay


def train_agent(episodes: int = 500) -> None:
    env = SimplesEnv()
    agent = QLearningAgent(state_size=env.size, action_size=len(env.action_space))
    
    print(f"Training agent for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            agent.update(state, action, float(reward), next_state)
            
            state = next_state
            steps += 1
            
            # Failsafe if it spins endlessly
            if steps > 100:
                break
                
        agent.decay_epsilon()
        
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} completed. Epsilon: {agent.epsilon:.3f}")
            
    print("\nTraining Complete!")
    print("Final Q-Table (Rows=States, Cols=Right/Left):")
    print(np.round(agent.q_table, 3))
    
    # Run a test episode to show it works
    print("\nRunning a greedy test episode:")
    state = env.reset()
    env.render()
    done = False
    
    while not done:
        # Greedy action choice
        action = int(np.argmax(agent.q_table[state]))
        state, _, done = env.step(action)
        env.render()
    print("\nGoal Reached!")


if __name__ == "__main__":
    # Choose to either play interactively or train the agent
    if False:
        env = SimplesEnv(verbose=True)
        env.play_interactive()
    else:    
        train_agent()

