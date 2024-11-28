"""
Random Shooting Algorithm.

This algorithm optimizes a sequence of future actions to maximize
the expected cumulative reward over a planning horizon using a
model of the environment. At each time step:

1. Randomly generate multiple candidate action sequences.
2. Simulate the future rewards for each sequence using the environment model.
3. Select the action sequence with the highest predicted cumulative reward.
4. Execute the first action of the optimal sequence in the environment and re-plan at the next step.
"""
import gymnasium as gym
import numpy as np

def random_shooting(
    horizon: int,
    num_samples: int,
    env: gym.Env,
):
    """
    Parameters:
        horizon: Length of each action sequence.
        num_samples: Number of random action sequences to sample.
        env: Gymnasium environment object.

    Returns:
        Total reward achieved using Random Shooting.
    """
    total_reward = 0
    state, _ = env.reset()
    done = False

    while not done:
        action_sequences = np.random.randint(
            0, env.action_space.n,
            size=(num_samples, horizon)
        )
        rewards = np.zeros(num_samples)

        # Evaluate action sequences
        for i in range(num_samples):
            sim_env = gym.make(env.unwrapped.spec.id)
            sim_env.reset()
            # Set to the true current state
            sim_env.unwrapped.state = state

            cumulative_reward = 0
            for action in action_sequences[i]:
                _, reward, terminated, truncated, _ = sim_env.step(action)
                cumulative_reward += reward
                if terminated or truncated:
                    break
            rewards[i] = cumulative_reward
            sim_env.close()

        best_action = action_sequences[np.argmax(rewards)][0]
        print(f"Best action: {best_action}")

        state, reward, done, truncated, _ = env.step(best_action)
        total_reward += reward

    return total_reward

if __name__ == "__main__":
    env = gym.make(
        "CartPole-v1",
        render_mode="human"
    )
    NB_ACTIONS = env.action_space.n
    print("Number of actions: ", NB_ACTIONS)
    obs, _ = env.reset()
    reward = random_shooting(horizon=20, num_samples=1000, env=env)
    print(f"Total Reward: {reward}")
