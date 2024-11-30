"""
Implementaion of Monte Carlo Tree Search in one file.

MCTS is an online planning algorithm that uses tree search and Monte Carlo simulations
to find the best action to take in a given state. At each time step:

1. Selection: Traverse the tree from the root node to a leaf node using an action
selection algorithm (UCB).
2. Expansion: Expand the leaf node by adding one or more child nodes.
3. Simulation: Simulate a rollout from the child node using a default policy.
4. Backup: Update the statistics of the nodes in the tree.
"""
import numpy as np
import gymnasium as gym
import time

from typing import Optional, Dict

def make_env(
    env_name: str,
    render_mode: str | None=None
) -> gym.Env:
    """
    """
    return gym.make(env_name, render_mode=render_mode)

class VNode(object):
    """
    """
    def __init__(
        self,
        state: np.ndarray,
        parent: Optional['QNode'],
        ucb_constant: float=np.sqrt(2)
    ):
        self.state = state
        self.ucb_constant = ucb_constant

        self.children: Dict[int, 'QNode'] = {}
        self.parent = parent
        self.visits = 0        

    def __repr__(self):
        return self.__repr__()

    def __str__(self):
        return f"Node({self.state}, {self.visits}, {list(self.children.keys())})"

    @property
    def is_leaf(self):
        return len(self.children) == 0

    def is_fully_expanded(
        self,
        action_space: gym.spaces.Discrete
    ):
        return len(self.children) == action_space.n

    def add_child(
        self,
        action: int,
        child: 'QNode'
    ):
        self.children[action] = child

    def get_child(
        self,
        action: int
    ):
        return self.children[action]

    def compute_ucb(
        self,
        action: int,
    ):
        """
        """
        fraction = np.log(self.visits+1)/(self.children[action].visits+1)
        exploration = self.ucb_constant*np.sqrt(fraction)
        return self.children[action].value + exploration

    def select_action(
        self,
        action_space: gym.spaces.Discrete
    ):
        """
        """
        if self.is_leaf:
            return np.random.randint(0, action_space.n)
        elif not self.is_fully_expanded(action_space):
            return np.random.choice([
                action for action in range(action_space.n) if action not in self.children.keys()
            ])
        else:
            ucb_values = np.array([
                self.compute_ucb(action) for action in range(action_space.n)
            ])
            selected_action = np.argmax(ucb_values)
            return selected_action

    def update(self):
        """
        """
        self.visits += 1

class QNode(object):
    """
    """
    def __init__(
        self,
        action: int,
        imediate_reward: float,
        parent: VNode,
        child: VNode
    ):
        self.action = action
        self.parent = parent
        self.child: VNode = child

        self.imediate_reward = imediate_reward
        self.value = 0
        self.visits = 0

    @property
    def get_child(self):
        return self.child

    def __str__(self):
        return f"QNode({self.action}, {self.visits}, {self.value})"

    def __repr__(self):
        return self.__str__()

    def update(
        self,
        total_reward: float
    ):
        """
        """
        self.visits += 1
        self.value += (total_reward-self.value)/self.visits

class MCTS(object):
    """
    TODO: Fix rollout horizon for continuing tasks.
    """
    def __init__(
        self,
        env: gym.Env,
        time_limit: int=60,
        ucb_constant: float=np.sqrt(2),
    ):
        self.sim_env = gym.make(env.unwrapped.spec.id)
        state, _ = self.sim_env.reset()

        # Root of the tree it contains the real state of the environment.
        self.root = VNode(state, None)

        self.time_limit = time_limit
        self.ucb_constant = ucb_constant

    def get_tree_depth(self) -> int:
        """
        """
        max_depth = 0
        stack = [(self.root, 1)]
        while stack:
            node, depth = stack.pop()
            max_depth = max(max_depth, depth)
            for child in node.children.values():
                stack.append((child.child, depth + 1))
        return max_depth

    def select_node_to_expand(self) -> VNode:
        """
        """
        node = self.root
        while not node.is_leaf and node.is_fully_expanded(self.sim_env.action_space):
            if isinstance(node, VNode):
                action = node.select_action(self.sim_env.action_space)
                node = node.get_child(action).child
        return node

    def expand(
        self,
        node: VNode,
        action: int
    ) -> VNode:
        """
        """
        self.sim_env.reset()
        self.sim_env.unwrapped.state = node.state
        new_state, reward, done, truncated, _ = self.sim_env.step(action)

        if done or truncated:
            return None

        new_qnode = QNode(action, reward, node, None)
        node.add_child(action, new_qnode)
        new_vnode = VNode(new_state, new_qnode)
        new_qnode.child = new_vnode
        return new_vnode

    def simulate(
        self,
        node: VNode
    ) -> float:
        """
        """
        state = node.state
        self.sim_env.reset()
        self.sim_env.unwrapped.state = state

        total_reward = 0
        done = False
        while not done:
            action = self.sim_env.action_space.sample()
            _, reward, done, truncated, _ = self.sim_env.step(action)
            total_reward += reward
            if truncated:
                break
        return total_reward

    def plan(self) -> int:
        """
        Returns the action to take in the real environment.
        """
        print("Planning...")
        start_time = time.time()
        while time.time()-start_time < self.time_limit:
            # Selection
            node = self.select_node_to_expand()
            action = node.select_action(self.sim_env.action_space)

            # Expansion
            child_node = self.expand(node, action)
            if child_node is None:
                continue

            # Simulation
            total_reward = self.simulate(child_node)

            # Backup
            iterator = child_node
            while iterator is not None:
                if isinstance(iterator, QNode):
                    total_reward += iterator.imediate_reward
                    iterator.update(total_reward)
                elif isinstance(iterator, VNode):
                    iterator.update()
                iterator = iterator.parent

        action_to_take = self.root.select_action(self.sim_env.action_space)
        print(f"Action to take: {action_to_take}")
        self.prune(action_to_take)
        return action_to_take

    def prune(
        self,
        action: int
    ):
        """
        Prune the tree after the action has been taken.
        """
        self.root = self.root.get_child(action).get_child
        self.root.parent = None

def run_mcts(
    env: gym.Env,
    time_limit: int=60,
    ucb_constant: float=np.sqrt(2),
    seed: int | None=None
):
    """
    Run MCTS on the given environment.
    """
    if seed is not None:
        np.random.seed(seed)

    obs, _ = env.reset(seed=seed)
    mcts = MCTS(
        env,
        time_limit,
        ucb_constant
    )

    total_reward = 0
    done = False
    while not done:
        action = mcts.plan()
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if truncated:
            break

    return total_reward

if __name__ == "__main__":
    ENV_NAME = "CartPole-v1"
    TIME_LIMIT = 20
    UCB_CONSTANT = np.sqrt(2)
    SEED = 42
    env = gym.make(ENV_NAME, render_mode="human")
    total_reward = run_mcts(env, TIME_LIMIT, UCB_CONSTANT, SEED)
    print(f"Total Reward: {total_reward}")
    env.close()
