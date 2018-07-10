"""
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import random
from collections import namedtuple
from itertools import count
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import seoulai_gym as gym
from seoulai_gym.envs.checkers.base import Constants
from seoulai_gym.envs.checkers.agents import Agent
from seoulai_gym.envs.checkers.agents import RandomAgentLight


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQNAgent(Agent):
    def __init__(
        self,
        name: str,
        ptype: int,
    ):
        """Initialize DQN agent.

        Args:
            name: name of agent.
            ptype: type of piece that agent is responsible for.
        """
        super().__init__(name, ptype)

    def act(
        self,
        board: List[List],
        reward: int,
        done: bool,
    ) -> Tuple[int, int, int, int]:
        """
        Choose a piece and its possible moves randomly.
        Pieces and moves are chosen from all current valid possibilities.

        Args:
            board: information about positions of pieces.
            reward: reward for perfomed step.
            done: information about end of game.

        Returns:
            Current and new location of piece.
        """
        board_size = len(board)
        valid_moves = self.generate_valid_moves(board, self.ptype, board_size)
        rand_from_row, rand_from_col = random.choice(list(valid_moves.keys()))
        rand_to_row, rand_to_col = random.choice(valid_moves[(rand_from_row, rand_from_col)])
        return rand_from_row, rand_from_col, rand_to_row, rand_to_col

    def select_action(
        self,
        obs,
    ):
        from_row = 0
        from_col = 0
        to_row = 0
        to_col = 0

        from_row = 0
        from_col = 0
        to_row = 0
        to_col = 0


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            # FIXME why not in one step?
            self.memory.append(None)
            self.memory[self.position] = Transition(*args)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


env = gym.make("Checkers")
num_episodes = 50   # FIXME
train_agent = DQNAgent("DQNLightAgent", ptype=Constants().LIGHT)
random_agent = RandomAgentLight("Agent 1")


for i_episode in range(num_episodes):
    # Initialize the environment and state
    obs = env.reset()
    # last_screen = get_screen()
    # current_screen = get_screen()
    # state = current_screen - last_screen

    for t in count():
        # Select and perform an action
        # action = select_action(state)
        last_obs = obs
        # action = tain_agent.select_action(obs)
        # _, reward, done, _ = env.step(action.item())
        obs, rew, done, _ = env.step(*action)
        reward = torch.tensor([reward], device=device)

        # Observe new state
        # last_screen = current_screen
        # current_screen = get_screen()
        if not done:
            next_obs = obs
        else:
            next_obs = None

        # Store the transition in memory
        # memory.push(state, action, next_state, reward)
        memory.push(last_obs, action, next_obs, rew)

        # Move to the next state
        obs = next_obs

        # Perform one step of the optimization (on the target network)
        from IPython import embed; embed()  # XXX DEBUG

        optimize_model()
        if done:
            # episode_durations.append(t + 1)
            # plot_durations()
            break

    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

# print('Complete')
env.close()

# if __name__ == "__main__":
    # main()
