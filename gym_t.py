import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

# Hyperparameters
ENV_NAME = "Pendulum-v1"
SEED = 1234
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
ALPHA = 0.2
HIDDEN_DIM = 256
BATCH_SIZE = 256
REPLAY_BUFFER_SIZE = 1e6
TARGET_UPDATE_INTERVAL = 1
NUM_EPISODES = 15
MAX_STEPS = 200

# Environment
env = gymnasium.make(ENV_NAME)
env.reset(seed=SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, size):
        self.size = int(size)
        self.buffer = []
        self.ptr = 0

    def push(self, transition):
        if len(self.buffer) < self.size:
            self.buffer.append(transition)
        else:
            self.buffer[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.size

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices]
        )
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


def check_weights(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in {name}")


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        for i, layer in enumerate(self.network):
            x = layer(x)
            if torch.isnan(x).any():
                print(f"NaN detected after layer {i} ({layer})")
                print(f"Input: {x}")
        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_range):
        super(Actor, self).__init__()
        self.action_range = action_range
        self.network = MLP(state_dim, 2 * action_dim, hidden_dim)

    def forward(self, state):
        mean, log_std = torch.chunk(self.network(state), 2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)

        # Clamping to avoid NaNs
        mean = torch.clamp(mean, -1e2, 1e2)
        std = torch.clamp(std, 1e-2, 1e2)

        # Debugging statements
        if torch.isnan(mean).any():
            print("NaN detected in mean")
        if torch.isnan(std).any():
            print("NaN detected in std")
        if torch.isnan(log_std).any():
            print("NaN detected in log_std")

        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick
        action = torch.tanh(x_t) * self.action_range
        log_prob = normal.log_prob(x_t) - torch.log(
            self.action_range * (1 - action.pow(2)) + 1e-6
        )

        # Debugging statements
        if (
            torch.isnan(mean).any()
            or torch.isnan(std).any()
            or torch.isnan(log_prob).any()
        ):
            print("NaN detected in sampling")

        return action, log_prob.sum(dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.network = MLP(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=-1))


# Initialize Networks
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_range = env.action_space.high[0]

actor = Actor(state_dim, action_dim, HIDDEN_DIM, action_range)
critic_1 = Critic(state_dim, action_dim, HIDDEN_DIM)
critic_2 = Critic(state_dim, action_dim, HIDDEN_DIM)
value = MLP(state_dim, 1, HIDDEN_DIM)
target_value = MLP(state_dim, 1, HIDDEN_DIM)

# Check for NaNs in initial weights
check_weights(actor)
check_weights(critic_1)
check_weights(critic_2)
check_weights(value)

# Initialize target value network
target_value.load_state_dict(value.state_dict())

# Optimizers
actor_optimizer = optim.Adam(actor.parameters(), lr=LR)
critic_1_optimizer = optim.Adam(critic_1.parameters(), lr=LR)
critic_2_optimizer = optim.Adam(critic_2.parameters(), lr=LR)
value_optimizer = optim.Adam(value.parameters(), lr=LR)

# Replay Buffer
replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)


def update_parameters(batch_size):
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    if torch.isnan(states).any():
        print("NaN detected in states")
    if torch.isnan(actions).any():
        print("NaN detected in actions")
    if torch.isnan(rewards).any():
        print("NaN detected in rewards")
    if torch.isnan(next_states).any():
        print("NaN detected in next_states")
    if torch.isnan(dones).any():
        print("NaN detected in dones")

    with torch.no_grad():
        next_state_actions, next_state_log_pis = actor.sample(next_states)
        q_target_1 = critic_1(next_states, next_state_actions)
        q_target_2 = critic_2(next_states, next_state_actions)
        min_q_target = torch.min(q_target_1, q_target_2) - ALPHA * next_state_log_pis
        q_target = rewards + (1 - dones) * GAMMA * min_q_target

    v = value(states)
    value_loss = ((v - q_target) ** 2).mean()
    if torch.isnan(value_loss).any():
        print("NaN detected in value_loss")

    value_optimizer.zero_grad()
    value_loss.backward()
    nn.utils.clip_grad_norm_(value.parameters(), max_norm=1.0)
    value_optimizer.step()

    q1 = critic_1(states, actions)
    q2 = critic_2(states, actions)
    v = value(states).detach()
    critic_1_loss = ((q1 - v) ** 2).mean()
    critic_2_loss = ((q2 - v) ** 2).mean()

    if torch.isnan(critic_1_loss).any():
        print("NaN detected in critic_1_loss")
    if torch.isnan(critic_2_loss).any():
        print("NaN detected in critic_2_loss")

    critic_1_optimizer.zero_grad()
    critic_1_loss.backward()
    nn.utils.clip_grad_norm_(critic_1.parameters(), max_norm=1.0)
    critic_1_optimizer.step()

    critic_2_optimizer.zero_grad()
    critic_2_loss.backward()
    nn.utils.clip_grad_norm_(critic_2.parameters(), max_norm=1.0)
    critic_2_optimizer.step()

    new_actions, log_pis = actor.sample(states)
    q1_new = critic_1(states, new_actions)
    q2_new = critic_2(states, new_actions)
    min_q_new = torch.min(q1_new, q2_new)
    policy_loss = (ALPHA * log_pis - min_q_new).mean()

    if torch.isnan(policy_loss).any():
        print("NaN detected in policy_loss")

    actor_optimizer.zero_grad()
    policy_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    actor_optimizer.step()

    for target_param, param in zip(target_value.parameters(), value.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


for episode in range(NUM_EPISODES):
    state = env.reset()[0]
    if torch.isnan(torch.FloatTensor(state)).any():
        print(f"NaN detected in initial state: {state}")
        continue  # Skip this episode if initial state is invalid

    episode_reward = 0

    for step in range(MAX_STEPS):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, _ = actor.sample(state_tensor)
        action = action.detach().cpu().numpy()[0]
        next_state, reward, done, _, _ = env.step(action)

        # Check for NaN in transitions
        if any(np.isnan([reward, done])) or np.isnan(next_state).any():
            print(
                f"NaN detected in transition: state={state}, action={action}, reward={reward}, next_state={next_state}, done={done}"
            )
            break  # Skip to next episode if NaN is detected

        replay_buffer.push((state, action, reward, next_state, done))
        state = next_state
        episode_reward += reward

        if len(replay_buffer) > BATCH_SIZE:
            update_parameters(BATCH_SIZE)

        if done:
            break

    print(f"Episode {episode + 1}, Reward: {episode_reward}")

env.close()
