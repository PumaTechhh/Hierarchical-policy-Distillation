import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- Neural Network Architecture ---
class QNetwork(nn.Module):
    """
    A simple Feed-Forward Neural Network for the Worker.
    Input: State Vector (4 for CartPole)
    Output: Q-Values for each Action (2 for CartPole)
    """
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# --- The Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)
        
    def clear(self):
        """Clears the buffer to prevent poisoning after a fundamental physics shift."""
        self.buffer.clear()

# --- The Worker Agent ---
class DQNWorker:
    """
    The 'Worker' Agent (Student).
    Capable of standard DQN learning AND Online Policy Distillation.
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = 64

        # Device configuration (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net is not trained directly

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()

        # THESIS ALIGNMENT: Track TD-error for rolling average
        self.td_error_history = deque(maxlen=3) # 10-step rolling window

        # DEDICATED DISTILLATION BUFFER
        self.expert_memory = deque(maxlen=1000)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()  # For standard RL
        self.bc_loss = nn.CrossEntropyLoss() # For Behavioral Cloning (Distillation)

    def select_action(self, state, evaluation_mode=False):
        """
        Selects an action using Epsilon-Greedy strategy.
        """
        if not evaluation_mode and np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def calculate_td_error(self, state, action, reward, next_state, done):
        """
        THESIS REQUIREMENT: Brittleness Proxy.
        Calculates the ROLLING AVERAGE of the TD-Error to detect 'surprise' 
        without triggering on momentary mathematical noise.
        """
        state_t = torch.FloatTensor(state).to(self.device)
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        reward_t = torch.tensor(reward, device=self.device)
        
        with torch.no_grad():
            current_q = self.policy_net(state_t)[action]
            max_next_q = self.target_net(next_state_t).max()
            target_q = reward_t + (1 - done) * self.gamma * max_next_q
            
            td_error = abs(target_q - current_q).item()
            
        # THESIS ALIGNMENT: Section 1.3.2.1 - Rolling Average
        self.td_error_history.append(td_error)
        rolling_avg = sum(self.td_error_history) / len(self.td_error_history)
            
        return rolling_avg

    def train_step(self):
        """
        THESIS ALIGNMENT: Interleaved Policy Distillation.
        Blends standard Q-learning (MSE) with Expert Distillation (BC) 
        to ensure robust recovery from any unnotified OOD state.
        """
        if len(self.memory) < self.batch_size:
            return None

        # --- 1. Standard Q-Learning (Self-Exploration) ---
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state_t = torch.FloatTensor(state).to(self.device)
        action_t = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward_t = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        done_t = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        q_values = self.policy_net(state_t).gather(1, action_t)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_t).max(1)[0].unsqueeze(1)
            expected_q_values = reward_t + (self.gamma * next_q_values * (1 - done_t))

        rl_loss = self.mse_loss(q_values, expected_q_values)
        
        # Initialize total loss as just the RL loss
        total_loss = rl_loss

        # --- 2. Interleaved Expert Distillation (Teacher Guidance) ---
        if len(self.expert_memory) > 0:
            exp_batch_size = min(16, len(self.expert_memory))
            exp_batch = random.sample(self.expert_memory, exp_batch_size)
            exp_states, exp_actions = zip(*exp_batch)
            
            exp_states_t = torch.FloatTensor(np.array(exp_states)).to(self.device)
            exp_actions_t = torch.LongTensor(exp_actions).to(self.device)
            
            exp_q_values = self.policy_net(exp_states_t)
            bc_loss = self.bc_loss(exp_q_values, exp_actions_t)
            
            # Gradient Blending: Combine RL exploration with LLM guidance
            bc_weight = 1.0  # Hyperparameter controlling trust in the Supervisor
            total_loss = rl_loss + (bc_weight * bc_loss)

        # --- 3. Unified Network Update ---
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Standard Epsilon Decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return total_loss.item()

    def distill_policy(self, state, expert_action):
        """
        THESIS REQUIREMENT: Online Policy Distillation.
        Safely performs multi-step Behavioral Cloning using a dedicated 
        expert buffer, preventing Q-learning pollution.
        """
        # Save the expert label
        self.expert_memory.append((state, expert_action))
        
        # Sample a batch of expert knowledge
        batch_size = min(32, len(self.expert_memory))
        batch = random.sample(self.expert_memory, batch_size)
        states, actions = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        
        self.optimizer.zero_grad()
        
        # Get Q-values for the batch and apply BC loss
        batch_q_values = self.policy_net(states_t)
        loss = self.bc_loss(batch_q_values, actions_t)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())