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
        Calculates the TD-Error for a SINGLE transition to detect 'surprise'.
        Returns the absolute scalar error value.
        """
        state_t = torch.FloatTensor(state).to(self.device)
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        reward_t = torch.tensor(reward, device=self.device)
        
        with torch.no_grad():
            # Current Q-value estimate
            current_q = self.policy_net(state_t)[action]
            
            # Target Q-value (Bellman)
            max_next_q = self.target_net(next_state_t).max()
            target_q = reward_t + (1 - done) * self.gamma * max_next_q
            
            # TD Error = |Target - Current|
            td_error = abs(target_q - current_q).item()
            
        return td_error

    def train_step(self):
        """
        Standard DQN Training Step (Offline/Online RL).
        """
        if len(self.memory) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # Q(s, a)
        q_values = self.policy_net(state).gather(1, action)

        # V(s') = max Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_state).max(1)[0].unsqueeze(1)
            expected_q_values = reward + (self.gamma * next_q_values * (1 - done))

        loss = self.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon Decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def distill_policy(self, state, expert_action):
        """
        THESIS REQUIREMENT: Online Policy Distillation.
        Performs a Behavioral Cloning update using the Supervisor's expert action.
        """
        self.optimizer.zero_grad()
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        expert_action_tensor = torch.LongTensor([expert_action]).to(self.device)
        
        # Get raw logits (Q-values) from the network
        q_values = self.policy_net(state_tensor)
        
        # We treat Q-values as logits for classification (which action is 'correct'?)
        # CrossEntropyLoss expects (Batch, Class_Logits) and (Batch, Target_Class_Indices)
        loss = self.bc_loss(q_values, expert_action_tensor)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())