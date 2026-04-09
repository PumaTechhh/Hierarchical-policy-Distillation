import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# --- Neural Network Architecture ---
class QNetwork(nn.Module):
    """
    Feed-Forward Neural Network for the Worker Agent.
    Input:  State Vector (4 for CartPole)
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


# --- Replay Buffer ---
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
        self.buffer.clear()


# --- The Worker Agent ---
class DQNWorker:
    """
    The 'Worker' Agent (Student).

    Architecture overview:
      - standard DQN with epsilon-greedy exploration
      - rolling TD-error as Brittleness Proxy
      - CONTEXT-ROUTED expert memory: each OOD context has its own isolated
        vault so contradictory sabotage types never contaminate each other
      - vault_warm_up(): called on RECURRENCE to re-anchor network weights
        from the stored vault before any episodes run — this is what achieves
        near-zero LLM calls on recurrence
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.batch_size    = 64

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy + Target networks
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory    = ReplayBuffer()

        # THESIS : Rolling average TD-error (window=3 for fast response)
        self.td_error_history = deque(maxlen=3)

        # ---------------------------------------------------------------
        # CONTEXT-ROUTED EXPERT MEMORY
        # ---------------------------------------------------------------
        # A dict keyed by sabotage_type. Each context stores up to 500
        # expert (state, action) pairs from the LLM Supervisor.
        # Isolation prevents inverted_controls and high_gravity advice
        # from contaminating each other during gradient blending.
        self.expert_memory          = {}   # {context_str: deque}
        self.expert_memory_capacity = 500

        # Loss functions
        self.mse_loss = nn.MSELoss()           # Standard Q-learning
        self.bc_loss  = nn.CrossEntropyLoss()  # Behavioral Cloning / Distillation

    # ------------------------------------------------------------------
    def _get_or_create_expert_buffer(self, context):
        if context not in self.expert_memory:
            self.expert_memory[context] = deque(maxlen=self.expert_memory_capacity)
        return self.expert_memory[context]

    def has_prior_knowledge(self, context):
        """Returns True if the vault for this context already has data."""
        return context in self.expert_memory and len(self.expert_memory[context]) > 0

    # ------------------------------------------------------------------
    def vault_warm_up(self, context, steps=150):
        """
        RECURRENCE MECHANISM — the key to near-zero LLM calls on recurrence.

        On a first encounter the network must explore from scratch (epsilon=1.0).
        On a RECURRENCE the vault already contains correct expert knowledge, but
        the Q-network weights have drifted during the intervening normal-operation
        phase. This method immediately re-anchors the network by performing `steps`
        Behavioral Cloning gradient updates from the stored vault — BEFORE any
        episode begins. The network is then already aligned with the correct policy,
        so it rarely makes the mistakes that would trigger the TD-error threshold
        and call the LLM.

        Args:
            context: The sabotage type string (routing key).
            steps:   Number of BC gradient updates to perform (default 150).
        """
        if not self.has_prior_knowledge(context):
            return  # Nothing to warm up from

        buf = self.expert_memory[context]
        print(f"  >> Vault warm-up: {steps} BC steps from '{context}' vault "
              f"({len(buf)} memories)...")

        for _ in range(steps):
            batch_size = min(32, len(buf))
            batch      = random.sample(buf, batch_size)
            states, actions = zip(*batch)

            states_t  = torch.FloatTensor(np.array(states)).to(self.device)
            actions_t = torch.LongTensor(actions).to(self.device)

            self.optimizer.zero_grad()
            q_values = self.policy_net(states_t)
            loss     = self.bc_loss(q_values, actions_t)
            loss.backward()
            self.optimizer.step()

        # Also sync target network after warm-up
        self.update_target_network()
        print(f"  >> Warm-up complete. Network re-anchored to stored expert policy.")

    # ------------------------------------------------------------------
    def select_action(self, state, evaluation_mode=False):
        if not evaluation_mode and np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    # ------------------------------------------------------------------
    def calculate_td_error(self, state, action, reward, next_state, done):
        """
        THESIS  — Brittleness Proxy.
        Returns the rolling average of the last 3 TD-errors to filter noise.
        """
        state_t      = torch.FloatTensor(state).to(self.device)
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        reward_t     = torch.tensor(reward, device=self.device)

        with torch.no_grad():
            current_q  = self.policy_net(state_t)[action]
            max_next_q = self.target_net(next_state_t).max()
            target_q   = reward_t + (1 - done) * self.gamma * max_next_q
            td_error   = abs(target_q - current_q).item()

        self.td_error_history.append(td_error)
        return sum(self.td_error_history) / len(self.td_error_history)

    # ------------------------------------------------------------------
    def distill_policy(self, state, expert_action, context):
        """
        Online Policy Distillation.
        Stores the expert label in the context-specific vault and performs
        an immediate BC update using only that vault.
        """
        buf = self._get_or_create_expert_buffer(context)
        buf.append((state, expert_action))

        batch_size = min(32, len(buf))
        batch      = random.sample(buf, batch_size)
        states, actions = zip(*batch)

        states_t  = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)

        self.optimizer.zero_grad()
        loss = self.bc_loss(self.policy_net(states_t), actions_t)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # ------------------------------------------------------------------
    def train_step(self, active_context=None):
        """
        Interleaved Policy Distillation — Gradient Blending (THESIS ALIGNMENT).

        Combines L_MSE (standard Q-learning) with L_BC (expert BC) in one
        update. The BC term ONLY samples from the vault matching active_context,
        ensuring zero cross-contamination between OOD failure types.
        """
        if len(self.memory) < self.batch_size:
            return None

        # --- Standard Q-Learning ---
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state_t      = torch.FloatTensor(state).to(self.device)
        action_t     = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward_t     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        done_t       = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        q_values = self.policy_net(state_t).gather(1, action_t)
        with torch.no_grad():
            next_q     = self.target_net(next_state_t).max(1)[0].unsqueeze(1)
            expected_q = reward_t + (self.gamma * next_q * (1 - done_t))

        rl_loss    = self.mse_loss(q_values, expected_q)
        total_loss = rl_loss

        # --- Context-Routed BC (only when sabotage active and vault has data) ---
        if active_context is not None and active_context in self.expert_memory:
            buf = self.expert_memory[active_context]
            if len(buf) > 0:
                exp_batch_size = min(16, len(buf))
                exp_batch      = random.sample(buf, exp_batch_size)
                exp_states, exp_actions = zip(*exp_batch)

                exp_states_t  = torch.FloatTensor(np.array(exp_states)).to(self.device)
                exp_actions_t = torch.LongTensor(exp_actions).to(self.device)

                bc_loss    = self.bc_loss(self.policy_net(exp_states_t), exp_actions_t)
                total_loss = rl_loss + 1.0 * bc_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return total_loss.item()

    # ------------------------------------------------------------------
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())