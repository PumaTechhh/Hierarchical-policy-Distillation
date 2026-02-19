import numpy as np
import matplotlib.pyplot as plt
from src.environment import SabotagedCartPole
from src.worker import DQNWorker
from src.supervisor import LocalLLaMASupervisor

def plot_metrics(rewards, supervisor_calls, sabotage_episode):
    """Generates the evaluation graphs defined in the thesis."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Reward / Time-to-Recovery (TTR)
    ax1.plot(rewards, label='Episode Reward', color='blue')
    ax1.axvline(x=sabotage_episode, color='red', linestyle='--', label='Sabotage Triggered')
    ax1.set_title('Worker Performance (Time-to-Recovery)')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    
    # Plot 2: Supervisor Activation Decay
    ax2.plot(supervisor_calls, label='Supervisor Calls', color='orange')
    ax2.axvline(x=sabotage_episode, color='red', linestyle='--')
    ax2.set_title('Supervisor Activation Decay')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Number of LLM Queries')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('hpd_results.png')
    print("\n>>> Results saved to 'hpd_results.png'")

def run_hpd_experiment():
    # --- 1. Initialization ---
    env = SabotagedCartPole(render_mode=None) # Set to 'human' if you want to watch it, but it slows training
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    worker = DQNWorker(state_dim, action_dim)
    supervisor = LocalLLaMASupervisor(model_name="llama3.2")
    
    # --- 2. Experiment Parameters ---
    num_episodes = 150
    sabotage_episode = 75       # Phase 2: Introduce perturbation here
    stability_threshold = 2.5   # Brittleness Proxy: TD-Error threshold to trigger Supervisor
    
    # --- Metrics Tracking ---
    episode_rewards = []
    supervisor_calls_history = []
    
    print("Starting HPD Experimental Protocol...")
    
    # --- 3. The Reactive Learning Loop ---
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        supervisor_calls_this_episode = 0
        
        # Phase 2: Trigger Sabotage Protocol
        if episode == sabotage_episode:
            print(f"\n{'='*50}\nEPISODE {episode}: INITIATING SABOTAGE PROTOCOL\n{'='*50}")
            env.trigger_sabotage(sabotage_type='inverted_controls')

        while not (done or truncated):
            # Worker selects an action
            action = worker.select_action(state)
            
            # Environment step
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            # THESIS REQUIREMENT: Calculate TD-Error to check Stability
            td_error = worker.calculate_td_error(state, action, reward, next_state, done)
            
            # Stability Check (The Gatekeeper)
            if td_error > stability_threshold and info.get('sabotage_active'):
                # --- SYNCHRONOUS CORRECTION ROUTINE ---
                # 1. System Pause (implicit by blocking execution)
                
                # 2. Supervisor Query & Symbol Grounding
                expert_action = supervisor.get_expert_action(state, info.get('sabotage_type'))
                supervisor_calls_this_episode += 1
                
                # 3. Online Policy Distillation (Behavioral Cloning)
                worker.distill_policy(state, expert_action)
                
                # Override the agent's action with the expert's for the replay buffer
                action = expert_action 
            
            # Standard DQN Buffer Push & Train
            worker.memory.push(state, action, reward, next_state, done)
            worker.train_step()
            
            state = next_state
            
        # Target Network Update
        if episode % 10 == 0:
            worker.update_target_network()
            
        # Logging
        episode_rewards.append(total_reward)
        supervisor_calls_history.append(supervisor_calls_this_episode)
        
        if episode % 10 == 0 or supervisor_calls_this_episode > 0:
            print(f"Episode {episode:3d} | Reward: {total_reward:5.1f} | Epsilon: {worker.epsilon:.2f} | Supervisor Calls: {supervisor_calls_this_episode}")

    env.close()
    
    # --- 4. Evaluation ---
    plot_metrics(episode_rewards, supervisor_calls_history, sabotage_episode)

if __name__ == "__main__":
    run_hpd_experiment()