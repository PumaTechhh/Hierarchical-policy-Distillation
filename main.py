import numpy as np
import matplotlib.pyplot as plt
from src.environment import SabotagedCartPole
from src.worker import DQNWorker
from src.supervisor import LocalLLaMASupervisor

def plot_metrics(rewards, supervisor_calls, sabotage_episode, sabotage_type):
    """Generates the evaluation graphs defined in the thesis."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Reward / Time-to-Recovery (TTR)
    ax1.plot(rewards, label='Episode Reward', color='blue')
    ax1.axvline(x=sabotage_episode, color='red', linestyle='--', label='Sabotage Triggered')
    ax1.set_title(f'Worker Performance (TTR) - {sabotage_type}')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    
    # Plot 2: Supervisor Activation Decay
    ax2.plot(supervisor_calls, label='Supervisor Calls', color='orange')
    ax2.axvline(x=sabotage_episode, color='red', linestyle='--')
    ax2.set_title(f'Supervisor Activation Decay - {sabotage_type}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Number of LLM Queries')
    ax2.legend()
    
    plt.tight_layout()
    filename = f'hpd_results_{sabotage_type}.png'
    plt.savefig(filename)
    print(f"\n>>> Results saved to '{filename}'")

def run_hpd_experiment(sabotage_type='inverted_controls'):
    # --- 1. Initialization ---
    env = SabotagedCartPole(render_mode=None) 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    worker = DQNWorker(state_dim, action_dim)
    supervisor = LocalLLaMASupervisor(model_name="llama3.2")
    
    # --- 2. Experiment Parameters ---
    num_episodes = 150
    sabotage_episode = 75       
    stability_threshold = 1.0
    
    # --- Metrics Tracking ---
    episode_rewards = []
    supervisor_calls_history = []
    
    print(f"\nStarting HPD Experimental Protocol: {sabotage_type.upper()}...")
    
    # --- 3. The Reactive Learning Loop ---
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        supervisor_calls_this_episode = 0
        
        # Phase 2: Trigger Sabotage Protocol dynamically
        if episode == sabotage_episode:
            print(f"\n{'='*50}\nEPISODE {episode}: INITIATING SABOTAGE PROTOCOL ({sabotage_type})\n{'='*50}")
            env.trigger_sabotage(sabotage_type=sabotage_type)
            
            # UNIVERSAL RECOVERY PROTOCOL (Thesis Aligned)
            # 1. Flush all prior assumptions about environment dynamics
            worker.memory.clear()
            worker.td_error_history.clear() 
            worker.expert_memory.clear() 
            print(">>> All Knowledge Buffers Flushed.")
            
            # 2. Universal Epsilon Reset to force OOD exploration
            worker.epsilon = 1.0 
            print(">>> Epsilon Reset to 1.0 for unbiased exploration.")

        while not (done or truncated):
            action = worker.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            td_error = worker.calculate_td_error(state, action, reward, next_state, done)
            
            if td_error > stability_threshold and info.get('sabotage_active'):
                expert_action = supervisor.get_expert_action(state, info.get('sabotage_type'))
                supervisor_calls_this_episode += 1
                worker.distill_policy(state, expert_action)
                action = expert_action 
            
            worker.memory.push(state, action, reward, next_state, done)
            worker.train_step()
            state = next_state
            
        if episode % 10 == 0:
            worker.update_target_network()
            
        episode_rewards.append(total_reward)
        supervisor_calls_history.append(supervisor_calls_this_episode)
        
        if episode % 10 == 0 or supervisor_calls_this_episode > 0:
            print(f"Episode {episode:3d} | Reward: {total_reward:5.1f} | Epsilon: {worker.epsilon:.2f} | Supervisor Calls: {supervisor_calls_this_episode}")

    env.close()
    
    # --- 4. Evaluation ---
    plot_metrics(episode_rewards, supervisor_calls_history, sabotage_episode, sabotage_type)

if __name__ == "__main__":
    # You can now easily comment out or run multiple experiments back-to-back!
    
    # Experiment 1
    run_hpd_experiment(sabotage_type='inverted_controls')
    
    # Experiment 2
    #run_hpd_experiment(sabotage_type='high_gravity')