import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator 
from src.environment import SabotagedCartPole
from src.worker import DQNWorker
from src.supervisor import LocalLLaMASupervisor


def plot_metrics(rewards, supervisor_calls, sabotage_episode, eval_episode, sabotage_type):
    """Generates the evaluation graphs defined in the thesis."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.2))
    
    # Plot 1: Worker Performance (TTR)
    ax1.plot(rewards, label='Episode Reward', color='blue', linewidth=1.5)
    ax1.axvline(x=sabotage_episode, color='red', linestyle='--', label='Sabotage Triggered', linewidth=1.5)
    ax1.axvline(x=eval_episode, color='green', linestyle=':', label='Evaluation Mode', linewidth=1.5)
    
    ax1.set_title('Worker Performance (TTR)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.tick_params(axis='both', labelsize=11)
    
    ax1.grid(True, linestyle='--', alpha=0.6) # Adds the grid
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8)) # Cleans up X-axis ticks
    ax1.legend(fontsize=11)
    
    # Plot 2: Supervisor Activation Decay
    ax2.plot(supervisor_calls, label='Supervisor Calls', color='orange', linewidth=1.5)
    ax2.axvline(x=sabotage_episode, color='red', linestyle='--', linewidth=1.5)
    ax2.axvline(x=eval_episode, color='green', linestyle=':', label='Evaluation Mode', linewidth=1.5)
    
    ax2.set_title('Supervisor Activation Decay', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Number of LLM Queries', fontsize=12)
    ax2.tick_params(axis='both', labelsize=11)
    
    ax2.grid(True, linestyle='--', alpha=0.6) # Adds the grid
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8)) # Cleans up X-axis ticks
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True)) # Keeps Y-axis whole numbers
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    filename = f'hpd_results_{sabotage_type}_paper.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight') # bbox_inches strips extra white space
    print(f"\n>>> Results saved to '{filename}'")
    plt.show()

#  Main Single-Sabotage Experiment (Inverted Controls)
def run_paper_experiment():
    # --- 1. Initialization ---
    env = SabotagedCartPole(render_mode=None) 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    worker = DQNWorker(state_dim, action_dim)
    supervisor = LocalLLaMASupervisor(model_name="llama3.2")
    
    # --- 2. Experiment Parameters ---
    num_episodes = 150
    sabotage_episode = 75       
    eval_episode = 125          # Point of strict independence testing
    sabotage_type = 'inverted_controls'
    
    stability_threshold = 1.0
    episode_rewards = []
    supervisor_calls_history = []
    
    print(f"\nStarting HPD Paper Protocol: {sabotage_type.upper()}...")
    
    # --- 3. The Reactive Learning Loop ---
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        supervisor_calls_this_episode = 0
        active_context = env.sabotage_type
        
        # Phase 2: Trigger Sabotage Protocol dynamically
        if episode == sabotage_episode:
            print(f"\n{'='*50}\nEPISODE {episode}: INITIATING SABOTAGE PROTOCOL ({sabotage_type})\n{'='*50}")
            env.trigger_sabotage(sabotage_type=sabotage_type)
            active_context = sabotage_type
            
            # UNIVERSAL RECOVERY PROTOCOL (Amnesia)
            worker.memory.clear()
            worker.td_error_history.clear() 
            worker.expert_memory.clear() 
            print(">>> All Knowledge Buffers Flushed.")
            
            worker.epsilon = 1.0 
            print(">>> Epsilon Reset to 1.0 for unbiased exploration.")

        # Phase 3: Evaluation Mode (Proving Knowledge Retention)
        if episode == eval_episode:
            print(f"\n{'='*50}\nEPISODE {episode}: ENTERING EVALUATION MODE\n{'='*50}")
            # Re-anchor the network to the expert vault before testing
            #worker.vault_warm_up(context=active_context, steps=150)
            
            # Override epsilon_min so it actually hits zero
            worker.epsilon = 0.0
            worker.epsilon_min = 0.0 
            
            stability_threshold = 3.0
            print(">>> Epsilon=0.0 (Exploitation), Threshold=3.0 (Strict Independence)")

        # Agent Execution Loop
        while not (done or truncated):
            action = worker.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            td_error = worker.calculate_td_error(state, action, reward, next_state, done)
            
            if td_error > stability_threshold and info.get('sabotage_active'):
                expert_action = supervisor.get_expert_action(state, active_context)
                supervisor_calls_this_episode += 1
                worker.distill_policy(state, expert_action, context=active_context)
                action = expert_action 
            
            worker.memory.push(state, action, reward, next_state, done)
            
            #Freeze the network during Evaluation Mode 
            if episode < eval_episode:
                worker.train_step(active_context=active_context)
                
            state = next_state
            
        if episode % 10 == 0:
            worker.update_target_network()
            
        episode_rewards.append(total_reward)
        supervisor_calls_history.append(supervisor_calls_this_episode)
        
        if episode % 10 == 0 or supervisor_calls_this_episode > 0:
            print(f"Episode {episode:3d} | Reward: {total_reward:5.1f} | Epsilon: {worker.epsilon:.2f} | Supervisor Calls: {supervisor_calls_this_episode}")

    env.close()
    
    # --- 4. Plot Generation ---
    plot_metrics(episode_rewards, supervisor_calls_history, sabotage_episode, eval_episode, sabotage_type)

if __name__ == "__main__":
    run_paper_experiment()