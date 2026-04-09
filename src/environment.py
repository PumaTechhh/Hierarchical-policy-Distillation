import gymnasium as gym
import numpy as np

class SabotagedCartPole(gym.Wrapper):
    """
    A wrapper for CartPole-v1 that allows for runtime 'sabotage' 
    (perturbations) to simulate unexpected environmental errors.
    
    Implements the 'Perturbation Protocol' defined in the Thesis Methodology.
    """
    def __init__(self, render_mode=None):
        env = gym.make('CartPole-v1', render_mode=render_mode)
        super().__init__(env)

        #sabotage ststes
        self.sabotage_active = False
        self.sabotage_type = None # 'inverted_controls' or 'high_gravity'

        #store original physics parameters for restoration
        self.original_gravity = self.env.unwrapped.gravity
        self.original_force_mag = self.env.unwrapped.force_mag
    
    def step(self,action):
        """
        Executes a step in the environment, applying sabotage if active.
        """
        actual_action = action

        # SABOTAGE LOGIC
        if self.sabotage_active:
            if self.sabotage_type == 'inverted_controls':
                # Thesis Phase 2: Inverting control inputs 
                # 0 becomes 1, 1 becomes 0
                actual_action = 1 - action

        # Execute the step with the (potentially modified) action 
        obs, reward, terminated, truncated, info = self.env.step(actual_action)

        #Inject sabotage info into the 'info' dict for logging   
        info['sabotage_active'] = self.sabotage_active
        info['sabotage_type'] = self.sabotage_type

        return obs, reward, terminated, truncated, info
    
    def trigger_sabotage(self, sabotage_type='inverted_controls'):
        """
        Activates a sabotage of the specified type.
        """
        self.sabotage_active = True
        self.sabotage_type = sabotage_type

        if sabotage_type == 'high_gravity':
            # Thesis Phase 2: Altering gravity coefficients 
            # Standard gravity is 9.8; we double it to make balancing harder
            self.env.unwrapped.gravity= 19.6

        print("! SABOTAGE TRIGGERED:", sabotage_type,"!")
    
    def reset_sabotage(self):
        """
        Restores the environment to Nominal Operation (Phase 1).
        """
        self.sabotage_active = False
        self.sabotage_type = None

        # Restore original physics parameters
        self.env.unwrapped.gravity = self.original_gravity
        self.env.unwrapped.force_mag = self.original_force_mag

        print(">> SABOTAGE RESET: Environment restored to Nominal Operation ")

# Quick Test Code
if __name__ == "__main__":
    # Create the environment
    env = SabotagedCartPole(render_mode="human")
    obs, _ = env.reset()
    
    print("Phase 1: Normal Operation (50 steps)")
    for i in range(150):
        # Trigger Sabotage at step 50
        if i == 50:
            env.trigger_sabotage('inverted_controls')
        
        # Reset Sabotage at step 100
        if i == 100:
            env.reset_sabotage()
            
        # Take a random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        env.render()
        
        if done or truncated:
            env.reset()
            
    env.close()