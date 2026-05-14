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

        # Runtime perturbation parameters
        self.sensor_noise_std = 0.03
        self._pending_action = None

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
            elif self.sabotage_type == 'actuator_delay':
                # Apply one-step delay: execute previous action, queue current.
                if self._pending_action is None:
                    self._pending_action = int(action)
                delayed_action = self._pending_action
                self._pending_action = int(action)
                actual_action = delayed_action

        # Execute the step with the (potentially modified) action
        obs, reward, terminated, truncated, info = self.env.step(actual_action)

        if self.sabotage_active:
            if self.sabotage_type == 'sensor_noise':
                noise = np.random.normal(0.0, self.sensor_noise_std, size=obs.shape)
                obs = obs + noise.astype(obs.dtype, copy=False)
            elif self.sabotage_type == 'sensor_inversion':
                # Negate the angle sensor: DQN perceives pole tipping opposite direction
                obs = obs.copy()
                obs[2] = -obs[2]

        #Inject sabotage info into the 'info' dict for logging   
        info['sabotage_active'] = self.sabotage_active
        info['sabotage_type'] = self.sabotage_type
        info['executed_action'] = int(actual_action)

        return obs, reward, terminated, truncated, info
    
    def trigger_sabotage(self, sabotage_type='inverted_controls'):
        """
        Activates a sabotage of the specified type.
        """
        self.sabotage_active = True
        self.sabotage_type = sabotage_type
        self._pending_action = None

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
        self._pending_action = None

        # Restore original physics parameters
        self.env.unwrapped.gravity = self.original_gravity
        self.env.unwrapped.force_mag = self.original_force_mag

        print(">> SABOTAGE RESET: Environment restored to Nominal Operation ")


class SabotagedLunarLander(gym.Wrapper):
    """
    A wrapper for LunarLander-v2 that mirrors the sabotage protocol used for
    CartPole so benchmark logic can stay unchanged.
    """

    def __init__(self, sabotage_type=None, render_mode=None):
        env = gym.make("LunarLander-v3", continuous=False, render_mode=render_mode)
        super().__init__(env)
        self.sabotage_type = sabotage_type
        self.sabotage_active = False

        self.sensor_noise_std = 0.1
        self._pending_action = None
        self._original_world_gravity = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._pending_action = None
        return obs, info

    def step(self, action):
        actual_action = int(action)

        if self.sabotage_active:
            if self.sabotage_type == "inverted_controls":
                # In discrete LunarLander: 1=left engine, 3=right engine.
                if actual_action == 1:
                    actual_action = 3
                elif actual_action == 3:
                    actual_action = 1
            elif self.sabotage_type == "actuator_delay":
                if self._pending_action is None:
                    self._pending_action = int(actual_action)
                delayed_action = self._pending_action
                self._pending_action = int(actual_action)
                actual_action = delayed_action

        obs, reward, terminated, truncated, info = self.env.step(actual_action)

        if self.sabotage_active and self.sabotage_type == "sensor_noise":
            noise = np.random.normal(0.0, self.sensor_noise_std, size=obs.shape)
            obs = obs + noise.astype(obs.dtype, copy=False)

        info["sabotage_active"] = self.sabotage_active
        info["sabotage_type"] = self.sabotage_type
        info["executed_action"] = int(actual_action)

        return obs, reward, terminated, truncated, info

    def trigger_sabotage(self, sabotage_type):
        self.sabotage_type = sabotage_type
        self.sabotage_active = True
        self._pending_action = None

        if sabotage_type == "high_gravity":
            world = getattr(self.env.unwrapped, "world", None)
            if world is not None and hasattr(world, "gravity"):
                self._original_world_gravity = tuple(world.gravity)
                world.gravity = (world.gravity[0], world.gravity[1] * 2.0)

        print("! SABOTAGE TRIGGERED:", sabotage_type, "!")

    def reset_sabotage(self):
        self.sabotage_active = False
        self.sabotage_type = None
        self._pending_action = None

        if self._original_world_gravity is not None:
            world = getattr(self.env.unwrapped, "world", None)
            if world is not None and hasattr(world, "gravity"):
                world.gravity = self._original_world_gravity
            self._original_world_gravity = None

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