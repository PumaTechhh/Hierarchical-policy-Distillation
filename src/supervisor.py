import requests
import json
import numpy as np

class LocalLLaMASupervisor:
    """
    The 'Supervisor' (Teacher) powered by a local LLaMA model via Ollama.
    It remains dormant until triggered by the Worker's stability failure.
    """
    def __init__(self, model_name="llama3.2", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.api_url = f"{self.host}/api/generate"

    def vector_to_text_bridge(self, state, sabotage_type=None):
        """
        THESIS REQUIREMENT: Symbol Grounding Interface.
        Serialises the numerical state vector (S_t) into a descriptive text prompt.
        CartPole State: [position, velocity, angle, angular_velocity]
        """
        position, velocity, angle_rad, angular_velocity = state
        
        # Convert radians to degrees for better LLM semantic understanding
        angle_deg = np.degrees(angle_rad)
        
        # Determine directions
        pole_direction = "right" if angle_deg > 0 else "left"
        cart_direction = "right" if velocity > 0 else "left"
        
        prompt = (
            f"You are an expert control system for a CartPole balancing task.\n"
            f"Current State:\n"
            f"- The pole is tipped {abs(angle_deg):.2f} degrees to the {pole_direction}.\n"
            f"- The pole is falling with an angular velocity of {angular_velocity:.2f}.\n"
            f"- The cart is at position {position:.2f} and moving {cart_direction} at a velocity of {velocity:.2f}.\n"
        )
        
        if sabotage_type == 'inverted_controls':
            prompt += "CRITICAL WARNING: The control physics have been inverted! Pushing Left now moves the cart Right, and vice versa.\n"
        elif sabotage_type == 'high_gravity':
            prompt += "CRITICAL WARNING: Gravity has been doubled! The pole will fall much faster.\n"
            
        prompt += (
            "Task: To balance the pole, which action should the cart take?\n"
            "Action 0: Push Left\n"
            "Action 1: Push Right\n"
            "Respond ONLY with the number 0 or 1. Do not include any other text, reasoning, or punctuation."
        )
        
        return prompt

    def get_expert_action(self, state, sabotage_type=None):
        """
        Queries the local LLaMA model and parses the response into a discrete executable action (a*).
        """
        prompt = self.vector_to_text_bridge(state, sabotage_type)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0  # We want deterministic, expert answers
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            # Extract the response text
            result_text = response.json().get("response", "").strip()
            
            # Parse back into a discrete action (0 or 1)
            # We use basic string matching in case the LLM is slightly chatty despite instructions
            if "0" in result_text and "1" not in result_text:
                return 0
            elif "1" in result_text and "0" not in result_text:
                return 1
            else:
                # Fallback heuristics if the LLM completely fails formatting
                print(f"  [Supervisor Parsing Error] Raw output: '{result_text}'. Defaulting to heuristic.")
                angle_rad = state[2]
                return 1 if angle_rad > 0 else 0
                
        except Exception as e:
            print(f"  [Supervisor Connection Error] {e}")
            # Fallback to keep the simulation running
            return 1 if state[2] > 0 else 0

# Quick Test
if __name__ == "__main__":
    supervisor = LocalLLaMASupervisor()
    
    # Mock CartPole state: Tipped far to the right (positive angle)
    mock_state = [0.0, 0.5, 0.2, 0.5] 
    print("Testing Vector-to-Text Bridge...")
    prompt = supervisor.vector_to_text_bridge(mock_state)
    print(f"\nPrompt Generated:\n{'-'*40}\n{prompt}\n{'-'*40}")
    
    print("\nQuerying Local LLaMA (ensure Ollama is running)...")
    action = supervisor.get_expert_action(mock_state)
    print(f"Expert Action Returned: {action}")