import json
import requests
import numpy as np
import re

class LocalLLaMASupervisor:
    """
    The 'Supervisor' (Teacher) powered by a local LLaMA model via Ollama.
    It remains dormant until triggered by the Worker's stability failure.
    """
    def __init__(self, model_name="qwen2.5:7b", host="http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.api_url = f"{self.host}/api/chat"
        self.tags_url = f"{self.host}/api/tags"
        self._ready_checked = False
        self._ready = False
        self._disabled_reason = None
        self.last_call_used_llm = False
        # Local larger models can respond slowly on first token.
        self.request_timeout_sec = 12
        self.max_retries = 0
        self._parse_error_count = 0

    def vector_to_text_bridge(self, state, sabotage_type=None):
        """Convert state vector to text description for LLM."""

        if len(state) == 4:
            # CartPole: state = [position, velocity, angle_rad, angular_velocity]
            # Actions: 0 = push cart LEFT, 1 = push cart RIGHT
            position, velocity, angle_rad, angular_velocity = state
            tipping = "RIGHT" if angle_rad > 0 else "LEFT"

            if sabotage_type == "inverted_controls":
                sabotage_note = (
                    "CRITICAL: Controls are INVERTED. "
                    "Sending action 0 physically pushes RIGHT. Sending action 1 physically pushes LEFT. "
                    f"The pole is tipping {tipping}. "
                    f"To push cart {tipping} and correct the pole, send action {'0' if tipping == 'RIGHT' else '1'}."
                )
            elif sabotage_type == "sensor_inversion":
                # angle_rad is the CORRUPTED value (negated by env); real pole tips opposite
                real_tipping = "LEFT" if angle_rad > 0 else "RIGHT"
                correct_action = "0" if angle_rad > 0 else "1"
                sabotage_note = (
                    "CRITICAL: Angle sensor is INVERTED. "
                    f"Sensor reads {angle_rad:.3f} rad but the pole is ACTUALLY tipping {real_tipping}. "
                    f"To balance, push cart {real_tipping}: send action {correct_action}."
                )
            elif sabotage_type == "high_gravity":
                sabotage_note = (
                    "Gravity is doubled. React quickly — balance the pole "
                    f"by pushing cart {tipping} (pole is tipping {tipping})."
                )
            elif sabotage_type == "sensor_noise":
                sabotage_note = (
                    "Observations have noise but apply the normal balancing rule: "
                    f"push cart {tipping} because pole is tipping {tipping}."
                )
            elif sabotage_type == "actuator_delay":
                sabotage_note = (
                    "There is a one-step actuator delay. Apply the correct action early: "
                    f"pole is tipping {tipping}, push cart {tipping}."
                )
            else:
                sabotage_note = f"Pole is tipping {tipping}. Push the cart {tipping} to balance."

            prompt = (
                f"CartPole balancing task.\n"
                f"State: Position={position:.2f}, Velocity={velocity:.2f}, "
                f"Angle={angle_rad:.3f} rad, AngularVel={angular_velocity:.2f}\n"
                f"{sabotage_note}\n"
                f"Actions: 0 = push LEFT, 1 = push RIGHT.\n"
                f"Reply with ONLY a single digit: 0 or 1."
            )
            return prompt

        # LunarLander: state = [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]
        # Actions: 0=do nothing, 1=left engine, 2=main engine, 3=right engine
        x, y, vx, vy, angle, angular_vel, left_leg, right_leg = state

        if sabotage_type == "inverted_controls":
            rule = (
                "CRITICAL: Left/Right engines are SWAPPED. "
                "To correct rightward drift send action 1 (physically fires right). "
                "To correct leftward drift send action 3 (physically fires left)."
            )
        elif sabotage_type == "high_gravity":
            rule = (
                "Gravity is doubled. Fire MAIN engine (action 2) aggressively to slow descent."
            )
        elif sabotage_type == "sensor_noise":
            rule = "Observations are noisy. Be conservative — prefer action 2 to slow descent."
        elif sabotage_type == "actuator_delay":
            rule = "One-step actuator delay is active. Anticipate drift and act one step early."
        else:
            rule = "Maintain stable flight and land safely."

        prompt = (
            f"LunarLander control task.\n"
            f"State: X={x:.2f}, Y={y:.2f}, Vx={vx:.2f}, Vy={vy:.2f}, "
            f"Angle={angle:.2f}, AngularVel={angular_vel:.2f}, "
            f"LeftLeg={int(left_leg)}, RightLeg={int(right_leg)}\n"
            f"{rule}\n"
            f"Actions: 0=nothing, 1=left engine, 2=main engine, 3=right engine.\n"
            f"Reply with ONLY a single digit: 0, 1, 2, or 3."
        )
        return prompt

    def _heuristic_fallback(self, state, sabotage_type=None):
        """Context-aware fallback to keep rollouts running without poisoning guidance logic."""
        self.last_call_used_llm = False
        if len(state) == 4:
            angle_rad = state[2]
            if sabotage_type == "inverted_controls":
                # Corrupted control: sending 0 pushes RIGHT, so flip the normal rule
                return 0 if angle_rad > 0 else 1
            if sabotage_type == "sensor_inversion":
                # angle_rad is negated by env: perceived > 0 means pole ACTUALLY tips LEFT
                return 0 if angle_rad > 0 else 1
            return 1 if angle_rad > 0 else 0

        x, y, vx, vy, angle, angular_vel, left_leg, right_leg = state

        if abs(angle) > 0.15:
            action = 1 if angle > 0 else 3
        elif vy < -0.35:
            action = 2
        elif vx > 0.15:
            action = 1
        elif vx < -0.15:
            action = 3
        else:
            action = 0

        if sabotage_type == "inverted_controls":
            if action == 1:
                action = 3
            elif action == 3:
                action = 1

        return int(action)

    def _ensure_backend_ready(self):
        """
        One-time readiness check:
        1) confirms Ollama endpoint is reachable
        2) confirms selected model is available
        """
        if self._ready_checked:
            return self._ready

        self._ready_checked = True
        try:
            response = requests.get(self.tags_url, timeout=3)
            response.raise_for_status()
            models = response.json().get("models", [])
            names = {m.get("name", "") for m in models}

            if self.model_name in names:
                self._ready = True
                return True

            self._disabled_reason = (
                f"Model '{self.model_name}' not found on Ollama. "
                f"Run: ollama pull {self.model_name}"
            )
            print(f"  [Supervisor Disabled] {self._disabled_reason}")
            self._ready = False
            return False

        except Exception as e:
            self._disabled_reason = (
                f"Cannot reach Ollama at {self.host}. "
                f"Start Ollama and verify /api/tags. Details: {e}"
            )
            print(f"  [Supervisor Disabled] {self._disabled_reason}")
            self._ready = False
            return False

    def get_expert_action(self, state, sabotage_type=None):
        """
        Queries the local LLaMA model and parses the response into a discrete executable action (a*).
        """
        self.last_call_used_llm = False
        if not self._ensure_backend_ready():
            return self._heuristic_fallback(state, sabotage_type)

        prompt = self.vector_to_text_bridge(state, sabotage_type)

        # Qwen requires the chat messages format.
        max_action = 1 if len(state) == 4 else 3

        payload = {
            "model": self.model_name,
            "stream": True,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise control system. Reply ONLY with a single digit number. No explanation.",
                },
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.0},
        }

        try:
            api_endpoint = f"{self.host}/api/chat"
            response = requests.post(api_endpoint, json=payload, stream=True, timeout=60)
            response.raise_for_status()

            collected = ""
            in_think = False

            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                chunk = json.loads(raw_line)
                token = chunk.get("message", {}).get("content", "")
                collected += token

                if "<think>" in collected:
                    in_think = True
                if "</think>" in collected:
                    in_think = False
                    collected = collected.split("</think>")[-1]

                if not in_think:
                    match = re.search(r'\b([0-9])\b', collected)
                    if match:
                        parsed = int(match.group(1))
                        if 0 <= parsed <= max_action:
                            self.last_call_used_llm = True
                            return parsed

                if chunk.get("done"):
                    break

            self._parse_error_count += 1
            return self._heuristic_fallback(state, sabotage_type)

        except Exception:
            return self._heuristic_fallback(state, sabotage_type)

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