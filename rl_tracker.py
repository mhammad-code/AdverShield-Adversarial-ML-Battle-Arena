import numpy as np
from collections import defaultdict
import json


class RLTracker:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.round_number = 0
        self.learning_log = []
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.action_space = [
            ("fgsm", "low"), ("fgsm", "medium"), ("fgsm", "high"),
            ("pgd", "low"), ("pgd", "medium"), ("pgd", "high"),
            ("cw", "low"), ("cw", "medium"), ("cw", "high"),
            ("gaussian_blur", "low"), ("gaussian_blur", "medium"), ("gaussian_blur", "high"),
            ("jpeg_compression", "low"), ("jpeg_compression", "medium"), ("jpeg_compression", "high"),
            ("feature_squeezing", "low"), ("feature_squeezing", "medium"), ("feature_squeezing", "high")
        ]
        self.state_history = []
        self.action_history = []
        self.reward_history = []

    def get_state(self, image_class, confidence):
        conf_level = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
        return f"{image_class}_{conf_level}"

    def choose_action(self, state, role="attacker"):
        if np.random.random() < self.epsilon:
            return self._random_action(role)

        state_actions = self.q_table[state]
        if not state_actions:
            return self._random_action(role)

        max_q = max(state_actions.values())
        best_actions = [a for a, q in state_actions.items() if abs(q - max_q) < 0.01]
        return np.random.choice(best_actions) if best_actions else self._random_action(role)

    def _random_action(self, role):
        if role == "attacker":
            candidates = [a for a in self.action_space if a[0] in ["fgsm", "pgd", "cw", "ai"]]
        else:
            candidates = [a for a in self.action_space if a[0] in ["gaussian_blur", "jpeg_compression", "feature_squeezing"]]
        return list(candidates[np.random.randint(0, len(candidates))])

    def update(self, state, action, reward, next_state=None):
        action_key = f"{action[0]}_{action[1]}"

        if next_state:
            max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
            td_target = reward + self.gamma * max_next_q
        else:
            td_target = reward

        old_q = self.q_table[state][action_key]
        td_error = td_target - old_q
        self.q_table[state][action_key] += self.alpha * td_error
        new_q = self.q_table[state][action_key]

        self.log_learning_event(state, action, reward, old_q, new_q)

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.state_history.append(state)
        self.action_history.append(action_key)
        self.reward_history.append(reward)

    def log_learning_event(self, state, action, reward, old_q, new_q):
        self.round_number += 1
        event = {
            "round": self.round_number,
            "state": state,
            "action": f"{action[0]}_{action[1]}",
            "reward": round(reward, 3),
            "old_q": round(old_q, 4),
            "new_q": round(new_q, 4),
            "delta": round(new_q - old_q, 4),
            "epsilon": round(self.epsilon, 4),
            "message": self._generate_learning_message(action, reward, old_q, new_q)
        }
        self.learning_log.append(event)
        if len(self.learning_log) > 100:
            self.learning_log.pop(0)
        return event

    def _generate_learning_message(self, action, reward, old_q, new_q):
        action_str = f"{action[0]} {action[1]}"
        if reward > 0:
            if new_q > old_q + 0.05:
                return f"✅ {action_str} worked well — Q-value increased from {old_q:.3f} to {new_q:.3f}. Will prefer this strategy more."
            else:
                return f"✅ {action_str} succeeded — reinforcing this approach."
        else:
            if new_q < old_q - 0.03:
                return f"❌ {action_str} failed — Q-value dropped from {old_q:.3f} to {new_q:.3f}. Will avoid this combination."
            else:
                return f"❌ {action_str} did not work — slight penalty applied."

    def get_learning_log(self):
        return self.learning_log

    def get_q_heatmap_data(self):
        attack_actions = ["fgsm_low","fgsm_medium","fgsm_high","pgd_low","pgd_medium","pgd_high","cw_low","cw_medium","cw_high"]
        defense_actions = ["gaussian_blur_low","gaussian_blur_medium","gaussian_blur_high","jpeg_compression_low","jpeg_compression_medium","jpeg_compression_high","feature_squeezing_low","feature_squeezing_medium","feature_squeezing_high"]
        
        all_states = list(self.q_table.keys()) or ["default"]
        
        attacker_matrix = []
        defender_matrix = []
        
        for state in all_states[:5]:
            atk_row = [self.q_table[state].get(a, 0.0) for a in attack_actions]
            def_row = [self.q_table[state].get(a, 0.0) for a in defense_actions]
            attacker_matrix.append({"state": state, "values": atk_row})
            defender_matrix.append({"state": state, "values": def_row})
        
        return {
            "attacker": attacker_matrix,
            "defender": defender_matrix,
            "attack_labels": attack_actions,
            "defense_labels": defense_actions
        }

    def get_learning_curve_data(self):
        return {
            "rounds": [e["round"] for e in self.learning_log],
            "rewards": [e["reward"] for e in self.learning_log],
            "q_values": [e["new_q"] for e in self.learning_log],
            "epsilon_values": [e["epsilon"] for e in self.learning_log]
        }

    def get_suggestion(self, state, role="attacker"):
        action = self.choose_action(state, role)
        return f"Use {action[0]} with {action[1]} strength"

    def get_q_values(self, state):
        return dict(self.q_table[state])

    def save(self, filepath):
        data = {
            "q_table": {k: dict(v) for k, v in self.q_table.items()},
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "round_number": self.round_number,
            "learning_log": self.learning_log
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float), {k: defaultdict(float, v) for k, v in data["q_table"].items()})
        self.alpha = data.get("alpha", 0.1)
        self.gamma = data.get("gamma", 0.9)
        self.epsilon = data.get("epsilon", 0.1)
        self.epsilon_min = data.get("epsilon_min", 0.01)
        self.epsilon_decay = data.get("epsilon_decay", 0.995)
        self.round_number = data.get("round_number", 0)
        self.learning_log = data.get("learning_log", [])