import numpy as np
from collections import defaultdict
import json
import os
from datetime import datetime

RL_PERSISTENCE_FILE = "rl_tracker.json"


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

        # Auto-load persisted data
        self._auto_load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _auto_load(self):
        if not os.path.exists(RL_PERSISTENCE_FILE):
            return
        try:
            self.load(RL_PERSISTENCE_FILE)
            print(f"[RL] Loaded: {self.round_number} rounds restored")
        except Exception as e:
            print(f"[RL] Could not load tracker: {e}")

    def _auto_save(self):
        try:
            self.save(RL_PERSISTENCE_FILE)
        except Exception as e:
            print(f"⚠️  Could not save RL tracker: {e}")

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
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float),
                                   {k: defaultdict(float, v) for k, v in data["q_table"].items()})
        self.alpha = data.get("alpha", 0.1)
        self.gamma = data.get("gamma", 0.9)
        self.epsilon = data.get("epsilon", 0.1)
        self.epsilon_min = data.get("epsilon_min", 0.01)
        self.epsilon_decay = data.get("epsilon_decay", 0.995)
        self.round_number = data.get("round_number", 0)
        self.learning_log = data.get("learning_log", [])

    def reset(self):
        """Clear all learned data back to initial state."""
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.learning_log = []
        self.round_number = 0
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.epsilon = 0.1
        print("[RL] Tracker reset to initial state.")

    # ── Core RL ───────────────────────────────────────────────────────────────

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
            candidates = [
                ("fgsm", "low"), ("fgsm", "medium"), ("fgsm", "high"),
                ("pgd", "low"), ("pgd", "medium"), ("pgd", "high"),
                ("cw", "low"), ("cw", "medium"), ("cw", "high"),
                ("bim", "low"), ("bim", "medium"), ("bim", "high"),
                ("deepfool", "low"), ("deepfool", "medium"), ("deepfool", "high"),
                ("salt_pepper", "low"), ("salt_pepper", "medium"), ("salt_pepper", "high"),
                ("contrast_shift", "low"), ("contrast_shift", "medium"), ("contrast_shift", "high"),
                ("jpeg_attack", "low"), ("jpeg_attack", "medium"), ("jpeg_attack", "high"),
                ("frequency_filter", "low"), ("frequency_filter", "medium"), ("frequency_filter", "high"),
                ("spatial_perturbation", "low"), ("spatial_perturbation", "medium"), ("spatial_perturbation", "high"),
                ("universal_perturbation", "low"), ("universal_perturbation", "medium"), ("universal_perturbation", "high"),
                ("ai", "low"), ("ai", "medium"), ("ai", "high"),
            ]
        else:
            candidates = [
                ("gaussian_blur", "low"), ("gaussian_blur", "medium"), ("gaussian_blur", "high"),
                ("jpeg_compression", "low"), ("jpeg_compression", "medium"), ("jpeg_compression", "high"),
                ("feature_squeezing", "low"), ("feature_squeezing", "medium"), ("feature_squeezing", "high"),
                ("ai_defense", "low"), ("ai_defense", "medium"), ("ai_defense", "high"),
                ("median_blur", "low"), ("median_blur", "medium"), ("median_blur", "high"),
                ("bilateral_filter", "low"), ("bilateral_filter", "medium"), ("bilateral_filter", "high"),
                ("tv_denoising", "low"), ("tv_denoising", "medium"), ("tv_denoising", "high"),
                ("randomized_smoothing", "low"), ("randomized_smoothing", "medium"), ("randomized_smoothing", "high"),
                ("pixel_deflection", "low"), ("pixel_deflection", "medium"), ("pixel_deflection", "high"),
                ("quilting", "low"), ("quilting", "medium"), ("quilting", "high"),
                ("autoencoder_restoration", "low"), ("autoencoder_restoration", "medium"), ("autoencoder_restoration", "high"),
                ("diff_jpeg", "low"), ("diff_jpeg", "medium"), ("diff_jpeg", "high"),
            ]
        return list(candidates[np.random.randint(0, len(candidates))])

    def update(self, state, action, reward, next_state=None, role="attacker"):
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

        self.log_learning_event(state, action, reward, old_q, new_q, role=role)

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.state_history.append(state)
        self.action_history.append(action_key)
        self.reward_history.append(reward)

        # Auto-save after every update
        self._auto_save()

    def log_learning_event(self, state, action, reward, old_q, new_q, role="attacker"):
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
            "role": role,  # "attacker" or "defender"
            "message": self._generate_learning_message(action, reward, old_q, new_q),
            "timestamp": datetime.now().isoformat()
        }
        self.learning_log.append(event)
        if len(self.learning_log) > 200:
            self.learning_log.pop(0)

        # Write to persistent log file
        self._write_to_log_file(event)
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

    def _write_to_log_file(self, event):
        filepath = "learning_log.jsonl"
        try:
            with open(filepath, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            print(f"Failed to write learning log: {e}")

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_learning_log(self):
        return self.learning_log

    def get_attacker_log(self):
        return [e for e in self.learning_log if e.get("role") == "attacker"]

    def get_defender_log(self):
        return [e for e in self.learning_log if e.get("role") == "defender"]

    def get_q_heatmap_data(self):
        attack_actions = [
            "fgsm_low", "fgsm_medium", "fgsm_high",
            "pgd_low", "pgd_medium", "pgd_high",
            "cw_low", "cw_medium", "cw_high",
            "bim_low", "bim_medium", "bim_high",
            "deepfool_low", "deepfool_medium", "deepfool_high",
            "salt_pepper_low", "salt_pepper_medium", "salt_pepper_high",
            "contrast_shift_low", "contrast_shift_medium", "contrast_shift_high",
            "jpeg_attack_low", "jpeg_attack_medium", "jpeg_attack_high",
            "frequency_filter_low", "frequency_filter_medium", "frequency_filter_high",
            "spatial_perturbation_low", "spatial_perturbation_medium", "spatial_perturbation_high",
            "universal_perturbation_low", "universal_perturbation_medium", "universal_perturbation_high",
            "ai_low", "ai_medium", "ai_high",
        ]
        defense_actions = [
            "gaussian_blur_low", "gaussian_blur_medium", "gaussian_blur_high",
            "jpeg_compression_low", "jpeg_compression_medium", "jpeg_compression_high",
            "feature_squeezing_low", "feature_squeezing_medium", "feature_squeezing_high",
            "ai_defense_low", "ai_defense_medium", "ai_defense_high",
            "median_blur_low", "median_blur_medium", "median_blur_high",
            "bilateral_filter_low", "bilateral_filter_medium", "bilateral_filter_high",
            "tv_denoising_low", "tv_denoising_medium", "tv_denoising_high",
            "randomized_smoothing_low", "randomized_smoothing_medium", "randomized_smoothing_high",
            "pixel_deflection_low", "pixel_deflection_medium", "pixel_deflection_high",
            "quilting_low", "quilting_medium", "quilting_high",
            "autoencoder_restoration_low", "autoencoder_restoration_medium", "autoencoder_restoration_high",
            "diff_jpeg_low", "diff_jpeg_medium", "diff_jpeg_high",
        ]

        all_states = list(self.q_table.keys()) or ["default"]

        attacker_matrix = []
        defender_matrix = []

        for state in all_states[:10]:
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
        """Returns combined + per-role curve data."""
        full = self.learning_log
        atk_log = [e for e in full if e.get("role") == "attacker"]
        def_log = [e for e in full if e.get("role") == "defender"]
        return {
            "rounds": [e["round"] for e in full],
            "rewards": [e["reward"] for e in full],
            "q_values": [e["new_q"] for e in full],
            "epsilon_values": [e["epsilon"] for e in full],
            # Per-role curves
            "attacker_rounds": [e["round"] for e in atk_log],
            "attacker_rewards": [e["reward"] for e in atk_log],
            "attacker_q_values": [e["new_q"] for e in atk_log],
            "defender_rounds": [e["round"] for e in def_log],
            "defender_rewards": [e["reward"] for e in def_log],
            "defender_q_values": [e["new_q"] for e in def_log],
        }

    def get_suggestion(self, state, role="attacker"):
        action = self.choose_action(state, role)
        return f"Use {action[0]} with {action[1]} strength"

    def get_q_values(self, state):
        return dict(self.q_table[state])

    def get_top_actions(self, role="attacker", n=5):
        """Return top N actions by highest Q-value across all states."""
        scores = defaultdict(float)
        counts = defaultdict(int)
        for state, actions in self.q_table.items():
            for action_key, q_val in actions.items():
                scores[action_key] += q_val
                counts[action_key] += 1
        averaged = {k: scores[k] / counts[k] for k in scores}
        sorted_actions = sorted(averaged.items(), key=lambda x: x[1], reverse=True)
        return sorted_actions[:n]