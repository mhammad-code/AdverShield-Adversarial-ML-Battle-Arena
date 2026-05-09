import json
import os

DEFAULT_SETTINGS = {
    "model": {
        "name": "resnet50",
        "confidence_threshold": 0.5,
        "top_k_predictions": 5
    },
    "gradcam": {
        "enabled": True,
        "alpha": 0.5,
        "target_layer": "layer4_conv2"
    },
    "attacks": {
        "default_type": "fgsm",
        "default_strength": "medium",
        "save_adversarial": True
    },
    "defenses": {
        "default_type": "gaussian_blur",
        "default_strength": "medium"
    },
    "rl": {
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "exploration_rate": 0.2
    },
    "display": {
        "show_top5": True,
        "show_heatmap": True,
        "show_overlay": True
    },
    "llm": {
        "enabled": True,
        "auto_coach": True
    }
}


class SettingsModel:
    def __init__(self, config_file="settings.json"):
        self.config_file = config_file
        self.settings = self._load()

    def _load(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                return self._merge_defaults(loaded)
            except Exception as e:
                print(f"Failed to load settings: {e}")
        return DEFAULT_SETTINGS.copy()

    def _merge_defaults(self, loaded):
        merged = DEFAULT_SETTINGS.copy()
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict):
                merged[key].update(value)
            else:
                merged[key] = value
        return merged

    def save(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.settings, f, indent=2)

    def get(self, key=None):
        if key is None:
            return self.settings
        keys = key.split('.')
        value = self.settings
        for k in keys:
            value = value.get(k)
            if value is None:
                return None
        return value

    def set(self, key, value):
        keys = key.split('.')
        current = self.settings
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
        self.save()

    def reset(self):
        self.settings = DEFAULT_SETTINGS.copy()
        self.save()

    def get_all_categories(self):
        return list(self.settings.keys())