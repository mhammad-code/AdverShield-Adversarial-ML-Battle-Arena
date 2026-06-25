import json
import os
from collections import deque
from config import MEMORY_SIZE

PERSISTENCE_FILE = "battle_memory.json"


class BattleMemory:
    def __init__(self, max_size=MEMORY_SIZE):
        self.max_size = max_size
        self.battles = deque(maxlen=max_size)
        self.battle_count = 0
        # Persistent score counters (survive memory rotation)
        self.total_attacker_wins = 0
        self.total_defender_wins = 0
        self.total_draws = 0
        # Load persisted data on startup
        self._load_from_file()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_from_file(self):
        if not os.path.exists(PERSISTENCE_FILE):
            return
        try:
            with open(PERSISTENCE_FILE, "r") as f:
                data = json.load(f)
            self.battle_count = data.get("battle_count", 0)
            self.total_attacker_wins = data.get("total_attacker_wins", 0)
            self.total_defender_wins = data.get("total_defender_wins", 0)
            self.total_draws = data.get("total_draws", 0)
            saved_battles = data.get("battles", [])
            # Respect maxlen – take the last max_size entries
            for b in saved_battles[-self.max_size:]:
                self.battles.append(b)
            print(f"[BM] Battle memory loaded: {len(self.battles)} battles restored")
        except Exception as e:
            print(f"[BM] Could not load battle memory: {e}")

    def _save_to_file(self):
        try:
            data = {
                "battle_count": self.battle_count,
                "total_attacker_wins": self.total_attacker_wins,
                "total_defender_wins": self.total_defender_wins,
                "total_draws": self.total_draws,
                "battles": list(self.battles),
            }
            with open(PERSISTENCE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save battle memory: {e}")

    # ── Core API ──────────────────────────────────────────────────────────────

    def add_battle(self, battle_record):
        battle_record["battle_id"] = self.battle_count
        battle_record["timestamp"] = self._get_timestamp()
        self.battles.append(battle_record)
        self.battle_count += 1

        mode = battle_record.get("mode", "")
        if battle_record.get("is_draw", False):
            self.total_draws += 1
        elif mode in ("proactive_ai_battle", "proactive"):
            if battle_record.get("attacker_won", False):
                self.total_attacker_wins += 1
            else:
                self.total_defender_wins += 1
        else:  # reactive / manual: both sides can score
            if battle_record.get("attack_success", False):
                self.total_attacker_wins += 1
            if battle_record.get("defense_success", False):
                self.total_defender_wins += 1

        self._save_to_file()

    def _get_timestamp(self):
        from datetime import datetime
        return datetime.now().isoformat()

    def get_recent(self, n=5):
        return list(self.battles)[-n:]

    def get_all(self):
        return list(self.battles)

    def get_stats(self):
        return {
            "total_battles": self.battle_count,
            "attacker_wins": self.total_attacker_wins,
            "defender_wins": self.total_defender_wins,
            "draws": self.total_draws,
            "attack_type_distribution": self._count_field("attack_type"),
            "defense_type_distribution": self._count_field("defense_type"),
        }

    def _count_field(self, field):
        counts = {}
        for b in self.battles:
            val = b.get(field, "unknown")
            if val:
                counts[val] = counts.get(val, 0) + 1
        return counts

    def clear(self):
        self.battles.clear()
        self.battle_count = 0
        self.total_attacker_wins = 0
        self.total_defender_wins = 0
        self.total_draws = 0
        self._save_to_file()

    def reset_scores(self):
        """Reset only score counters, not battle history."""
        self.total_attacker_wins = 0
        self.total_defender_wins = 0
        self.total_draws = 0
        self._save_to_file()