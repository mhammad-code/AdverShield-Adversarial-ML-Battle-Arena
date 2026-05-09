from collections import deque
from config import MEMORY_SIZE


class BattleMemory:
    def __init__(self, max_size=MEMORY_SIZE):
        self.max_size = max_size
        self.battles = deque(maxlen=max_size)
        self.battle_count = 0
        # Persistent score counters (survive memory rotation)
        self.total_attacker_wins = 0
        self.total_defender_wins = 0
        self.total_draws = 0

    def add_battle(self, battle_record):
        battle_record['battle_id'] = self.battle_count
        battle_record['timestamp'] = self._get_timestamp()
        self.battles.append(battle_record)
        self.battle_count += 1

        # Update persistent score counters
        if battle_record.get('is_draw', False):
            self.total_draws += 1
        elif battle_record.get('attacker_won', False):
            self.total_attacker_wins += 1
        else:
            self.total_defender_wins += 1

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
            "attack_type_distribution": self._count_field('attack_type'),
            "defense_type_distribution": self._count_field('defense_type'),
        }

    def _count_field(self, field):
        counts = {}
        for b in self.battles:
            val = b.get(field, 'unknown')
            if val:
                counts[val] = counts.get(val, 0) + 1
        return counts

    def clear(self):
        self.battles.clear()
        self.battle_count = 0
        self.total_attacker_wins = 0
        self.total_defender_wins = 0
        self.total_draws = 0

    def reset_scores(self):
        """Reset only score counters, not battle history."""
        self.total_attacker_wins = 0
        self.total_defender_wins = 0
        self.total_draws = 0