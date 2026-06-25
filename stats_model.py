from collections import defaultdict
from datetime import datetime, timedelta


class StatsModel:
    def __init__(self, battle_memory):
        self.battle_memory = battle_memory

    def get_comprehensive_stats(self):
        battles = self.battle_memory.get_all()
        basic = self.battle_memory.get_stats()
        
        return {
            "basic": basic,
            "detailed": self._get_detailed_stats(battles),
            "performance": self._get_performance_metrics(battles),
            "trends": self._get_trends(battles),
            "insights": self._generate_insights(battles)
        }

    def _get_detailed_stats(self, battles):
        if not battles:
            return {}

        total = len(battles)
        attack_battles = [b for b in battles if b.get('mode') == 'attack']
        defend_battles = [b for b in battles if b.get('mode') == 'defend']

        # Use independent attack_success / defense_success fields
        attack_wins = sum(1 for b in attack_battles if b.get('attack_success', False) or b.get('attacker_won', False))
        defend_wins = sum(1 for b in defend_battles if b.get('defense_success', False))

        avg_original_conf = sum(b.get('original_confidence', 0) for b in battles) / total if total > 0 else 0
        avg_adv_conf = sum(b.get('adversarial_confidence', 0) for b in battles) / total if total > 0 else 0

        return {
            "total_battles": total,
            "attack_battles": len(attack_battles),
            "defend_battles": len(defend_battles),
            "attack_win_rate": attack_wins / len(attack_battles) if attack_battles else 0,
            "defense_win_rate": defend_wins / len(defend_battles) if defend_battles else 0,
            "avg_original_confidence": round(avg_original_conf, 4),
            "avg_adversarial_confidence": round(avg_adv_conf, 4),
            "confidence_drop": round(avg_original_conf - avg_adv_conf, 4)
        }

    def _get_performance_metrics(self, battles):
        if not battles:
            return {}

        attack_types = defaultdict(lambda: {"wins": 0, "total": 0})
        defense_types = defaultdict(lambda: {"wins": 0, "total": 0})

        for b in battles:
            at = b.get('attack_type', 'unknown')
            dt = b.get('defense_type', 'unknown')
            
            if b.get('mode') == 'attack':
                attack_types[at]["total"] += 1
                if b.get('attack_success', False) or b.get('attacker_won', False):
                    attack_types[at]["wins"] += 1
            elif b.get('mode') == 'defend':
                defense_types[dt]["total"] += 1
                if b.get('defense_success', False):
                    defense_types[dt]["wins"] += 1

        return {
            "attack_types": {
                k: {
                    "wins": v["wins"],
                    "total": v["total"],
                    "win_rate": round(v["wins"] / v["total"], 4) if v["total"] > 0 else 0
                }
                for k, v in attack_types.items()
            },
            "defense_types": {
                k: {
                    "wins": v["wins"],
                    "total": v["total"],
                    "win_rate": round(v["wins"] / v["total"], 4) if v["total"] > 0 else 0
                }
                for k, v in defense_types.items()
            }
        }

    def _get_trends(self, battles):
        if len(battles) < 2:
            return {"message": "Need more battles for trend analysis"}
        
        recent = battles[-10:]
        early = battles[:10] if len(battles) > 10 else battles[:-10]
        
        if not early:
            return {"message": "Not enough data for trends"}
        
        early_win_rate = sum(1 for b in early if b.get('attack_success', False) or b.get('attacker_won', False)) / len(early) if early else 0
        recent_win_rate = sum(1 for b in recent if b.get('attack_success', False) or b.get('attacker_won', False)) / len(recent) if recent else 0
        
        return {
            "attacker_trend": "increasing" if recent_win_rate > early_win_rate else "decreasing",
            "attacker_early_rate": round(early_win_rate, 4),
            "attacker_recent_rate": round(recent_win_rate, 4),
            "change": round(recent_win_rate - early_win_rate, 4)
        }

    def _generate_insights(self, battles):
        insights = []
        
        if not battles:
            insights.append("No battles yet. Start a battle to see insights!")
            return insights

        basic = self._get_detailed_stats(battles)
        
        if basic.get("attack_win_rate", 0) > 0.7:
            insights.append("Attack is highly effective - consider strengthening defenses")
        elif basic.get("attack_win_rate", 0) < 0.3:
            insights.append("Defenses are strong - try different attack strategies")
        
        if basic.get("confidence_drop", 0) > 0.3:
            insights.append("Large confidence drop indicates successful adversarial attacks")
        
        perf = self._get_performance_metrics(battles)
        if perf.get("attack_types"):
            best_attack = max(perf["attack_types"].items(), key=lambda x: x[1].get("win_rate", 0))
            insights.append(f"Best performing attack: {best_attack[0]} ({best_attack[1].get('win_rate', 0)*100:.1f}%)")
        
        if perf.get("defense_types"):
            best_defense = max(perf["defense_types"].items(), key=lambda x: x[1].get("win_rate", 0))
            insights.append(f"Best performing defense: {best_defense[0]} ({best_defense[1].get('win_rate', 0)*100:.1f}%)")
        
        return insights