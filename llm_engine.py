import os
import json
import re
from groq import Groq
from config import LLM_MODEL


class LLMEngine:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        self.client = Groq(api_key=api_key)
        self.model = LLM_MODEL

    def generate(self, prompt, system_prompt=None):
        if system_prompt is None:
            system_prompt = "You are an expert in adversarial machine learning. Generate strategic advice for attacking or defending image classifiers."
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_structured(self, prompt, system_prompt, fallback):
        """Always returns a dict, never crashes."""
        try:
            raw = self.generate(prompt, system_prompt)
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        return fallback

    def generate_attack_strategy(self, image_info, history, rl_suggestion=None):
        """Now returns a structured dict (but old interface still works if string used)."""
        result = self.generate_attack_strategy_structured(image_info, history, rl_suggestion)
        # Keep backward compatibility: return JSON string for old frontends
        return json.dumps(result)

    def generate_defense_strategy(self, image_info, history, rl_suggestion=None):
        result = self.generate_defense_strategy_structured(image_info, history, rl_suggestion)
        return json.dumps(result)

    def generate_attack_strategy_structured(self, image_info, history, rl_suggestion=None):
        prompt = self._build_attack_prompt(image_info, history, rl_suggestion)
        fallback = {"attack_type": "fgsm", "strength": "medium", "reasoning": "Default strategy"}
        return self.generate_structured(prompt,
            "You are an adversarial ML attacker. Reply ONLY in valid JSON.", fallback)

    def generate_defense_strategy_structured(self, image_info, history, rl_suggestion=None):
        prompt = self._build_defense_prompt(image_info, history, rl_suggestion)
        fallback = {"defense_type": "gaussian_blur", "strength": "medium", "reasoning": "Default defense"}
        return self.generate_structured(prompt,
            "You are an adversarial ML defender. Reply ONLY in valid JSON.", fallback)

    def generate_outcome_analysis(self, battle_data):
        """Call 3 – explains why attack/defense succeeded or failed."""
        prompt = f"""Analyze this adversarial ML battle round in plain English:

Attack: {battle_data.get('attack_type')} at {battle_data.get('attack_strength')} strength
Defense: {battle_data.get('defense_type')} at {battle_data.get('defense_strength')} strength
Original classification: {battle_data.get('original_class')} ({battle_data.get('original_confidence', 0):.1%})
After attack: {battle_data.get('adversarial_class')} ({battle_data.get('adversarial_confidence', 0):.1%})
After defense: {battle_data.get('defended_class')} ({battle_data.get('defended_confidence', 0):.1%})
Attack succeeded: {battle_data.get('attacker_won')}
Defense succeeded: {battle_data.get('defense_success')}

Reply in JSON:
{{
  "why_attack_succeeded_or_failed": "detailed explanation",
  "why_defense_succeeded_or_failed": "detailed explanation",
  "what_model_focused_on": "what GradCAM would show at each stage",
  "key_insight": "one sentence takeaway",
  "learning_note": "what the RL agent should learn from this round"
}}"""
        fallback = {
            "why_attack_succeeded_or_failed": "Analysis unavailable",
            "why_defense_succeeded_or_failed": "Analysis unavailable",
            "what_model_focused_on": "GradCAM shows model attention regions",
            "key_insight": "Battle completed",
            "learning_note": "RL tracker updated"
        }
        return self.generate_structured(prompt,
            "You are an adversarial ML analyst. Explain battle results clearly. Reply ONLY in valid JSON.",
            fallback)

    def _build_attack_prompt(self, image_info, history, rl_suggestion):
        prompt = f"""You are an attacker trying to fool an image classifier (ResNet-50 pretrained on ImageNet).

Current Image:
- Original class: {image_info.get('original_class', 'Unknown')}
- Confidence: {image_info.get('confidence', 0):.2%}

Attack Types Available: FGSM, PGD, Carlini-Wagner, AI
Strength Levels: low (epsilon=0.01), medium (0.03), high (0.08)

Previous Attacks: {history[-3:] if history else 'None'}
"""
        if rl_suggestion:
            prompt += f"\nRL Suggestion: {rl_suggestion}\n"
        prompt += """
Suggest the best attack type and strength. Return JSON format:
{"attack_type": "fgsm/pgd/cw/ai", "strength": "low/medium/high", "reasoning": "brief explanation"}
"""
        return prompt

    def _build_defense_prompt(self, image_info, history, rl_suggestion):
        prompt = f"""You are a defender trying to protect an image classifier (ResNet-50 pretrained on ImageNet).

Current Image:
- Original class: {image_info.get('original_class', 'Unknown')}
- Confidence: {image_info.get('confidence', 0):.2%}
- Adversarial class: {image_info.get('adversarial_class', 'Unknown')}

Defense Types Available: Gaussian Blur, JPEG Compression, Feature Squeezing
Strength Levels vary by method.

Previous Defenses: {history[-3:] if history else 'None'}
"""
        if rl_suggestion:
            prompt += f"\nRL Suggestion: {rl_suggestion}\n"
        prompt += """
Suggest the best defense type and strength. Return JSON format:
{"defense_type": "gaussian_blur/jpeg_compression/feature_squeezing", "strength": "low/medium/high", "reasoning": "brief explanation"}
"""
        return prompt

    def coach(self, battle_history):
        if len(battle_history) < 3:
            return "Keep battling to gather more data for coaching."

        wins = sum(1 for b in battle_history if b.get('attacker_won', False))
        losses = len(battle_history) - wins

        prompt = f"""Analyze the following battle history and provide strategic coaching advice:

- Total battles: {len(battle_history)}
- Wins (attacker): {wins}
- Losses (attacker): {losses}

Recent battles:
{chr(10).join([str(b) for b in battle_history[-5:]])}

Provide concise advice on which strategies are working and which need adjustment."""
        return self.generate(prompt, "You are an expert ML coach analyzing adversarial battle strategies.")