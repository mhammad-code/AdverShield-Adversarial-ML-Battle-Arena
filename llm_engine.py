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
            system_prompt = (
                "You are an expert in adversarial machine learning. "
                "Generate strategic advice for attacking or defending image classifiers."
            )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=800,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_structured(self, prompt, system_prompt, fallback):
        """Always returns a dict, never crashes."""
        try:
            raw = self.generate(prompt, system_prompt)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return fallback

    # ── Attack Strategy ───────────────────────────────────────────────────────

    def generate_attack_strategy(self, image_info, history, rl_suggestion=None):
        result = self.generate_attack_strategy_structured(image_info, history, rl_suggestion)
        return json.dumps(result)

    def generate_attack_strategy_structured(self, image_info, history, rl_suggestion=None):
        prompt = self._build_attack_prompt(image_info, history, rl_suggestion)
        fallback = {
            "attack_type": "fgsm",
            "strength": "medium",
            "reasoning": "Defaulting to FGSM medium — reliable gradient-based attack that works on most classifiers.",
            "strategy": "standard_gradient",
            "why_chosen": "FGSM is the baseline adversarial attack; chosen when no prior history is available.",
        }
        return self.generate_structured(
            prompt,
            "You are an adversarial ML red-teamer. Reply ONLY in valid JSON, no extra text.",
            fallback,
        )

    def _build_attack_prompt(self, image_info, history, rl_suggestion):
        cls = image_info.get("original_class", "Unknown")
        conf = image_info.get("confidence", 0)
        recent = history[-3:] if history else []
        history_str = json.dumps([
            {"attack": b.get("attack_type"), "strength": b.get("attack_strength", b.get("strength")),
             "success": b.get("attack_success"), "defense": b.get("defense_type")}
            for b in recent
        ]) if recent else "None"

        prompt = f"""You are a red-team adversarial ML attacker targeting a ResNet-50/ResNet-18 ImageNet classifier.

Image info:
- Original class: {cls}
- Confidence: {conf:.2%}

Available attack types: fgsm, pgd, cw, bim, deepfool, salt_pepper, contrast_shift, jpeg_attack, frequency_filter, spatial_perturbation, universal_perturbation, ai
Strength levels: low, medium, high

Recent battle history: {history_str}
"""
        if rl_suggestion:
            prompt += f"\nRL reinforcement learning suggestion: {rl_suggestion}\n"
        prompt += """
Choose the attack type and strength most likely to cause a misclassification for this specific image class.
Consider: gradient-based attacks (fgsm, pgd, bim) work well on high-confidence images; perturbation attacks (salt_pepper, contrast_shift) work on texture-sensitive models.

Reply ONLY in JSON:
{
  "attack_type": "<type>",
  "strength": "<low|medium|high>",
  "reasoning": "2-3 sentence technical explanation of why this attack targets the model's specific vulnerability for this image class",
  "strategy": "<short strategy name>",
  "why_chosen": "1-2 sentence explanation of why this specific combination was chosen given the history and RL suggestion"
}"""
        return prompt

    # ── Defense Strategy ──────────────────────────────────────────────────────

    def generate_defense_strategy(self, image_info, history, rl_suggestion=None):
        result = self.generate_defense_strategy_structured(image_info, history, rl_suggestion)
        return json.dumps(result)

    def generate_defense_strategy_structured(self, image_info, history, rl_suggestion=None):
        prompt = self._build_defense_prompt(image_info, history, rl_suggestion)
        fallback = {
            "defense_type": "gaussian_blur",
            "strength": "medium",
            "reasoning": "Gaussian blur removes high-frequency adversarial noise while preserving low-frequency class structure.",
            "strategy": "frequency_smoothing",
            "why_chosen": "Chosen as a robust baseline defense that works across a wide range of attack types.",
        }
        return self.generate_structured(
            prompt,
            "You are an adversarial ML defender. Reply ONLY in valid JSON, no extra text.",
            fallback,
        )

    def _build_defense_prompt(self, image_info, history, rl_suggestion):
        cls = image_info.get("original_class", "Unknown")
        conf = image_info.get("confidence", 0)
        adv_cls = image_info.get("adversarial_class", "Unknown")
        recent = history[-3:] if history else []
        history_str = json.dumps([
            {"attack": b.get("attack_type"), "defense": b.get("defense_type"),
             "success": b.get("defense_success")}
            for b in recent
        ]) if recent else "None"

        prompt = f"""You are an adversarial ML defender protecting a ResNet-50/ResNet-18 ImageNet classifier.

Image info:
- Original class: {cls} (confidence {conf:.2%})
- After attack class: {adv_cls}

Available defense types: gaussian_blur, jpeg_compression, feature_squeezing, ai_defense, median_blur, bilateral_filter, tv_denoising, randomized_smoothing, pixel_deflection, quilting, autoencoder_restoration, diff_jpeg
Strength levels: low, medium, high

Recent battle history: {history_str}
"""
        if rl_suggestion:
            prompt += f"\nRL reinforcement learning suggestion: {rl_suggestion}\n"
        prompt += """
Choose the defense type and strength most likely to restore correct classification.
Consider: gaussian_blur removes L∞ perturbations; jpeg_compression destroys high-frequency noise; feature_squeezing reduces color depth; randomized_smoothing adds certified robustness.

Reply ONLY in JSON:
{
  "defense_type": "<type>",
  "strength": "<low|medium|high>",
  "reasoning": "2-3 sentence technical explanation of why this defense counters the attack mechanism",
  "strategy": "<short strategy name>",
  "why_chosen": "1-2 sentence explanation referencing the specific attack type and history"
}"""
        return prompt

    # ── Proactive Attacker ────────────────────────────────────────────────────

    def generate_proactive_attacker_strategy_structured(
        self, image_info, defense_type, defense_strength, history, rl_suggestion=None
    ):
        prompt = f"""You are a red-team adversarial ML attacker in PROACTIVE mode.
The target image has already been pre-hardened by the defender.

Original class: {image_info.get('original_class', 'Unknown')} (confidence {image_info.get('confidence', 0):.2%})
Defense already applied: {defense_type} at {defense_strength} strength.

Your task: pick an attack type and strength that specifically DEFEATS this defense.
Defense-attack counter-strategy knowledge:
- gaussian_blur → use high-frequency attacks: frequency_filter, jpeg_attack, or high-epsilon fgsm
- jpeg_compression → use gradient attacks: pgd, bim (jpeg-resilient perturbations)
- feature_squeezing → use spatial attacks: spatial_perturbation, deepfool
- randomized_smoothing → use pgd with many steps to overcome smoothing
- bilateral_filter → use structured noise: universal_perturbation, contrast_shift
- ai_defense → use ai attack (adaptive)

Available attacks: fgsm, pgd, cw, bim, deepfool, salt_pepper, contrast_shift, jpeg_attack, frequency_filter, spatial_perturbation, universal_perturbation, ai
Strengths: low, medium, high

Recent battle history (proactive rounds): {history[-5:] if history else 'None'}
"""
        if rl_suggestion:
            prompt += f"\nRL suggestion: {rl_suggestion}\n"
        prompt += """
Reply ONLY in JSON:
{
  "attack_type": "<type>",
  "strength": "<low|medium|high>",
  "reasoning": "2-3 sentence technical explanation of why this attack specifically defeats the applied defense",
  "strategy": "<strategy name>",
  "why_chosen": "1-2 sentence explanation referencing the specific defense and why this attack bypasses it"
}"""
        fallback = {
            "attack_type": "fgsm",
            "strength": "high",
            "reasoning": "Defaulting to high-strength FGSM which works through most defenses.",
            "strategy": "standard",
            "why_chosen": "Fallback strategy when no specific counter is available.",
        }
        return self.generate_structured(
            prompt,
            "You are a red-team AI. Choose the best attack to defeat the given defense. Reply only in JSON.",
            fallback,
        )

    # ── Multi-Algorithm Strategy ──────────────────────────────────────────────

    def generate_multi_attack_strategies(self, image_info, history, rl_suggestion=None, top_actions=None, n=3):
        """Return a list of N attack strategies, best candidates from RL + reasoning."""
        top_str = ""
        if top_actions:
            top_str = "Top RL-learned attack actions (by avg Q-value):\n" + "\n".join(
                [f"  - {a}: Q={q:.4f}" for a, q in top_actions[:6]]
            )
        prompt = f"""You are a red-team adversarial ML attacker selecting MULTIPLE strategies for a comprehensive attack.

Image: {image_info.get('original_class', 'Unknown')} ({image_info.get('confidence', 0):.2%} confidence)
{top_str}
Recent history: {json.dumps(history[-3:]) if history else 'None'}
{'RL suggestion: ' + rl_suggestion if rl_suggestion else ''}

Select between 2 and {n} distinct attack strategies (different attack_type values), ranked from most to least promising. Based on the battle history and Q-values, YOU must intelligently decide the optimal number of strategies to combine in order to overwhelm the defense. Do NOT use more strategies than necessary.

Available attack types: fgsm, pgd, cw, bim, deepfool, salt_pepper, contrast_shift, jpeg_attack, frequency_filter, spatial_perturbation, universal_perturbation, ai
Strengths: low, medium, high

Reply ONLY in JSON:
{{
  "strategies": [
    {{"attack_type": "...", "strength": "...", "reasoning": "why this is best", "rank": 1}},
    {{"attack_type": "...", "strength": "...", "reasoning": "why this is second", "rank": 2}}
    // Add up to {n} strategies total, depending on what you think is necessary
  ],
  "overall_reasoning": "2-3 sentence explanation of the multi-strategy approach and WHY you chose this specific number of attackers"
}}"""
        fallback = {
            "strategies": [
                {"attack_type": "pgd", "strength": "high", "reasoning": "Strong gradient attack", "rank": 1},
                {"attack_type": "fgsm", "strength": "medium", "reasoning": "Fast baseline attack", "rank": 2},
                {"attack_type": "cw", "strength": "medium", "reasoning": "Optimization-based attack", "rank": 3},
            ],
            "overall_reasoning": "Multi-strategy approach covering gradient, fast, and optimization-based attacks.",
        }
        return self.generate_structured(
            prompt,
            "You are a red-team AI selecting multiple attack strategies. Reply only in valid JSON.",
            fallback,
        )

    def generate_multi_defense_strategies(self, image_info, history, rl_suggestion=None, top_actions=None, n=3):
        """Return a list of N defense strategies."""
        top_str = ""
        if top_actions:
            top_str = "Top RL-learned defense actions (by avg Q-value):\n" + "\n".join(
                [f"  - {a}: Q={q:.4f}" for a, q in top_actions[:6]]
            )
        prompt = f"""You are an adversarial ML defender selecting MULTIPLE defense strategies for comprehensive protection.

Image: {image_info.get('original_class', 'Unknown')} ({image_info.get('confidence', 0):.2%} confidence)
{top_str}
Recent history: {json.dumps(history[-3:]) if history else 'None'}
{'RL suggestion: ' + rl_suggestion if rl_suggestion else ''}

Select between 2 and {n} distinct defense strategies (different defense_type values), ranked from most to least promising. Based on the battle history and Q-values, YOU must intelligently decide the optimal number of strategies to combine in order to protect against severe attacks. Do NOT use more strategies than necessary.

Available defense types: gaussian_blur, jpeg_compression, feature_squeezing, ai_defense, median_blur, bilateral_filter, tv_denoising, randomized_smoothing, pixel_deflection, quilting, autoencoder_restoration, diff_jpeg
Strengths: low, medium, high

Reply ONLY in JSON:
{{
  "strategies": [
    {{"defense_type": "...", "strength": "...", "reasoning": "why this is best", "rank": 1}},
    {{"defense_type": "...", "strength": "...", "reasoning": "why this is second", "rank": 2}}
    // Add up to {n} strategies total, depending on what you think is necessary
  ],
  "overall_reasoning": "2-3 sentence explanation of the multi-defense approach and WHY you chose this specific number of defenders"
}}"""
        fallback = {
            "strategies": [
                {"defense_type": "gaussian_blur", "strength": "medium", "reasoning": "Removes high-freq noise", "rank": 1},
                {"defense_type": "jpeg_compression", "strength": "medium", "reasoning": "Destroys adversarial structure", "rank": 2},
                {"defense_type": "feature_squeezing", "strength": "high", "reasoning": "Reduces color space", "rank": 3},
            ],
            "overall_reasoning": "Multi-defense approach covering spatial, compression, and feature-level defenses.",
        }
        return self.generate_structured(
            prompt,
            "You are a defender AI selecting multiple defense strategies. Reply only in valid JSON.",
            fallback,
        )

    # ── Outcome Analysis ──────────────────────────────────────────────────────

    def generate_outcome_analysis(self, battle_data):
        """Rich analysis explaining the full battle outcome using Groq API."""
        battle_mode = battle_data.get("battle_mode", "reactive")
        atk = battle_data.get("attack_type", "unknown")
        atk_str = battle_data.get("attack_strength", "medium")
        dfn = battle_data.get("defense_type", "unknown")
        dfn_str = battle_data.get("defense_strength", "medium")
        orig_cls = battle_data.get("original_class", "?")
        orig_conf = battle_data.get("original_confidence", 0)
        adv_cls = battle_data.get("adversarial_class", "?")
        adv_conf = battle_data.get("adversarial_confidence", 0)
        def_cls = battle_data.get("defended_class", "?")
        def_conf = battle_data.get("defended_confidence", 0)
        atk_ok = battle_data.get("attack_success", False)
        def_ok = battle_data.get("defense_success", False)

        conf_drop = (orig_conf - adv_conf) * 100
        conf_restore = (def_conf - adv_conf) * 100

        prompt = f"""You are an expert adversarial ML analyst. Analyze this battle round in detail.

=== BATTLE REPORT ({battle_mode.upper()} MODE) ===
Attack: {atk} at {atk_str} strength
Defense: {dfn} at {dfn_str} strength

Original classification: {orig_cls} ({orig_conf:.1%} confidence)
After attack:            {adv_cls} ({adv_conf:.1%} confidence)  [class changed: {orig_cls != adv_cls}, conf drop: {conf_drop:.1f}%]
After defense:           {def_cls} ({def_conf:.1%} confidence)  [restored: {def_cls == orig_cls}, conf gain: {conf_restore:.1f}%]

Attack succeeded: {atk_ok}
Defense succeeded: {def_ok}

Provide detailed, technically accurate analysis for each field below.
For attacker/defender strategy explanations: explain WHY they chose this type based on technical properties.
For success/failure: explain the MECHANISM (e.g., epsilon size, gradient direction, filter frequency response).

Reply ONLY in valid JSON (no extra text):
{{
  "attacker_logic": "2-3 sentences: what the {atk} attack does at a technical level and how it manipulates pixel gradients to fool the classifier",
  "defender_strategy": "2-3 sentences: how {dfn} defense works mechanically and why it was chosen to counter adversarial perturbations",
  "why_attacker_chose_this": "2 sentences: specific strategic reason this attack type was selected for this image class and confidence level",
  "why_defender_chose_this": "2 sentences: specific strategic reason this defense type was selected given the attack and model architecture",
  "why_attack_succeeded_or_failed": "2-3 sentences: precise technical explanation of WHY the attack {'succeeded' if atk_ok else 'failed'} — reference epsilon={atk_str}, gradient properties, or model vulnerability",
  "why_defense_succeeded_or_failed": "2-3 sentences: precise technical explanation of WHY the defense {'succeeded' if def_ok else 'failed'} — reference filter properties, residual noise, or frequency spectrum",
  "what_model_focused_on": "2 sentences: what GradCAM would show the model attending to at each stage (original, attacked, defended)",
  "key_insight": "One concise takeaway sentence about what this round reveals about the attack-defense dynamics",
  "learning_note": "One sentence about what the RL agent should update in its Q-table based on this outcome"
}}"""

        fallback = {
            "attacker_logic": f"{atk} attack perturbs pixel values along gradient directions to maximize model loss.",
            "defender_strategy": f"{dfn} defense applies signal processing to remove adversarial noise patterns.",
            "why_attacker_chose_this": f"{atk} was chosen based on RL Q-value learning from previous rounds.",
            "why_defender_chose_this": f"{dfn} was selected as the highest-Q defense for this image state.",
            "why_attack_succeeded_or_failed": f"Attack {'caused misclassification by exceeding the model decision boundary' if atk_ok else 'failed to push confidence below the classification threshold'}.",
            "why_defense_succeeded_or_failed": f"Defense {'successfully removed adversarial perturbations' if def_ok else 'could not fully eliminate the adversarial noise'}.",
            "what_model_focused_on": "GradCAM shows model attention shifting from semantic regions to noise artifacts during attack.",
            "key_insight": f"{'Attack broke through' if atk_ok else 'Defense held'} at {atk_str} strength — this updates the RL Q-table accordingly.",
            "learning_note": f"RL agent should {'increase Q-value for' if atk_ok else 'decrease Q-value for'} {atk}_{atk_str}.",
        }
        return self.generate_structured(
            prompt,
            "You are an adversarial ML analyst. Provide detailed technical analysis. Reply ONLY in valid JSON.",
            fallback,
        )

    # ── Coach ─────────────────────────────────────────────────────────────────

    def coach(self, battle_history):
        if len(battle_history) < 3:
            return "Keep battling to gather more data for coaching."

        wins = sum(1 for b in battle_history if b.get("attacker_won", False))
        losses = len(battle_history) - wins
        attack_types = {}
        for b in battle_history:
            at = b.get("attack_type", "unknown")
            attack_types[at] = attack_types.get(at, 0) + 1

        prompt = f"""Analyze this AdverShield battle history and provide strategic coaching:

- Total battles: {len(battle_history)}
- Attacker wins: {wins} ({wins/len(battle_history):.0%})
- Defender wins: {losses} ({losses/len(battle_history):.0%})
- Attack type distribution: {json.dumps(attack_types)}

Recent battles:
{chr(10).join([str(b) for b in battle_history[-5:]])}

Provide actionable advice: which strategies are working, which to avoid, and what to try next."""
        return self.generate(
            prompt,
            "You are an expert ML coach analyzing adversarial battle strategies. Be specific and actionable.",
        )