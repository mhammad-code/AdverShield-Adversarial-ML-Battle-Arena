from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import json
import time
import random
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import torch
import numpy as np
import cv2
from PIL import Image
from werkzeug.utils import secure_filename

from config import STATIC_DIR, ATTACK_TYPES, DEFENSE_TYPES, STRENGTHS, COACH_INTERVAL, SAMPLES_DIR
from target_model import TargetModel
from gradcam import GradCAM
from attacks import AttackEngine
from defenses import DefenseEngine
from llm_engine import LLMEngine
from rl_tracker import RLTracker
from battle_memory import BattleMemory
from image_loader import ImageLoader
from settings_model import SettingsModel
from stats_model import StatsModel

# Rolling ETA tracker: {mode: [duration_seconds, ...]}
_battle_timings = {"manual": [], "reactive": [], "proactive": [], "proactive_multi": []}


app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs('templates', exist_ok=True)

print("Loading target model...")
target_model = TargetModel()
print("Target model loaded")

print("Setting up GradCAM...")
gradcam = GradCAM(target_model)
print("GradCAM ready")

print("Setting up attack engine...")
attack_engine = AttackEngine(target_model)
print("Attack engine ready")

try:
    llm_engine = LLMEngine()
    print("LLM Engine loaded")
except Exception as e:
    print(f"⚠️  LLM disabled: {e}")
    llm_engine = None

# Inject LLM into both engines
if llm_engine is not None:
    attack_engine.llm_engine = llm_engine

defense_engine = DefenseEngine(llm_engine=llm_engine)

rl_tracker = RLTracker()
battle_memory = BattleMemory()
image_loader = ImageLoader()
settings_model = SettingsModel()
stats_model = StatsModel(battle_memory)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _save_rgb01_hwc_to_static(rgb01_hwc, filename):
    rgb01_hwc = np.asarray(rgb01_hwc, dtype=np.float32)
    rgb_uint8 = np.clip(rgb01_hwc * 255.0, 0, 255).astype(np.uint8)
    bgr_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(STATIC_DIR, filename), bgr_uint8)
    return filename


# ─── Image Routes ────────────────────────────────────────────────────────────

@app.route('/api/images', methods=['GET'])
def list_images():
    images = image_loader.list_available_images()
    return jsonify({"images": images})


@app.route('/api/random-image', methods=['GET'])
def random_image():
    images = image_loader.list_available_images()
    if not images:
        return jsonify({"error": "No images in samples folder"}), 404
    return jsonify({"filename": random.choice(images)})


@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_name = f"upload_{uuid.uuid4().hex[:6]}_{filename}"
        filepath = os.path.join(SAMPLES_DIR, unique_name)
        file.save(filepath)
        return jsonify({"filename": unique_name, "message": "Upload successful"})
    return jsonify({"error": "File type not allowed"}), 400


@app.route('/samples/<path:filename>')
def serve_sample(filename):
    return send_from_directory(SAMPLES_DIR, filename)


@app.route('/api/image-preview/<path:filename>')
def image_preview(filename):
    return send_from_directory(SAMPLES_DIR, filename)


# ─── Classification ──────────────────────────────────────────────────────────

@app.route('/api/classify', methods=['POST'])
def classify_image():
    data = request.json
    filename = data.get('image')

    try:
        image_tensor = image_loader.load_image(filename)
        original_class, class_name, confidence, top5 = target_model.predict(image_tensor)

        heatmap = gradcam.generate(image_tensor)
        if heatmap is not None:
            heatmap_filename = f"original_{uuid.uuid4().hex[:8]}.png"
            heatmap_path = os.path.join(STATIC_DIR, heatmap_filename)
            gradcam.save_heatmap(
                heatmap,
                heatmap_path,
                output_shape=(image_tensor.shape[2], image_tensor.shape[3]),
                colored=True,
            )

            original_image_np = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
            overlay_filename = f"overlay_{uuid.uuid4().hex[:8]}.png"
            overlay_path = os.path.join(STATIC_DIR, overlay_filename)
            gradcam.save_overlay(original_image_np, heatmap, overlay_path)
        else:
            heatmap_filename = None
            overlay_filename = None

        return jsonify({
            "original_class": class_name,
            "confidence": confidence,
            "top5": top5,
            "heatmap": heatmap_filename,
            "overlay": overlay_filename
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ─── Attack ──────────────────────────────────────────────────────────────────

@app.route('/api/attack', methods=['POST'])
def attack():
    data = request.json
    filename = data.get('image')
    attack_type = data.get('attack_type', 'fgsm')
    strength = data.get('strength', 'medium')
    target_class = data.get('target_class')

    try:
        image_tensor = image_loader.load_image(filename)
        original_class, original_name, original_conf, original_top5 = target_model.predict(image_tensor)

        adversarial_tensor, llm_instructions = attack_engine.generate_attack(
            image_tensor, attack_type, strength, target_class,
            battle_context={"recent_battles": battle_memory.get_recent(3)}
        )

        adv_class, adv_name, adv_conf, adv_top5 = target_model.predict(adversarial_tensor)

        state = rl_tracker.get_state(original_name, original_conf)
        rl_suggestion = rl_tracker.get_suggestion(state, "attacker")

        llm_response = "LLM unavailable"
        if llm_engine is not None:
            if llm_instructions:
                llm_response = json.dumps(llm_instructions)
            else:
                llm_response = llm_engine.generate_attack_strategy(
                    {"original_class": original_name, "confidence": original_conf},
                    battle_memory.get_recent(3),
                    rl_suggestion
                )

        heatmap_original = gradcam.generate(image_tensor, original_class)
        heatmap_adv = gradcam.generate(adversarial_tensor, adv_class)

        hm_orig_name = None
        hm_adv_name = None
        overlay_orig_name = None
        overlay_adv_name = None

        if heatmap_original is not None:
            hm_orig_name = f"hm_orig_{uuid.uuid4().hex[:8]}.png"
            original_image_np = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
            gradcam.save_heatmap(
                heatmap_original,
                os.path.join(STATIC_DIR, hm_orig_name),
                output_shape=original_image_np.shape[:2],
                colored=True,
            )
            overlay_orig_name = f"overlay_orig_{uuid.uuid4().hex[:8]}.png"
            gradcam.save_overlay(original_image_np, heatmap_original, os.path.join(STATIC_DIR, overlay_orig_name))

        if heatmap_adv is not None:
            hm_adv_name = f"hm_adv_{uuid.uuid4().hex[:8]}.png"
            adv_image_np = adversarial_tensor[0].cpu().numpy().transpose(1, 2, 0)
            gradcam.save_heatmap(
                heatmap_adv,
                os.path.join(STATIC_DIR, hm_adv_name),
                output_shape=adv_image_np.shape[:2],
                colored=True,
            )
            overlay_adv_name = f"overlay_adv_{uuid.uuid4().hex[:8]}.png"
            gradcam.save_overlay(adv_image_np, heatmap_adv, os.path.join(STATIC_DIR, overlay_adv_name))

        adversarial_image_name = f"img_adv_{uuid.uuid4().hex[:8]}.png"
        adv_image_np = adversarial_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        _save_rgb01_hwc_to_static(adv_image_np, adversarial_image_name)

        attack_success = original_class != adv_class

        battle_record = {
            "image": filename,
            "attack_type": attack_type,
            "strength": strength,
            "original_class": original_name,
            "original_confidence": original_conf,
            "adversarial_class": adv_name,
            "adversarial_confidence": adv_conf,
            "attack_success": attack_success,
            "attacker_won": attack_success,
            "defense_success": False,
            "is_draw": False,
            "mode": "attack",
            "attack_llm_response": llm_response
        }
        battle_memory.add_battle(battle_record)

        reward = 1.0 if attack_success else -0.5
        rl_tracker.update(state, (attack_type, strength), reward, role="attacker")

        return jsonify({
            "original_class": original_name,
            "original_confidence": original_conf,
            "original_top5": original_top5,
            "adversarial_class": adv_name,
            "adversarial_confidence": adv_conf,
            "adversarial_top5": adv_top5,
            "attack_success": attack_success,
            "llm_strategy": llm_response,
            "rl_suggestion": rl_suggestion,
            "heatmap_original": hm_orig_name,
            "heatmap_adversarial": hm_adv_name,
            "overlay_original": overlay_orig_name,
            "overlay_adversarial": overlay_adv_name,
            "adversarial_image": adversarial_image_name,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ─── Defense ─────────────────────────────────────────────────────────────────

@app.route('/api/defend', methods=['POST'])
def defend():
    data = request.json
    filename = data.get('image')
    attack_type = data.get('attack_type', 'fgsm')
    attack_strength = data.get('attack_strength', 'medium')
    defense_type = data.get('defense_type', 'gaussian_blur')
    defense_strength = data.get('defense_strength', 'medium')

    try:
        image_tensor = image_loader.load_image(filename)

        original_class, original_name, original_conf, original_top5 = target_model.predict(image_tensor)

        adversarial_tensor, _ = attack_engine.generate_attack(
            image_tensor, attack_type, attack_strength,
            battle_context={"recent_battles": battle_memory.get_recent(3)}
        )
        adv_class, adv_name, adv_conf, adv_top5 = target_model.predict(adversarial_tensor)

        defended_tensor, defense_llm_instructions = defense_engine.apply_defense(
            adversarial_tensor, defense_type, defense_strength,
            battle_context={"recent_battles": battle_memory.get_recent(3)}
        )
        defended_class, defended_name, defended_conf, defended_top5 = target_model.predict(defended_tensor)

        state = rl_tracker.get_state(original_name, original_conf)
        rl_suggestion = rl_tracker.get_suggestion(state, "defender")

        llm_response = "LLM unavailable"
        if llm_engine is not None:
            llm_response = llm_engine.generate_defense_strategy(
                {"original_class": original_name, "confidence": original_conf, "adversarial_class": adv_name},
                battle_memory.get_recent(3),
                rl_suggestion
            )

        attack_success = original_class != adv_class
        defense_success = defended_class == original_class

        outcome_analysis = {}
        if llm_engine is not None:
            outcome_analysis = llm_engine.generate_outcome_analysis({
                "attack_type": attack_type,
                "attack_strength": attack_strength,
                "defense_type": defense_type,
                "defense_strength": defense_strength,
                "original_class": original_name,
                "original_confidence": original_conf,
                "adversarial_class": adv_name,
                "adversarial_confidence": adv_conf,
                "defended_class": defended_name,
                "defended_confidence": defended_conf,
                "attack_success": attack_success,
                "defense_success": defense_success
            })

        hm_orig_name = hm_adv_name = hm_def_name = None
        overlay_orig_name = overlay_adv_name = overlay_def_name = None

        heatmap_original = gradcam.generate(image_tensor, original_class)
        if heatmap_original is not None:
            hm_orig_name = f"hm_orig_{uuid.uuid4().hex[:8]}.png"
            orig_np = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
            gradcam.save_heatmap(
                heatmap_original,
                os.path.join(STATIC_DIR, hm_orig_name),
                output_shape=orig_np.shape[:2],
                colored=True,
            )
            overlay_orig_name = f"overlay_orig_{uuid.uuid4().hex[:8]}.png"
            gradcam.save_overlay(orig_np, heatmap_original, os.path.join(STATIC_DIR, overlay_orig_name))

        heatmap_adv = gradcam.generate(adversarial_tensor, adv_class)
        if heatmap_adv is not None:
            hm_adv_name = f"hm_adv_{uuid.uuid4().hex[:8]}.png"
            adv_np = adversarial_tensor[0].cpu().numpy().transpose(1, 2, 0)
            gradcam.save_heatmap(
                heatmap_adv,
                os.path.join(STATIC_DIR, hm_adv_name),
                output_shape=adv_np.shape[:2],
                colored=True,
            )
            overlay_adv_name = f"overlay_adv_{uuid.uuid4().hex[:8]}.png"
            gradcam.save_overlay(adv_np, heatmap_adv, os.path.join(STATIC_DIR, overlay_adv_name))

        heatmap_defended = gradcam.generate(defended_tensor, defended_class)
        if heatmap_defended is not None:
            hm_def_name = f"hm_def_{uuid.uuid4().hex[:8]}.png"
            def_np = defended_tensor[0].cpu().numpy().transpose(1, 2, 0)
            gradcam.save_heatmap(
                heatmap_defended,
                os.path.join(STATIC_DIR, hm_def_name),
                output_shape=def_np.shape[:2],
                colored=True,
            )
            overlay_def_name = f"overlay_def_{uuid.uuid4().hex[:8]}.png"
            gradcam.save_overlay(def_np, heatmap_defended, os.path.join(STATIC_DIR, overlay_def_name))

        adversarial_image_name = f"img_adv_{uuid.uuid4().hex[:8]}.png"
        defended_image_name = f"img_def_{uuid.uuid4().hex[:8]}.png"
        adv_np_img = adversarial_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        def_np_img = defended_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        _save_rgb01_hwc_to_static(adv_np_img, adversarial_image_name)
        _save_rgb01_hwc_to_static(def_np_img, defended_image_name)

        battle_record = {
            "image": filename,
            "attack_type": attack_type,
            "attack_strength": attack_strength,
            "defense_type": defense_type,
            "defense_strength": defense_strength,
            "original_class": original_name,
            "original_confidence": original_conf,
            "adversarial_class": adv_name,
            "adversarial_confidence": adv_conf,
            "defended_class": defended_name,
            "defended_confidence": defended_conf,
            "attack_success": attack_success,
            "attacker_won": attack_success,
            "defense_success": defense_success,
            "is_draw": False,
            "mode": "defend"
        }
        battle_memory.add_battle(battle_record)

        reward_att = 1.0 if attack_success else -0.5
        rl_tracker.update(state, (attack_type, attack_strength), reward_att, role="attacker")
        reward_def = 1.0 if defense_success else -0.5
        rl_tracker.update(state, (defense_type, defense_strength), reward_def, role="defender")

        return jsonify({
            "original_class": original_name,
            "original_confidence": original_conf,
            "original_top5": original_top5,
            "adversarial_class": adv_name,
            "adversarial_confidence": adv_conf,
            "adversarial_top5": adv_top5,
            "defended_class": defended_name,
            "defended_confidence": defended_conf,
            "defended_top5": defended_top5,
            "attack_success": attack_success,
            "defense_success": defense_success,
            "llm_strategy": llm_response,
            "defense_llm_strategy": defense_llm_instructions,
            "rl_suggestion": rl_suggestion,
            "heatmap_original": hm_orig_name,
            "heatmap_adversarial": hm_adv_name,
            "heatmap_defended": hm_def_name,
            "overlay_original": overlay_orig_name,
            "overlay_adversarial": overlay_adv_name,
            "overlay_defended": overlay_def_name,
            "adversarial_image": adversarial_image_name,
            "defended_image": defended_image_name,
            "outcome_analysis": outcome_analysis,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ─── AI Battle (Reactive – classic attack‑then‑defense) ──────────────────────

@app.route('/api/ai-battle', methods=['POST'])
def ai_battle():
    """Fully autonomous AI battle (reactive). LLM chooses attack & defense types + strengths."""
    data = request.json
    filename = data.get('image')

    if not llm_engine:
        return jsonify({"error": "LLM not available"}), 400

    try:
        image_tensor = image_loader.load_image(filename)
        original_class, original_name, original_conf, _ = target_model.predict(image_tensor)

        state = rl_tracker.get_state(original_name, original_conf)

        # ----- Attacker AI decision (structured) -----
        atk_strat = llm_engine.generate_attack_strategy_structured(
            {"original_class": original_name, "confidence": original_conf},
            battle_memory.get_recent(5),
            rl_tracker.get_suggestion(state, "attacker")
        )
        attack_type = atk_strat.get("attack_type", "fgsm")
        if attack_type not in ATTACK_TYPES:
            attack_type = "fgsm"
        attack_strength = atk_strat.get("strength", "medium")
        if attack_strength not in STRENGTHS:
            attack_strength = "medium"

        # ----- Defender AI decision (structured) -----
        def_strat = llm_engine.generate_defense_strategy_structured(
            {"original_class": original_name, "confidence": original_conf},
            battle_memory.get_recent(5),
            rl_tracker.get_suggestion(state, "defender")
        )
        defense_type = def_strat.get("defense_type", "gaussian_blur")
        if defense_type not in DEFENSE_TYPES:
            defense_type = "gaussian_blur"
        defense_strength = def_strat.get("strength", "medium")
        if defense_strength not in STRENGTHS:
            defense_strength = "medium"

        # ----- Build attacker strategy object with reasoning from LLM decision -----
        attacker_strategy = {
            "reasoning": atk_strat.get("reasoning", "No specific reasoning."),
            "strategy": atk_strat.get("strategy", "Standard attack"),
            "attack_type": attack_type,
            "strength": attack_strength,
            "operations": []
        }

        # ----- Build defender strategy object with reasoning from LLM decision -----
        defender_strategy = {
            "reasoning": def_strat.get("reasoning", "No specific reasoning."),
            "strategy": def_strat.get("strategy", "Standard defense"),
            "defense_type": defense_type,
            "strength": defense_strength,
            "operations": []
        }

        # ----- Execute attack -----
        adversarial_tensor, atk_llm_instructions = attack_engine.generate_attack(
            image_tensor, attack_type, attack_strength,
            battle_context={"recent_battles": battle_memory.get_recent(5)}
        )
        adv_class, adv_name, adv_conf, _ = target_model.predict(adversarial_tensor)

        # If attack was AI‑generated, merge its operations
        if atk_llm_instructions and "operations" in atk_llm_instructions:
            attacker_strategy["operations"] = atk_llm_instructions["operations"]
            if atk_llm_instructions.get("reasoning"):
                attacker_strategy["reasoning"] += " | " + atk_llm_instructions["reasoning"]

        # ----- Execute defense -----
        defended_tensor, def_llm_instructions = defense_engine.apply_defense(
            adversarial_tensor, defense_type, defense_strength,
            battle_context={"recent_battles": battle_memory.get_recent(5)}
        )
        defended_class, defended_name, defended_conf, _ = target_model.predict(defended_tensor)

        # If defense was AI‑generated, merge its operations
        if def_llm_instructions and "operations" in def_llm_instructions:
            defender_strategy["operations"] = def_llm_instructions["operations"]
            if def_llm_instructions.get("reasoning"):
                defender_strategy["reasoning"] += " | " + def_llm_instructions["reasoning"]

        attack_success = original_class != adv_class
        defense_success = defended_class == original_class

        # ----- Grad‑CAM overlays (unchanged) -----
        hm_orig, overlay_orig = None, None
        heatmap_original = gradcam.generate(image_tensor, original_class)
        if heatmap_original is not None:
            hm_orig = f"ai_orig_{uuid.uuid4().hex[:8]}.png"
            orig_np = image_tensor[0].cpu().numpy().transpose(1,2,0)
            gradcam.save_heatmap(heatmap_original, os.path.join(STATIC_DIR, hm_orig),
                                 output_shape=orig_np.shape[:2], colored=True)
            overlay_orig = f"ai_overlay_orig_{uuid.uuid4().hex[:8]}.png"
            gradcam.save_overlay(orig_np, heatmap_original, os.path.join(STATIC_DIR, overlay_orig))

        hm_adv, overlay_adv = None, None
        heatmap_adv = gradcam.generate(adversarial_tensor, adv_class)
        if heatmap_adv is not None:
            hm_adv = f"ai_adv_{uuid.uuid4().hex[:8]}.png"
            adv_np = adversarial_tensor[0].cpu().numpy().transpose(1,2,0)
            gradcam.save_heatmap(heatmap_adv, os.path.join(STATIC_DIR, hm_adv),
                                 output_shape=adv_np.shape[:2], colored=True)
            overlay_adv = f"ai_overlay_adv_{uuid.uuid4().hex[:8]}.png"
            gradcam.save_overlay(adv_np, heatmap_adv, os.path.join(STATIC_DIR, overlay_adv))

        hm_def, overlay_def = None, None
        heatmap_def = gradcam.generate(defended_tensor, defended_class)
        if heatmap_def is not None:
            hm_def = f"ai_def_{uuid.uuid4().hex[:8]}.png"
            def_np = defended_tensor[0].cpu().numpy().transpose(1,2,0)
            gradcam.save_heatmap(heatmap_def, os.path.join(STATIC_DIR, hm_def),
                                 output_shape=def_np.shape[:2], colored=True)
            overlay_def = f"ai_overlay_def_{uuid.uuid4().hex[:8]}.png"
            gradcam.save_overlay(def_np, heatmap_def, os.path.join(STATIC_DIR, overlay_def))

        # ----- LLM outcome analysis -----
        outcome_analysis = llm_engine.generate_outcome_analysis({
            "attack_type": attack_type, "attack_strength": attack_strength,
            "defense_type": defense_type, "defense_strength": defense_strength,
            "original_class": original_name, "original_confidence": original_conf,
            "adversarial_class": adv_name, "adversarial_confidence": adv_conf,
            "defended_class": defended_name, "defended_confidence": defended_conf,
            "attack_success": attack_success, "defense_success": defense_success
        }) if llm_engine else {}

        # ----- Update RL and battle memory -----
        battle_record = {
            "image": filename,
            "attack_type": attack_type, "attack_strength": attack_strength,
            "defense_type": defense_type, "defense_strength": defense_strength,
            "original_class": original_name, "original_confidence": original_conf,
            "adversarial_class": adv_name, "adversarial_confidence": adv_conf,
            "defended_class": defended_name, "defended_confidence": defended_conf,
            "attack_success": attack_success,
            "attacker_won": attack_success,
            "defense_success": defense_success,
            "is_draw": False, "mode": "ai_battle"
        }
        battle_memory.add_battle(battle_record)

        reward_att = 1.0 if attack_success else -0.5
        rl_tracker.update(state, (attack_type, attack_strength), reward_att, role="attacker")
        reward_def = 1.0 if defense_success else -0.5
        rl_tracker.update(state, (defense_type, defense_strength), reward_def, role="defender")

        # ----- Return full result, with combined strategies -----
        return jsonify({
            "attack_type": attack_type, "attack_strength": attack_strength,
            "defense_type": defense_type, "defense_strength": defense_strength,
            "original_class": original_name, "original_confidence": original_conf,
            "adversarial_class": adv_name, "adversarial_confidence": adv_conf,
            "defended_class": defended_name, "defended_confidence": defended_conf,
            "attack_success": attack_success, "defense_success": defense_success,
            "attacker_llm_strategy": attacker_strategy,
            "defender_llm_strategy": defender_strategy,
            "outcome_analysis": outcome_analysis,
            "overlay_original": overlay_orig,
            "overlay_adversarial": overlay_adv,
            "overlay_defended": overlay_def,
            "heatmap_original": hm_orig, "heatmap_adversarial": hm_adv, "heatmap_defended": hm_def
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ─── AI Battle PROACTIVE (defense first, then attack) ───────────────────────

@app.route('/api/ai-battle-proactive', methods=['POST'])
def ai_battle_proactive():
    """Proactive defense-first AI battle. Defender pre-hardens, attacker tries to break."""
    data = request.json
    filename = data.get('image')

    if not llm_engine:
        return jsonify({"error": "LLM not available"}), 400

    try:
        image_tensor = image_loader.load_image(filename)
        original_class, original_name, original_conf, _ = target_model.predict(image_tensor)

        state = rl_tracker.get_state(original_name, original_conf)

        # ----- 1. Defender AI chooses defense (proactive) -----
        def_strat = llm_engine.generate_defense_strategy_structured(
            {"original_class": original_name, "confidence": original_conf},
            battle_memory.get_recent(5),
            rl_tracker.get_suggestion(state, "defender")
        )
        defense_type = def_strat.get("defense_type", "gaussian_blur")
        if defense_type not in DEFENSE_TYPES:
            defense_type = "gaussian_blur"
        defense_strength = def_strat.get("strength", "medium")
        if defense_strength not in STRENGTHS:
            defense_strength = "medium"

        # Apply defense onto the original clean image
        defended_tensor, def_llm_instructions = defense_engine.apply_defense(
            image_tensor, defense_type, defense_strength,
            battle_context={"recent_battles": battle_memory.get_recent(5)}
        )

        # ✅ Predict the defended image so we have real values
        defended_class, defended_name, defended_conf, _ = target_model.predict(defended_tensor)

        # ----- 2. Attacker AI chooses attack, knowing the defense used -----
        atk_strat = llm_engine.generate_proactive_attacker_strategy_structured(
            {"original_class": original_name, "confidence": original_conf},
            defense_type, defense_strength,
            battle_memory.get_recent(5),
            rl_tracker.get_suggestion(state, "attacker")
        )
        attack_type = atk_strat.get("attack_type", "fgsm")
        if attack_type not in ATTACK_TYPES:
            attack_type = "fgsm"
        attack_strength = atk_strat.get("strength", "medium")
        if attack_strength not in STRENGTHS:
            attack_strength = "medium"

        adversarial_tensor, atk_llm_instructions = attack_engine.generate_attack(
            defended_tensor, attack_type, attack_strength,
            battle_context={"recent_battles": battle_memory.get_recent(5)}
        )
        final_class, final_name, final_conf, _ = target_model.predict(adversarial_tensor)

        # Winner is based on the final classification
        attacker_wins = final_class != original_class
        defender_wins = not attacker_wins

        # ----- Build strategy summaries -----
        attacker_strategy = {
            "reasoning": atk_strat.get("reasoning", ""),
            "strategy": atk_strat.get("strategy", "Targeted attack"),
            "attack_type": attack_type,
            "strength": attack_strength,
            "operations": atk_llm_instructions.get("operations", []) if atk_llm_instructions else []
        }
        defender_strategy = {
            "reasoning": def_strat.get("reasoning", ""),
            "strategy": def_strat.get("strategy", "Proactive hardening"),
            "defense_type": defense_type,
            "strength": defense_strength,
            "operations": def_llm_instructions.get("operations", []) if def_llm_instructions else []
        }

        # ----- GradCAM overlays (original, attacked) only – proactive uses 2 images -----
        hm_orig, overlay_orig = None, None
        heatmap_original = gradcam.generate(image_tensor, original_class)
        if heatmap_original is not None:
            hm_orig = f"pro_orig_{uuid.uuid4().hex[:8]}.png"
            orig_np = image_tensor[0].cpu().numpy().transpose(1,2,0)
            gradcam.save_heatmap(heatmap_original, os.path.join(STATIC_DIR, hm_orig),
                                 output_shape=orig_np.shape[:2], colored=True)
            overlay_orig = f"pro_overlay_orig_{uuid.uuid4().hex[:8]}.png"
            gradcam.save_overlay(orig_np, heatmap_original, os.path.join(STATIC_DIR, overlay_orig))

        hm_adv, overlay_adv = None, None
        heatmap_adv = gradcam.generate(adversarial_tensor, final_class)
        if heatmap_adv is not None:
            hm_adv = f"pro_adv_{uuid.uuid4().hex[:8]}.png"
            adv_np = adversarial_tensor[0].cpu().numpy().transpose(1,2,0)
            gradcam.save_heatmap(heatmap_adv, os.path.join(STATIC_DIR, hm_adv),
                                 output_shape=adv_np.shape[:2], colored=True)
            overlay_adv = f"pro_overlay_adv_{uuid.uuid4().hex[:8]}.png"
            gradcam.save_overlay(adv_np, heatmap_adv, os.path.join(STATIC_DIR, overlay_adv))

        # ----- LLM outcome analysis -----
        outcome_analysis = {}
        if llm_engine:
            outcome_analysis = llm_engine.generate_outcome_analysis({
                "attack_type": attack_type, "attack_strength": attack_strength,
                "defense_type": defense_type, "defense_strength": defense_strength,
                "original_class": original_name, "original_confidence": original_conf,
                "adversarial_class": final_name, "adversarial_confidence": final_conf,
                "defended_class": defended_name, "defended_confidence": defended_conf,
                "attack_success": attacker_wins,
                "defense_success": defender_wins,
                "battle_mode": "proactive"
            })

        # ----- Record battle (single‑winner) -----
        battle_record = {
            "image": filename,
            "attack_type": attack_type, "attack_strength": attack_strength,
            "defense_type": defense_type, "defense_strength": defense_strength,
            "original_class": original_name, "original_confidence": original_conf,
            "defended_class": defended_name,
            "defended_confidence": defended_conf,
            "adversarial_class": final_name, "adversarial_confidence": final_conf,
            "attack_success": attacker_wins,
            "attacker_won": attacker_wins,
            "defense_success": defender_wins,
            "is_draw": False,
            "mode": "proactive_ai_battle"
        }
        battle_memory.add_battle(battle_record)

        # ----- RL rewards (zero‑sum style) -----
        reward_att = 1.0 if attacker_wins else -0.5
        rl_tracker.update(state, (attack_type, attack_strength), reward_att, role="attacker")
        reward_def = 1.0 if defender_wins else -0.5
        rl_tracker.update(state, (defense_type, defense_strength), reward_def, role="defender")

        return jsonify({
            "defense_type": defense_type, "defense_strength": defense_strength,
            "attack_type": attack_type, "attack_strength": attack_strength,
            "original_class": original_name, "original_confidence": original_conf,
            "adversarial_class": final_name, "adversarial_confidence": final_conf,
            "defended_class": defended_name,           # provided for record
            "defended_confidence": defended_conf,     # provided for record
            "attacker_wins": attacker_wins, "defender_wins": defender_wins,
            "attacker_strategy": attacker_strategy,
            "defender_strategy": defender_strategy,
            "outcome_analysis": outcome_analysis,
            "overlay_original": overlay_orig,
            "overlay_adversarial": overlay_adv,
            "heatmap_original": hm_orig,
            "heatmap_adversarial": hm_adv
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ─── Proactive Multi-Algorithm AI Battle ─────────────────────────────────────

@app.route('/api/ai-battle-proactive-multi', methods=['POST'])
def ai_battle_proactive_multi():
    """Multi-algorithm proactive battle: 1 to N combinations."""
    import random
    data = request.json
    filename = data.get('image')
    
    # The LLM is instructed to pick between 2 and N strategies intelligently
    n_atk_max = len(ATTACK_TYPES)
    n_def_max = len(DEFENSE_TYPES)

    if not llm_engine:
        return jsonify({"error": "LLM not available"}), 400

    try:
        t_start = time.time()
        image_tensor = image_loader.load_image(filename)
        original_class, original_name, original_conf, _ = target_model.predict(image_tensor)
        state = rl_tracker.get_state(original_name, original_conf)

        image_info = {"original_class": original_name, "confidence": original_conf}
        recent = battle_memory.get_recent(5)
        atk_suggestion = rl_tracker.get_suggestion(state, "attacker")
        def_suggestion = rl_tracker.get_suggestion(state, "defender")
        atk_top = rl_tracker.get_top_actions(role="attacker", n=6)
        def_top = rl_tracker.get_top_actions(role="defender", n=6)

        def_multi = llm_engine.generate_multi_defense_strategies(image_info, recent, def_suggestion, def_top, n=n_def_max)
        defense_strategies = def_multi.get("strategies", [{"defense_type": "gaussian_blur", "strength": "medium"}])

        atk_multi = llm_engine.generate_multi_attack_strategies(image_info, recent, atk_suggestion, atk_top, n=n_atk_max)
        attack_strategies = atk_multi.get("strategies", [{"attack_type": "fgsm", "strength": "high"}])

        # 1. Attacker Alliance (Sequential Attacks)
        adv_tensor = image_tensor.clone()
        attack_ops = [{"type": f"Formed Alliance of {len(attack_strategies)} Attackers"}]
        used_attacks = []
        for atk_strat in attack_strategies:
            a_type = atk_strat.get("attack_type", "fgsm")
            a_str = atk_strat.get("strength", "medium")
            if a_type not in ATTACK_TYPES: a_type = "fgsm"
            if a_str not in STRENGTHS: a_str = "medium"

            adv_tensor, _ = attack_engine.generate_attack(adv_tensor, a_type, a_str, battle_context={"recent_battles": recent})
            attack_ops.append({"type": f"Executed Attack", "params": {"attack": a_type, "strength": a_str, "reasoning": atk_strat.get("reasoning", "")}})
            used_attacks.append((a_type, a_str))

        # Check attacker success before defense
        atk_only_class, atk_only_name, atk_only_conf, _ = target_model.predict(adv_tensor)

        # 2. Defender Alliance (Sequential Defenses)
        defended_tensor = adv_tensor.clone()
        defense_ops = [{"type": f"Formed Alliance of {len(defense_strategies)} Defenders"}]
        used_defenses = []
        for def_strat in defense_strategies:
            d_type = def_strat.get("defense_type", "gaussian_blur")
            d_str = def_strat.get("strength", "medium")
            if d_type not in DEFENSE_TYPES: d_type = "gaussian_blur"
            if d_str not in STRENGTHS: d_str = "medium"

            defended_tensor, _ = defense_engine.apply_defense(defended_tensor, d_type, d_str, battle_context={"recent_battles": recent})
            defense_ops.append({"type": f"Deployed Defense", "params": {"defense": d_type, "strength": d_str, "reasoning": def_strat.get("reasoning", "")}})
            used_defenses.append((d_type, d_str))

        # Final Outcome
        final_class, final_name, final_conf, _ = target_model.predict(defended_tensor)
        attacker_wins = final_class != original_class
        defender_wins = not attacker_wins

        overlay_orig = overlay_adv = None
        heatmap_original = gradcam.generate(image_tensor, original_class)
        if heatmap_original is not None:
            orig_np = image_tensor[0].cpu().numpy().transpose(1, 2, 0)
            fn = f"pm_overlay_orig_{uuid.uuid4().hex[:8]}.png"
            gradcam.save_overlay(orig_np, heatmap_original, os.path.join(STATIC_DIR, fn))
            overlay_orig = fn

        heatmap_adv = gradcam.generate(adv_tensor, atk_only_class)
        if heatmap_adv is not None:
            adv_np = adv_tensor[0].cpu().numpy().transpose(1, 2, 0)
            fn2 = f"pm_overlay_adv_{uuid.uuid4().hex[:8]}.png"
            gradcam.save_overlay(adv_np, heatmap_adv, os.path.join(STATIC_DIR, fn2))
            overlay_adv = fn2

        outcome_analysis = {}
        if llm_engine:
            outcome_analysis = llm_engine.generate_outcome_analysis({
                "attack_type": "Alliance", "attack_strength": "Multi",
                "defense_type": "Alliance", "defense_strength": "Multi",
                "original_class": original_name, "original_confidence": float(original_conf),
                "adversarial_class": final_name, "adversarial_confidence": float(final_conf),
                "defended_class": final_name, "defended_confidence": float(final_conf),
                "attack_success": attacker_wins, "defense_success": defender_wins,
                "battle_mode": "proactive_multi"
            })

        # Generate exact strings for history and UI
        atk_alliance_names = " -> ".join([a[0].upper() for a in used_attacks])
        def_alliance_names = " -> ".join([d[0].upper() for d in used_defenses])
        
        rep_a_type = f"STACK ({'+'.join([a[0].upper() for a in used_attacks])})" if len(used_attacks) > 1 else (used_attacks[0][0] if used_attacks else "fgsm")
        rep_a_str = "multi"
        rep_d_type = f"STACK ({'+'.join([d[0].upper() for d in used_defenses])})" if len(used_defenses) > 1 else (used_defenses[0][0] if used_defenses else "gaussian_blur")
        rep_d_str = "multi"

        battle_record = {
            "image": filename, "attack_type": rep_a_type, "attack_strength": rep_a_str,
            "defense_type": rep_d_type, "defense_strength": rep_d_str,
            "original_class": original_name, "original_confidence": float(original_conf),
            "adversarial_class": final_name, "adversarial_confidence": float(final_conf),
            "defended_class": final_name, "defended_confidence": float(final_conf),
            "attack_success": attacker_wins, "attacker_won": attacker_wins,
            "defense_success": defender_wins, "is_draw": False,
            "mode": "proactive_ai_battle", "multi_algo": True,
        }
        battle_memory.add_battle(battle_record)
        
        # RL Rewards for all alliance members
        for (a_t, a_s) in used_attacks:
            rl_tracker.update(state, (a_t, a_s), 1.0 if attacker_wins else -0.5, role="attacker")
        for (d_t, d_s) in used_defenses:
            rl_tracker.update(state, (d_t, d_s), 1.0 if defender_wins else -0.5, role="defender")

        elapsed = time.time() - t_start
        _battle_timings["proactive_multi"].append(elapsed)
        if len(_battle_timings["proactive_multi"]) > 20:
            _battle_timings["proactive_multi"].pop(0)

        return jsonify({
            "mode": "proactive_multi",
            "original_class": original_name, "original_confidence": float(original_conf),
            "adversarial_class": final_name, "adversarial_confidence": float(final_conf),
            "attacker_wins": attacker_wins, "defender_wins": defender_wins,
            "tested_attack_strategies": attack_strategies,
            "tested_defense_strategies": defense_strategies,
            "attacker_strategy": {
                "reasoning": f"Alliance: {atk_alliance_names}",
                "strategy": f"Multi-algo Proactive Attack: Sequentially applied {len(used_attacks)} attacks.",
                "why_chosen": atk_multi.get("overall_reasoning", "Formed an alliance to overwhelm defenses."),
                "operations": attack_ops
            },
            "defender_strategy": {
                "reasoning": f"Alliance: {def_alliance_names}",
                "strategy": f"Multi-algo Proactive Defense: Sequentially applied {len(used_defenses)} defenses.",
                "why_chosen": def_multi.get("overall_reasoning", "Formed an alliance to restore class stability."),
                "operations": defense_ops
            },
            "outcome_analysis": outcome_analysis,
            "overlay_original": overlay_orig,
            "overlay_adversarial": overlay_adv,
            "elapsed_seconds": round(elapsed, 1),
            "n_strategies_tested": len(used_attacks) + len(used_defenses),
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ─── Stats / History / Coach ─────────────────────────────────────────────────

@app.route('/api/battle-history', methods=['GET'])
def get_battle_history():
    return jsonify(battle_memory.get_all())


@app.route('/api/stats', methods=['GET'])
def get_stats():
    stats = stats_model.get_comprehensive_stats()
    basic = stats.get("basic", {})
    stats["total_battles"] = basic.get("total_battles", 0)
    stats["attacker_wins"] = basic.get("attacker_wins", 0)
    stats["defender_wins"] = basic.get("defender_wins", 0)
    stats["draws"] = basic.get("draws", 0)
    return jsonify(stats)


@app.route('/api/reset-scores', methods=['POST'])
def reset_scores():
    battle_memory.reset_scores()
    return jsonify({"message": "Scores reset", "stats": battle_memory.get_stats()})


@app.route('/api/reset-all', methods=['POST'])
def reset_all():
    """Wipe ALL persisted data: battle history, scores, Q-table, learning log."""
    battle_memory.clear()
    rl_tracker.reset()          # clears Q-table + learning log + round counter
    # Delete persistence files so they start fresh
    import glob
    for f in glob.glob("battle_memory.json") + glob.glob("rl_tracker.json") + glob.glob("learning_log.jsonl"):
        try:
            os.remove(f)
        except OSError:
            pass
    return jsonify({
        "message": "All data reset",
        "battles": 0,
        "rl_rounds": 0
    })


@app.route('/api/coach', methods=['GET'])
def get_coach():
    history = battle_memory.get_all()
    if len(history) < COACH_INTERVAL:
        return jsonify({"message": f"Complete at least {COACH_INTERVAL} battles for coaching."})
    if llm_engine is None:
        return jsonify({"message": "LLM not configured. Set GROQ_API_KEY in .env"})
    return jsonify({"coaching": llm_engine.coach(history)})


@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        "attack_types": ATTACK_TYPES,
        "defense_types": DEFENSE_TYPES,
        "strengths": STRENGTHS,
        "llm_available": llm_engine is not None
    })


@app.route('/api/help', methods=['GET'])
def get_help():
    return jsonify({
        "name": "AdverShield Battle System",
        "description": "AI-powered adversarial attack and defense testing platform",
        "endpoints": {}   # placeholder, can be filled later
    })


@app.route('/api/settings', methods=['GET'])
def get_settings():
    return jsonify(settings_model.get())


@app.route('/api/settings/<path:key>', methods=['GET'])
def get_setting(key):
    value = settings_model.get(key)
    if value is None:
        return jsonify({"error": f"Setting '{key}' not found"}), 404
    return jsonify({key: value})


@app.route('/api/settings', methods=['POST'])
def update_setting():
    data = request.json
    key = data.get('key')
    value = data.get('value')
    if not key or value is None:
        return jsonify({"error": "Missing 'key' or 'value'"}), 400
    settings_model.set(key, value)
    return jsonify({"message": "Setting updated", "key": key, "value": value})


@app.route('/api/settings/reset', methods=['POST'])
def reset_settings():
    settings_model.reset()
    return jsonify({"message": "Settings reset to defaults", "settings": settings_model.get()})


@app.route('/static/heatmaps/<path:filename>')
def serve_heatmap(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route('/api/learning-log', methods=['GET'])
def get_learning_log():
    curve = rl_tracker.get_learning_curve_data()
    return jsonify({
        "log": rl_tracker.get_learning_log(),
        "attacker_log": rl_tracker.get_attacker_log(),
        "defender_log": rl_tracker.get_defender_log(),
        "q_heatmap": rl_tracker.get_q_heatmap_data(),
        "curve": curve,
        "epsilon": rl_tracker.epsilon,
        "total_rounds": rl_tracker.round_number
    })


@app.route('/api/battle-eta', methods=['GET'])
def get_battle_eta():
    """Return rolling average battle durations per mode."""
    eta = {}
    for mode, times in _battle_timings.items():
        if times:
            eta[mode] = round(sum(times) / len(times), 1)
        else:
            eta[mode] = None
    return jsonify(eta)


@app.route('/api/matchup-matrix', methods=['GET'])
def get_matchup_matrix():
    battles = battle_memory.get_all()
    matrix = {}
    for atk in ATTACK_TYPES:
        matrix[atk] = {}
        for dfn in DEFENSE_TYPES:
            relevant = [b for b in battles if b.get("attack_type") == atk and b.get("defense_type") == dfn]
            if relevant:
                wins = sum(1 for b in relevant if b.get("attacker_won"))
                matrix[atk][dfn] = {
                    "attacker_win_rate": round(wins / len(relevant), 2),
                    "n": len(relevant)
                }
            else:
                matrix[atk][dfn] = {"attacker_win_rate": None, "n": 0}
    return jsonify(matrix)


@app.route('/')
def index():
    tmpl = os.path.join('templates', 'index.html')
    if os.path.exists(tmpl):
        return render_template('index.html')
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  AdverShield Battle System")
    print("  http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)