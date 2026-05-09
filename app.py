from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import json
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

# Inject LLM engine into attack and defense engines
if llm_engine is not None:
    attack_engine.llm_engine = llm_engine

defense_engine = DefenseEngine(llm_engine=llm_engine)   # AI defender ready

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

        # GradCAM for original
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

        # Save adversarial image
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
            "attacker_won": attack_success,
            "is_draw": False,
            "mode": "attack",
            "attack_llm_response": llm_response
        }
        battle_memory.add_battle(battle_record)

        reward = 1.0 if attack_success else -0.5
        rl_tracker.update(state, (attack_type, strength), reward)

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

        # Apply defense – now returns (defended_tensor, defense_llm_instructions)
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

        defense_success = defended_class == original_class

        # Outcome analysis
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
                "attacker_won": not defense_success,
                "defense_success": defense_success
            })

        # GradCAM
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

        # Save base images for pipeline UI
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
            "defense_success": defense_success,
            "attacker_won": not defense_success,
            "is_draw": False,
            "mode": "defend"
        }
        battle_memory.add_battle(battle_record)

        reward = 1.0 if defense_success else -0.5
        rl_tracker.update(state, (defense_type, defense_strength), reward)

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
            "defense_success": defense_success,
            "llm_strategy": llm_response,                       # attacker LLM suggestion
            "defense_llm_strategy": defense_llm_instructions,   # defender LLM reasoning
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
    battle_memory.clear()
    return jsonify({"message": "All history and scores cleared"})


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


# ─── Help ────────────────────────────────────────────────────────────────────

@app.route('/api/help', methods=['GET'])
def get_help():
    return jsonify({
        "name": "AdverShield Battle System",
        "description": "AI-powered adversarial attack and defense testing platform",
        "endpoints": {
            "images": {
                "GET /api/images": "List all available images in samples folder",
                "GET /api/random-image": "Get a random image filename",
                "POST /api/upload-image": "Upload an image file (multipart/form-data, field: 'file')"
            },
            "classification": {
                "POST /api/classify": "Classify an image. Body: {'image': 'filename'}. Returns: class, confidence, top5, heatmap, overlay"
            },
            "attack": {
                "POST /api/attack": "Run an adversarial attack. Body: {'image': 'filename', 'attack_type': 'fgsm|pgd|cw|ai', 'strength': 'low|medium|high', 'target_class': 'optional'}"
            },
            "defense": {
                "POST /api/defend": "Test a defense against attack. Body: {'image': 'filename', 'attack_type': 'fgsm|pgd|cw|ai', 'attack_strength': 'low|medium|high', 'defense_type': 'gaussian_blur|jpeg_compression|feature_squeezing|ai_defense', 'defense_strength': 'low|medium|high'}"
            },
            "stats": {
                "GET /api/stats": "Get comprehensive battle statistics",
                "GET /api/battle-history": "Get all battle records",
                "POST /api/reset-scores": "Reset win/loss counters",
                "POST /api/reset-all": "Clear all history and scores"
            },
            "coach": {
                "GET /api/coach": "Get AI coaching based on battle history (requires 5+ battles)"
            },
            "settings": {
                "GET /api/settings": "Get all settings",
                "GET /api/settings/<key>": "Get a specific setting",
                "POST /api/settings": "Update settings. Body: {'key': 'setting.key', 'value': 'new_value'}",
                "POST /api/settings/reset": "Reset all settings to defaults"
            },
            "config": {
                "GET /api/config": "Get available attack types, defense types, strengths"
            }
        },
        "gradcam": {
            "description": "GradCAM visualizes which image regions the model focuses on for classification",
            "usage": "Automatically generated during classify/attack/defend endpoints",
            "files": {
                "heatmap": "Raw heatmap showing activation intensity",
                "overlay": "Heatmap overlaid on original image (alpha=0.5)"
            }
        }
    })


# ─── Settings ────────────────────────────────────────────────────────────────

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


# ─── Static / Heatmaps ───────────────────────────────────────────────────────

@app.route('/static/heatmaps/<path:filename>')
def serve_heatmap(filename):
    return send_from_directory(STATIC_DIR, filename)


# ─── AI Learning & Matchup ───────────────────────────────────────────────────

@app.route('/api/learning-log', methods=['GET'])
def get_learning_log():
    return jsonify({
        "log": rl_tracker.get_learning_log(),
        "q_heatmap": rl_tracker.get_q_heatmap_data(),
        "curve": rl_tracker.get_learning_curve_data(),
        "epsilon": rl_tracker.epsilon,
        "total_rounds": rl_tracker.round_number
    })

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


# ─── Frontend ────────────────────────────────────────────────────────────────

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