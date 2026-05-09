import torch
import numpy as np
import cv2
from PIL import Image, ImageFilter
import io
import json
import re
from config import JPEG_QUALITY_MAP, BLUR_KERNEL_MAP, SQUEEZE_BITS_MAP


class DefenseEngine:
    def __init__(self, llm_engine=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm_engine = llm_engine   # <-- injected from app.py

    def apply_defense(self, image_tensor, defense_type, strength, battle_context=None):
        if defense_type == "gaussian_blur":
            return self._gaussian_blur(image_tensor, strength), None
        elif defense_type == "jpeg_compression":
            return self._jpeg_compression(image_tensor, strength), None
        elif defense_type == "feature_squeezing":
            return self._feature_squeezing(image_tensor, strength), None
        elif defense_type == "ai_defense" and self.llm_engine is not None:
            return self._llm_defense(image_tensor, strength, battle_context)
        else:
            return self._gaussian_blur(image_tensor, strength), None

    def _gaussian_blur(self, image_tensor, strength):
        kernel_size = BLUR_KERNEL_MAP.get(strength, 5)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        image_np = image_tensor.cpu().numpy()
        blurred = cv2.GaussianBlur(
            image_np[0].transpose(1, 2, 0),
            (kernel_size, kernel_size), 0
        )
        result = torch.from_numpy(blurred.transpose(2, 0, 1)).unsqueeze(0)
        return result.to(self.device).float()

    def _jpeg_compression(self, image_tensor, strength):
        quality = JPEG_QUALITY_MAP.get(strength, 70)
        image_np = (image_tensor.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        buffer = io.BytesIO()
        image_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert('RGB')
        result = torch.from_numpy(np.array(compressed)).permute(2, 0, 1).float() / 255.0
        return result.unsqueeze(0).to(self.device)

    def _feature_squeezing(self, image_tensor, strength):
        bits = SQUEEZE_BITS_MAP.get(strength, 5)
        levels = 2 ** bits
        squeezed = torch.floor(image_tensor * levels) / levels
        return squeezed

    # ----- AI Defense (LLM) -----
    def _llm_defense(self, image_tensor, strength, battle_context):
        prompt = self._build_llm_defense_prompt(strength, battle_context)
        try:
            raw = self.llm_engine.generate(prompt, system_prompt="""You are an adversarial ML defender.
Your job is to restore an image that has been adversarially attacked so that ResNet-50 correctly classifies it again.
You must specify exact pixel-level operations using available OpenCV/PIL tools.
Reply ONLY in valid JSON. No extra text.""")
            instructions = self._parse_json_safe(raw)
            if not instructions or "operations" not in instructions:
                return self._gaussian_blur(image_tensor, "medium"), None
            defended = self._execute_llm_operations(image_tensor, instructions["operations"], strength)
            return defended, instructions
        except Exception as e:
            print(f"AI Defense failed, falling back to Gaussian blur: {e}")
            return self._gaussian_blur(image_tensor, "medium"), None

    def _build_llm_defense_prompt(self, strength, battle_context):
        strength_guide = {"low": "subtle, nearly invisible", "medium": "moderate", "high": "aggressive"}
        history_str = "None"
        if battle_context and battle_context.get("recent_battles"):
            recent = battle_context["recent_battles"][-3:]
            history_str = json.dumps([{
                "attack": b.get("attack_type"),
                "defense": b.get("defense_type"),
                "success": not b.get("attacker_won")
            } for b in recent])

        return f"""You are defending an image against adversarial attacks for a ResNet-50 classifier.
Target: restore the original correct classification.
Strength: {strength} ({strength_guide.get(strength, 'moderate')})
Recent battle history: {history_str}

Available operations (use 2-4):
- gaussian_blur: params: {{"kernel_size": 3-9 (odd)}}
- median_blur: params: {{"kernel_size": 3-9 (odd)}}
- denoise_bilateral: params: {{"d": 5, "sigma_color": 50-100, "sigma_space": 50-100}}
- jpeg_compress: params: {{"quality": 50-90}}
- color_correction: params: {{"r_factor": 0.9-1.1, "g_factor": 0.9-1.1, "b_factor": 0.9-1.1}}
- unsharp_mask: params: {{"amount": 50-150}}
- contrast_enhance: params: {{"alpha": 0.8-1.2}}

Reply ONLY in JSON:
{{
  "operations": [
    {{"type": "operation_name", "params": {{...}}}},
    ...
  ],
  "reasoning": "Why these operations should restore the original classification",
  "strategy": "one word strategy name"
}}"""

    def _execute_llm_operations(self, image_tensor, operations, strength):
        img = image_tensor.clone().detach()
        img_np = img[0].cpu().numpy().transpose(1, 2, 0)  # HWC float32 [0,1]
        multiplier = {"low": 0.4, "medium": 1.0, "high": 1.8}.get(strength, 1.0)

        for op in operations:
            op_type = op.get("type", "")
            params = op.get("params", {})
            try:
                if op_type == "gaussian_blur":
                    k = int(params.get("kernel_size", 3))
                    if k % 2 == 0: k += 1
                    img_np = cv2.GaussianBlur(img_np, (k, k), 0)
                elif op_type == "median_blur":
                    k = int(params.get("kernel_size", 3))
                    if k % 2 == 0: k += 1
                    img_np = cv2.medianBlur((img_np * 255).astype(np.uint8), k).astype(np.float32) / 255.0
                elif op_type == "denoise_bilateral":
                    d = int(params.get("d", 5))
                    sc = float(params.get("sigma_color", 75)) * multiplier
                    ss = float(params.get("sigma_space", 75)) * multiplier
                    img_np = cv2.bilateralFilter(img_np, d, sc, ss)
                elif op_type == "jpeg_compress":
                    quality = int(params.get("quality", 70))
                    img_uint8 = (img_np * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_uint8)
                    buf = io.BytesIO()
                    pil_img.save(buf, format='JPEG', quality=quality)
                    buf.seek(0)
                    img_np = np.array(Image.open(buf).convert('RGB')).astype(np.float32) / 255.0
                elif op_type == "color_correction":
                    rf = float(params.get("r_factor", 1.0)) * multiplier
                    gf = float(params.get("g_factor", 1.0)) * multiplier
                    bf = float(params.get("b_factor", 1.0)) * multiplier
                    img_np[:,:,0] *= rf
                    img_np[:,:,1] *= gf
                    img_np[:,:,2] *= bf
                elif op_type == "unsharp_mask":
                    amount = float(params.get("amount", 100)) * multiplier
                    blur = cv2.GaussianBlur(img_np, (0,0), 3)
                    img_np = cv2.addWeighted(img_np, 1.0 + amount/100, blur, -amount/100, 0)
                elif op_type == "contrast_enhance":
                    alpha = float(params.get("alpha", 1.0)) * multiplier
                    img_np = np.clip((img_np - 0.5) * alpha + 0.5, 0, 1)
            except Exception as e:
                print(f"Defense operation {op_type} failed: {e}, skipping")
                continue

        img_np = np.clip(img_np, 0.0, 1.0).astype(np.float32)
        result = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return result

    def _parse_json_safe(self, text):
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group())
        except:
            return {}