import torch
import numpy as np
import cv2
import json
import re
import io
from PIL import Image, ImageFilter
import torch.nn as nn
from config import DEVICE, EPSILON_MAP, STEPS_MAP

try:
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool as ARTDeepFool
    HAS_ART = True
except ImportError:
    HAS_ART = False


class AttackEngine:
    def __init__(self, target_model, llm_engine=None):
        self.target_model = target_model
        self.device = DEVICE
        self.llm_engine = llm_engine

        if HAS_ART:
            self.classifier = PyTorchClassifier(
                model=target_model.model,
                loss=nn.CrossEntropyLoss(),
                input_shape=(3, 224, 224),
                nb_classes=1000,
                clip_values=(0.0, 1.0),
                device_type='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.classifier = None

    def generate_attack(self, image_tensor, attack_type, strength, target_class=None, battle_context=None):
        if attack_type == "ai" and self.llm_engine is not None:
            return self._llm_attack(image_tensor, strength, battle_context)
        elif attack_type == "fgsm":
            return self._fgsm(image_tensor, strength), None
        elif attack_type == "pgd":
            return self._pgd(image_tensor, strength), None
        elif attack_type == "cw":
            return self._cw(image_tensor, strength), None
        elif attack_type == "bim":
            return self._bim(image_tensor, strength), None
        elif attack_type == "deepfool":
            return self._deepfool(image_tensor, strength), None
        elif attack_type == "salt_pepper":
            return self._salt_pepper(image_tensor, strength), None
        elif attack_type == "contrast_shift":
            return self._contrast_shift(image_tensor, strength), None
        elif attack_type == "jpeg_attack":
            return self._jpeg_attack(image_tensor, strength), None
        elif attack_type == "frequency_filter":
            return self._frequency_filter(image_tensor, strength), None
        elif attack_type == "spatial_perturbation":
            return self._spatial_perturbation(image_tensor, strength), None
        elif attack_type == "universal_perturbation":
            return self._universal_perturbation(image_tensor, strength), None
        else:
            return self._fgsm(image_tensor, strength), None

    # --------------- LLM Attack ---------------
    def _llm_attack(self, image_tensor, strength, battle_context):
        prompt = self._build_llm_attack_prompt(image_tensor, strength, battle_context)
        try:
            raw = self.llm_engine.generate(prompt, system_prompt="""You are an adversarial ML attacker.
Your job is to corrupt an image to fool ResNet-50 into misclassifying it.
You must specify exact pixel-level operations using available OpenCV/PIL tools.
Reply ONLY in valid JSON. No extra text.""")
            instructions = self._parse_json_safe(raw)
            if not instructions or "operations" not in instructions:
                return self._fgsm(image_tensor, strength), None
            adversarial = self._execute_llm_operations(image_tensor, instructions["operations"], strength)
            return adversarial, instructions
        except Exception as e:
            print(f"LLM attack failed, falling back to FGSM: {e}")
            return self._fgsm(image_tensor, strength), None

    def _build_llm_attack_prompt(self, image_tensor, strength, battle_context):
        strength_guide = {"low": "subtle, nearly invisible", "medium": "moderate, slight texture changes", "high": "aggressive, noticeable noise"}
        history_str = "None"
        if battle_context and battle_context.get("recent_battles"):
            recent = battle_context["recent_battles"][-3:]
            history_str = json.dumps([{
                "attack": b.get("attack_type"),
                "success": b.get("attacker_won"),
                "defense": b.get("defense_type")
            } for b in recent])

        return f"""You are attacking a ResNet-50 image classifier.
Target: fool it into misclassifying the image.
Strength: {strength} ({strength_guide.get(strength, 'moderate')})
Recent battle history: {history_str}

Available operations (choose 2-4 that work together):
- gaussian_noise: adds random pixel noise. params: {{"sigma": 0.01-0.15}}
- salt_pepper: adds random black/white pixels. params: {{"density": 0.01-0.1}}
- edge_enhance: sharpens edges to confuse texture detection. params: {{"factor": 1.5-4.0}}
- hue_shift: shifts color hue. params: {{"degrees": 10-60}}
- brightness_patch: darkens/brightens specific region. params: {{"x": 0-1, "y": 0-1, "w": 0.1-0.5, "h": 0.1-0.5, "delta": -0.3-0.3}}
- frequency_noise: adds structured high-frequency noise. params: {{"amplitude": 0.02-0.1, "frequency": 4-16}}
- checkerboard: overlays subtle checkerboard pattern. params: {{"size": 4-16, "alpha": 0.05-0.2}}
- color_jitter: shifts RGB channels independently. params: {{"r_delta": -0.1-0.1, "g_delta": -0.1-0.1, "b_delta": -0.1-0.1}}

Reply with this exact JSON:
{{
  "operations": [
    {{"type": "operation_name", "params": {{...}}}},
    ...
  ],
  "reasoning": "Why these operations should fool ResNet-50 for this image",
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
                if op_type == "gaussian_noise":
                    sigma = float(params.get("sigma", 0.03)) * multiplier
                    noise = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
                    img_np = img_np + noise

                elif op_type == "salt_pepper":
                    density = float(params.get("density", 0.02)) * multiplier
                    mask = np.random.random(img_np.shape[:2])
                    img_np[mask < density/2] = 0.0
                    img_np[mask > 1 - density/2] = 1.0

                elif op_type == "edge_enhance":
                    factor = float(params.get("factor", 2.0)) * multiplier
                    img_uint8 = np.uint8(np.clip(img_np, 0, 1) * 255)
                    pil_img = Image.fromarray(img_uint8)
                    enhanced = ImageFilter.UnsharpMask(radius=2, percent=int(factor*100), threshold=3)
                    img_np = np.array(pil_img.filter(enhanced)).astype(np.float32) / 255.0

                elif op_type == "hue_shift":
                    degrees = float(params.get("degrees", 20)) * multiplier
                    img_uint8 = np.uint8(np.clip(img_np, 0, 1) * 255)
                    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
                    hsv[:, :, 0] = (hsv[:, :, 0] + degrees) % 180
                    img_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

                elif op_type == "brightness_patch":
                    h, w = img_np.shape[:2]
                    x = int(float(params.get("x", 0.25)) * w)
                    y = int(float(params.get("y", 0.25)) * h)
                    pw = int(float(params.get("w", 0.3)) * w)
                    ph = int(float(params.get("h", 0.3)) * h)
                    delta = float(params.get("delta", 0.1)) * multiplier
                    img_np[y:y+ph, x:x+pw] = np.clip(img_np[y:y+ph, x:x+pw] + delta, 0, 1)

                elif op_type == "frequency_noise":
                    amplitude = float(params.get("amplitude", 0.04)) * multiplier
                    frequency = int(params.get("frequency", 8))
                    h, w = img_np.shape[:2]
                    x_grid = np.linspace(0, frequency * np.pi, w)
                    y_grid = np.linspace(0, frequency * np.pi, h)
                    xx, yy = np.meshgrid(x_grid, y_grid)
                    pattern = amplitude * np.sin(xx + yy)
                    img_np = img_np + pattern[:, :, np.newaxis]

                elif op_type == "checkerboard":
                    size = int(params.get("size", 8))
                    alpha = float(params.get("alpha", 0.1)) * multiplier
                    h, w = img_np.shape[:2]
                    checker = np.zeros((h, w), dtype=np.float32)
                    for i in range(h):
                        for j in range(w):
                            if (i // size + j // size) % 2 == 0:
                                checker[i, j] = alpha
                    img_np = img_np + checker[:, :, np.newaxis]

                elif op_type == "color_jitter":
                    r_delta = float(params.get("r_delta", 0.05)) * multiplier
                    g_delta = float(params.get("g_delta", -0.05)) * multiplier
                    b_delta = float(params.get("b_delta", 0.05)) * multiplier
                    img_np[:, :, 0] += r_delta
                    img_np[:, :, 1] += g_delta
                    img_np[:, :, 2] += b_delta

            except Exception as e:
                print(f"Operation {op_type} failed: {e}, skipping")
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

    # --------------- Standard Attacks ---------------
    def _fgsm(self, image_tensor, strength):
        if not HAS_ART or self.classifier is None:
            return self._simple_noise(image_tensor, strength)
        epsilon = EPSILON_MAP.get(strength, 0.03)
        attack = FastGradientMethod(estimator=self.classifier, eps=epsilon)
        image_np = image_tensor.cpu().numpy()
        adversarial = np.clip(attack.generate(x=image_np), 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(adversarial).to(self.device)

    def _pgd(self, image_tensor, strength):
        if not HAS_ART or self.classifier is None:
            return self._simple_noise(image_tensor, strength)
        epsilon = EPSILON_MAP.get(strength, 0.03)
        steps = STEPS_MAP.get(strength, 20)
        attack = ProjectedGradientDescent(estimator=self.classifier, eps=epsilon, max_iter=steps, verbose=False)
        image_np = image_tensor.cpu().numpy()
        adversarial = np.clip(attack.generate(x=image_np), 0.0, 1.0).astype(np.float32)
        return torch.from_numpy(adversarial).to(self.device)

    def _cw(self, image_tensor, strength):
        epsilon = EPSILON_MAP.get(strength, 0.03)
        img = image_tensor.clone().detach().to(self.device)
        with torch.no_grad():
            logits = self.target_model.model(img)
            orig_class = logits.argmax(dim=1).item()
            top2 = logits.topk(2, dim=1).indices[0]
            target_class = top2[1].item() if top2[0].item() == orig_class else top2[0].item()

        delta = torch.zeros_like(img, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=0.01)

        for _ in range(100):
            optimizer.zero_grad()
            perturbed = torch.clamp(img + delta, 0, 1)
            logits = self.target_model.model(perturbed)
            loss = -logits[0, target_class] + logits[0, orig_class]
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)

        adversarial = torch.clamp(img + delta.detach(), 0, 1)
        return adversarial

    # ---------- New Attacks ----------
    def _bim(self, image_tensor, strength):
        epsilon = EPSILON_MAP.get(strength, 0.03)
        steps = STEPS_MAP.get(strength, 20)
        alpha = epsilon / steps * 2
        img = image_tensor.clone().detach().to(self.device)
        for _ in range(steps):
            img.requires_grad_(True)
            with torch.enable_grad():
                logits = self.target_model.model(img)
            self.target_model.model.zero_grad()
            loss = -logits[0, logits.argmax()]
            loss.backward()
            grad = img.grad.data.sign()
            img = img.detach() + alpha * grad
            img = torch.min(torch.max(img, image_tensor - epsilon), image_tensor + epsilon)
            img = torch.clamp(img, 0, 1)
        return img

    def _deepfool(self, image_tensor, strength):
        if HAS_ART:
            attack = ARTDeepFool(classifier=self.classifier, max_iter=STEPS_MAP.get(strength, 20), epsilon=EPSILON_MAP.get(strength, 0.03))
            adv_np = attack.generate(x=image_tensor.cpu().numpy())
            return torch.from_numpy(np.clip(adv_np, 0, 1)).to(self.device)
        return self._fgsm(image_tensor, strength)

    def _salt_pepper(self, image_tensor, strength):
        density = {"low": 0.01, "medium": 0.03, "high": 0.08}.get(strength, 0.03)
        img = image_tensor.clone().cpu().numpy()[0].transpose(1,2,0)
        mask = np.random.random(img.shape[:2])
        img[mask < density/2] = 0.0
        img[mask > 1 - density/2] = 1.0
        return torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(self.device)

    def _contrast_shift(self, image_tensor, strength):
        factor = {"low": 0.9, "medium": 0.7, "high": 0.5}[strength]
        img = image_tensor.clone().cpu().numpy()[0].transpose(1,2,0)
        img = (img - 0.5) * factor + 0.5
        img = np.clip(img, 0, 1)
        return torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(self.device)

    def _jpeg_attack(self, image_tensor, strength):
        quality = {"low": 50, "medium": 30, "high": 10}[strength]
        img_np = (image_tensor.cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        jpeg_img = np.array(Image.open(buf).convert('RGB')).astype(np.float32)/255.0
        return torch.from_numpy(jpeg_img.transpose(2,0,1)).unsqueeze(0).to(self.device)

    def _frequency_filter(self, image_tensor, strength):
        cutoff = {"low": 20, "medium": 10, "high": 5}[strength]
        img = image_tensor.clone().cpu().numpy()[0].transpose(1,2,0)
        f = np.fft.fftshift(np.fft.fft2(img, axes=(0,1)), axes=(0,1))
        h,w = img.shape[:2]
        cy,cx = h//2, w//2
        mask = np.ones((h,w))
        for y in range(h):
            for x in range(w):
                if np.sqrt((y-cy)**2 + (x-cx)**2) < cutoff:
                    mask[y,x] = 0
        f *= mask[..., None]
        img = np.real(np.fft.ifft2(np.fft.ifftshift(f, axes=(0,1)), axes=(0,1)))
        img = np.clip(img, 0, 1)
        return torch.from_numpy(img.transpose(2,0,1)).unsqueeze(0).to(self.device)

    def _spatial_perturbation(self, image_tensor, strength):
        degrees = {"low": 2, "medium": 5, "high": 10}[strength]
        img_np = (image_tensor.cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
        rows,cols = img_np.shape[:2]
        angle = np.random.uniform(-degrees, degrees)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        img_rot = cv2.warpAffine(img_np, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
        return torch.from_numpy(img_rot.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(self.device)

    def _universal_perturbation(self, image_tensor, strength):
        epsilon = EPSILON_MAP.get(strength, 0.03)
        seed = hash(str(image_tensor.shape))
        rng = np.random.default_rng(seed)
        perturbation = rng.normal(0, epsilon, image_tensor.shape).astype(np.float32)
        return torch.clamp(image_tensor + torch.from_numpy(perturbation).to(self.device), 0, 1)

    def _simple_noise(self, image_tensor, strength):
        epsilon = EPSILON_MAP.get(strength, 0.03)
        noise = torch.randn_like(image_tensor) * epsilon
        return torch.clamp(image_tensor + noise, 0, 1)