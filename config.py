import os

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
SAMPLES_DIR = "./samples"
STATIC_DIR = "./static/heatmaps"
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ATTACK_TYPES = ["fgsm", "pgd", "cw", "ai"]
DEFENSE_TYPES = ["gaussian_blur", "jpeg_compression", "feature_squeezing", "ai_defense"]
STRENGTHS = ["low", "medium", "high"]

EPSILON_MAP = {"low": 0.01, "medium": 0.03, "high": 0.08}
STEPS_MAP = {"low": 10, "medium": 20, "high": 40}
JPEG_QUALITY_MAP = {"low": 90, "medium": 70, "high": 40}
BLUR_KERNEL_MAP = {"low": 3, "medium": 5, "high": 9}
SQUEEZE_BITS_MAP = {"low": 7, "medium": 5, "high": 3}

MEMORY_SIZE = 30
COACH_INTERVAL = 5