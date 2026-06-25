import os

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
SAMPLES_DIR = "./samples"
STATIC_DIR = "./static/heatmaps"
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ATTACK_TYPES = [
    "fgsm", "pgd", "cw", "ai",
    "bim", "deepfool", "salt_pepper", "contrast_shift",
    "jpeg_attack", "frequency_filter", "spatial_perturbation", "universal_perturbation"
]

DEFENSE_TYPES = [
    "gaussian_blur", "jpeg_compression", "feature_squeezing", "ai_defense",
    "median_blur", "bilateral_filter", "tv_denoising", "randomized_smoothing",
    "pixel_deflection", "quilting", "autoencoder_restoration", "diff_jpeg"
]

STRENGTHS = ["low", "medium", "high"]

EPSILON_MAP = {"low": 0.02, "medium": 0.06, "high": 0.15}
STEPS_MAP = {"low": 10, "medium": 20, "high": 40}
JPEG_QUALITY_MAP = {"low": 95, "medium": 85, "high": 65}
BLUR_KERNEL_MAP = {"low": 1, "medium": 3, "high": 5}
SQUEEZE_BITS_MAP = {"low": 7, "medium": 6, "high": 4}

MEMORY_SIZE = 30
COACH_INTERVAL = 5