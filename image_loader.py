import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from config import SAMPLES_DIR, IMAGE_SIZE
import urllib.request
import ssl


class ImageLoader:
    def __init__(self, samples_dir=SAMPLES_DIR):
        self.samples_dir = samples_dir
        os.makedirs(samples_dir, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize(232),
            transforms.CenterCrop(IMAGE_SIZE[0]),
            transforms.ToTensor()
        ])

        self.default_images = [
            ("dog.jpg", "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"),
            ("cat.jpg", "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"),
        ]

    def list_available_images(self):
        images = [f for f in os.listdir(self.samples_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if not images:
            self._download_defaults()
            images = [f for f in os.listdir(self.samples_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        return images

    def _download_defaults(self):
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        for filename, url in self.default_images:
            filepath = os.path.join(self.samples_dir, filename)
            if not os.path.exists(filepath):
                try:
                    with urllib.request.urlopen(url, context=ssl_context) as response, open(filepath, "wb") as output:
                        output.write(response.read())
                except Exception as e:
                    print(f"Failed to download {filename}: {e}")

    def load_image(self, filename):
        filepath = os.path.join(self.samples_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image not found: {filepath}")

        image = Image.open(filepath).convert('RGB')
        tensor = self.transform(image)
        return tensor.unsqueeze(0)

    def get_image_path(self, filename):
        return os.path.join(self.samples_dir, filename)
