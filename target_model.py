import torch
import torch.nn as nn
import torchvision.models as models

from config import DEVICE, IMAGENET_MEAN, IMAGENET_STD


class NormalizeWrapper(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.resnet(x)


class TargetModel:
    def __init__(self):
        self.device = DEVICE
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        resnet.eval()

        self.model = NormalizeWrapper(resnet).to(self.device)
        self.model.eval()
        self.confidence_temperature = 0.7

        self.categories = weights.meta["categories"]

    def _predict_from_logits(self, logits):
        calibrated_logits = logits / self.confidence_temperature
        probs = torch.softmax(calibrated_logits, dim=1)
        top1_prob, top1_idx = probs.max(1)
        top5_probs, top5_idx = probs.topk(5, dim=1)

        top1_class_name = self.categories[top1_idx.item()]
        top5_list = [
            {"class": self.categories[idx.item()], "prob": prob.item()}
            for idx, prob in zip(top5_idx[0], top5_probs[0])
        ]

        return top1_idx.item(), top1_class_name, top1_prob.item(), top5_list

    def predict(self, image_tensor):
        with torch.no_grad():
            logits = self.model(image_tensor.to(self.device))
            return self._predict_from_logits(logits)

    def predict_with_grad(self, image_tensor):
        logits = self.model(image_tensor.to(self.device))
        return self._predict_from_logits(logits)

    def get_last_conv(self):
        return self.model.resnet.layer4[-1].conv3
