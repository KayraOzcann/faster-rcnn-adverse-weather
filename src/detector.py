"""Faster R-CNN ResNet-50 FPN — torchvision pretrained, COCO sınıflarıyla."""
from __future__ import annotations
import numpy as np
import torch
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from . import config as C


class FasterRCNNDetector:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model.eval().to(self.device)
        self.weights = weights

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray, score_thresh: float = C.CONFIDENCE_THRESHOLD) -> dict:
        """Tek görüntü için tahmin. Dönen dict torchmetrics formatına yakın:
        {boxes: (N,4) xyxy, scores: (N,), labels: (N,)} — eşik altı filtrelendi.
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)
        out = self.model([tensor])[0]

        boxes = out["boxes"].cpu()
        scores = out["scores"].cpu()
        labels = out["labels"].cpu()

        keep = scores >= score_thresh
        return {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
        }
