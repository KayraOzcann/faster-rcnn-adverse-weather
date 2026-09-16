"""mAP@0.5, ortalama güven, tespit sayısı.

Güven ve tespit sayısı görüntü başına; mAP ise her (condition, treatment) için
tüm görüntüler biriktirilip tek seferde hesaplanır.
"""
from __future__ import annotations
import torch
from torchmetrics.detection import MeanAveragePrecision

from . import config as C


def filter_target(pred: dict) -> dict:
    """Sadece TARGET_CLASS_IDS içindeki tahminleri tut."""
    mask = torch.tensor([int(l) in C.TARGET_CLASS_IDS for l in pred["labels"]], dtype=torch.bool)
    return {
        "boxes": pred["boxes"][mask],
        "scores": pred["scores"][mask],
        "labels": pred["labels"][mask],
    }


def gt_to_tensors(gt: dict) -> dict:
    return {
        "boxes": torch.from_numpy(gt["boxes"]),
        "labels": torch.from_numpy(gt["labels"]),
    }


def mean_confidence(pred: dict) -> float:
    s = pred["scores"]
    return float(s.mean()) if len(s) > 0 else 0.0


def detection_count(pred: dict) -> int:
    return int(len(pred["scores"]))


class GroupedMAP:
    """Bir (condition, treatment) çiftini biriktirip tek seferde mAP hesaplar.

    Bütün görüntüler bittikten sonra .compute() çağırılır.
    """

    def __init__(self):
        self.metric = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox",
            iou_thresholds=[C.IOU_THRESHOLD],
        )

    def update(self, pred: dict, gt: dict) -> None:
        self.metric.update([filter_target(pred)], [gt_to_tensors(gt)])

    def compute(self) -> float:
        try:
            res = self.metric.compute()
            return float(res["map"])
        except Exception:
            return float("nan")
