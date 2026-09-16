"""COCO val2017 alt kümesini yükleme ve ground-truth çıkarma."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from pycocotools.coco import COCO

from . import config as C


def load_coco() -> COCO:
    if not C.ANN_PATH.exists():
        raise FileNotFoundError(
            f"Annotation dosyası yok: {C.ANN_PATH}. Önce scripts/1_download_data.py çalıştır."
        )
    return COCO(str(C.ANN_PATH))


def select_subset_ids(coco: COCO, n: int = C.N_IMAGES, seed: int = C.RANDOM_SEED) -> list[int]:
    """Trafik sahnelerini seç:
    - TRAFFIC_REQUIRED_IDS (araç/işaret) sınıflarından **en az 1 nesne** olmalı,
    - TARGET_CLASS_IDS toplamından (person dahil) **en az 2 nesne** olmalı.

    Bu kombinasyon selfie/spor/yemek sahnelerini eler, gerçek sokak/trafik
    görüntülerini bırakır.
    """
    # Trafik sınıflarından en az birini içeren görüntüler
    candidate_ids = set()
    for cat_id in C.TRAFFIC_REQUIRED_IDS:
        candidate_ids.update(coco.getImgIds(catIds=[cat_id]))

    rich = []
    for img_id in candidate_ids:
        traffic_anns = coco.getAnnIds(
            imgIds=[img_id], catIds=list(C.TRAFFIC_REQUIRED_IDS), iscrowd=False
        )
        target_anns = coco.getAnnIds(
            imgIds=[img_id], catIds=list(C.TARGET_CLASS_IDS), iscrowd=False
        )
        if len(traffic_anns) >= 1 and len(target_anns) >= 2:
            rich.append(img_id)

    rng = np.random.default_rng(seed)
    rich = sorted(rich)
    rng.shuffle(rich)
    return rich[:n]


def save_subset(ids: list[int]) -> None:
    C.SUBSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    C.SUBSET_FILE.write_text(json.dumps(ids), encoding="utf-8")


def load_subset() -> list[int]:
    if not C.SUBSET_FILE.exists():
        raise FileNotFoundError(
            f"{C.SUBSET_FILE} bulunamadı. Önce scripts/2_select_subset.py çalıştır."
        )
    return json.loads(C.SUBSET_FILE.read_text(encoding="utf-8"))


def iter_images(coco: COCO, image_ids: list[int]) -> Iterator[tuple[int, np.ndarray, dict]]:
    """(image_id, BGR görüntü, ground-truth dict) verir.

    GT dict: {boxes: (N,4) xyxy float, labels: (N,) int} — sadece TARGET sınıflar.
    """
    for img_id in image_ids:
        info = coco.loadImgs([img_id])[0]
        path = C.IMAGE_DIR / info["file_name"]
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue

        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=list(C.TARGET_CLASS_IDS), iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w < 1 or h < 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])

        gt = {
            "boxes": np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
        }
        yield img_id, img, gt
