"""COCO val2017 görüntülerini ve annotations'ı indirir.

Toplam ~1 GB. Hızlı internet bağlantısında ~5-10 dk.
Daha önce indirilmişse atlar.
"""
from __future__ import annotations
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402

IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


MIN_BYTES = 1_000_000  # 1 MB altındaysa yarım kalmış say


def download(url: str, dst: Path) -> None:
    if dst.exists():
        size = dst.stat().st_size
        if size >= MIN_BYTES:
            print(f"[atlandı] {dst.name} zaten var ({size/1e6:.1f} MB)")
            return
        print(f"[uyarı] {dst.name} eksik görünüyor ({size} bayt) — siliniyor ve yeniden indirilecek")
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[indiriliyor] {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp.rename(dst)


def unzip(zip_path: Path, target_dir: Path, marker: Path) -> None:
    if marker.exists():
        print(f"[atlandı] {marker.name} zaten çıkarılmış")
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[açılıyor] {zip_path.name} → {target_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        for name in tqdm(zf.namelist()):
            zf.extract(name, target_dir)


def main() -> None:
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    images_zip = C.DATA_DIR / "val2017.zip"
    ann_zip = C.DATA_DIR / "annotations_trainval2017.zip"

    download(IMAGES_URL, images_zip)
    download(ANN_URL, ann_zip)

    # val2017.zip içinde "val2017/" klasörü var → onu coco_val2017'ye taşıyalım
    if not C.IMAGE_DIR.exists() or not any(C.IMAGE_DIR.iterdir()):
        print("[açılıyor] val2017 görüntüleri")
        with zipfile.ZipFile(images_zip) as zf:
            zf.extractall(C.DATA_DIR)
        extracted = C.DATA_DIR / "val2017"
        if extracted.exists():
            extracted.rename(C.IMAGE_DIR)

    if not C.ANN_PATH.exists():
        print("[açılıyor] annotations")
        with zipfile.ZipFile(ann_zip) as zf:
            zf.extractall(C.DATA_DIR)

    n_imgs = sum(1 for _ in C.IMAGE_DIR.glob("*.jpg"))
    print(f"\n[tamam] {n_imgs} görüntü, annotations: {C.ANN_PATH.exists()}")


if __name__ == "__main__":
    main()
