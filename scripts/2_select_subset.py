"""COCO val2017'den otonom-ilgili sınıflar içeren ~100 görüntülük subset seçer.

Çıktı: data/subset_ids.json
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.data_loader import load_coco, select_subset_ids, save_subset  # noqa: E402


def main() -> None:
    coco = load_coco()
    ids = select_subset_ids(coco, n=C.N_IMAGES, seed=C.RANDOM_SEED)
    save_subset(ids)
    print(f"[tamam] {len(ids)} görüntü seçildi → {C.SUBSET_FILE}")
    print(f"  Örnek id'ler: {ids[:5]}...")


if __name__ == "__main__":
    main()
