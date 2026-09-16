"""Ana deney: temiz / 4 bozulma / 4 iyileştirme = 9 senaryo × N görüntü.

Her görüntü için her senaryoda:
  - Faster R-CNN ile tahmin
  - Per-image: tespit sayısı, ortalama güven skoru
  - Toplu mAP için (condition, treatment) altında biriktirilir

Çıktılar:
  - results/metrics.csv            (condition, treatment başına mAP, mean_conf, n_detections)
  - results/per_image_metrics.csv  (görüntü başına mean_conf, n_detections)
  - results/comparison_grids/grid_<idx>.png (ilk N_GRID_SAMPLES için)
  - results/degraded_samples/, results/enhanced_samples/

Kullanım:
  python scripts/3_run_experiment.py            # tam (config.N_IMAGES)
  python scripts/3_run_experiment.py --n 5      # smoke test
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.data_loader import load_coco, load_subset, iter_images  # noqa: E402
from src.degradations import degrade, DEGRADATIONS  # noqa: E402
from src.enhancements import auto_enhance  # noqa: E402
from src.detector import FasterRCNNDetector  # noqa: E402
from src.metrics import GroupedMAP, mean_confidence, detection_count, filter_target  # noqa: E402
from src.visualize import draw_boxes, make_comparison_grid  # noqa: E402


CONDITIONS = list(DEGRADATIONS.keys())  # ["fog", "night", "rain", "blur"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None,
                   help="Override görüntü sayısı (smoke test için)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    coco = load_coco()
    ids = load_subset()
    if args.n is not None:
        ids = ids[: args.n]
    print(f"[bilgi] {len(ids)} görüntü ile çalışılacak")

    # Klasörler hazır olsun
    for d in (C.DEGRADED_SAMPLES_DIR, C.ENHANCED_SAMPLES_DIR,
              C.GRIDS_DIR, C.PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("[bilgi] Faster R-CNN yükleniyor (ilk seferde ~160 MB inecek)...")
    detector = FasterRCNNDetector(device="cpu")
    print("[bilgi] model hazır.")

    # (condition, treatment) → GroupedMAP
    grouped_maps: dict[tuple[str, str], GroupedMAP] = {
        ("clean", "none"): GroupedMAP(),
    }
    for cond in CONDITIONS:
        grouped_maps[(cond, "degraded")] = GroupedMAP()
        grouped_maps[(cond, "enhanced")] = GroupedMAP()

    per_image_records: list[dict] = []

    np.random.seed(C.RANDOM_SEED)  # rain stochastic — tekrarlanabilirlik
    t0 = time.time()

    for idx, (img_id, img, gt) in enumerate(tqdm(list(iter_images(coco, ids)), desc="görüntüler")):
        # Baseline
        pred = detector.predict(img)
        pred_t = filter_target(pred)
        grouped_maps[("clean", "none")].update(pred, gt)
        per_image_records.append({
            "image_id": img_id, "condition": "clean", "treatment": "none",
            "mean_conf": mean_confidence(pred_t),
            "n_detections": detection_count(pred_t),
        })
        clean_panel = draw_boxes(img, pred_t)

        grid_panels = [(f"CLEAN ({detection_count(pred_t)} tespit)", clean_panel)]

        for cond in CONDITIONS:
            deg_img = degrade(img, cond)
            deg_pred = detector.predict(deg_img)
            deg_pred_t = filter_target(deg_pred)
            grouped_maps[(cond, "degraded")].update(deg_pred, gt)
            per_image_records.append({
                "image_id": img_id, "condition": cond, "treatment": "degraded",
                "mean_conf": mean_confidence(deg_pred_t),
                "n_detections": detection_count(deg_pred_t),
            })

            enh_img = auto_enhance(deg_img, cond)
            enh_pred = detector.predict(enh_img)
            enh_pred_t = filter_target(enh_pred)
            grouped_maps[(cond, "enhanced")].update(enh_pred, gt)
            per_image_records.append({
                "image_id": img_id, "condition": cond, "treatment": "enhanced",
                "mean_conf": mean_confidence(enh_pred_t),
                "n_detections": detection_count(enh_pred_t),
            })

            # İlk N örnekte görsel kayıt
            if idx < C.N_GRID_SAMPLES:
                deg_panel = draw_boxes(deg_img, deg_pred_t, color=(0, 0, 220))
                enh_panel = draw_boxes(enh_img, enh_pred_t, color=(0, 200, 0))
                grid_panels.extend([
                    (f"{cond.upper()} ({detection_count(deg_pred_t)})", deg_panel),
                    (f"{cond.upper()}+enh ({detection_count(enh_pred_t)})", enh_panel),
                ])

                # Tekli örnek görüntü dosyaları
                cv2.imwrite(str(C.DEGRADED_SAMPLES_DIR / f"img{idx}_{cond}.jpg"), deg_img)
                cv2.imwrite(str(C.ENHANCED_SAMPLES_DIR / f"img{idx}_{cond}_enhanced.jpg"), enh_img)

        if idx < C.N_GRID_SAMPLES:
            make_comparison_grid(grid_panels, C.GRIDS_DIR / f"grid_{idx:02d}_id{img_id}.png")

    # mAP'leri topla
    print("\n[bilgi] mAP hesaplanıyor...")
    map_records: list[dict] = []
    for (cond, treatment), grouper in grouped_maps.items():
        m = grouper.compute()
        map_records.append({"condition": cond, "treatment": treatment, "mAP": m})
        print(f"  {cond:>6} | {treatment:>9} | mAP@0.5 = {m:.3f}")

    # Per-image mean_conf, n_det'leri condition+treatment bazında ortalayıp mAP ile birleştir
    df_per_image = pd.DataFrame(per_image_records)
    df_summary = df_per_image.groupby(["condition", "treatment"]).agg(
        mean_conf=("mean_conf", "mean"),
        n_detections=("n_detections", "mean"),
    ).reset_index()

    df_map = pd.DataFrame(map_records)
    df_final = df_summary.merge(df_map, on=["condition", "treatment"])

    # Ham (per-image) ve özet birlikte yazılır — özet ana CSV
    df_final.to_csv(C.METRICS_CSV, index=False)
    df_per_image.to_csv(C.RESULTS_DIR / "per_image_metrics.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n[tamam] {len(ids)} görüntü, {elapsed/60:.1f} dk")
    print(f"  Özet CSV: {C.METRICS_CSV}")
    print(f"  Karşılaştırma grid'leri: {C.GRIDS_DIR}")
    print(f"\nŞimdi: python scripts/4_make_report.py")


if __name__ == "__main__":
    main()
