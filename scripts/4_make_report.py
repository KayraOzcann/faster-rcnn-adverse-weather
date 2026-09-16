"""results/metrics.csv'den grafikleri ve kısa metin özetini üretir."""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C  # noqa: E402
from src.visualize import (
    bar_chart_per_condition,
    confidence_distribution_plot,
    detection_count_plot,
    per_image_scatter,
)  # noqa: E402


def main() -> None:
    if not C.METRICS_CSV.exists():
        raise SystemExit(f"{C.METRICS_CSV} yok. Önce scripts/3_run_experiment.py çalıştır.")

    df = pd.read_csv(C.METRICS_CSV)
    print("=== Özet Tablo ===")
    print(df.to_string(index=False))

    bar_chart_per_condition(df, C.PLOTS_DIR / "map_per_condition.png")
    print(f"\n[grafik] {C.PLOTS_DIR / 'map_per_condition.png'}")

    detection_count_plot(df, C.PLOTS_DIR / "detection_count.png")
    print(f"[grafik] {C.PLOTS_DIR / 'detection_count.png'}")

    per_image_path = C.RESULTS_DIR / "per_image_metrics.csv"
    if per_image_path.exists():
        per_image = pd.read_csv(per_image_path)
        confidence_distribution_plot(per_image, C.PLOTS_DIR / "confidence_drop.png")
        print(f"[grafik] {C.PLOTS_DIR / 'confidence_drop.png'}")

        per_image_scatter(per_image, C.PLOTS_DIR / "scatter_n_detections.png",
                          metric="n_detections")
        print(f"[grafik] {C.PLOTS_DIR / 'scatter_n_detections.png'}")
        per_image_scatter(per_image, C.PLOTS_DIR / "scatter_mean_conf.png",
                          metric="mean_conf")
        print(f"[grafik] {C.PLOTS_DIR / 'scatter_mean_conf.png'}")

    # Kısa sayısal özet
    summary_lines = ["# Sayısal Özet\n"]
    clean_map = df.query("condition == 'clean'")["mAP"].values
    base = float(clean_map[0]) if len(clean_map) else float("nan")
    summary_lines.append(f"- Temiz görüntülerde baseline mAP@0.5: **{base:.3f}**\n")

    for cond in ["fog", "night", "rain", "blur"]:
        sub = df.query("condition == @cond")
        deg = sub.query("treatment == 'degraded'")["mAP"]
        enh = sub.query("treatment == 'enhanced'")["mAP"]
        if len(deg) and len(enh):
            d = float(deg.iloc[0])
            e = float(enh.iloc[0])
            drop = base - d
            recover = e - d
            summary_lines.append(
                f"- **{cond.upper()}**: bozuk mAP={d:.3f} (Δ={-drop:+.3f}), "
                f"iyileştirilmiş mAP={e:.3f} (geri kazanım={recover:+.3f})"
            )

    summary_path = C.RESULTS_DIR / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\n[özet] {summary_path}")


if __name__ == "__main__":
    main()
