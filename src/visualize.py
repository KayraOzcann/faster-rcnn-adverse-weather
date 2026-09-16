"""Görsel çıktılar: bbox çizimi, karşılaştırma grid'i, bar/line/scatter grafikleri."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config as C


def draw_boxes(img: np.ndarray, pred: dict, color=(0, 200, 0)) -> np.ndarray:
    out = img.copy()
    boxes = pred["boxes"].numpy() if hasattr(pred["boxes"], "numpy") else pred["boxes"]
    scores = pred["scores"].numpy() if hasattr(pred["scores"], "numpy") else pred["scores"]
    labels = pred["labels"].numpy() if hasattr(pred["labels"], "numpy") else pred["labels"]
    for box, score, lbl in zip(boxes, scores, labels):
        if int(lbl) not in C.TARGET_CLASS_IDS:
            continue
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        name = C.TARGET_CLASSES.get(int(lbl), str(int(lbl)))
        text = f"{name} {float(score):.2f}"
        cv2.putText(out, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    return out


def make_comparison_grid(panels: list[tuple[str, np.ndarray]], save_path: Path) -> None:
    """panels: [(başlık, BGR görüntü), ...] — yatay yan yana, başlıklı."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def bar_chart_per_condition(df: pd.DataFrame, save_path: Path) -> None:
    """X: koşul (clean, fog, night, rain, blur). Her koşulda 'degraded' ve 'enhanced' bar'ları.
    Clean tek sütun (referans).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df.pivot(index="condition", columns="treatment", values="mAP")
    order = ["clean", "fog", "night", "rain", "blur"]
    pivot = pivot.reindex([c for c in order if c in pivot.index])

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    x = np.arange(len(pivot.index))

    deg_vals = pivot.get("degraded", pd.Series(index=pivot.index, dtype=float)).fillna(0).values
    enh_vals = pivot.get("enhanced", pd.Series(index=pivot.index, dtype=float)).fillna(0).values
    none_vals = pivot.get("none", pd.Series(index=pivot.index, dtype=float)).fillna(0).values

    # 'clean' satırında 'none' kullanılır; diğerlerinde 'degraded' & 'enhanced'.
    bars_left = np.where(pivot.index == "clean", none_vals, deg_vals)
    bars_right = np.where(pivot.index == "clean", none_vals, enh_vals)

    ax.bar(x - width / 2, bars_left, width, label="Bozuk / Temiz", color="#d96459")
    ax.bar(x + width / 2, bars_right, width, label="İyileştirilmiş", color="#5ea864")

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in pivot.index])
    ax.set_ylabel("mAP@0.5")
    # NaN-güvenli y-limit: CLEAN satırında degraded/enhanced boş olduğu için
    # pivot.values.max() NaN dönebilir. fillna(0) ile güvene al.
    y_max = float(pivot.fillna(0).values.max())
    ax.set_ylim(0, max(0.05, y_max * 1.15))
    ax.set_title("Koşul Bazlı Tespit Performansı (Faster R-CNN)")
    # Bar değerlerini üstüne yaz (sunum için okunaklı)
    for i, (l, r) in enumerate(zip(bars_left, bars_right)):
        if l > 0:
            ax.text(i - width / 2, l + y_max * 0.01, f"{l:.2f}", ha="center", fontsize=9)
        if r > 0:
            ax.text(i + width / 2, r + y_max * 0.01, f"{r:.2f}", ha="center", fontsize=9)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def confidence_distribution_plot(records_df: pd.DataFrame, save_path: Path) -> None:
    """Her (condition, treatment) için ortalama güven skoru — line plot."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = records_df.groupby(["condition", "treatment"])["mean_conf"].mean().reset_index()
    pivot = grouped.pivot(index="condition", columns="treatment", values="mean_conf")
    order = ["clean", "fog", "night", "rain", "blur"]
    pivot = pivot.reindex([c for c in order if c in pivot.index])

    fig, ax = plt.subplots(figsize=(9, 5))
    if "none" in pivot.columns:
        ax.plot(pivot.index, pivot["none"], "o-", label="Bozuk / Temiz", color="#d96459")
    if "degraded" in pivot.columns:
        ax.plot(pivot.index, pivot["degraded"], "o-", label="Bozuk", color="#d96459")
    if "enhanced" in pivot.columns:
        ax.plot(pivot.index, pivot["enhanced"], "s--", label="İyileştirilmiş", color="#5ea864")
    ax.set_ylabel("Ortalama tespit güveni")
    ax.set_xlabel("Koşul")
    ax.set_title("Güven Skoru Düşüşü ve İyileşme")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def per_image_scatter(per_image_df: pd.DataFrame, save_path: Path,
                       metric: str = "n_detections") -> None:
    """Her bozulma koşulu için 2x2 scatter:
       x = bozuk metrik (per-image)
       y = iyileştirilmiş metrik (per-image)
       Köşegenin altı = iyileştirme zarar verdi.
       Köşegenin üstü = iyileştirme kazandırdı.

    metric: 'n_detections' veya 'mean_conf'.
    Sun 2022'nin "iyileştirme her zaman kazandırmaz" bulgusunu görsel olarak gösterir.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    conditions = ["fog", "night", "rain", "blur"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    for ax, cond in zip(axes, conditions):
        deg = per_image_df[(per_image_df["condition"] == cond) &
                           (per_image_df["treatment"] == "degraded")][["image_id", metric]]
        enh = per_image_df[(per_image_df["condition"] == cond) &
                           (per_image_df["treatment"] == "enhanced")][["image_id", metric]]
        merged = deg.merge(enh, on="image_id", suffixes=("_deg", "_enh"))
        if merged.empty:
            ax.set_visible(False)
            continue

        x = merged[f"{metric}_deg"]
        y = merged[f"{metric}_enh"]
        worse = (y < x).sum()
        better = (y > x).sum()
        same = (y == x).sum()

        ax.scatter(x, y, alpha=0.55, s=40, edgecolor="k", linewidth=0.4,
                   c=["#5ea864" if yi > xi else ("#d96459" if yi < xi else "#888888")
                      for xi, yi in zip(x, y)])
        lo = min(x.min(), y.min()) - 0.5
        hi = max(x.max(), y.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="x=y (değişim yok)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"Bozuk ({cond}) {metric}")
        ax.set_ylabel(f"İyileştirilmiş ({cond}) {metric}")
        ax.set_title(f"{cond.upper()} — kazandı: {better}, zarar: {worse}, eşit: {same}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        f"Per-Image Karşılaştırma ({metric}) — Sun 2022: iyileştirme her zaman kazandırmaz",
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def detection_count_plot(records_df: pd.DataFrame, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grouped = records_df.groupby(["condition", "treatment"])["n_detections"].mean().reset_index()
    pivot = grouped.pivot(index="condition", columns="treatment", values="n_detections")
    order = ["clean", "fog", "night", "rain", "blur"]
    pivot = pivot.reindex([c for c in order if c in pivot.index])

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    x = np.arange(len(pivot.index))
    deg = pivot.get("degraded", pd.Series(index=pivot.index, dtype=float)).fillna(0).values
    enh = pivot.get("enhanced", pd.Series(index=pivot.index, dtype=float)).fillna(0).values
    none = pivot.get("none", pd.Series(index=pivot.index, dtype=float)).fillna(0).values
    left = np.where(pivot.index == "clean", none, deg)
    right = np.where(pivot.index == "clean", none, enh)

    ax.bar(x - width / 2, left, width, label="Bozuk / Temiz", color="#d96459")
    ax.bar(x + width / 2, right, width, label="İyileştirilmiş", color="#5ea864")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in pivot.index])
    ax.set_ylabel("Görüntü başına ortalama tespit sayısı")
    ax.set_title("Tespit Sayısının Hava Koşullarına Göre Değişimi")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
