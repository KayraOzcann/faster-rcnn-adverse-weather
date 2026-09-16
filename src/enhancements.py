"""Klasik görüntü iyileştirme yöntemleri (modelden bağımsız)."""
import numpy as np
import cv2

from . import config as C


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """LAB uzayında L kanalına CLAHE — kontrast artırır, renkleri korur."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP, tileGridSize=C.CLAHE_GRID)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def apply_gamma(img: np.ndarray, gamma: float = C.GAMMA_BRIGHT) -> np.ndarray:
    """out = in^gamma; gamma<1 → görüntüyü aydınlatır (gece için)."""
    table = ((np.arange(256) / 255.0) ** max(gamma, 1e-3) * 255).astype(np.uint8)
    return cv2.LUT(img, table)


def _dark_channel(img: np.ndarray, patch: int) -> np.ndarray:
    min_chan = img.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch))
    return cv2.erode(min_chan, kernel)


def _atmospheric_light(img: np.ndarray, dark: np.ndarray) -> np.ndarray:
    h, w = dark.shape
    n_pixels = max(int(h * w * 0.001), 1)
    flat_dark = dark.ravel()
    idx = np.argpartition(-flat_dark, n_pixels)[:n_pixels]
    flat_img = img.reshape(-1, 3)
    return flat_img[idx].mean(axis=0)


def apply_dehaze(img: np.ndarray, patch: int = C.DEHAZE_PATCH,
                 omega: float = C.DEHAZE_OMEGA, t0: float = C.DEHAZE_T0) -> np.ndarray:
    """Dark Channel Prior (He et al. 2009) — sentetik sis için iyi sonuç verir."""
    f = img.astype(np.float32)
    dark = _dark_channel(f, patch)
    A = _atmospheric_light(f, dark)
    A_safe = np.where(A < 1e-3, 1e-3, A)

    norm = f / A_safe.reshape(1, 1, 3)
    t = 1.0 - omega * _dark_channel(norm, patch)
    t = np.clip(t, t0, 1.0)
    t3 = t[:, :, None]

    J = (f - A.reshape(1, 1, 3)) / t3 + A.reshape(1, 1, 3)
    return np.clip(J, 0, 255).astype(np.uint8)


def apply_unsharp(img: np.ndarray, sigma: float = 2.0, amount: float = 1.5) -> np.ndarray:
    """Hareket bulanıklığını kısmen tersine çevirmek için unsharp mask."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    out = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_median_clahe(img: np.ndarray) -> np.ndarray:
    """Yağmur için: median ile çizgileri yumuşat, sonra kontrast geri kazan."""
    den = cv2.medianBlur(img, 3)
    return apply_clahe(den)


def auto_enhance(img: np.ndarray, condition: str) -> np.ndarray:
    """Hangi bozulma için hangi iyileştirme — tek kapı."""
    if condition == "fog":
        return apply_dehaze(img)
    if condition == "night":
        return apply_clahe(apply_gamma(img))
    if condition == "rain":
        return apply_median_clahe(img)
    if condition == "blur":
        return apply_unsharp(img)
    raise ValueError(f"Unknown condition: {condition}")
