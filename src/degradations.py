"""Sentetik kötü hava bozulmaları. Tüm fonksiyonlar BGR uint8 görüntü alır, BGR uint8 döndürür."""
import numpy as np
import cv2

from . import config as C


def add_fog(img: np.ndarray, beta: float = C.FOG_BETA) -> np.ndarray:
    """Koschmieder atmosferik saçılma modeli: I = J*t + A*(1-t), t = exp(-beta*d).

    d: merkeze normalize uzaklık (0..~1.4) — yatay düzlemde sabit derinlik yerine
    bu basit yaklaşım, görüntü kenarlarında daha yoğun sis hissi verir ve
    görsel olarak inandırıcıdır.
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (np.sqrt(cx ** 2 + cy ** 2))
    t = np.exp(-beta * d)
    t = t[:, :, None]

    A = np.array([240, 240, 240], dtype=np.float32)  # parlak gri atmosfer
    J = img.astype(np.float32)
    foggy = J * t + A * (1.0 - t)
    return np.clip(foggy, 0, 255).astype(np.uint8)


def add_night(img: np.ndarray, gamma: float = C.NIGHT_GAMMA,
              noise_std: float = C.NIGHT_NOISE_STD) -> np.ndarray:
    """Karartma + hafif gauss gürültüsü ile gece/düşük ışık simülasyonu."""
    norm = img.astype(np.float32) / 255.0
    dark = np.power(norm, gamma) * 255.0
    noise = np.random.normal(0, noise_std, dark.shape)
    out = dark + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_rain(img: np.ndarray, density: float = C.RAIN_DENSITY,
             length: int = C.RAIN_LENGTH, angle: float = -75) -> np.ndarray:
    """Rastgele yağmur çizgileri + motion blur + orijinal üstüne bindirme."""
    h, w = img.shape[:2]
    n_drops = int(h * w * density)

    rain_layer = np.zeros((h, w), dtype=np.uint8)
    ys = np.random.randint(0, h, n_drops)
    xs = np.random.randint(0, w, n_drops)
    rain_layer[ys, xs] = 255

    # Yağmur çizgileri için açılı motion blur kerneli
    k = np.zeros((length, length), dtype=np.float32)
    cv2.line(k, (length // 2, 0), (length // 2, length - 1), 1.0, 1)
    M = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
    k = cv2.warpAffine(k, M, (length, length))
    k = k / (k.sum() + 1e-8)
    rain_layer = cv2.filter2D(rain_layer, -1, k)

    rain_bgr = cv2.cvtColor(rain_layer, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(img, 0.85, rain_bgr, 0.7, 0)
    # Hafif genel bulanıklık (yağmurda görüş azalır)
    out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


def add_blur(img: np.ndarray, kernel: int = C.BLUR_KERNEL) -> np.ndarray:
    """Yatay hareket bulanıklığı (kamera/araç hareketi)."""
    k = np.zeros((kernel, kernel), dtype=np.float32)
    k[kernel // 2, :] = 1.0
    k /= kernel
    return cv2.filter2D(img, -1, k)


DEGRADATIONS = {
    "fog": add_fog,
    "night": add_night,
    "rain": add_rain,
    "blur": add_blur,
}


def degrade(img: np.ndarray, condition: str) -> np.ndarray:
    return DEGRADATIONS[condition](img)
