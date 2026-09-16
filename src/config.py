from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "coco_val2017"
ANN_PATH = DATA_DIR / "annotations" / "instances_val2017.json"
SUBSET_FILE = DATA_DIR / "subset_ids.json"

RESULTS_DIR = ROOT / "results"
DEGRADED_SAMPLES_DIR = RESULTS_DIR / "degraded_samples"
ENHANCED_SAMPLES_DIR = RESULTS_DIR / "enhanced_samples"
GRIDS_DIR = RESULTS_DIR / "comparison_grids"
PLOTS_DIR = RESULTS_DIR / "plots"
METRICS_CSV = RESULTS_DIR / "metrics.csv"

# COCO sınıf id'leri (otonom sürüşle ilgili olanlar)
TARGET_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
    10: "traffic light",
    13: "stop sign",
}
TARGET_CLASS_IDS = set(TARGET_CLASSES.keys())

# Trafik sahnesi olduğunu garanti etmek için: bu sınıflardan en az biri olmalı.
# (person tek başına bir trafik sahnesi göstermez — selfie/spor sahneleri elenir.)
TRAFFIC_REQUIRED_IDS = {2, 3, 4, 6, 8, 10, 13}  # bicycle, car, motorcycle, bus, truck, traffic light, stop sign

# Örnekleme
N_IMAGES = 100
RANDOM_SEED = 42

# Tespit eşikleri
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# Bozulma şiddetleri (tek noktadan ayar)
FOG_BETA = 1.6
NIGHT_GAMMA = 3.2
NIGHT_NOISE_STD = 12
RAIN_DENSITY = 0.020
RAIN_LENGTH = 18
BLUR_KERNEL = 17

# İyileştirme parametreleri
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
GAMMA_BRIGHT = 0.5
DEHAZE_PATCH = 15
DEHAZE_OMEGA = 0.95
DEHAZE_T0 = 0.1

# Sonuç görselleştirmesi
N_GRID_SAMPLES = 6
