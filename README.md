# Kötü Hava Koşullarında Nesne Tespiti: Faster R-CNN Performans Analizi

Otonom sürüşte kullanılan nesne tespit modellerinin sis, gece, yağmur ve hareket bulanıklığı altında ne kadar başarı kaybettiğini ölçen, ardından klasik görüntü iyileştirme yöntemlerinin bu kaybı ne ölçüde telafi edebildiğini inceleyen deneysel bir çalışma.

**Teknolojiler:** Python, PyTorch, torchvision, OpenCV, torchmetrics, pycocotools, pandas, matplotlib

## Motivasyon

Otonom araçların kamera tabanlı algı modelleri çoğunlukla açık havada toplanmış verilerle eğitilir. Model sis, karanlık veya yağmur gibi eğitimde görmediği koşullarla karşılaştığında performansı düşer. Literatürde *domain shift* (alan kayması) olarak bilinen bu durum, bir yayanın veya aracın gözden kaçmasına yol açabilir.

Literatürde bu soruna iki temel çözüm önerilir: modeli kötü hava verisiyle yeniden eğitmek ya da görüntüyü modele vermeden önce iyileştirmek. Bu proje ikinci yaklaşımı inceler. Hazır bir modelin yeniden eğitilmeden, yalnızca klasik ön işleme ile ne kadar kurtarılabileceğini test eder.

## Araştırma Soruları

1. COCO üzerinde eğitilmiş bir Faster R-CNN modeli sis, gece, yağmur ve hareket bulanıklığı altında ne kadar başarı kaybediyor?
2. Klasik görüntü iyileştirme yöntemleri bu kaybın ne kadarını telafi edebiliyor?

**Hipotez:** Sun ve diğerlerinin (NeurIPS 2022) bulgusuna dayanarak klasik iyileştirmenin tespit başarısını her koşulda artırmayacağı, bazı durumlarda düşürebileceği öngörülmüştür.

## Yöntem

### Veri

- **Kaynak:** COCO val2017 (5000 görüntü, etiketli)
- **Hedef sınıflar:** person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
- **Seçim ölçütü:** Görüntüde en az bir araç veya trafik işareti, hedef sınıflardan ise toplam en az iki nesne bulunmalıdır. Bu ölçüt, trafikle ilgisi olmayan portre ve spor sahnelerini eler.
- **Örneklem:** Ölçütü sağlayan görüntüler arasından `seed=42` ile rastgele seçilen **100 trafik sahnesi**. Seçilen görüntülerin id'leri `data/subset_ids.json` dosyasında yer alır.

Bozulmalar sentetik olarak üretilmiştir. Böylece aynı sahnenin temiz ve bozuk hâli birebir karşılaştırılabilir. Gerçek kötü hava veri setlerinde bu eşleştirmeyi yapmak mümkün değildir.

### Model

- **Faster R-CNN ResNet-50 FPN**, torchvision `COCO_V1` ağırlıklarıyla, ince ayar yapılmadan kullanılmıştır.
- Güven eşiği 0.5'tir, çıkarım CPU üzerinde yapılmıştır.
- Faster R-CNN, kötü hava ve alan uyarlaması üzerine yapılmış çalışmalarda (ör. Chen vd., 2018) yaygın olarak referans model olarak kullanıldığı için seçilmiştir.

### Sentetik Bozulmalar (`src/degradations.py`)

| Koşul | Uygulama | Parametreler |
|---|---|---|
| Sis | Koschmieder atmosferik saçılma modeli: `I = J·t + A·(1 − t)`, `t = exp(−β·d)`. Burada `d`, merkeze olan normalize mesafedir. | β = 1.6, A = 240 |
| Gece | Gamma ile karartma (`I^γ`) ve Gauss gürültüsü | γ = 3.2, σ = 12 |
| Yağmur | Rastgele damlalar, −75° açılı hareket kerneliyle çizgiye dönüştürülüp görüntüye bindirilir | yoğunluk 0.02, çizgi uzunluğu 18 px |
| Hareket bulanıklığı | Yatay doğrusal bulanıklık kerneli | kernel 17 px |

### Görüntü İyileştirme Yöntemleri (`src/enhancements.py`)

| Bozulma | Yöntem | Gerekçe | Parametreler |
|---|---|---|---|
| Sis | Dark Channel Prior (He vd., 2009) | Sis modelini tersine çevirerek iletim haritasını ve atmosfer ışığını tahmin eder | yama 15, ω = 0.95, t₀ = 0.1 |
| Gece | Gamma düzeltme + CLAHE | Gamma görüntüyü aydınlatır, CLAHE yerel kontrastı artırır | γ = 0.5, clip 2.0, 8×8 |
| Yağmur | Median filtre + CLAHE | Median filtre ince yağmur çizgilerini bastırır, CLAHE kontrastı geri kazandırır | kernel 3 |
| Hareket bulanıklığı | Unsharp mask | Kenar kontrastını artırarak keskinleştirir | σ = 2.0, miktar 1.5 |

### Deney Düzeni

Her görüntü 9 senaryoda modele verilmiştir (100 × 9 = 900 çıkarım):

1. **Temiz** görüntü (referans)
2. 4 bozulmanın her biri uygulanmış **bozuk** görüntü
3. Her bozuk görüntünün **iyileştirilmiş** hâli

**Metrikler:**

- **mAP@0.5:** Her senaryo için 100 görüntü üzerinden, yalnızca hedef sınıflar dikkate alınarak hesaplanır (`torchmetrics`).
- **Görüntü başına tespit sayısı** ve **ortalama güven skoru**
- **Görüntü bazlı karşılaştırma:** Her görüntüde iyileştirme sonrası tespit sayısının arttığı, azaldığı veya değişmediği durumlar sayılmıştır. Böylece ortalamanın gizleyebileceği tutarsızlıklar ortaya çıkarılır.

## Sonuçlar

### Tespit Başarısı (mAP@0.5)

| Koşul | Bozuk | Temize göre kayıp | İyileştirilmiş | Geri kazanım |
|---|:---:|:---:|:---:|:---:|
| Temiz (referans) | 0.570 | — | — | — |
| Sis | 0.490 | −14.1% | **0.561** | **+0.071** |
| Gece | 0.394 | −30.9% | —\* | —\* |
| Yağmur | 0.528 | −7.5% | 0.469 | −0.059 |
| Hareket bulanıklığı | 0.218 | **−61.8%** | 0.221 | +0.003 |

### Tespit Sayısı ve Güven Skoru

| Koşul | Tespit / görüntü (bozuk → iyileştirilmiş) | Ortalama güven (bozuk → iyileştirilmiş) |
|---|:---:|:---:|
| Temiz (referans) | 10.47 | 0.836 |
| Sis | 8.68 → 10.35 | 0.833 → 0.828 |
| Gece | 7.50 → —\* | 0.804 → —\* |
| Yağmur | 9.24 → 9.46 | 0.835 → 0.823 |
| Hareket bulanıklığı | 5.26 → 5.36 | 0.744 → 0.722 |

### Görüntü Bazlı Analiz (100 görüntü)

| Koşul | İyileştirme sonrası tespit sayısı arttı | Azaldı | Değişmedi |
|---|:---:|:---:|:---:|
| Sis | **67** | 10 | 23 |
| Gece | —\* | —\* | —\* |
| Yağmur | 40 | 30 | 30 |
| Hareket bulanıklığı | 32 | 32 | 36 |

<sub>\* Gece iyileştirmesindeki gamma hatası düzeltildi. Bu değerler deney yeniden çalıştırılarak güncellenecek. Tablolardaki diğer değerler bu hatadan etkilenmez.</sub>

## Bulgular ve Değerlendirme

**Sis: Klasik iyileştirme işe yarıyor.** Sis mAP'i %14 düşürdü. Dark Channel Prior bu kaybın yaklaşık %88'ini geri kazandırdı (0.490 → 0.561). Tespit sayısı da temiz seviyeye yaklaştı (8.7 → 10.4; temiz: 10.5). Görüntü bazında iyileştirme 67 görüntüde yardımcı oldu, yalnızca 10 görüntüde zarar verdi. Başarının nedeni, sentetik sisin fiziksel modelinin (Koschmieder) iyi bilinmesi ve DCP'nin tam olarak bu modeli tersine çevirmek için tasarlanmış olmasıdır.

**Hareket bulanıklığı: En zor problem.** En büyük kayıp bu koşulda yaşandı (%62). Görüntü başına tespit sayısı yarıya indi (10.5 → 5.3), ortalama güven 0.84'ten 0.74'e düştü. Unsharp mask ortalamada neredeyse hiç etki yaratmadı. Görüntü bazında ise kazanım ve kayıplar birebir dengede kaldı (32'ye 32). Bulanıklık, sis gibi görüntünün üzerine eklenen bir katman değil, piksellerin karışmasıyla oluşan bir bilgi kaybıdır. Basit keskinleştirme bu bilgiyi geri getiremez.

**Yağmur: İyileştirme zarar verdi.** Model yağmura görece dayanıklıydı (%7.5 kayıp). Median filtre ve CLAHE uygulandığında ise tespit sayısı hafifçe arttığı hâlde mAP 0.528'den 0.469'a düştü. Bu durum, iyileştirmenin nesne detaylarını da yumuşatarak yanlış veya daha az isabetli tespitlere yol açtığına işaret ediyor.

**Güven skoru:** İyileştirme sonrasında ortalama güven, sonucu geçerli olan üç koşulun üçünde de düştü. Tespit sayısını belirgin şekilde artıran sis iyileştirmesinde bile model, bulduğu nesnelerden daha az emin oldu.

**Genel sonuç:** Klasik iyileştirme yalnızca bozulmanın matematiksel modeli iyi bilindiğinde (sis) güvenilir bir kazanım sağladı. Diğer koşullarda etkisi zayıf, tutarsız veya olumsuz oldu. Bu sonuç, Sun ve diğerlerinin (2022) *"görüntü restorasyonu tespiti her zaman iyileştirmez"* bulgusunu destekliyor. Ayrıca literatürün neden IA-YOLO ve BAD-Net gibi tespit görevine göre uçtan uca öğrenilen yöntemlere yöneldiğini açıklıyor.

## Sınırlamalar ve Gelecek Çalışmalar

- Bozulmalar sentetiktir ve gerçek hava koşullarını tam olarak temsil etmez. Sonuçlar ACDC veya Foggy Cityscapes gibi gerçek kötü hava veri setlerinde doğrulanabilir.
- Örneklem 100 görüntüyle sınırlıdır ve istatistiksel anlamlılık testi yapılmamıştır.
- Yalnızca Faster R-CNN test edilmiştir. Deney YOLOv8 ve SSD ile tekrarlanarak sonuçların modelden bağımsız olup olmadığı incelenebilir.
- Modele ince ayar veya veri artırma uygulanmamıştır. Kötü hava verisiyle yeniden eğitim ve tespit odaklı öğrenilen iyileştirme yöntemleri (IA-YOLO, BAD-Net) ile karşılaştırma yapılabilir.

## Kurulum ve Çalıştırma

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python scripts/1_download_data.py     # COCO val2017 + etiketler (~1 GB)
python scripts/2_select_subset.py     # 100 görüntülük alt küme (data/subset_ids.json)
python scripts/3_run_experiment.py    # Deney (CPU'da ~30-60 dk; hızlı deneme için --n 5)
python scripts/4_make_report.py       # Grafikler ve özet → results/
```

Tüm parametreler (bozulma şiddetleri, eşikler, seed) `src/config.py` dosyasındadır. Deney sonunda `results/` klasöründe metrik CSV dosyaları, grafikler ve temiz, bozuk ve iyileştirilmiş görüntülerin karşılaştırma görselleri oluşur.

## Proje Yapısı

```
├── src/
│   ├── config.py          # Parametreler
│   ├── data_loader.py     # COCO alt kümesi ve etiketler
│   ├── degradations.py    # Sentetik bozulmalar
│   ├── enhancements.py    # İyileştirme yöntemleri
│   ├── detector.py        # Faster R-CNN
│   ├── metrics.py         # mAP, güven, tespit sayısı
│   └── visualize.py       # Grafikler
├── scripts/               # Deney adımları (1 → 4)
├── data/subset_ids.json   # Kullanılan görüntü id'leri
└── requirements.txt
```

## Kaynaklar

- Ren, S., He, K., Girshick, R., Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* NeurIPS.
- Lin, T.-Y. vd. (2014). *Microsoft COCO: Common Objects in Context.* ECCV.
- He, K., Sun, J., Tang, X. (2009). *Single Image Haze Removal Using Dark Channel Prior.* CVPR.
- Sakaridis, C., Dai, D., Van Gool, L. (2018). *Semantic Foggy Scene Understanding with Synthetic Data.* IJCV.
- Chen, Y. vd. (2018). *Domain Adaptive Faster R-CNN for Object Detection in the Wild.* CVPR.
- Liu, W. vd. (2022). *Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions.* AAAI.
- Li, C. vd. (2023). *Detection-Friendly Dehazing: Object Detection in Real-World Hazy Scenes.* IEEE TPAMI.
- Sun, S., Ren, W., Wang, T., Cao, X. (2022). *Rethinking Image Restoration for Object Detection.* NeurIPS.
