# 🎯 Object Detection

YOLOv8 ile resim, video ve canlı kamera üzerinde nesne tespiti yapan Streamlit uygulaması.
COCO veri setiyle eğitilmiş hazır model sayesinde insan, araba, köpek, çanta gibi **80 farklı nesneyi** tanır.

> 📸 *Buraya bir ekran görüntüsü / demo GIF ekle — README'nin en çok bakılan yeri burası.*

---

## Neler yapabiliyor?

| Özellik | Açıklama |
|---|---|
| 📷 **Resim** | JPG/PNG yükle, tespit edilen nesneleri kutularla gör, sonucu indir |
| 🎬 **Video** | MP4 yükle, kare kare işle, işlenmiş videoyu indir |
| 📹 **Webcam** | Bilgisayar kamerasından canlı tespit |
| 🖼️ **Örnekler** | Repoda hazır gelen görsellerle tek tıkla dene |
| ⚙️ **Ayarlar** | Model boyutu (n/s/m), güven eşiği ve sınıf filtresi |

## Kurulum

```bash
git clone <repo-url>
cd Object-Detection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_samples.py   # örnek görselleri indir (opsiyonel)
```

## Çalıştırma

```bash
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` açılır. İlk çalıştırmada model ağırlığı (~6 MB)
otomatik olarak indirilip `models/` klasörüne kaydedilir.

## Proje yapısı

```
Object-Detection/
├── app.py                      # Streamlit arayüzü (tüm sekmeler)
├── src/
│   ├── config.py               # yollar, model listesi, varsayılan ayarlar
│   ├── detector.py             # YOLO sarmalayıcı — detect() burada
│   └── video.py                # video dosyasını kare kare işleme
├── scripts/
│   └── download_samples.py     # örnek görselleri indirir
├── samples/                    # örnek görseller
├── models/                     # model ağırlıkları (git'e girmez)
├── outputs/                    # işlenmiş videolar (git'e girmez)
├── requirements.txt
└── CLAUDE.md                   # geliştirme günlüğü / yol haritası
```

## Nasıl çalışıyor?

1. `Detector` sınıfı ultralytics'in `YOLO` modelini yükler ve bellekte tutar
   (Streamlit'te `@st.cache_resource` ile bir kez yüklenir).
2. Yüklenen resim OpenCV ile BGR bir numpy dizisine çevrilir.
3. Model çıktısı hem çizilmiş görsel hem de `Detection(label, confidence, box)`
   listesi olarak döner — arayüz ikisini de kullanır.
4. Videoda her kare aynı yoldan geçer; "kare atlama" ayarı ile hız/doğruluk
   dengesi kurulabilir.

## Yol haritası

- [x] **M1** — Resim, video, webcam ve örneklerle çalışan temel uygulama
- [ ] **M2** — Nesne takibi (aynı nesneye ID verip video boyunca izleme)
- [ ] **M3** — Kendi veri setiyle fine-tune (özel sınıflar)
- [ ] **M4** — Testler + GitHub Actions
- [ ] **M5** — Docker + canlı demo (Streamlit Cloud / Hugging Face Spaces)

Detaylar ve her milestone'un notları için [CLAUDE.md](CLAUDE.md).

## Notlar

- CPU'da çalışır; GPU varsa ultralytics otomatik kullanır.
- Webcam sekmesi macOS'ta kamera izni ister; izin verdikten sonra terminali
  yeniden başlatman gerekebilir.
- Örnek görseller [Ultralytics](https://ultralytics.com)'in herkese açık demo görselleridir.

## Lisans

MIT
