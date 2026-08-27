# 🎯 Object Detection

YOLOv8 ile resim, video ve canlı kamera üzerinde **nesne tespiti ve takibi** yapan Streamlit uygulaması.
COCO veri setiyle eğitilmiş hazır model sayesinde insan, araba, köpek, çanta gibi **80 farklı nesneyi** tanır.
Takip modu her nesneye kalıcı bir ID vererek "bu videodan toplam kaç farklı araba geçti" sorusunu cevaplar.

> 📸 *Buraya bir ekran görüntüsü / demo GIF ekle — README'nin en çok bakılan yeri burası.*

---

## Neler yapabiliyor?

| Özellik | Açıklama |
|---|---|
| 📷 **Resim** | JPG/PNG yükle, tespit edilen nesneleri kutularla gör, sonucu indir |
| 🎬 **Video** | MP4 yükle, kare kare işle, işlenmiş videoyu indir |
| 📹 **Webcam** | Bilgisayar kamerasından canlı tespit |
| 🖼️ **Örnekler** | Repoda hazır gelen görsellerle tek tıkla dene |
| 🎯 **Takip** | ByteTrack ile kalıcı ID, benzersiz sayım, çizgi geçişi, hareket izi |
| ⚙️ **Ayarlar** | Model boyutu (n/s/m), güven eşiği ve sınıf filtresi |

### Takip modu neler veriyor?

Video ve Webcam sekmelerindeki **Takip modu** anahtarı açıldığında:

- **Benzersiz sayım** — aynı nesneyi iki kez saymadan "3 farklı insan, 1 otobüs"
- **Çizgi geçiş sayımı** — ekrana sanal bir çizgi koy, geçenleri yönüyle say
  (yatay çizgide `aşağı`/`yukarı`, dikey çizgide `sağa`/`sola`)
- **Hareket izi** — her nesnenin son N karedeki yolu, ID'ye özel renkte
- **Nesne başına süre** — hangi ID kaç saniye ekranda kaldı; CSV olarak indirilebilir

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
│   ├── tracker.py              # takip oturumu, çizgi sayacı, izler
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
   dengesi kurulabilir. `process_video` işin ne olduğunu bilmez — kendisine
   verilen `on_frame` fonksiyonunu çağırır, böylece aynı döngü hem tespit hem
   takip için kullanılır.
5. Takipte `TrackSession` bir oturumun durumunu (ID'ler, izler, sayaçlar) tutar.
   Çizgi geçişi, nesne merkezinin çizgiye göre hangi tarafta olduğunun
   (vektörel çarpımın işareti) kareler arasında değişmesiyle tespit edilir.

## Yol haritası

- [x] **M1** — Resim, video, webcam ve örneklerle çalışan temel uygulama
- [x] **M2** — Nesne takibi: ByteTrack, benzersiz sayım, çizgi geçişi, hareket izi
- [ ] **M3** — Kendi veri setiyle fine-tune (özel sınıflar)
- [ ] **M4** — Testler + GitHub Actions
- [ ] **M5** — Docker + canlı demo (Streamlit Cloud / Hugging Face Spaces)

Detaylar ve her milestone'un notları için [CLAUDE.md](CLAUDE.md).

## Notlar

- CPU'da çalışır; GPU varsa ultralytics otomatik kullanır.
- Takip için `lap` paketi gerekir (`requirements.txt`'de var); eksikse
  ultralytics kurulumu kendi başlatmaya çalışır ama yeniden başlatma ister.
- Webcam sekmesi macOS'ta kamera izni ister; izin verdikten sonra terminali
  yeniden başlatman gerekebilir.
- Örnek görseller [Ultralytics](https://ultralytics.com)'in herkese açık demo görselleridir.

## Lisans

MIT
