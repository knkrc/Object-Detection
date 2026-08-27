# CLAUDE.md

Bu dosya, projede çalışan Claude (ve gelecekteki ben) için bağlam ve
geliştirme günlüğüdür. **Her milestone sonunda güncellenir.**

---

## Proje özeti

GitHub portfolyosu / CV için geliştirilen bir **nesne tespiti (object detection)**
uygulaması. YOLOv8'in COCO ile eğitilmiş hazır modelini kullanarak resim, video
ve canlı kamera üzerinde 80 sınıfı tespit eder. Arayüz Streamlit.

**Hedef:** Çalışan, gösterilebilir, anlaşılır bir proje. Aşırı mühendislik yok —
kod okunduğunda ne yaptığı anlaşılmalı.

## Teknoloji

| Ne | Neden |
|---|---|
| Python 3.13 (`.venv`) | Proje içi izole ortam, Anaconda base'i kirletmiyoruz |
| ultralytics (YOLOv8) | Hazır COCO modeli, tek satırda inference |
| OpenCV | Resim/video okuma-yazma, kare işleme |
| Streamlit | Hızlı, görsel arayüz — portfolyoda ekran görüntüsü almak kolay |

## Komutlar

```bash
source .venv/bin/activate        # ortamı aktif et
streamlit run app.py             # uygulamayı çalıştır
python scripts/download_samples.py   # örnek görselleri indir
```

## Mimari kararlar

- **`src/detector.py` tek giriş noktası.** Arayüz katmanı (`app.py`) ultralytics'i
  doğrudan tanımaz; sadece `Detector.detect()` çağırır ve
  `(çizilmiş_görsel, [Detection, ...])` alır. Böylece ileride model değiştirmek
  (YOLOv11, RT-DETR, kendi eğittiğimiz model) tek dosyayı ilgilendirir.
- **Görüntüler BGR numpy dizisi olarak dolaşır.** OpenCV'nin varsayılanı bu;
  RGB'ye çevirme sadece ekrana basarken (`to_rgb`) yapılır. Karışıklığı önlemek
  için bu kurala sadık kalıyoruz.
- **Model `@st.cache_resource` ile bir kez yüklenir.** Yoksa her etkileşimde
  yeniden yüklenir ve uygulama kullanılamaz hale gelir.
- **Ağırlıklar `models/` altında, git'e girmez.** İlk çalıştırmada otomatik iner
  (`Detector.__init__` indirilen dosyayı `models/`'e taşır).
- **Video işleme `src/video.py`'de ayrı.** `app.py`'nin şişmemesi için.
  "Kare atlama" (stride) ayarı hız/doğruluk dengesi kurar.

## Kod kuralları

- Yorumlar Türkçe, kod/değişken isimleri İngilizce.
- Docstring'ler *ne yaptığını* değil *neden öyle yaptığını* anlatsın.
- Yeni bir özellik `src/` altında kendi modülüne; `app.py` sadece arayüz olsun.
- Bağımlılık eklerken hem `requirements.txt` (gevşek) hem
  `requirements-lock.txt` (`pip freeze`) güncellenir.

---

## Milestone günlüğü

### ✅ M1 — Temel uygulama (2026-08-27)

**Yapılanlar**
- `.venv` kuruldu; ultralytics 8.4.131, torch 2.13.0, opencv 5.0.0, streamlit 1.62.0.
- `src/config.py` — yollar, model listesi (n/s/m), varsayılan ayarlar.
- `src/detector.py` — `Detector` sınıfı + `Detection` dataclass + `summarize()`.
- `src/video.py` — video dosyasını kare kare işleyip mp4 yazma, ilerleme callback'i.
- `app.py` — 4 sekme (Resim / Video / Webcam / Örnekler) + kenar çubuğunda
  model seçimi, güven eşiği, sınıf filtresi.
- `scripts/download_samples.py` — Ultralytics'in açık demo görsellerini indirir.
- README, .gitignore, requirements.
- `LICENSE` (MIT) — README'de belirtilen lisansın karşılığı. **Copyright satırındaki
  ismi tam adınla değiştir.**
- `git init` yapıldı, dosyalar stage'lendi (commit atılmadı).

**Doğrulandı**
- `samples/bus.jpg`: 4 tespit (3× person, 1× bus). Sınıf filtresi (`keep_classes`)
  çalışıyor, ağırlık `models/` altına iniyor.
- Streamlit arayüzü ayağa kalkıyor, "Örnekler" sekmesi orijinal/sonuç
  karşılaştırmasını doğru gösteriyor.
- Video hattı: 40 karelik test videosu, stride=2 ile 40 kare yazıldı,
  ilerleme callback'i 1.0'a ulaşıyor, çıktı mp4 tekrar okunabiliyor.

**Bilinen eksikler / notlar**
- Webcam döngüsü Streamlit'in rerun mekanizmasına dayanıyor; "Durdur" butonu
  script'i yeniden çalıştırarak döngüyü kesiyor. Basit ama kırılgan — sorun
  çıkarsa `streamlit-webrtc`'ye geçilebilir.
- Video çıktısı `avc1` codec'i ile yazılmaya çalışılıyor, olmazsa `mp4v`.
  `mp4v` bazı tarayıcılarda oynamayabilir — indirme butonu her hâlükârda var.
- Henüz test yok.

---

## Sıradaki milestone'lar

### 🔜 M2 — Nesne takibi (tracking)
Videoda aynı nesneye kalıcı bir ID verip kare boyunca izlemek.
`model.track(persist=True)` ile ByteTrack/BoTSORT. Çıktı: "bu videoda toplam
7 farklı araba geçti" gibi gerçek bir sayım. Portfolyoda en çok fark yaratacak adım.

### 📋 M3 — Kendi veri setiyle fine-tune
Küçük bir özel veri seti (2-3 sınıf) toplayıp/etiketleyip YOLOv8'i fine-tune etmek.
`train.py` + `data.yaml` + eğitim metrikleri (mAP, confusion matrix) README'de.
"Hazır model kullandı" ile "model eğitti" arasındaki farkı gösterir.

### 📋 M4 — Testler + CI
`pytest` ile `Detector` ve `video.py` için birkaç anlamlı test
(örn. bilinen görselde beklenen sınıflar bulunuyor mu). GitHub Actions ile
her push'ta çalışsın.

### 📋 M5 — Dağıtım
Dockerfile + Hugging Face Spaces veya Streamlit Cloud'da canlı demo.
README'ye "Live Demo" rozeti — işe alım yapan kişi kodu indirmeden deneyebilsin.

### 💡 Fikir havuzu (sıralı değil)
- CLI arayüzü (`python detect.py --image foo.jpg`) — batch işler için.
- Isı haritası / yoğunluk görselleştirmesi.
- Tespit sonuçlarını JSON/CSV olarak dışa aktarma.
- İngilizce README (uluslararası başvurular için).
- Model karşılaştırma sekmesi: aynı görselde n/s/m sonuçları yan yana.
