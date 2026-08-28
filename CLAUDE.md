# CLAUDE.md

Bu dosya, projede çalışan Claude (ve gelecekteki ben) için bağlam ve
geliştirme günlüğüdür. **Her milestone sonunda güncellenir.**

---

## Proje özeti

GitHub portfolyosu / CV için geliştirilen bir **nesne tespiti (object detection)**
ve **takip (tracking)** uygulaması. YOLOv8'in COCO ile eğitilmiş hazır modelini
kullanarak resim, video ve canlı kamera üzerinde 80 sınıfı tespit eder; takip
modunda her nesneye kalıcı ID verip benzersiz sayım ve çizgi geçişi hesaplar.
Ayrıca African Wildlife veri setiyle fine-tune edilmiş kendi modelimiz de
arayüzden seçilebiliyor. Arayüz Streamlit.

**Hedef:** Çalışan, gösterilebilir, anlaşılır bir proje. Aşırı mühendislik yok —
kod okunduğunda ne yaptığı anlaşılmalı.

## Teknoloji

| Ne | Neden |
|---|---|
| Python 3.13 (`.venv`) | Proje içi izole ortam, Anaconda base'i kirletmiyoruz |
| ultralytics (YOLOv8) | Hazır COCO modeli, tek satırda inference |
| ByteTrack (+ `lap`) | Takip için; hızlı, CPU'da rahat çalışır, ultralytics'e gömülü |
| OpenCV | Resim/video okuma-yazma, kare işleme |
| Streamlit | Hızlı, görsel arayüz — portfolyoda ekran görüntüsü almak kolay |
| MPS (Apple Silicon) | Eğitim burada dönüyor; Colab notebook'u GPU alternatifi |
| pytest + ruff | Test ve lint; ikisi de `pyproject.toml`'da yapılandırılıyor |

## Komutlar

```bash
source .venv/bin/activate        # ortamı aktif et
streamlit run app.py             # uygulamayı çalıştır
python scripts/download_samples.py   # örnek görselleri indir

python scripts/train.py --epochs 30  # fine-tune (models/<isim>.pt üretir)
python scripts/evaluate.py           # metrikler → docs/metrics.{json,md} + docs/plots/
python scripts/compare.py            # önce/sonra görselleri → docs/comparison/

pytest                               # tüm testler
pytest -m "not slow"                 # hızlı olanlar (CI'ın koştuğu)
pytest -m slow                       # gerçek modeli çalıştıranlar
ruff check . && ruff format .        # lint + format
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
- **Video işleme `src/video.py`'de ayrı ve *işten bağımsız*.** `process_video`
  ne yaptığını bilmez; kendisine verilen `on_frame(kare) -> kare` fonksiyonunu
  çağırır. Böylece tespit ve takip aynı döngüyü paylaşır, kod ikiye bölünmez.
  "Kare atlama" (stride) ayarı hız/doğruluk dengesi kurar.
- **Takip durumu `TrackSession` içinde.** ID'ler, izler ve sayaçlar oturuma ait;
  her video/webcam açılışında yeni bir oturum kurulur. Model
  `@st.cache_resource` ile paylaşıldığı için `TrackSession.__post_init__`
  ultralytics'in tracker durumunu da sıfırlar — yoksa önceki videonun ID'leri
  yenisine sızar.
- **Eğitilen modeller `models/` altında, arayüz onları kendi bulur.**
  `config.custom_models()` hazır listede olmayan `.pt` dosyalarını tarar ve
  "Özel: <isim>" olarak model listesine ekler. Yeni bir model eğitmek arayüzde
  hiçbir kod değişikliği gerektirmiyor — dosyayı `models/`'a koymak yeterli.
- **`models/african-wildlife.pt` bilerek git'e dahil (5.9 MB).** `.gitignore`'da
  `*.pt` kuralına özel bir istisna var. Repoyu klonlayan biri 31 dakika eğitim
  beklemeden "Özel:" modelini deneyebilsin diye.
- **Metrikler ve grafikler `docs/` altında commit'leniyor.** `runs/` git'e
  girmiyor; `evaluate.py` gösterilmeye değer grafikleri `docs/plots/`'a
  kopyalıyor. README ve Streamlit sekmesi aynı dosyaları okuyor.
- **Testler `slow` işaretiyle ikiye ayrılıyor.** Hızlı olanlar `tests/conftest.py`
  içindeki sahte model katmanını kullanıyor: `FakeModel.track()` ultralytics
  çıktısının yalnızca `TrackSession`'ın dokunduğu kadarını taklit ediyor.
  Bu sayede takip mantığı torch'a hiç dokunmadan test ediliyor ve CI dakikalar
  yerine saniyeler sürüyor. Gerçek modelle çalışanlar `slow` işaretli.
- **`pythonpath = ["."]` pytest yapılandırmasında şart.** CI `pytest` komutunu
  doğrudan çağırıyor; `python -m pytest`ten farklı olarak çalışma dizinini
  `sys.path`'e eklemiyor ve `import src` patlıyor.
- **Çizgi geçişi vektörel çarpımın işaretiyle bulunur.** Nesne merkezinin
  çizgiye göre tarafı iki kare arasında değiştiyse geçmiştir; işaretin yönü de
  giriş/çıkış ayrımını verir. Kesişim hesabı yapmaya gerek yok.

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

### ✅ M2 — Nesne takibi (2026-08-28)

**Yapılanlar**
- `src/tracker.py` — `TrackSession` (oturum durumu), `Track` (kimlikli nesne),
  `LineCounter` (çizgi geçiş sayacı), `color_for` (ID'ye özel renk),
  `line_from_ratio` (arayüz seçimi → piksel koordinatı).
- ByteTrack (`bytetrack.yaml`) kullanılıyor. `lap>=0.5.12` bağımlılığı eklendi —
  ultralytics eksikse kendi kurmaya çalışıyor ama yeniden başlatma istiyor,
  o yüzden `requirements.txt`'e açıkça yazıldı.
- `src/video.py` yeniden düzenlendi: `process_video` artık `on_frame` callback'i
  alıyor, tespit/takip ayrımını bilmiyor. `video_info()` eklendi (fps/boyutu
  işlemeden önce okumak için).
- `Detector._class_ids` → `class_ids` (tracker da aynı dönüşüme ihtiyaç duyuyor).
- `app.py` — Video ve Webcam sekmelerine "Takip modu" anahtarı, iz uzunluğu,
  çizgi yönü/konumu kontrolleri; benzersiz sayım + çizgi sayacı + süre tablosu
  ve CSV indirme.

**Doğrulandı**
- Sentetik videoda 60 kare boyunca 5 ID hiç değişmeden korundu (ID switch yok).
- Çizgi yönü üç senaryoda test edildi: sağa giden → `saga: 4`, aşağı giden →
  `asagi: 4`, sola giden → `sola: 4`. Yanlış yön hatası düzeltildi (aşağıya bak).
- Tracker sıfırlama: aynı model nesnesiyle art arda iki oturum açıldığında
  ikincisi de ID 1'den başlıyor — sızma yok.
- Arayüz: takip kontrolleri doğru render ediliyor, çizgi ayarları
  "Çizgi geçiş sayımı" işaretlenene kadar pasif.

**Yol boyunca düzeltilen**
- Dikey çizgide soldan sağa hareket "geri" olarak sayılıyordu. Sebep: çizgi
  yukarıdan aşağı çiziliyordu, vektörel çarpımın pozitif tarafı sol kalıyordu.
  Çizgi aşağıdan yukarı çizilecek şekilde değiştirildi. Ayrıca yön isimleri
  `ileri/geri` yerine yöne göre `asagi/yukari` ve `saga/sola` yapıldı.

**Bilinen eksikler / notlar**
- Yüksek `stride` değeri ID kararlılığını bozabilir; arayüzde uyarı var ama
  engellenmiyor.
- Benzersiz sayım ByteTrack'in ID'lerine güveniyor. Nesne uzun süre kaybolup
  geri gelirse yeni ID alır ve iki kez sayılır. BoT-SORT (re-ID) bunu iyileştirir
  — istenirse kenar çubuğuna seçim eklenebilir.
- Trail sözlüğü oturum boyunca büyür (her ID için `deque`). Saatlerce süren
  webcam oturumunda bellek sorun olabilir; şimdilik önemsiz.

---

### ✅ M3 — Kendi veri setiyle fine-tune (2026-08-28)

**Veri seti:** `african-wildlife` (ultralytics'in hazır seti, 100 MB, 4 sınıf:
buffalo, elephant, rhino, zebra). Seçim sebebi: elephant ve zebra COCO'da var,
buffalo ve rhino yok — "önce/sonra" farkı hem gerçek hem de dürüst görünüyor.
API anahtarı gerektirmiyor, `data=african-wildlife.yaml` deyince kendi iniyor.

**Yapılanlar**
- `scripts/train.py` — fine-tune CLI'ı. Cihazı otomatik seçiyor (cuda → mps → cpu),
  `patience` ile erken durdurma, bitince en iyi ağırlığı `models/<isim>.pt`'ye
  kopyalıyor.
- `scripts/evaluate.py` — doğrulama metriklerini `docs/metrics.json` ve
  `docs/metrics.md`'ye yazıyor, eğitim grafiklerini `docs/plots/`'a kopyalıyor.
- `scripts/compare.py` — hazır COCO modeli ile kendi modelimizi aynı görsellerde
  yan yana koyuyor.
- `notebooks/train_colab.ipynb` — aynı eğitimin Colab GPU sürümü.
- `config.custom_models()` + `app.py` kenar çubuğu — eğitilen model tüm
  sekmelerde (resim/video/webcam/takip) kullanılabiliyor.
- `app.py` "📊 Model performansı" sekmesi — metrik kartları, sınıf tablosu,
  önce/sonra seçici, eğitim grafikleri.

**Sonuçlar** (YOLOv8n, 30 epoch, 640px, MPS, 31 dakika)

| Metrik | Değer |
|---|---|
| mAP50 | 0.957 |
| mAP50-95 | 0.791 |
| Precision | 0.954 |
| Recall | 0.895 |

Sınıf bazında mAP50: buffalo 0.970, elephant 0.927, rhino 0.972, zebra 0.958.

**Önce/sonra kanıtı:** COCO modeli gergedanı `cow 0.56` + hayalet bir `horse`
olarak görüyor, kendi modelimiz `rhino 0.97` diyor. Bufalo için de COCO `cow`
diyor. Elephant ve zebra'da ikisi de doğru — beklenen, çünkü bunlar COCO'da var.

**Yol boyunca düzeltilen**
- `compare.py` veri setinin `valid/images` düzeninde olduğunu varsayıyordu, oysa
  bu set `images/val` kullanıyor. Yaygın dört düzeni de deneyen bir arama eklendi.
- Rastgele görsel seçimi veri setinde çok olan sınıfa (fil) yığılıyordu ve
  karşılaştırma anlamsız görünüyordu. Etiket dosyalarından sınıf okunup her
  sınıftan eşit örnek alınacak şekilde değiştirildi.
- Karşılaştırma başlıkları sadece tespit *sayısını* yazıyordu; asıl fark
  etiketlerde olduğu için başlıklara etiket listesi konuldu.
- Veri setindeki dosya adlarında boşluk ve parantez var (`3 (226).jpg`) — çıktılar
  içeriğe göre yeniden adlandırılıyor (`rhino.jpg`, `buffalo-2.jpg`).

**Bilinen eksikler / notlar**
- 30 epoch keyfi bir sayı; `patience=15` erken durdurma tetiklenmedi, yani daha
  uzun eğitim biraz daha iyileştirebilir.
- Sadece YOLOv8n denendi. `--model yolov8s.pt` ile daha büyük model muhtemelen
  mAP50-95'i yükseltir.
- Colab notebook'u yazıldı ama Colab'da **çalıştırılmadı** — yerelde aynı
  ultralytics çağrılarını kullanıyor, yine de ilk kullanımda gözden geçir.
- `docs/` şu an tek bir modelin sonuçlarını tutuyor. İkinci bir model eğitilirse
  dosyalar üzerine yazılır; gerekirse model adına göre klasörlenmeli.

---

### ✅ M4 — Testler + CI (2026-08-28)

**Yapılanlar**
- `tests/conftest.py` — sahte model katmanı (`FakeModel`, `FakeResult`, `FakeBox`,
  `FakeDetector`) ve sentetik video fixture'ı. Ultralytics çıktısının sadece
  `TrackSession`'ın kullandığı kadarı taklit ediliyor.
- `tests/test_tracker.py` (26 test) — çizgi sayacı yön mantığı, ilk görülmede
  saymama, aynı tarafta kalınca saymama, gidip gelme, çizgi üstündeki nokta;
  `TrackSession` benzersiz sayım, süre, iz uzunluğu, sınıf filtresi, reset.
- `tests/test_video.py` (10 test) — kare sayısı, stride davranışı (atlanan
  karelerin son çizilmiş kareyle doldurulması dahil), ilerleme callback'i,
  hatalı dosya.
- `tests/test_config.py` (8 test) — `custom_models()` keşfi ve hazır modelleri
  dışarıda bırakması.
- `tests/test_detector.py` (11 test, 7'si `slow`) — `summarize()` mantığı hızlı;
  gerçek modelle sınıf sayısı, tespit, filtre, eşik davranışı `slow`.
- `pyproject.toml` — pytest (marker, testpaths, pythonpath) ve ruff (E/F/I/B/UP,
  100 karakter) yapılandırması.
- `.github/workflows/ci.yml` — ruff işi + Python 3.11/3.12/3.13 test matrisi.
- `requirements-dev.txt`, README'ye CI rozeti ve test bölümü.

**Sonuç:** 55 test, hızlı paket 0.7 sn'de koşuyor, `src/` kapsamı **%91**
(tracker %98, video %95, config %100). `detector` %54 — model gerektiren
kısımları yalnızca `slow` testlerde.

**Yol boyunca düzeltilen**
- CI `pytest`i doğrudan çağırıyordu ve `import src` patlıyordu; `python -m pytest`
  çalışma dizinini `sys.path`'e eklediği için yerelde sorun görünmüyordu.
  `pythonpath = ["."]` eklendi — yerelde bare `pytest` ile doğrulandı.
- Ruff 15 sorun buldu (import sırası, uzun satırlar, `%` formatı); 9'u otomatik,
  kalanı elle düzeltildi. 9 dosya yeniden formatlandı.

**Bilinen eksikler / notlar**
- CI'da her iş `torch`u CPU deposundan kuruyor (PyPI sürümü Linux'ta CUDA
  paketlerini de çekiyor, ~2.5 GB). Yine de kurulum işin çoğu zamanını alıyor.
- `app.py` test edilmiyor. Streamlit arayüzünü test etmek `streamlit.testing`
  gerektirir; şimdilik değmez, arayüz elle doğrulanıyor.
- `scripts/` altındaki eğitim hattı test edilmiyor — gerçek eğitim gerektirdiği
  için CI'a uygun değil.
- Coverage rozeti yok, sadece CI çıktısında görünüyor. İstenirse Codecov eklenebilir.

---

## Sıradaki milestone'lar

### 🔜 M5 — Dağıtım
Dockerfile + Hugging Face Spaces veya Streamlit Cloud'da canlı demo.
README'ye "Live Demo" rozeti — işe alım yapan kişi kodu indirmeden deneyebilsin.

### 💡 Fikir havuzu (sıralı değil)
- CLI arayüzü (`python detect.py --image foo.jpg`) — batch işler için.
- Isı haritası / yoğunluk görselleştirmesi.
- Tespit sonuçlarını JSON olarak dışa aktarma (takip CSV'si M2'de eklendi).
- BoT-SORT seçeneği: uzun süre kaybolan nesneyi re-ID ile hatırlar.
- İngilizce README (uluslararası başvurular için).
- Kendi topladığın görsellerle ikinci bir veri seti (M3 altyapısı hazır).
- Daha büyük model (`yolov8s/m`) ile eğitim ve karşılaştırma.
- `streamlit.testing` ile arayüz testleri.
- Codecov entegrasyonu ve kapsam rozeti.
- Model karşılaştırma sekmesi: aynı görselde n/s/m sonuçları yan yana.
