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
| Docker | Dağıtım; aynı imaj hem yerelde hem Hugging Face Spaces'te çalışıyor |

## Komutlar

```bash
source .venv/bin/activate        # ortamı aktif et
streamlit run app.py             # uygulamayı çalıştır
python scripts/download_samples.py   # örnek görselleri indir

python scripts/train.py --epochs 30  # fine-tune (models/<isim>.pt üretir)
python scripts/evaluate.py           # metrikler → docs/metrics.{json,md} + docs/plots/
python scripts/compare.py            # önce/sonra görselleri → docs/comparison/
python scripts/screenshot.py         # README ekran görüntüleri → docs/screenshots/
python scripts/make_demo_gif.py      # README demo GIF'i → docs/demo.gif (ffmpeg gerekir)

pytest                               # tüm testler
pytest -m "not slow"                 # hızlı olanlar (CI'ın koştuğu)
pytest -m slow                       # gerçek modeli çalıştıranlar
ruff check . && ruff format .        # lint + format

docker compose up --build            # konteynerde çalıştır
./deploy/push_to_hf.sh <user>/<space>  # HF Spaces'e gönder (HF_TOKEN gerekir)
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
- **Ağırlıklar `models/` altında, git'e girmez.** İlk çalıştırmada otomatik iner.
  `detector.resolve_weights()` / `stash_weights()` bu işi yapıyor ve hem
  `Detector` hem `scripts/train.py` bunları kullanıyor — ultralytics indirmeyi
  çalışma dizinine yaptığı için, ortak bir yerde tutulmazsa proje kökü kirleniyor.
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
- **Demo GIF'i de script'le.** `scripts/make_demo_gif.py` playwright'ın video
  kaydıyla arayüz turunu çekiyor, ffmpeg iki geçişli palet yöntemiyle GIF'e
  çeviriyor (tek geçiş 256 renk sınırında berbat görünüyor). Ortak "uygulamayı
  başlat / sekmede gez" mantığı `scripts/_preview.py`'ye alındı.
- **Ekran görüntüleri script'le alınıyor, elle değil.** `scripts/screenshot.py`
  playwright ile uygulamayı başlatıp gezerek `docs/screenshots/` altına yazıyor.
  Arayüz değiştikçe tek komutla yenilenebilsin diye. JPEG kullanılıyor: içerik
  ağırlıklı olarak fotoğraf, PNG gereksiz şişiyor (1.4 MB → 576 KB).
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
- **Arayüz metni değişirse görselleri yenile.** `docs/screenshots/`, `docs/demo.gif`
  ve `docs/comparison/` uygulamanın ekranını ve çıktılarını gösteriyor; metin
  değişince eskiyorlar. Üç komut: `scripts/screenshot.py`, `scripts/make_demo_gif.py`,
  `scripts/compare.py`. Metrik anahtarları değişirse `scripts/evaluate.py` de.
- **`is_deployed()` webcam sekmesini sunucuda gizliyor.** `cv2.VideoCapture(0)`
  uygulamanın *çalıştığı makinenin* kamerasını açıyor; sunucuda bu ziyaretçinin
  değil sunucunun kamerası olurdu. Sekme oluşturulmuyor bile — blok bir
  `if show_webcam:` altında, yoksa widget'lar ana sayfaya sızardı.
- **Docker imajı kendi kendine yeter.** Model ağırlıkları, örnekler ve metrikler
  imaja kopyalanıyor; konteyner ilk açılışta hiçbir şey indirmiyor. torch CPU
  deposundan kuruluyor (PyPI sürümü Linux'ta CUDA paketlerini de çekiyor).
- **Space, reponun kopyası değil.** `deploy/push_to_hf.sh` yalnızca uygulamanın
  çalışması için gerekenleri gönderiyor; eğitim scriptleri, testler ve veri
  setleri Space'e gitmiyor. Space'in README'si ayrı bir dosya
  (`deploy/space-README.md`) çünkü HF yapılandırmayı README frontmatter'ından
  okuyor ve bizim README'miz onu taşıyamaz.
- **Çizgi geçişi vektörel çarpımın işaretiyle bulunur.** Nesne merkezinin
  çizgiye göre tarafı iki kare arasında değiştiyse geçmiştir; işaretin yönü de
  giriş/çıkış ayrımını verir. Kesişim hesabı yapmaya gerek yok.

## Kod kuralları

- **Kod tabanının tamamı İngilizce.** Yorumlar, docstring'ler, test isimleri,
  değişken isimleri, arayüz metinleri, `metrics.json` anahtarları, Dockerfile
  ve CI yorumları — hepsi. Yeni kod yazarken Türkçe yorum ekleme.
- **Bu dosya (CLAUDE.md) ve README.tr.md Türkçe kalıyor.** Biri geliştirme
  günlüğü, diğeri Türkçe okuyanlar için; ikisi de kod değil.
- Docstring'ler *ne yaptığını* değil *neden öyle yaptığını* anlatsın.
- **İki README var:** `README.md` İngilizce (birincil, uluslararası başvurular
  için), `README.tr.md` Türkçe. İkisi de karşılıklı link veriyor. Bir şey
  değişince **ikisini birden** güncelle — özellikle test sayısı, kapsam yüzdesi
  ve metrikler gibi sayılar.
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

### ✅ M5 — Docker + dağıtım hattı (2026-08-28)

**Yapılanlar**
- `Dockerfile` — python:3.12-slim, torch CPU deposundan, root olmayan kullanıcı
  (uid 1000, HF Spaces gereği), healthcheck, 8501 portu. Model ağırlıkları,
  örnekler ve metrikler imaja gömülü.
- `.dockerignore` — veri setleri, `runs/`, testler ve scriptler imaja girmiyor.
- `docker-compose.yml` — tek komutla yerel çalıştırma, `outputs/` dışarı bağlı.
- `config.is_deployed()` — `DEPLOYED` veya HF'in eklediği `SPACE_ID` değişkenine
  bakıyor; `app.py` webcam sekmesini buna göre hiç oluşturmuyor.
- `deploy/space-README.md` — HF Space'in kendi README'si (frontmatter'da
  `sdk: docker`, `app_port: 8501`).
- `deploy/push_to_hf.sh` — Space'i klonlar, gerekli dosyaları kopyalar, push eder.
- CI'a `docker` işi: imajı derler, konteyneri başlatır, sağlık kontrolü yapar ve
  sunucu modunun gerçekten aktif olduğunu doğrular.
- `tests/test_config.py`'ye 5 test daha (`is_deployed` davranışı). Toplam 60 test.

**Doğrulandı**
- Yerel mod: 5 sekme, webcam var, alt başlıkta "canlı kamera" geçiyor.
- Sunucu modu (`DEPLOYED=1`): 4 sekme, webcam yok, alt başlık da değişiyor.
- Docker imajı **bu makinede test edilemedi** (docker kurulu değil); build ve
  smoke testi CI'da yapılıyor ve geçiyor: imaj derleniyor (~170 sn), konteyner
  ayağa kalkıyor, sağlık kontrolü yanıt veriyor, sunucu modu aktif.

**Yol boyunca düzeltilen**
- Webcam bloğunu ilk denemede `with tab_webcam if show_webcam else nullcontext():`
  ile sarmıştım. Bu blok içindeki kodu yine çalıştırıyor ve widget'lar ana sayfaya
  düşüyordu. `if show_webcam:` altına alındı.
- Bloğun girintisi artınca bir satır 100 karakteri aştı, ruff yakaladı.

**Bilinen eksikler / notlar**
- **Canlı demo henüz yayında değil** — HF hesabı ve Space'i Kaan'ın açması
  gerekiyor. Script ve yapılandırma hazır, README'de link için yer bırakıldı.
- Ücretsiz katmanda CPU ile video işleme yavaş olacak; uzun videolarda kullanıcı
  beklemek zorunda. İstenirse arayüze bir süre/boyut sınırı konabilir.
- İmaj **2.16 GB** (CI'da ölçüldü). Çoğu torch + ultralytics. Küçültmek için
  `opencv-python-headless`'a geçilebilir (o zaman `libgl1` de gerekmez) veya
  multi-stage build denenebilir; şimdilik değmez.
- `docker-compose.yml` içinde `DEPLOYED=1` sabit. Konteynerde webcam zaten
  çalışmayacağı için doğru, ama Linux'ta `--device /dev/video0` ile denenebilir.

---

### ✅ Arayüzü İngilizceye çevirme (2026-08-28)

M5 sonrası, README İngilizceye çevrildikten sonra gelen tamamlayıcı adım.

**Çevrilenler**
- `app.py` — bütün etiketler, yardım metinleri, mesajlar, buton ve sekme isimleri,
  indirilen dosya adları (`detected_*.png`, `tracking-data.csv`).
- `src/config.py` — model etiketleri (`YOLOv8n (fast)` vb.), `Ozel:` → `Custom:`
  (artık `CUSTOM_PREFIX` sabiti).
- `src/tracker.py` — çizgi yön isimleri (`asagi/yukari` → `down/up`,
  `saga/sola` → `right/left`), süre tablosu sütunları (`object`, `seconds`,
  `frames`, `first_frame`, `last_frame`), özet anahtarları (`total_objects`,
  `line`, `frames`). `line_from_ratio` artık `horizontal`/`vertical` alıyor.
- `src/video.py` — hata mesajları.
- `scripts/evaluate.py` — `metrics.json` anahtarları (`overall`, `per_class`,
  `class`) ve markdown başlıkları.
- `scripts/compare.py` — görsel üstündeki şeritler ("Pretrained COCO model" /
  "Our own model"), özet ızgara adı `ozet.jpg` → `summary.jpg`.
- `deploy/space-README.md` — "arayüz Türkçe" notu kaldırıldı.

**Yeniden üretilenler:** `docs/metrics.{json,md}`, `docs/comparison/*` (7 görsel),
`docs/screenshots/*` (`tespit.jpg` → `detection.jpg`,
`model-performansi.jpg` → `model-performance.jpg`), `docs/demo.gif`.

**Testler:** 65 test, hepsi yeni isimlere göre güncellendi. Çizgi yön mantığı
üç senaryoda tekrar doğrulandı (`right: 4`, `down: 4`, `left: 4`).

**Yol boyunca düzeltilen**
- Demo GIF'inde modele geçiş sırası değiştirildi. Önce yaban hayatı modeline
  geçip sonra metrik sekmesine gidince, model otobüs fotoğrafını yeniden
  değerlendirip "elephant 0.43" gibi alan dışı tespitler üretiyordu — doğru ama
  izleyene modelin kötü olduğunu düşündüren bir kare. Artık önce sekmeye geçilip
  sonra model değiştiriliyor.
- `metrics.json` anahtarları değişince `app.py`'nin okuduğu alanlar da
  değişmek zorundaydı; `evaluate.py` yeniden çalıştırılarak dosya üretildi.

**Kalan:** Kod yorumları ve CLAUDE.md hâlâ Türkçe. Uluslararası bir kod
incelemesi için yorumların da çevrilmesi gerekebilir — ayrı bir karar.

---

### ✅ Kod yorumlarını İngilizceye çevirme (2026-08-28)

Arayüz çevirisinin devamı. Artık **kod tabanında Türkçe metin kalmadı**.

**Çevrilenler**
- `src/` (4 dosya), `app.py` — bütün docstring'ler ve satır içi yorumlar
- `scripts/` (7 dosya) — docstring'ler, argparse yardım metinleri, konsol çıktısı
- `tests/` (5 dosya) — yorumlar, docstring'ler, yerel değişken isimleri ve
  **65 test fonksiyonunun tamamının adı** (`test_soldan_saga_gecis_saga_sayilir`
  → `test_left_to_right_counts_as_right`)
- `Dockerfile`, `docker-compose.yml`, `.gitignore`, `.dockerignore`,
  `pyproject.toml`, `requirements*.txt`, `.github/workflows/ci.yml`,
  `deploy/push_to_hf.sh`
- `notebooks/train_colab.ipynb` — markdown hücreleri ve kod yorumları

**Doğrulandı:** 65 test geçiyor, ruff temiz, altı script de `--help` ile
çalışıyor, `download_samples.py` gerçekten koştu. Türkçe kelime taraması
(regex ile 50+ Türkçe kelime/ek) sıfır sonuç veriyor.

**Çevrilmeyenler (bilerek):** bu dosya ve `README.tr.md`. İkisi de kod değil —
biri geliştirme günlüğü, diğeri Türkçe okuyanlar için README.

---

## Sıradaki milestone'lar

### 🔜 Sonraki adım — demoyu yayına al
HF hesabı aç, Docker SDK'sıyla bir Space oluştur, `./deploy/push_to_hf.sh` ile
gönder ve linki README'ye ekle. Kalan tek iş bu; altyapı hazır.

### 💡 Fikir havuzu (sıralı değil)
- CLI arayüzü (`python detect.py --image foo.jpg`) — batch işler için.
- Isı haritası / yoğunluk görselleştirmesi.
- Tespit sonuçlarını JSON olarak dışa aktarma (takip CSV'si M2'de eklendi).
- BoT-SORT seçeneği: uzun süre kaybolan nesneyi re-ID ile hatırlar.
- Kendi topladığın görsellerle ikinci bir veri seti (M3 altyapısı hazır).
- Daha büyük model (`yolov8s/m`) ile eğitim ve karşılaştırma.
- `streamlit.testing` ile arayüz testleri.
- Codecov entegrasyonu ve kapsam rozeti.
- Demoda video boyutu/süresi sınırı (ücretsiz CPU'yu korumak için).
- Takip modunun hareketli demosu — arayüz turu GIF'i var ama takip modu
  içinde yok. Ultralytics'in örnek videolarının hepsi ya kendi demo çıktıları
  (üzerinde başkasının kutuları basılı) ya da 1 saniyeden kısa. Telifsiz bir
  stok video ya da Kaan'ın kendi çektiği görüntü gerekiyor.
- Model karşılaştırma sekmesi: aynı görselde n/s/m sonuçları yan yana.
