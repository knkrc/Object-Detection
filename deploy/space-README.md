---
title: Object Detection
emoji: 🎯
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# 🎯 Object Detection

YOLOv8 ile nesne tespiti ve takibi. [Kaynak kod ve detaylar](https://github.com/knkrc/Object-Detection).

- **Resim** — bir görsel yükle, tespit edilen nesneleri kutularla gör
- **Video** — MP4 yükle, kare kare işle, işlenmiş videoyu indir
- **Takip** — her nesneye kalıcı ID, benzersiz sayım, çizgi geçişi, hareket izi
- **Örnekler** — dosya yüklemeden hemen dene
- **Model performansı** — kendi eğittiğimiz modelin metrikleri ve önce/sonra karşılaştırması

İki model var: hazır COCO modeli (80 sınıf) ve African Wildlife veri setiyle
fine-tune ettiğimiz model (buffalo, elephant, rhino, zebra — mAP50 0.957).
Kenar çubuğundan seçilebilir.

> Webcam sekmesi burada yok: sunucuda kamera açmak ziyaretçinin değil sunucunun
> kamerasını açardı. Repoyu yerelde çalıştırırsan o sekme de geliyor.
