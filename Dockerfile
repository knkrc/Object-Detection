# Object Detection — Streamlit uygulamasi
#
# Yerelde calistirmak icin:
#   docker build -t object-detection .
#   docker run -p 8501:8501 object-detection
#
# Imaj Hugging Face Spaces ile uyumlu: uygulama root olmayan bir kullanici
# (uid 1000) altinda calisiyor ve 8501 portunu dinliyor.

FROM python:3.12-slim

# opencv'nin ihtiyac duydugu sistem kutuphaneleri + saglik kontrolu icin curl
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces konteynerleri root olmayan kullaniciyla calisiyor; ayni sekilde kuruyoruz
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# PyPI'daki torch Linux'ta CUDA paketlerini de cekiyor (~2.5 GB).
# CPU deposundan kurmak imaji ciddi olcude kucultuyor.
RUN pip install --no-cache-dir --user \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Uygulama ve birlikte gelen varliklar. Model agirliklari imajda gomulu geliyor
# ki konteyner ilk acilista indirme beklemesin.
COPY --chown=user app.py .
COPY --chown=user src/ src/
COPY --chown=user models/ models/
COPY --chown=user samples/ samples/
COPY --chown=user docs/ docs/

# Sunucuda webcam sekmesi anlamsiz (kamera ziyaretcinin degil sunucunun olurdu),
# bu degisken onu gizliyor. Bkz. src/config.py: is_deployed()
ENV DEPLOYED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
