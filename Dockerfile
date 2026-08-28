# Object Detection — Streamlit app
#
# To run it locally:
#   docker build -t object-detection .
#   docker run -p 8501:8501 object-detection
#
# The image is Hugging Face Spaces compatible: the app runs as a non-root user
# (uid 1000) and listens on port 8501.

FROM python:3.12-slim

# System libraries opencv needs, plus curl for the healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces containers run as a non-root user; set the image up the same way
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# The PyPI torch build pulls CUDA packages on Linux (~2.5 GB).
# Installing from the CPU index shrinks the image considerably.
RUN pip install --no-cache-dir --user \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# The app and the assets shipped with it. Model weights are baked in so the
# container does not have to download anything on first start.
COPY --chown=user app.py .
COPY --chown=user src/ src/
COPY --chown=user models/ models/
COPY --chown=user samples/ samples/
COPY --chown=user docs/ docs/

# The webcam tab is meaningless on a server (the camera would be the server's,
# not the visitor's); this variable hides it. See src/config.py: is_deployed()
ENV DEPLOYED=1

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
