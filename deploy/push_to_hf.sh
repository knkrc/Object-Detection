#!/usr/bin/env bash
# Pushes the app to Hugging Face Spaces.
#
# Prerequisites:
#   1. Create an account on huggingface.co and a Space with the "Docker" SDK
#   2. Generate a token with write access: https://huggingface.co/settings/tokens
#   3. export HF_TOKEN=hf_...
#
# Usage:
#   ./deploy/push_to_hf.sh <your-username>/<space-name>
#
# The Space is not a copy of this repo: only the files the app needs to run are
# pushed (training scripts, tests and datasets are left out).
#
# The Space runs on the Streamlit SDK, not Docker — HF only offers Docker Spaces
# on a paid plan. So the Dockerfile is not pushed; HF installs
# space-requirements.txt and space-packages.txt and runs app.py itself. The
# Dockerfile still works for running the app locally or self-hosting.

set -euo pipefail

SPACE="${1:-}"
if [[ -z "$SPACE" ]]; then
    echo "Usage: $0 <your-username>/<space-name>" >&2
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set. Generate a token with write access at" >&2
    echo "https://huggingface.co/settings/tokens and run 'export HF_TOKEN=hf_...'." >&2
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "-> Cloning the Space: $SPACE"
git clone "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE}" "$WORK/space"

echo "-> Copying files"
cd "$WORK/space"
# Clear out whatever the previous version left behind. .gitattributes is kept:
# HF puts the LFS patterns there, and deleting it makes the push bounce off the
# "binary files must use Xet storage" hook.
find . -mindepth 1 -maxdepth 1 -not -name .git -not -name .gitattributes -exec rm -rf {} +

cp "$ROOT/app.py" .
cp -r "$ROOT/src" "$ROOT/models" "$ROOT/samples" "$ROOT/docs" .

# The Space's own README: HF reads its configuration from the frontmatter
cp "$ROOT/deploy/space-README.md" README.md
# Space-specific dependency files (CPU torch, apt packages for opencv)
cp "$ROOT/deploy/space-requirements.txt" requirements.txt
cp "$ROOT/deploy/space-packages.txt" packages.txt

# The app reads docs/metrics.json, docs/comparison and docs/plots for its
# performance tab. The demo GIF and screenshots are README-only, so drop them
# rather than shipping ~3 MB the Space never serves.
rm -rf docs/screenshots docs/demo.gif

# Keep compiled files and macOS clutter out
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
find . -name .DS_Store -delete 2>/dev/null || true

# HF rejects binary files that are not in LFS. Its default .gitattributes covers
# weights (*.pt) but not images, and our screenshots and plots are well over the
# size threshold. Tracking has to happen before `git add`, or the files go in as
# raw blobs and the push is rejected.
git lfs install --local >/dev/null
git lfs track "*.jpg" "*.jpeg" "*.png" "*.gif" >/dev/null

echo "-> Pushing"
git add -A
if git diff --cached --quiet; then
    echo "No changes, nothing to push."
    exit 0
fi
git -c user.email="deploy@local" -c user.name="deploy" \
    commit -q -m "Update the Object Detection app"
git push

echo
echo "Done: https://huggingface.co/spaces/${SPACE}"
echo "The first build takes a few minutes; watch the logs on the Space page."
