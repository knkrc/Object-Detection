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
# Clear out whatever the previous version left behind (except .git)
find . -mindepth 1 -maxdepth 1 -not -name .git -exec rm -rf {} +

cp "$ROOT/Dockerfile" "$ROOT/requirements.txt" "$ROOT/app.py" .
cp -r "$ROOT/src" "$ROOT/models" "$ROOT/samples" "$ROOT/docs" .
# The Space's own README: HF reads its configuration from the frontmatter
cp "$ROOT/deploy/space-README.md" README.md

# Keep training output and compiled files out
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

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
