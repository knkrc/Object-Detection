#!/usr/bin/env bash
# Uygulamayi Hugging Face Spaces'e gonderir.
#
# Onkosullar:
#   1. huggingface.co'da bir hesap ac ve "Docker" SDK ile bir Space olustur
#   2. Write yetkili bir token uret: https://huggingface.co/settings/tokens
#   3. export HF_TOKEN=hf_...
#
# Kullanim:
#   ./deploy/push_to_hf.sh <kullanici-adin>/<space-adi>
#
# Space, bu reponun bir kopyasi degil: sadece uygulamanin calismasi icin
# gereken dosyalar gonderiliyor (egitim scriptleri, testler, veri setleri haric).

set -euo pipefail

SPACE="${1:-}"
if [[ -z "$SPACE" ]]; then
    echo "Kullanim: $0 <kullanici-adin>/<space-adi>" >&2
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN ayarli degil. https://huggingface.co/settings/tokens adresinden" >&2
    echo "write yetkili bir token uretip 'export HF_TOKEN=hf_...' yap." >&2
    exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "-> Space klonlaniyor: $SPACE"
git clone "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE}" "$WORK/space"

echo "-> Dosyalar kopyalaniyor"
cd "$WORK/space"
# Onceki surumden kalanlari temizle (.git haric)
find . -mindepth 1 -maxdepth 1 -not -name .git -exec rm -rf {} +

cp "$ROOT/Dockerfile" "$ROOT/requirements.txt" "$ROOT/app.py" .
cp -r "$ROOT/src" "$ROOT/models" "$ROOT/samples" "$ROOT/docs" .
# Space'in kendi README'si: HF yapilandirmasi bu dosyanin frontmatter'inda
cp "$ROOT/deploy/space-README.md" README.md

# Egitim ciktilari ve derlenmis dosyalar gitmesin
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

echo "-> Gonderiliyor"
git add -A
if git diff --cached --quiet; then
    echo "Degisiklik yok, gonderilecek bir sey de yok."
    exit 0
fi
git -c user.email="deploy@local" -c user.name="deploy" \
    commit -q -m "Object Detection uygulamasini guncelle"
git push

echo
echo "Bitti: https://huggingface.co/spaces/${SPACE}"
echo "Ilk derleme birkac dakika surer; Space sayfasindan loglari izleyebilirsin."
