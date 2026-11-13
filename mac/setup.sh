#!/bin/bash
# 3. Mac にインストール (最初の 1 回だけ)。Python の環境を作り、torch・gradio などを入れる。
#
#   ./mac/setup.sh                                  → このリポジトリの .local/ に入れる
#   MESA_HOME=/Volumes/外付け/mesa ./mac/setup.sh    → 外付けなど好きな場所に入れる (次からは mac/.env に覚えておく)
#   PYTHON=/path/to/python3.11 ./mac/setup.sh       → 使う Python を指定する (3.10〜3.12)
#
# 外付けが exFAT でも動くように、Python の環境はシンボリックリンクを使わない形 (venv --copies) で作る。
set -euo pipefail
. "$(dirname "$0")/env.sh"

mkdir -p "$MESA_HOME"
MESA_HOME="$(cd "$MESA_HOME" && pwd)"
save_env MESA_HOME "$MESA_HOME"
. "$(dirname "$0")/env.sh"                    # 絶対パスにした MESA_HOME で、置き場所を決め直す
echo "置き場所 (MESA_HOME): $MESA_HOME"

# ── Python 3.10〜3.12 を探す (Hugging Face の Space は 3.10) ──
find_python() {
  for p in "${PYTHON:-}" python3.10 python3.11 python3.12 \
           /opt/homebrew/bin/python3.10 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.12 \
           /usr/local/bin/python3.10 /usr/local/bin/python3.11 /usr/local/bin/python3.12 python3; do
    [ -n "$p" ] && command -v "$p" > /dev/null 2>&1 || continue
    if "$p" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' 2> /dev/null; then
      command -v "$p"; return 0
    fi
  done
  return 1
}

if [ ! -x "$PY" ]; then
  BASE="$(find_python)" || {
    echo "Python 3.10〜3.12 が見つかりません。Homebrew で入れてから、もう一度実行してください:"
    echo "  brew install python@3.10"
    exit 1
  }
  echo "Python の環境を作ります: $BASE → $VENV"
  "$BASE" -m venv --copies "$VENV"
fi

echo "必要なものを入れます (torch, gradio など。初回は数分かかります)"
"$PY" -m pip install --disable-pip-version-check -r "$MAC_DIR/requirements.txt"
"$PY" -c 'import torch, gradio; print(f"torch {torch.__version__} / gradio {gradio.__version__}")'

echo
echo "インストールできました ($(du -sh "$MESA_HOME" | cut -f1))。次はモデルを落とします:"
echo "  ./mac/download_model.sh"
