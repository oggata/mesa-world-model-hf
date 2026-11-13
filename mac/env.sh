# 共通の設定。setup.sh / download_model.sh / run.sh が読み込む (これだけを実行することはない)
#
# Python の環境・重み・DINOv2・一時ファイルは、すべて MESA_HOME の下に置く (Mac 本体の ~/.cache などは使わない)。
# MESA_HOME の既定はこのリポジトリの .local/。setup.sh に MESA_HOME=... を渡すと mac/.env に覚えておき、次からはそこを使う。

MAC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$MAC_DIR")"

# setup.sh / download_model.sh が書いた設定 (MESA_HOME, MODEL_DIR) を読む。環境変数で渡したものがあれば、そちらが優先
if [ -f "$MAC_DIR/.env" ]; then
  _mh="${MESA_HOME:-}"; _md="${MODEL_DIR:-}"
  . "$MAC_DIR/.env"
  [ -n "$_mh" ] && MESA_HOME="$_mh"
  [ -n "$_md" ] && MODEL_DIR="$_md"
fi
MESA_HOME="${MESA_HOME:-$REPO_DIR/.local}"

VENV="$MESA_HOME/venv"
PY="$VENV/bin/python"
export PIP_CACHE_DIR="$MESA_HOME/cache/pip"
export HF_HOME="$MESA_HOME/cache/huggingface"                        # Hugging Face から落とすもの
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}" # ログイン (hf auth login) はいつもの場所のものを使う
export TORCH_HOME="$MESA_HOME/cache/torch"                           # DINOv2
export MESA_WM_CACHE="$MESA_HOME/cache/mesa_wm"                      # 街を描く写真 (mesa-clouds-renderer の textures/)
export TMPDIR="$MESA_HOME/tmp"                                       # 作った GIF など
export GRADIO_TEMP_DIR="$MESA_HOME/tmp/gradio"
export PYTHONDONTWRITEBYTECODE=1                                     # app/ に __pycache__ を作らない
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$MESA_WM_CACHE" "$GRADIO_TEMP_DIR"

# mac/.env に KEY="値" を書く (同じ KEY があれば置き換える)
save_env() {
  local f="$MAC_DIR/.env"
  touch "$f"
  grep -v "^$1=" "$f" > "$f.tmp" || true
  printf '%s="%s"\n' "$1" "$2" >> "$f.tmp"
  mv "$f.tmp" "$f"
}
