#!/bin/bash
# 5. シミュレーターを起動する。起動したらブラウザで http://127.0.0.1:7860 を開く (止めるときは Ctrl-C)。
#
#   ./mac/run.sh                                     → download_model.sh で落としたモデルを使う
#   MODEL_DIR=/path/to/export_sharp ./mac/run.sh     → Colab の書き出し (Google Drive の export_<PRESET>/) を直接使う
#   PORT=7861 ./mac/run.sh                           → ポートを変える
#
# 計算は CPU で行う (Mac の GPU は使わない)。1 回の指示 (12 歩) に数秒かかる。
set -euo pipefail
. "$(dirname "$0")/env.sh"
[ -x "$PY" ] || { echo "先に ./mac/setup.sh を実行してください"; exit 1; }

if [ -n "${MODEL_DIR:-}" ]; then
  [ -f "$MODEL_DIR/config.json" ] || { echo "$MODEL_DIR に config.json がありません"; exit 1; }
  export MODEL_DIR
  echo "モデル: $MODEL_DIR"
else
  echo "モデルを落としていないので、起動のときに Hugging Face (MODEL_REPO) から取ってきます"
fi
export GRADIO_SERVER_PORT="${PORT:-7860}"

cd "$REPO_DIR/app"
exec "$PY" app.py
