#!/bin/bash
# 拡散 (フローマッチング) 版のシミュレーターを起動する。起動したらブラウザで http://127.0.0.1:7860 を開く (止めるときは Ctrl-C)。
# 1 回で当てる版 (run.sh) とは別のモデルを使う。
#
#   ./mac/run_flow.sh                                       → download_model.sh で落とした拡散版のモデル (FLOW_MODEL_DIR)
#   MODEL_DIR=/path/to/export_flow_sharp ./mac/run_flow.sh  → Colab の書き出し (Google Drive の export_flow_<PRESET>/) を直接使う
#   MODEL_REPO=ユーザー名/リポジトリ名 ./mac/run_flow.sh       → 起動のときに Hugging Face から取ってくる
#   PORT=7861 ./mac/run_flow.sh                             → ポートを変える (run.sh と同時に動かすとき)
#
# 計算は CPU で行う。1 歩ごとにノイズを何回も取り除くので、1 回で当てる版より遅い (メニューの「ノイズを取り除く回数」で変えられる)。
set -euo pipefail
_user_md="${MODEL_DIR:-}"                 # env.sh は mac/.env の MODEL_DIR (1 回で当てる版) を読むので、渡されたものを先に取っておく
. "$(dirname "$0")/env.sh"
[ -x "$PY" ] || { echo "先に ./mac/setup.sh を実行してください"; exit 1; }

MODEL_DIR="${_user_md:-${FLOW_MODEL_DIR:-}}"
if [ -n "$MODEL_DIR" ]; then
  [ -f "$MODEL_DIR/config.json" ] || { echo "$MODEL_DIR に config.json がありません"; exit 1; }
  export MODEL_DIR
  echo "モデル: $MODEL_DIR"
else
  unset MODEL_DIR
  [ -n "${MODEL_REPO:-}" ] || { echo "拡散版のモデルがありません。./mac/download_model.sh ユーザー名/リポジトリ名 で落とすか、MODEL_DIR=... を付けてください"; exit 1; }
  export MODEL_REPO
fi
export GRADIO_SERVER_PORT="${PORT:-7860}"

cd "$REPO_DIR/app"
exec "$PY" app_flow.py
