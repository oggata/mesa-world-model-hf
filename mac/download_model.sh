#!/bin/bash
# 4. 学習したモデルを Hugging Face から落とす。DINOv2 (特徴を取り出す部分) も一緒に落としておく。
#
#   ./mac/download_model.sh                        → app/app.py の MODEL_REPO (既定のモデル)
#   ./mac/download_model.sh ユーザー名/リポジトリ名   → 自分で学習して上げたモデル
#
# 落としたモデルの場所は mac/.env に覚えておき、run.sh はそれを使う。
# 拡散版 (config.json の model_type が "flow") のモデルは別の名前 (FLOW_MODEL_DIR) で覚えておき、run_flow.sh が使う。
# モデルリポジトリが非公開なら、先に hf auth login (読み取りできるトークン) が要る。
set -euo pipefail
. "$(dirname "$0")/env.sh"
[ -x "$PY" ] || { echo "先に ./mac/setup.sh を実行してください"; exit 1; }

REPO="${1:-}"
if [ -z "$REPO" ]; then
  REPO="$(sed -n 's/^MODEL_REPO = os.environ.get("MODEL_REPO", "\([^"]*\)").*/\1/p' "$REPO_DIR/app/app.py")"
fi
REPO="${REPO#https://huggingface.co/}"; REPO="${REPO%/}"
DEST="$MESA_HOME/models/$(echo "$REPO" | tr '/' '_')"
echo "モデル: $REPO → $DEST"

"$PY" - "$REPO" "$DEST" << 'EOF'
import os, sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1:]
path = snapshot_download(repo, local_dir=dest)
files = sorted(f for f in os.listdir(path) if not f.startswith("."))
print("  ", ", ".join(files))
if "config.json" not in files:
    sys.exit(f"{repo} に config.json がありません。Colab のノートブックの最後のセルで上げたリポジトリか確かめてください")

import torch
print("DINOv2 を落とします →", os.environ["TORCH_HOME"])
torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
EOF

echo
if grep -q '"model_type": *"flow"' "$DEST/config.json"; then
  # 拡散版 (world_model_mesa_flow.ipynb) のモデルは別に覚えておく (1 回で当てる版の MODEL_DIR は上書きしない)
  save_env FLOW_MODEL_DIR "$DEST"
  echo "落とせました (拡散版)。次はシミュレーターを起動します:"
  echo "  ./mac/run_flow.sh"
else
  save_env MODEL_DIR "$DEST"
  echo "落とせました。次はシミュレーターを起動します:"
  echo "  ./mac/run.sh"
fi
