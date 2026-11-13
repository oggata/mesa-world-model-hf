---
title: Mesa World Model Simulator
emoji: 🏙️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: "3.10"
app_file: app.py
pinned: false
short_description: Walk a Mesa town inside a DINO-WM style world model
---

# Mesa の街を、世界モデルの想像の中で歩く

`colab/world_model_mesa.ipynb` で学習した小さな世界モデル (DINO-WM と同じ形) を、
**前進 / 左 / 右 / 後退** の指示で動かすシミュレーターです。

- 1 回の指示で、数秒ぶん (6 歩 = 1 秒) の動画を作ります
- **大きく映るのが世界モデルの想像、右上の小窓 (REAL) が本物の描画器**です。想像は「いま見えている景色の DINOv2 の特徴」と
  行動だけから、次に見える景色を 1 歩ずつ予測し、復元器で画像に戻したものです (fast で学習したモデルは 64×64、sharp 以上は 128×128)。
  座標や地図は世界モデルに渡していません
- sharp 以上のモデルは大きいので、CPU の Space では 1 回の指示に fast の数倍の時間がかかる見込みです (測ってはいません)
- 「毎回、本当に見えている景色から想像し直す」を外すと、想像の続きを想像し続けます (誤差が積もっていく様子が見えます)
- 「復元器に本物の手がかり画像を渡す」を外すと、壁の模様や色を本物から写さず、世界モデルが予測した特徴だけで絵にします
  (sharp 以上の重みのとき。想像がどこまで世界モデル自身の力かを確かめられます)

## 重み

モデルの重みは別のモデルリポジトリに置き、起動時に取ってきます。
Space の **Settings → Variables** に `MODEL_REPO` (例: `あなたのユーザー名/mesa-world-model`) を入れるか、
`app.py` の `MODEL_REPO` を書き換えてください。

## ハードウェア

Space の **Settings → Space hardware** は、次のどれでも動きます。

- **CPU basic** (無料): いちばん手軽。sharp 以上のモデルだと 1 回の指示に時間がかかる
- **ZeroGPU**: ボタンを押したときだけ GPU を借りて計算する (`app.py` の `gpu` で包んだ関数)
- GPU 付きのハードウェア: 最初から GPU で計算する

## 中身

| ファイル | 役割 |
| --- | --- |
| `app.py` | Gradio の画面 |
| `mesa_wm/world.py` | 街の地図 (ノートブックと同じ seed なら同じ街) |
| `mesa_wm/render.py` | 目線の描画 (mesa-clouds-renderer に寄せたレイキャスタ。テクスチャは GitHub から取得) |
| `mesa_wm/models.py` | 世界モデル・復元器・逆動力学・DINOv2 |
| `mesa_wm/sim.py` | 想像と本物の実行、地図の絵、GIF |

ノートブックの地図・描画・モデルの形を変えたら、`mesa_wm/` も同じように直してください (重みのキーと見え方がずれると、想像が崩れます)。

CPU (無料の CPU basic) で動きます。1 回の指示 (12 歩) で数秒かかります。
