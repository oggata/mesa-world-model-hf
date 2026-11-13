# MESA World Model

MESA の街を歩く住民の「目線」を学習した、小さな世界モデル (DINO-WM と同じ形) です。
いま見えている景色と行動 (前進 / 左 / 右 / 後退) だけから、**次に何が見えるか**を 1 歩ずつ想像します。
座標や地図は世界モデルに渡していません。

このリポジトリには、学習から手元の Mac で動かすまでの一式が入っています。

```
 1. Google Colab で学習する         colab/world_model_mesa.ipynb
        │  重み (safetensors) と config.json を書き出す
        ▼
 2. Hugging Face に置く             モデルリポジトリ (重み) + Space (ブラウザで試せるデモ。なくてもよい)
        │
        ▼
 3. Mac にインストールする          ./mac/setup.sh
        │
        ▼
 4. モデルを落とす                  ./mac/download_model.sh
        │
        ▼
 5. 動かす                          ./mac/run.sh  →  http://127.0.0.1:7860
```

すでに学習済みのモデル ([oggata/mesa-world-model-sharp-v01](https://huggingface.co/oggata/mesa-world-model-sharp-v01)) を使うだけなら、
**1 と 2 は飛ばして 3 から**始められます。

## フォルダの中身

| 場所 | 中身 |
| --- | --- |
| `colab/world_model_mesa.ipynb` | 学習のノートブック。データ作り・世界モデル・復元器・逆動力学・評価・計画・書き出しまで、これ 1 つ |
| `huggingface/model_card.md` | モデルリポジトリの説明 (Hugging Face では `README.md` として置く) |
| `app/` | シミュレーター (Gradio)。そのまま Hugging Face の Space として置ける。Mac でもこれを動かす |
| `app/mesa_wm/` | 世界モデル・描画器・住民 (NPC)・シミュレーターの中身 |
| `mac/` | Mac でのインストール・モデルのダウンロード・起動のスクリプト |

---

## 1. Google Colab で学習する

1. `colab/world_model_mesa.ipynb` を Google Colab で開く
   (GitHub に置いたあとなら `https://colab.research.google.com/github/<ユーザー名>/<リポジトリ名>/blob/main/colab/world_model_mesa.ipynb` で開ける)
2. ランタイムを **A100** にする (L4 でも動くが、`sharp` 以上はメモリと時間がきつい)
3. 「設定」のセルで学習の量を選ぶ

   | 設定 | 意味 |
   | --- | --- |
   | `PRESET` | `"fast"` (A100 で約 10 分・64×64) / `"sharp"` (40〜60 分・128×128) / `"sharper"` (1.5〜2.5 時間) |
   | `NPCS` | 街を歩くほかの住民の数。0 = いない。5 にすると、住民の位置もテンソルで世界モデルに渡して一緒に学習する |
   | `OVERRIDE` | 1 つだけ変えたいとき。例: `dict(WM_STEPS=40000)` |

4. 最後のセルの `HF_MODEL_REPO` に、上げ先のモデルリポジトリ (例: `"あなたのユーザー名/mesa-world-model"`) を書く。
   Colab の左の鍵のマーク (シークレット) に、Hugging Face の**書き込み用トークン**を `HF_TOKEN` として入れておく
5. 「すべてのセルを実行」。最初に Google Drive への接続を求められるので許可する

学習したものは Google Drive の `MyDrive/mesa_world_model/export_<PRESET>/` に書き出され、`HF_MODEL_REPO` を書いていれば
Hugging Face にも上がります。

| 書き出すファイル | 中身 |
| --- | --- |
| `world_model.safetensors` | 世界モデル (順動力学) |
| `decoder.safetensors` | 特徴 → 画像の復元器 |
| `inverse_dynamics.safetensors` | 逆動力学 (2 枚の景色の間の行動を当てる) |
| `feat_stats.safetensors` | DINOv2 の特徴を標準化する平均と標準偏差 |
| `config.json` | 大きさ・行動・描画・住民の設定。シミュレーターはここを読んで同じ条件で描く |

## 2. Hugging Face に置く

[hf コマンド](https://huggingface.co/docs/huggingface_hub/guides/cli) を使います (`pip install -U huggingface_hub` で入る)。
書き込み用のトークンでログインしておきます。

```bash
hf auth login
```

### モデルリポジトリ (重み)

重みは 1 の最後のセルで上がっています。説明 (モデルカード) を上げます。

```bash
hf upload あなたのユーザー名/mesa-world-model huggingface/model_card.md README.md
```

ノートブックで上げなかったときは、Google Drive から書き出しのフォルダを落としてきて、まとめて上げます。

```bash
hf upload あなたのユーザー名/mesa-world-model /path/to/export_sharp .
```

### Space (ブラウザで試せるデモ。なくてもよい)

`app/` がそのまま Space になります。起動したときに、`MODEL_REPO` に書いたモデルリポジトリから重みを取ってきます。

```bash
hf repos create あなたのユーザー名/mesa-world-model-sim --repo-type space --space-sdk gradio
```

```bash
hf upload あなたのユーザー名/mesa-world-model-sim app . --repo-type space --exclude "weights/*" --exclude "*/__pycache__/*" --exclude "._*" --exclude "*/._*"
```

```bash
hf spaces variables add あなたのユーザー名/mesa-world-model-sim -e MODEL_REPO=あなたのユーザー名/mesa-world-model
```

モデルリポジトリが非公開なら、Space の Settings → Secrets に、読み取りできるトークンを `HF_TOKEN` として入れます。
重みを上げ直すと、Space は再起動したときに新しい重みを使います (Space のコードは上げ直さなくてよい)。

## 3. Mac にインストールする

Apple Silicon の Mac で確かめています。Python 3.10〜3.12 が要ります (無ければ `brew install python@3.10`)。

```bash
./mac/setup.sh
```

Python の環境・重み・DINOv2・一時ファイルは、すべて 1 つのフォルダ (`MESA_HOME`) にまとめて置きます。
既定はこのリポジトリの `.local/` です。**Mac 本体の空きが少ないときは、外付けのドライブを指定**できます
(1 回指定すれば `mac/.env` に覚えておき、ほかのスクリプトもそこを使います)。

```bash
MESA_HOME=/Volumes/外付けの名前/mesa ./mac/setup.sh
```

外付けが exFAT でも動きます。中身は 2 GB ほどですが、exFAT は小さなファイルでも 1 つあたりの場所を大きく取る (torch は
ファイルが何千もある) ので、**exFAT のドライブでは 10 GB ほど**使います。APFS (Mac の普通の形式) なら 2〜3 GB です。

## 4. モデルを落とす

```bash
./mac/download_model.sh
```

引数を省くと、学習済みのモデル (`app/app.py` の `MODEL_REPO`) を落とします。自分で学習して上げたモデルを使うときは、リポジトリ名を渡します。

```bash
./mac/download_model.sh あなたのユーザー名/mesa-world-model
```

特徴を取り出す DINOv2 (約 85 MB) も、ここで一緒に落としておきます。

## 5. 動かす

```bash
./mac/run.sh
```

`Running on local URL: http://127.0.0.1:7860` と出たら、ブラウザで開きます。止めるときは Ctrl-C です。

- 画面いっぱいに映るのが**世界モデルの想像**です。左下の **W A S D** (キーボードの W A S D・矢印キーでも可) で歩きます。
  W 前進 / A 左 / S 後退 / D 右
- 右上のボタンでメニューが開きます。地図・いまの状態・街の選び直し・まとめて指示と、次の切り替えが入っています

  | 切り替え | 外す / 入れると |
  | --- | --- |
  | 右上の小窓に本物 (REAL) を出す | 入れると、本物の描画器の景色を小窓で重ねる (答え合わせ用) |
  | 毎回、本当に見えている景色から想像し直す | 外すと、想像の続きを想像し続ける (誤差が積もっていく様子が見える) |
  | 復元器に本物の手がかり画像を渡す | 外すと、壁の模様や色を本物から写さず、世界モデルが予測した特徴だけで絵にする |

Colab で書き出したフォルダを Hugging Face を通さずに直接使うこともできます。

```bash
MODEL_DIR=/path/to/export_sharp ./mac/run.sh
```

計算は CPU で行います (12 歩で数秒)。ポートを変えるときは `PORT=7861 ./mac/run.sh` です。

---

## しくみ (ひとこと)

```
目線の画像 ──DINOv2 (凍結)──> 特徴 z_t ─┐
                                        ├─ 世界モデル (Transformer) ─> 次の特徴 z_{t+1}   ← 想像
行動 a_t (前進/左/右/後退 の番号) ──────┘
z ── 復元器 ──> 画像                     ← 想像を目で見るためだけに使う
```

- 行動は文章 (プロンプト) ではなく**番号**で渡します。世界モデルの中で学習済みのベクトルに変わり、画像のトークンに足されます
- 想像は、直前に**本当に見た** 3 枚の特徴から始まり、そこからは自分の予測を次の入力にして進みます
- 住民 (NPC) ありで学習したモデルでは、住民の位置をシミュレーターが決め、「プレイヤーから見た相対位置」を数の並び (テンソル) で渡します

詳しくは `huggingface/model_card.md` を見てください。

## 困ったとき

| 症状 | 直し方 |
| --- | --- |
| `Python 3.10〜3.12 が見つかりません` | `brew install python@3.10` のあと、もう一度 `./mac/setup.sh` |
| `先に ./mac/setup.sh を実行してください` | 3 から順に実行する。`MESA_HOME` を変えたときは `mac/.env` の中身を確かめる |
| モデルを落とせない (401 / 404) | リポジトリ名を確かめる。非公開なら `hf auth login` |
| `address already in use` / 別のポートで起動した | すでに 7860 で動いているものがある。止めるか `PORT=7861 ./mac/run.sh` |
| 地図に住民の点が出ない | 学習で `NPCS = 0` だったモデル。住民ありで学習し直す |
