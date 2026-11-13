"""Mesa の街を、世界モデルの「想像の中」で歩くシミュレーター (Gradio)。

前進 / 左 / 右 / 後退 のボタンか、「前進 10, 左 5」のような指示で、数秒ぶんの動画を作る。
画面いっぱいに映るのが世界モデルの想像 (DINOv2 の特徴を予測し、復元器で画像に戻したもの。元の大きさは学習の設定による: fast は 64×64、sharp 以上は 128×128)。
左下の W A S D (キーボードでも可) で歩く。地図・本物 (REAL) の小窓・細かい設定は、右のメニューにしまってある。

重みの場所:
  ・このフォルダに weights/config.json があればそれを使う (手元で試すとき)
  ・無ければ環境変数 MODEL_REPO の Hugging Face のモデルリポジトリから取ってくる
"""
import functools
import os
import tempfile

import gradio as gr
import torch

from mesa_wm import Simulator, parse_commands, save_gif

MODEL_REPO = os.environ.get("MODEL_REPO", "oggata/mesa-world-model-sharp-v01")   # ← 自分のモデルリポジトリに書き換える
# URL ごと貼ったときや、前後に空白が入ったときも「ユーザー名/リポジトリ名」に直す
MODEL_REPO = MODEL_REPO.strip().removeprefix("https://huggingface.co/").removeprefix("huggingface.co/").strip("/")
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.environ.get("MODEL_DIR", os.path.join(HERE, "weights"))
FPS = 6

if os.path.exists(os.path.join(LOCAL, "config.json")):
    model_dir = LOCAL
else:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError
    token = os.environ.get("HF_TOKEN") or None   # モデルリポジトリが非公開のときは、Space のシークレットに HF_TOKEN が要る
    print(f"モデルを取ってきます: {MODEL_REPO} (HF_TOKEN: {'あり' if token else 'なし'})")
    try:
        model_dir = snapshot_download(MODEL_REPO, token=token)
    except RepositoryNotFoundError as e:
        raise SystemExit(
            f"モデルリポジトリ {MODEL_REPO!r} を読めません。次を確かめてください:\n"
            f"  1. Space の Settings → Variables の MODEL_REPO が「ユーザー名/リポジトリ名」と完全に一致しているか (大文字・小文字も)\n"
            f"  2. モデルリポジトリが非公開なら、Space の Settings → Secrets に、読み取りできるトークンを HF_TOKEN として入れたか\n"
            f"     (または、モデルリポジトリを公開にする)\n"
            f"  3. モデルリポジトリの一番上に config.json などの 5 ファイルがあるか"
        ) from e

# ── どこで計算するか ──
#   ・ZeroGPU の Space: 起動時は CPU に置き、ボタンを押したときだけ @spaces.GPU の中で GPU に移して計算する
#     (ZeroGPU は @spaces.GPU の関数が 1 つも無いと起動しない)
#   ・GPU 付きの Space / 手元の GPU: 最初から GPU
#   ・CPU の Space: CPU のまま
try:
    import spaces
    ZERO_GPU = bool(os.environ.get("SPACES_ZERO_GPU"))
except ImportError:
    spaces, ZERO_GPU = None, False

torch.set_num_threads(max(1, os.cpu_count() or 1))
DEVICE = "cuda" if torch.cuda.is_available() and not ZERO_GPU else "cpu"
print(f"計算する場所: {'ZeroGPU (ボタンを押したときだけ GPU)' if ZERO_GPU else DEVICE}")
SIM = Simulator(model_dir, DEVICE)


def gpu(fn):
    """fn を @spaces.GPU で包む。spaces が入っていなければそのまま返す。
    @spaces.GPU は ZeroGPU 以外では何もしない。GPU への出し入れは ZeroGPU のときだけ行う"""
    if spaces is None:
        return fn

    @spaces.GPU(duration=60)
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if not ZERO_GPU:
            return fn(*args, **kwargs)
        SIM.to("cuda")
        try:
            return fn(*args, **kwargs)
        finally:
            SIM.to("cpu")   # GPU は関数を抜けると返すので、中身は CPU に戻しておく
    return wrapped
ACTIONS = SIM.actions
TEST_SEEDS = SIM.cfg["maps"]["test_seeds"]
TRAIN_SEEDS = SIM.cfg["maps"]["train_seeds"]


def info(state, extra=""):
    x, y, th = state["poses"][-1]
    seen = "学習で見た街" if state["seed"] in TRAIN_SEEDS else "学習で見ていない街"
    return f"街 seed={state['seed']} ({seen}) / 位置 ({x:.1f}, {y:.1f}) / 向き {th * 57.3 % 360:.0f}°" + (f"\n{extra}" if extra else "")


def _start(seed, start_seed, hint=True, show_real=False):
    st = SIM.reset(int(seed), int(start_seed))
    real = st["last_img"].permute(1, 2, 0).numpy()
    imag = SIM.decode(st["z_real"][-1:].to(SIM.device), st["last_img"], hint).permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()
    path = save_gif([real], [imag[0]], tempfile.mktemp(suffix=".gif"), FPS, show_real=show_real)
    return st, path, SIM.minimap(st), info(st, "スタートしました。W A S D (キーボードでも可) で歩いてください。")


start = gpu(_start)


@gpu
def walk(st, acts, sync, hint=True, show_real=False):
    if st is None:
        st, *_ = _start(TEST_SEEDS[0], 0, hint, show_real)   # walk の中ですでに GPU を借りているので、包んでいない方を呼ぶ
    if not acts:
        return st, None, SIM.minimap(st), info(st, "歩数が 0 です")
    real, imag, st = SIM.run(st, acts, sync=sync, hint=hint)
    path = save_gif(real, imag, tempfile.mktemp(suffix=".gif"), FPS, show_real=show_real)
    sec = len(acts) / FPS
    return st, path, SIM.minimap(st), info(st, f"{len(acts)} 歩 ({sec:.1f} 秒) を想像しました")


def press(name):
    def f(st, steps, sync, hint, show_real):
        return walk(st, [ACTIONS.index(name)] * int(steps), sync, hint, show_real)
    return f


def run_text(st, text, steps, sync, hint, show_real):
    try:
        acts = parse_commands(text, ACTIONS, int(steps))
    except ValueError as e:
        raise gr.Error(str(e))
    return walk(st, acts, sync, hint, show_real)


# ── 見た目: 想像を横いっぱいに出し、左下に W A S D。ほかの操作は右のメニュー (サイドバー) にしまう ──
KEYS = (("W", "前進"), ("A", "左"), ("S", "後退"), ("D", "右"))   # キー → 行動の名前
CSS = """
#stage { position: relative; background: #000; border-radius: 14px; overflow: hidden; padding: 0 !important; border: none !important; }
#view, #view > div, #view .image-container, #view .image-frame { background: #000; border: none !important; border-radius: 0; height: auto !important; min-height: 0 !important; }
#view .image-frame, #view button { width: 100% !important; }
#view img { display: block; width: 100% !important; height: auto !important; max-height: calc(100vh - 32px); object-fit: contain; margin: 0 auto; }
#wasd { position: absolute; left: 20px; bottom: 20px; z-index: 20; width: auto !important; min-width: 0 !important;
        flex: none !important; gap: 6px !important; background: transparent; border: none; padding: 0 !important; }
#wasd .row { gap: 6px !important; flex-wrap: nowrap !important; justify-content: center; background: transparent; border: none; }
#wasd button.key { flex: none !important; width: 44px !important; min-width: 44px !important; height: 44px; padding: 0 !important;
        border-radius: 9px; font-weight: 600; font-size: 15px; color: #e8e8e8 !important;
        background: rgba(70, 70, 70, 0.55) !important; border: 1px solid rgba(255, 255, 255, 0.14) !important;
        backdrop-filter: blur(6px); box-shadow: none !important; }
#wasd button.key:hover { background: rgba(110, 110, 110, 0.65) !important; }
#wasd button.key.pressed, #wasd button.key:active { background: rgba(255, 255, 255, 0.40) !important; color: #111 !important; }
#wasd .spacer { width: 44px; height: 44px; flex: none; }
"""
# キーボードの W A S D と矢印キーでも歩けるようにする (文字を打つ欄にいるときは何もしない)
JS = """
() => {
  const ids = {w: "key-W", arrowup: "key-W", a: "key-A", arrowleft: "key-A",
               s: "key-S", arrowdown: "key-S", d: "key-D", arrowright: "key-D"};
  document.addEventListener("keydown", (e) => {
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
    const id = ids[e.key.toLowerCase()];
    const b = id && document.getElementById(id);
    if (!b) return;
    e.preventDefault();
    b.click();
    b.classList.add("pressed");
    setTimeout(() => b.classList.remove("pressed"), 150);
  });
}
"""

with gr.Blocks(title="Mesa world model simulator", fill_width=True) as demo:
    state = gr.State(None)
    with gr.Column(elem_id="stage"):
        video = gr.Image(type="filepath", show_label=False, container=False, buttons=[], interactive=False, elem_id="view")
        with gr.Column(elem_id="wasd"):
            keys = {}
            with gr.Row():
                keys["W"] = gr.Button("W", elem_id="key-W", elem_classes="key", min_width=44)
            with gr.Row():
                for k in "ASD":
                    keys[k] = gr.Button(k, elem_id=f"key-{k}", elem_classes="key", min_width=44)

    with gr.Sidebar(label="メニュー", open=False, position="right", width=380):
        gr.Markdown(
            "### Mesa の街を、世界モデルの想像の中で歩く\n"
            "画面は世界モデルの**想像**です。いま見えている景色の特徴 (DINOv2) と行動だけから、次に見える景色を 1 歩ずつ予測して"
            "画像に戻しています (座標や地図は渡していません)。\n\n"
            "**W** 前進 / **A** 左 / **S** 後退 / **D** 右 (キーボードの W A S D・矢印キーでも歩けます)")
        status = gr.Textbox(label="いまの状態", lines=2, interactive=False)
        mini = gr.Image(label="地図 (赤 = 歩いた道、黄 = いまの位置と向き、色の点 = ほかの住民)", type="pil", height=240,
                        buttons=[])
        show_real = gr.Checkbox(value=False, label="右上の小窓に本物 (REAL) を出す (次に歩いたときから)")
        steps = gr.Slider(1, 60, value=12, step=1, label=f"1 回の歩数 ({FPS} 歩 = 1 秒)")
        sync = gr.Checkbox(value=True, label="毎回、本当に見えている景色から想像し直す (外すと夢の続きを歩く)")
        hint = gr.Checkbox(value=True, interactive=SIM.has_hint,
                           label="復元器に本物の手がかり画像を渡す (外すと、世界モデルが予測した特徴だけで絵にする)"
                                 + ("" if SIM.has_hint else " ※ この重みの復元器は手がかりを使わない形です"))
        seed = gr.Dropdown(TEST_SEEDS + TRAIN_SEEDS, value=TEST_SEEDS[0], label="街 (seed)")
        start_seed = gr.Number(value=0, precision=0, label="立つ場所 (乱数)")
        btn_start = gr.Button("この街に立つ", variant="secondary")
        text = gr.Textbox(label="まとめて指示", value="前進 12, 左 6, 前進 12, 右 6, 後退 6", lines=2)
        btn_text = gr.Button("この順に歩く")

    outs = [state, video, mini, status]
    btn_start.click(start, [seed, start_seed, hint, show_real], outs, show_progress="minimal")
    for k, name in KEYS:
        if name in ACTIONS:                           # 後退の無い古い重みでは S は何もしない
            keys[k].click(press(name), [state, steps, sync, hint, show_real], outs, show_progress="minimal")
    btn_text.click(run_text, [state, text, steps, sync, hint, show_real], outs, show_progress="minimal")
    demo.load(start, [seed, start_seed, hint, show_real], outs)

if __name__ == "__main__":
    demo.queue(max_size=16).launch(css=CSS, js=JS)
