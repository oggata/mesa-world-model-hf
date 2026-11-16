"""Mesa の街を、拡散 (フローマッチング) 版の世界モデルの「想像の中」で歩くシミュレーター (Gradio)。

colab/world_model_mesa_flow.ipynb で学習したモデル (config.json の model_type が "flow") を動かす。
1 回で当てる版 (world_model_mesa.ipynb) のモデルは app.py で動かす。

使い方は app.py と同じ (左下の W A S D で歩く。ほかの操作は右のメニュー)。拡散版ならではの操作:
  ・右下の「別の未来」(キーボードの R): 直前の歩きを、同じ場所・同じ行動のまま、乱数だけ変えて描き直す
  ・メニューの「ノイズを取り除く回数」「ゆらぎ」「乱数」: 1 歩の描き方を変える

重みの場所:
  ・環境変数 MODEL_DIR (Colab の書き出し export_flow_<PRESET>/ など)
  ・無ければ、このフォルダの weights_flow/
  ・それも無ければ、環境変数 MODEL_REPO の Hugging Face のモデルリポジトリから取ってくる
"""
import functools
import os
import random
import tempfile

import gradio as gr
import torch

from mesa_wm import parse_commands, save_gif
from mesa_wm.sim_flow import FlowSimulator

MODEL_REPO = os.environ.get("MODEL_REPO", "")     # 例: "あなたのユーザー名/mesa-world-model-flow"
MODEL_REPO = MODEL_REPO.strip().removeprefix("https://huggingface.co/").removeprefix("huggingface.co/").strip("/")
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.environ.get("MODEL_DIR", os.path.join(HERE, "weights_flow"))
FPS = 6

if os.path.exists(os.path.join(LOCAL, "config.json")):
    model_dir = LOCAL
elif MODEL_REPO:
    from huggingface_hub import snapshot_download
    print(f"モデルを取ってきます: {MODEL_REPO}")
    model_dir = snapshot_download(MODEL_REPO, token=os.environ.get("HF_TOKEN") or None)
else:
    raise SystemExit("拡散版のモデルが見つかりません。次のどれかで場所を教えてください:\n"
                     "  ・MODEL_DIR=/path/to/export_flow_sharp (Colab の書き出しのフォルダ)\n"
                     "  ・MODEL_REPO=ユーザー名/リポジトリ名 (Hugging Face に上げたもの)\n"
                     f"  ・{os.path.join(HERE, 'weights_flow')} に書き出しの中身を置く")

try:
    import spaces
    ZERO_GPU = bool(os.environ.get("SPACES_ZERO_GPU"))
except ImportError:
    spaces, ZERO_GPU = None, False

torch.set_num_threads(max(1, os.cpu_count() or 1))
DEVICE = "cuda" if torch.cuda.is_available() and not ZERO_GPU else "cpu"
print(f"計算する場所: {'ZeroGPU (ボタンを押したときだけ GPU)' if ZERO_GPU else DEVICE}")
SIM = FlowSimulator(model_dir, DEVICE)


def gpu(fn):
    """fn を @spaces.GPU で包む (app.py と同じ)。spaces が入っていなければそのまま返す"""
    if spaces is None:
        return fn

    @spaces.GPU(duration=120)
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if not ZERO_GPU:
            return fn(*args, **kwargs)
        SIM.to("cuda")
        try:
            return fn(*args, **kwargs)
        finally:
            SIM.to("cpu")
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
    return st, None, path, SIM.minimap(st), info(st, "スタートしました。W A S D (キーボードでも可) で歩いてください。")


start = gpu(_start)


def _walk(st, acts, sync, hint, show_real, fsteps, temp, seed):
    """st から acts を歩いて描く。返り値の last = (歩く前の state, acts) は「別の未来」で描き直すのに使う"""
    real, imag, new = SIM.run(st, acts, sync=sync, hint=hint, steps=fsteps, temp=temp, seed=seed)
    path = save_gif(real, imag, tempfile.mktemp(suffix=".gif"), FPS, show_real=show_real)
    msg = (f"{len(acts)} 歩 ({len(acts) / FPS:.1f} 秒) を想像しました。乱数 {seed} / ノイズを取り除く回数 {int(fsteps)} / "
           f"ゆらぎ {float(temp):.2f}")
    return new, (st, list(acts)), path, SIM.minimap(new), info(new, msg)


@gpu
def walk(st, acts, sync, hint, show_real, fsteps, temp, rand, seed):
    if st is None:
        st, *_ = _start(TEST_SEEDS[0], 0, hint, show_real)
    if not acts:
        return st, None, None, SIM.minimap(st), info(st, "歩数が 0 です")
    return _walk(st, acts, sync, hint, show_real, fsteps, temp, random.randrange(10 ** 6) if rand else int(seed))


@gpu
def again(st, last, sync, hint, show_real, fsteps, temp):
    """直前の歩きを、同じ場所・同じ行動のまま、乱数だけ変えて描き直す"""
    if not last:
        raise gr.Error("まだ歩いていません。先に W A S D で歩いてください")
    prev, acts = last
    return _walk(prev, acts, sync, hint, show_real, fsteps, temp, random.randrange(10 ** 6))


def press(name):
    def f(st, steps, sync, hint, show_real, fsteps, temp, rand, seed):
        return walk(st, [ACTIONS.index(name)] * int(steps), sync, hint, show_real, fsteps, temp, rand, seed)
    return f


def run_text(st, text, steps, sync, hint, show_real, fsteps, temp, rand, seed):
    try:
        acts = parse_commands(text, ACTIONS, int(steps))
    except ValueError as e:
        raise gr.Error(str(e))
    return walk(st, acts, sync, hint, show_real, fsteps, temp, rand, seed)


# ── 見た目: app.py と同じ。右下に「別の未来」のボタンを足す ──
KEYS = (("W", "前進"), ("A", "左"), ("S", "後退"), ("D", "右"))
CSS = """
#stage { position: relative; background: #000; border-radius: 14px; overflow: hidden; padding: 0 !important; border: none !important; }
#view, #view > div, #view .image-container, #view .image-frame { background: #000; border: none !important; border-radius: 0; height: auto !important; min-height: 0 !important; }
#view .image-frame, #view button { width: 100% !important; }
#view img { display: block; width: 100% !important; height: auto !important; max-height: calc(100vh - 32px); object-fit: contain; margin: 0 auto; }
#wasd { position: absolute; left: 20px; bottom: 20px; z-index: 20; width: auto !important; min-width: 0 !important;
        flex: none !important; gap: 6px !important; background: transparent; border: none; padding: 0 !important; }
#wasd .row { gap: 6px !important; flex-wrap: nowrap !important; justify-content: center; background: transparent; border: none; }
#wasd button.key, #again { flex: none !important; height: 44px; padding: 0 !important;
        border-radius: 9px; font-weight: 600; font-size: 15px; color: #e8e8e8 !important;
        background: rgba(70, 70, 70, 0.55) !important; border: 1px solid rgba(255, 255, 255, 0.14) !important;
        backdrop-filter: blur(6px); box-shadow: none !important; }
#wasd button.key { width: 44px !important; min-width: 44px !important; }
#again { position: absolute; right: 20px; bottom: 20px; z-index: 20; width: auto !important; min-width: 0 !important; padding: 0 14px !important; }
#wasd button.key:hover, #again:hover { background: rgba(110, 110, 110, 0.65) !important; }
#wasd button.key.pressed, #wasd button.key:active, #again.pressed, #again:active { background: rgba(255, 255, 255, 0.40) !important; color: #111 !important; }
"""
JS = """
() => {
  const ids = {w: "key-W", arrowup: "key-W", a: "key-A", arrowleft: "key-A",
               s: "key-S", arrowdown: "key-S", d: "key-D", arrowright: "key-D", r: "again"};
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

with gr.Blocks(title="Mesa world model simulator (flow)", fill_width=True) as demo:
    state = gr.State(None)
    last = gr.State(None)                             # 直前の歩き (歩く前の state, 行動)
    with gr.Column(elem_id="stage"):
        video = gr.Image(type="filepath", show_label=False, container=False, buttons=[], interactive=False, elem_id="view")
        with gr.Column(elem_id="wasd"):
            keys = {}
            with gr.Row():
                keys["W"] = gr.Button("W", elem_id="key-W", elem_classes="key", min_width=44)
            with gr.Row():
                for k in "ASD":
                    keys[k] = gr.Button(k, elem_id=f"key-{k}", elem_classes="key", min_width=44)
        btn_again = gr.Button("↻ 別の未来 (R)", elem_id="again", min_width=0)

    with gr.Sidebar(label="メニュー", open=False, position="right", width=380):
        gr.Markdown(
            "### Mesa の街を、拡散版の世界モデルの想像の中で歩く\n"
            "画面は世界モデルの**想像**です。いま見えている景色の特徴 (DINOv2) と行動から、次のフレームを"
            "**ノイズから少しずつ描いて**います (フローマッチング。Cosmos などの動画の世界モデルと同じ考え方)。\n\n"
            "見えていない所 (角の先・振り返った先) は、乱数によって違う未来になります。"
            "右下の **別の未来** (キーボードの R) で、直前の歩きを同じ場所・同じ行動のまま描き直せます。\n\n"
            "**W** 前進 / **A** 左 / **S** 後退 / **D** 右 (キーボードの W A S D・矢印キーでも歩けます)")
        status = gr.Textbox(label="いまの状態", lines=3, interactive=False)
        mini = gr.Image(label="地図 (赤 = 歩いた道、黄 = いまの位置と向き、色の点 = ほかの住民)", type="pil", height=240,
                        buttons=[])
        with gr.Accordion("拡散の描き方", open=True):
            fsteps = gr.Slider(1, 32, value=SIM.flow_steps, step=1,
                               label="ノイズを取り除く回数 (多いほど丁寧・遅い)")
            temp = gr.Slider(0.0, 1.5, value=SIM.flow_temp, step=0.05,
                             label="ゆらぎ (最初のノイズの大きさ。0 = いつも同じ無難な未来 / 1 = 学習どおり)")
            rand = gr.Checkbox(value=True, label="毎回ちがう乱数で描く (外すと、下の乱数で描く = 同じ指示なら同じ未来)")
            seed_box = gr.Number(value=0, precision=0, label="乱数")
        show_real = gr.Checkbox(value=False, label="右上の小窓に本物 (REAL) を出す (次に歩いたときから)")
        steps = gr.Slider(1, 60, value=12, step=1, label=f"1 回の歩数 ({FPS} 歩 = 1 秒)")
        sync = gr.Checkbox(value=True, label="毎回、本当に見えている景色から想像し直す (外すと夢の続きを歩く)")
        hint = gr.Checkbox(value=True, interactive=SIM.has_hint,
                           label="復元器に本物の手がかり画像を渡す (外すと、世界モデルが描いた特徴だけで絵にする)"
                                 + ("" if SIM.has_hint else " ※ この重みの復元器は手がかりを使わない形です"))
        seed = gr.Dropdown(TEST_SEEDS + TRAIN_SEEDS, value=TEST_SEEDS[0], label="街 (seed)")
        start_seed = gr.Number(value=0, precision=0, label="立つ場所 (乱数)")
        btn_start = gr.Button("この街に立つ", variant="secondary")
        text = gr.Textbox(label="まとめて指示", value="前進 12, 左 6, 前進 12, 右 6, 後退 6", lines=2)
        btn_text = gr.Button("この順に歩く")

    outs = [state, last, video, mini, status]
    flow_in = [fsteps, temp, rand, seed_box]
    btn_start.click(start, [seed, start_seed, hint, show_real], outs, show_progress="minimal")
    for k, name in KEYS:
        if name in ACTIONS:
            keys[k].click(press(name), [state, steps, sync, hint, show_real] + flow_in, outs, show_progress="minimal")
    btn_text.click(run_text, [state, text, steps, sync, hint, show_real] + flow_in, outs, show_progress="minimal")
    btn_again.click(again, [state, last, sync, hint, show_real, fsteps, temp], outs, show_progress="minimal")
    demo.load(start, [seed, start_seed, hint, show_real], outs)

if __name__ == "__main__":
    demo.queue(max_size=16).launch(css=CSS, js=JS)
