---
license: apache-2.0
language:
  - en
  - ja
library_name: pytorch
pipeline_tag: other
tags:
  - world-model
  - dino-wm
  - dinov2
  - simulation
  - video-prediction
  - model-based-planning
  - inverse-dynamics
  - multi-agent
---

# MESA World Model

<p align="center">
    <img src="https://static.zenn.studio/user-upload/4b2a18cffa5b-20261111.png" width="400"/>
</p>

<p align="center">
    🖥️ <a href="https://github.com/oggata/mesa-clouds-renderer">GitHub</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    🤗 <a href="https://huggingface.co/oggata">Hugging Face</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    🎮 <a href="https://huggingface.co/spaces/oggata/mesa-world-model-sim">Demo (Space)</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    📑 <a href="https://zenn.dev/oggata/books/fccaa80b4c38bb">Blog</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    🎬 <a href="https://www.youtube.com/watch?v=jidF7Oky-XU">Video</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    🔴 <a href="https://www.youtube.com/live/MXz9-ZCeH6w">24/7 Live</a>
</p>

-----

**MESA** (**M**ulti **E**ntity **S**imulation **A**rchitecture) is a town simulation in which many residents live and walk around a procedurally generated city.
The MESA town is **streamed live on YouTube 24 hours a day** ([watch the live stream](https://www.youtube.com/live/MXz9-ZCeH6w)), and this project builds world models on top of that always-running world.

This repository contains a small **action-conditioned world model** that has learned what a MESA resident *sees* when they walk.

Given the last few first-person frames and an action (**forward / turn left / turn right / backward**), the model predicts **what the resident will see next**.
It does this in the feature space of a frozen DINOv2 encoder, following the [DINO-WM](https://arxiv.org/abs/2411.04983) recipe, and a separate decoder turns the predicted features back into images so you can watch the model "dream".

- 🧠 **Learned simulator**: the forward-dynamics Transformer replaces the renderer. Chain its predictions and you can walk through the town entirely inside the model's imagination.
- 🗺️ **Generalizes to unseen towns**: trained on 64 procedurally generated towns and evaluated on 16 towns it has never seen. It learns *how the view moves* when you walk or turn, not a memorized map.
- 🔁 **Forward and inverse dynamics**: besides predicting the future, an inverse-dynamics head recovers the action taken between two frames (video → action).
- 🎯 **Planning in imagination**: given only a goal image, the agent can search for action sequences inside the model (CEM + MPC) and then execute them in the real environment.
- 🪶 **Small and cheap**: about 11M trainable parameters in the world model. It runs on a free CPU Space, and the whole pipeline trains from scratch in one Colab notebook (about 40–60 minutes on an A100).

## 🔴 The Live World: MESA on YouTube, 24/7

<p align="center">
    <a href="https://www.youtube.com/live/MXz9-ZCeH6w">
        <img src="https://img.youtube.com/vi/MXz9-ZCeH6w/hqdefault.jpg" width="80%" alt="MESA 24/7 live stream on YouTube"/>
    </a>
</p>

MESA is not just a dataset generator. It is a living town that runs around the clock and is [**broadcast live on YouTube 24/7**](https://www.youtube.com/live/MXz9-ZCeH6w). Residents walk the streets, talk to each other, make friends and go about their day in real time. Viewers can even move into the town from the YouTube chat (`!join`).

This world model is built on that world:

- **Same world, learned from the inside**: the towns, building types, heights and textures used for training come from the same MESA map generator and [mesa-clouds-renderer](https://github.com/oggata/mesa-clouds-renderer) that power the live stream.
- **Through a resident's eyes**: the live stream shows the town from a bird's-eye 3D view. The world model learns the **first-person view** of a single resident walking through it, and how that view changes with each step.
- **Toward a learned MESA**: the long-term goal is a world model that can stand in for the simulation itself. Residents could then imagine, plan and act inside a learned copy of the world that is running live.

## Video Demos

<p align="center">
    <a href="https://www.youtube.com/watch?v=jidF7Oky-XU">
        <img src="https://img.youtube.com/vi/jidF7Oky-XU/maxresdefault.jpg" width="80%" alt="MESA World Model demo video"/>
    </a>
</p>

## Try it in the browser

The [**MESA World Model Simulator**](https://huggingface.co/spaces/oggata/mesa-world-model-sim) Space loads the weights from this repository.
Walk with the **W A S D** buttons in the bottom-left corner (or the W A S D / arrow keys). The whole screen is the **world model's imagination**: it starts from DINOv2 features of what the resident currently sees and predicts each next step from the action alone. No coordinates or map are given to the model.

The menu on the right holds the rest: the map, a small **REAL** inset showing the actual renderer for comparison, and these switches:

- Untick "re-imagine from what is actually visible each time" to keep dreaming on top of the model's own predictions, and watch the errors build up over time.
- Untick "pass the real hint image to the decoder" to draw the imagination from the predicted features alone, without copying textures from the last real frame.

## Model Overview

```
first-person image ──DINOv2 ViT-S/14 (frozen)──> z_t  (16×16 patch tokens × 384) ─┐
                                                                                   ├─ World model (Transformer) ─> z_{t+1}
action a_t ∈ {forward, left, right, backward} ─────────────────────────────────────┘
(z_t, z_{t+1}) ── Inverse dynamics ──> a_t
z ── Decoder ──> 128×128 image   (only used to look at the imagination, never for planning)
```

| Component | Architecture | Params | Notes |
| --- | --- | --- | --- |
| Encoder | DINOv2 ViT-S/14 (`facebookresearch/dinov2:dinov2_vits14`) | 21M (frozen) | 224×224 image → 16×16 patch tokens, 384-d each. Not stored here; loaded from `torch.hub` |
| World model (forward dynamics) | Pre-norm Transformer encoder, width 384, 6 layers, 8 heads | 11.0M | Context of 3 frames (768 tokens). Action embedding is added to every token. A frame-level causal mask lets frame *i* attend only to frames 0..*i*. Predicts a residual on top of the previous frame's features, so training starts from "nothing changes" |
| Decoder | Conv up-sampler (16×16 → 128×128) with residual blocks | 4.9M | Also receives the last real frame as a context image, so wall textures and colors can be copied from it. Trained with L1 + VGG16 perceptual + PatchGAN losses and EMA weights |
| Inverse dynamics | Linear reduction (384 → 32) + 3-layer MLP | 8.7M | (z_t, z_{t+1}) → logits over the 4 actions |

### Action space

Each step either moves the resident 0.2 cells or rotates them by 12°.

| Index | Action | `config.json` label |
| --- | --- | --- |
| 0 | forward | `前進` |
| 1 | turn left | `左` |
| 2 | turn right | `右` |
| 3 | backward | `後退` |

Walking into a building or tree is blocked, and the view stays the same. The model has learned to predict this as well.

## Files

| File | Contents |
| --- | --- |
| `world_model.safetensors` | World model (forward dynamics) |
| `decoder.safetensors` | Features → `thumb`×`thumb` image decoder |
| `inverse_dynamics.safetensors` | (current, next) features → action |
| `feat_stats.safetensors` | Mean `mu` and std `sd` used to standardize DINOv2 features (required before feeding the world model) |
| `config.json` | Model sizes, action order, renderer settings, the town seeds used for training and evaluation, and the full training config (`train.preset`) |

## Quickstart

The loader, renderer and simulator code live in the `mesa_wm` package of the Space.

```sh
pip install torch numpy pillow safetensors huggingface_hub
hf download oggata/mesa-world-model-sim --repo-type space --local-dir mesa-world-model-sim
cd mesa-world-model-sim
```

```python
from huggingface_hub import snapshot_download
from mesa_wm import Simulator, save_gif

sim = Simulator(snapshot_download("oggata/mesa-world-model-sharp-v01"), device="cpu")  # or "cuda"

# Stand on a road in a town that was never used for training
st = sim.reset(seed=sim.cfg["maps"]["test_seeds"][0], start_seed=0)

# Walk forward 6 steps, then turn left 6 steps
real, imagined, st = sim.run(st, [0] * 6 + [1] * 6)

save_gif(real, imagined, "walk.gif", show_real=True)   # imagination, with the real renderer in a small inset
```

`sim.run(..., sync=False)` keeps imagining from the previous imagination instead of re-encoding what is actually visible.

To run the Gradio app locally:

```sh
MODEL_REPO=oggata/mesa-world-model-sharp-v01 python app.py
```

## Training

### Data

All data is generated on the fly, so no external dataset is needed.

- **Towns**: 30×30-cell procedural cities built with the same map generator as the MESA simulation (a road grid with some roads randomly removed, buildings and trees). There are 64 towns for training and a separate 16 for evaluation.
- **Renderer**: a textured DDA ray caster matched to [mesa-clouds-renderer](https://github.com/oggata/mesa-clouds-renderer). It uses the same building photos and per-business heights (including 2×2 buildings), asphalt with sidewalks and road markings, grass, dirt, trees, a daytime sky and distance fog.
- **Behavior**: a random walker that keeps an action for a while and occasionally switches (forward 45%, left / right 20% each, backward 15%), so all actions are well covered.
- **Size**: 64 towns × 8 episodes × 101 frames = **51,712 training frames**, plus 6,464 evaluation frames from unseen towns.

### Procedure

Everything is trained from scratch in a single notebook ([`colab/world_model_mesa.ipynb`](colab/world_model_mesa.ipynb)). The notebook has three presets. **The weights in this repository use `sharp`.**

| Preset | Tokens | Decoded image | World model (width · layers · params) | WM steps | Rollout | Decoder | A100 time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fast` | 8×8 | 64×64 | 256 · 6 · ~5M | 12k | 1 | L1 only | ~10 min |
| **`sharp`** ✅ | 16×16 | 128×128 | 384 · 6 · ~11M | 20k | 2 | L1 + perceptual + GAN, context image | 40–60 min |
| `sharper` | 16×16 | 128×128 | 512 · 8 · ~26M | 30k | 2 | wider, 2 res blocks | 1.5–2.5 h |

> **Note:** the current notebook's `sharp` / `sharper` presets add several improvements that the released `sharp-v01` weights do **not** use yet: a long-term memory of older frames (14 frames pooled to 4×4 tokens, every 2nd frame, reaching 28 steps back), a longer rollout curriculum (up to 4–6 steps) with input noise, an extra cosine loss, scripted maneuvers in the data (spin in place, turn around, go and come back), more training towns, and a 256×256 decoder. Weights trained this way store the extra settings in `config.json` (`world_model.n_mem`, `render_px`, …), and the Space code reads both kinds of weights.

Key hyper-parameters for `sharp-v01`: batch 32, AdamW with lr 3e-4. The world model is trained with a 2-step rollout loss, feeding its own prediction back as input, which reduces blurring over long horizons. The decoder is trained for 10k steps with perceptual weight 0.2 and GAN weight 0.1. To make it robust to imagined features, training adds noise of 0.3 to the features and uses imagined features for 25% of samples. The inverse-dynamics head is trained for 3k steps.

## Evaluation

The model is evaluated on **towns never seen during training**. It gets the last 3 real frames plus the next 15 actions, and its imagined features are compared with the features from actually walking those actions. The baseline predicts "nothing changes", reusing the last observed frame.

Reference numbers from the `fast` preset (book, chapter 14). Feature MSE, lower is better:

| Horizon | World model | "Nothing changes" baseline |
| --- | --- | --- |
| 1 step | ≈ 0.22 | ≈ 0.49 |
| 15 steps | ≈ 1.0 | ≈ 1.54 |

The world model beats the baseline at every horizon. The `sharp` model in this repository uses 4× more tokens, a larger model and a rollout loss. Its own numbers are printed by the notebook but have not been reported in this card yet.

The notebook also evaluates:

- **Counterfactuals ("what if?")**: imagining different action plans from the same spot, side by side with the real renderer. This includes walking straight into a wall and stopping.
- **Inverse dynamics**: accuracy of recovering the action from two consecutive frames. A blocked forward step is inherently ambiguous, because both frames are identical.
- **Goal-image planning**: CEM in imagination (10-step plans, MPC re-planning every step), compared with a random walker. Success means getting within 0.5 cells and 25° of the goal.

## Limitations

- Trained **only on the synthetic MESA renderer** (30×30-cell towns). It will not work on photos or real street imagery.
- Imagination is autoregressive: each prediction is fed back as input, so frames **get blurrier the further ahead you go**.
- Decoded images are 128×128 (64×64 for `fast`), so fine details such as text on shop signs are not reproduced.
- The released `sharp` weights were trained **without other residents (NPCs)** (`NPCS = 0`). The architecture supports NPCs as extra tokens, which pass the residents' positions as a tensor, but these weights do not use them.
- `config.json` stores action labels in Japanese. Use the integer indices above.

## Intended Use

This is a research and education model. It is meant for learning how latent world models work (forward and inverse dynamics, imagination rollouts, planning in latent space) on a small, fully controllable world, and as a building block for experiments with agents in the MESA town simulation.

## Related

- 📑 Blog / book (Japanese): [NVIDIA Cosmos 3 practical guide](https://zenn.dev/oggata/books/fccaa80b4c38bb). Chapter 14 builds this world model, chapter 14.5 improves the decoder, and chapter 17 drives it with language goals through Cosmos Reason.
- 🔴 Live stream (24/7): [MESA town on YouTube](https://www.youtube.com/live/MXz9-ZCeH6w)
- 🖥️ Renderer: [oggata/mesa-clouds-renderer](https://github.com/oggata/mesa-clouds-renderer)
- 🎮 Space: [oggata/mesa-world-model-sim](https://huggingface.co/spaces/oggata/mesa-world-model-sim)

## Citation

If you find this work useful, please cite this repository and the works it builds on:

```bibtex
@misc{mesa_world_model_2026,
  title        = {MESA World Model: an action-conditioned DINO-WM style world model for the MESA town simulation},
  author       = {oggata},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/oggata/mesa-world-model-sharp-v01}}
}

@article{zhou2024dinowm,
  title   = {DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning},
  author  = {Zhou, Gaoyue and Pan, Hengkai and LeCun, Yann and Pinto, Lerrel},
  journal = {arXiv preprint arXiv:2411.04983},
  year    = {2024}
}

@article{oquab2023dinov2,
  title   = {DINOv2: Learning Robust Visual Features without Supervision},
  author  = {Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and others},
  journal = {arXiv preprint arXiv:2304.07193},
  year    = {2023}
}
```

## License

The weights in this repository are released under the Apache 2.0 License. The frozen DINOv2 encoder is downloaded separately from [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) and is subject to its own license.

## Acknowledgements

Thanks to the authors of [DINO-WM](https://github.com/gaoyuezhou/dino_wm) for the world-model recipe, [DINOv2](https://github.com/facebookresearch/dinov2) for the visual features, and [Hugging Face](https://huggingface.co) for hosting the model and the Space.
