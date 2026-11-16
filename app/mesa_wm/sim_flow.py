"""拡散 (フローマッチング) 版の世界モデルを「歩けるシミュレーター」として動かす。

sim.Simulator (1 回で当てる版) と同じ使い方で、違うのは想像の描き方だけ:
  ・1 歩ごとに、ノイズから steps 回かけて次のフレームを描く
  ・seed (乱数) を変えると、同じ場所・同じ行動からでも別の未来になる。temp = 0 なら、いつも同じ「無難な」未来
街・本物の描画器・住民・地図・GIF は sim.Simulator のものをそのまま使う。
"""
import copy
import json
import math
import os

import torch
from safetensors.torch import load_file

from .flow import FlowWorldModel, imagine_flow
from .models import Decoder, DinoEncoder, InverseDynamics
from .render import Renderer
from .sim import Simulator


class FlowSimulator(Simulator):
    def __init__(self, model_dir, device="cpu"):
        # Simulator.__init__ は 1 回で当てる版の世界モデルを作るので呼ばず、同じことを拡散版で行う
        self.device = device
        with open(os.path.join(model_dir, "config.json")) as f:
            self.cfg = cfg = json.load(f)
        if cfg.get("model_type") != "flow":
            raise SystemExit(f"{model_dir} は拡散版のモデルではありません (config.json の model_type が \"flow\" ではない)。"
                             f"1 回で当てる版は app.py で動かしてください")
        self.actions = cfg["actions"]
        self.hist, self.grid_tok = cfg["hist"], cfg["tok_grid"]
        self.move, self.rot = cfg["move"], math.radians(cfg["rot_deg"])
        self.img, self.thumb, self.size = cfg["img"], cfg["thumb"], cfg["grid"]
        self.render_px = cfg.get("render_px", self.img)
        self.flow_steps = cfg.get("flow", {}).get("steps", 8)      # 想像の既定 (画面で変えられる)
        self.flow_temp = cfg.get("flow", {}).get("temp", 1.0)
        n_tok, D = self.grid_tok ** 2, cfg["feat_dim"]
        ld = lambda name: load_file(os.path.join(model_dir, name), device=device)

        self.wm = FlowWorldModel(D, n_tok, self.hist, n_actions=len(self.actions), **cfg["world_model"]).to(device).eval()
        self.wm.load_state_dict(ld("world_model.safetensors"))
        self.span = self.wm.span
        self.dec = Decoder(D, self.grid_tok, **cfg["decoder"]).to(device).eval()
        self.dec.load_state_dict(ld("decoder.safetensors"))
        self.inv = None
        if os.path.exists(os.path.join(model_dir, "inverse_dynamics.safetensors")):
            self.inv = InverseDynamics(D, n_tok, len(self.actions), **cfg["inverse_dynamics"]).to(device).eval()
            self.inv.load_state_dict(ld("inverse_dynamics.safetensors"))
        st = ld("feat_stats.safetensors")
        self.mu, self.sd = st["mu"], st["sd"]
        self.dino = DinoEncoder(device)
        self.renderer = Renderer(cfg, device)
        self.npc = cfg.get("npc")
        self._cities, self._cur = {}, None

    @torch.no_grad()
    def run(self, state, acts, sync=True, hint=True, steps=None, temp=None, seed=None):
        """Simulator.run と同じ。steps / temp / seed で描き方を変える (省くと config.json の既定、seed=None は毎回違う未来)。
        state は書き換えないので、同じ state から seed だけ変えて呼び直すと「別の未来」を描ける"""
        steps = self.flow_steps if steps is None else int(steps)
        temp = self.flow_temp if temp is None else float(temp)
        city = self.city(state["seed"])
        H = self.span
        z_ctx = (state["z_real"] if sync else state["z_dream"]).to(self.device)
        a_seq = torch.tensor([state["a_hist"] + list(acts)], device=self.device)
        poses = list(state["poses"])
        ppl = copy.deepcopy(state.get("people"))
        pps, n_new = [], []
        for a in acts:
            poses.append(self.step_pose(city, *poses[-1], a))
            if ppl is not None:
                ppl.step(); pps.append(ppl.pose()); n_new.append(self.npc_feats(poses[-1], pps[-1]))
        n_all = state.get("n_hist", []) + n_new
        n_seq = torch.stack(n_all)[None].to(self.device) if ppl is not None else None
        z_imag = imagine_flow(self.wm, z_ctx[None], a_seq, n_seq, steps=steps, temp=temp, seed=seed)[0]   # (K,N,D)
        z_all = torch.cat([z_ctx[-1:], z_imag]).float()
        imag = self.decode(z_all, state["last_img"], hint)
        new = poses[-len(acts):]
        real = self.draw(new, pps if ppl is not None else None)
        z_new = self.encode(real[-H:]).cpu()
        z_real = torch.cat([state["z_real"], z_new])[-H:]
        z_dream = torch.cat([z_ctx.cpu(), z_imag.cpu()])[-H:]
        real = torch.cat([state["last_img"][None].to(self.device), real])
        st = dict(state, poses=poses[-H:], z_real=z_real, z_dream=z_dream, people=ppl, n_hist=n_all[-(H - 1):] if ppl else [],
                  a_hist=(state["a_hist"] + list(acts))[-(H - 1):], last_img=real[-1].cpu(),
                  trail=(state["trail"] + [p[:2] for p in new])[-400:])
        to_np = lambda t: t.permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()
        return to_np(real), to_np(imag), st
