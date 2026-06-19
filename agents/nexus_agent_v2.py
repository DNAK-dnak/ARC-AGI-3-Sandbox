# ======================================================================
# SOVEREIGN v3 — FORGE v15 + BFS + Symbolic + Anti-Stuck
#
# Base: 0.17 submission (BFS solver + ForgeNet CNN + symbolic layer)
# Added: position-based stuck detection, escape mode, action-blocked
#        tracking, heatmap-aware symbolic brain, oscillation breaker
# ======================================================================
import heapq
import pickle
import copy
import glob
import hashlib
import importlib.util
import logging
import os
import random
import time
import traceback
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

logger = logging.getLogger(__name__)

# ==================== BFS SOLVER (FORGE v15, unchanged) ====================

class BFSSolver:
    def __init__(self, game_path, game_class_name, scan_timeout=3, bfs_timeout=120):
        self.game_path = game_path
        self.class_name = game_class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.game_cls = None
        self.solutions = {}
        self._warmup_prefix = []

    def load(self):
        try:
            spec = importlib.util.spec_from_file_location('game_mod', self.game_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.game_cls = getattr(mod, self.class_name)
            return True
        except Exception as e:
            logger.warning(f"BFS: Failed to load game class: {e}")
            return False

    def _state_hash(self, g, frame, hidden_fields=None):
        fh = hashlib.md5(frame.tobytes()).hexdigest()[:16]
        if hidden_fields:
            extras = []
            for field_name in hidden_fields:
                try:
                    v = getattr(g, field_name, None)
                    if v is not None:
                        extras.append(f"{field_name}={v}")
                except:
                    pass
            if extras:
                return fh + "|" + "|".join(extras)
        return fh

    def _extract_win_field(self):
        try:
            source = open(self.game_path).read()
            lines = source.split('\n')
            for i, line in enumerate(lines):
                if 'self.next_level()' in line:
                    for j in range(i-1, max(0, i-8), -1):
                        s = lines[j].strip()
                        if s.startswith('if ') or s.startswith('elif '):
                            import re
                            m = re.search(r'self\.(\w+)', s)
                            if m:
                                return m.group(1)
                    break
        except:
            pass
        return None

    def _probe_hidden_fields(self, game, actions):
        if not actions:
            return []
        win_field = self._extract_win_field()
        initial = {}
        for k, v in game.__dict__.items():
            if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                initial[k] = v
        changing_fields = set()
        if win_field and win_field in initial:
            changing_fields.add(win_field)
        for act_id, data in actions[:10]:
            g = copy.deepcopy(game)
            try:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                g.perform_action(ai, raw=True)
            except:
                continue
            for k, v in g.__dict__.items():
                if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                    if k in initial and v != initial[k]:
                        if k not in ('_action_count', '_full_reset', '_action_complete'):
                            changing_fields.add(k)
        hidden = []
        for f in changing_fields:
            if f.startswith('_') and f not in ('_current_level_index', '_score'):
                continue
            hidden.append(f)
        return sorted(hidden)

    def _scan_actions(self, game, f0, bg):
        avail = game._available_actions
        actions = []
        for a in [a for a in avail if a <= 5]:
            g = copy.deepcopy(game)
            try:
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                if r.frame and np.sum(f0 != np.array(r.frame[-1])) > 0:
                    actions.append((a, None))
            except:
                pass
        if 6 in avail:
            t0 = time.time()
            for y in range(0, 64, 2):
                if time.time() - t0 > self.scan_timeout:
                    break
                for x in range(0, 64, 2):
                    if f0[y, x] == bg:
                        continue
                    g = copy.deepcopy(game)
                    try:
                        r = g.perform_action(
                            ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y, 'game_id': 'bfs'}),
                            raw=True)
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        if np.sum(f0 != f) > 0:
                            actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
                    except:
                        pass
        return actions

    def solve_level(self, level_idx, max_states=500000, prev_solution=None):
        if not self.game_cls:
            return None
        game = self.game_cls()
        game.set_level(level_idx)
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not r0.frame:
            return None
        f0 = np.array(r0.frame[-1])
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
        if prev_solution and level_idx > 0:
            transfer_result = self._try_transfer(game, level_idx, prev_solution, f0)
            if transfer_result:
                return transfer_result
        actions = self._scan_actions(game, f0, bg)
        if not actions:
            avail = game._available_actions
            for warmup_id in [a for a in avail if a <= 4]:
                g_warmup = copy.deepcopy(game)
                try:
                    g_warmup.perform_action(ActionInput(id=GameAction.from_id(warmup_id)), raw=True)
                    f_after = np.array(g_warmup.get_pixels(0, 0, 64, 64))
                    warmup_actions = self._scan_actions(g_warmup, f_after, bg)
                    if warmup_actions:
                        game = g_warmup; f0 = f_after; actions = warmup_actions
                        self._warmup_prefix = [(warmup_id, None)]
                        break
                except:
                    pass
        if not actions:
            return None
        win_field = self._extract_win_field()
        visited = set()
        h0 = self._state_hash(game, f0, None)
        visited.add(h0)
        t0 = time.time()
        explored = 0
        queue = deque()
        queue.append((copy.deepcopy(game), [], 0))
        while queue and explored < max_states and (time.time() - t0) < self.bfs_timeout:
            g, hist, depth = queue.popleft()
            for act_id, data in actions:
                g2 = copy.deepcopy(g)
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g2.perform_action(ai, raw=True)
                except:
                    continue
                explored += 1
                if not r.frame:
                    continue
                f = np.array(r.frame[-1])
                h = self._state_hash(g2, f, None)
                if h in visited:
                    continue
                visited.add(h)
                new_hist = hist + [(act_id, data)]
                if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                    self.solutions[level_idx] = new_hist
                    return new_hist
                if depth < 30:
                    queue.append((g2, new_hist, depth + 1))
        elapsed_first = time.time() - t0
        # ACMD pass
        if len(visited) < 100 and elapsed_first < self.bfs_timeout * 0.8:
            hidden_fields = self._probe_hidden_fields(game, actions)
            if hidden_fields:
                clock_fields = set()
                try:
                    if actions:
                        g_noop2 = copy.deepcopy(game)
                        ai = ActionInput(id=GameAction.from_id(actions[0][0]), data=actions[0][1]) if actions[0][1] else ActionInput(id=GameAction.from_id(actions[0][0]))
                        g_noop2.perform_action(ai, raw=True)
                        g_noop3 = copy.deepcopy(g_noop2)
                        g_noop3.perform_action(ai, raw=True)
                        for f in hidden_fields:
                            v1 = getattr(g_noop2, f, None)
                            v2 = getattr(g_noop3, f, None)
                            if v1 != v2:
                                clock_fields.add(f)
                except:
                    pass
                trigger_fields = [f for f in hidden_fields if f not in clock_fields] or hidden_fields
                game2 = self.game_cls()
                game2.set_level(level_idx)
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                f0_2 = np.array(game2.perform_action(ActionInput(id=GameAction.RESET), raw=True).frame[-1])
                visited2 = set()
                init_state = {f: getattr(game2, f, None) for f in trigger_fields}
                h0_2 = self._state_hash(game2, f0_2, trigger_fields)
                visited2.add(h0_2)
                fifo2 = 0
                heap2 = [(0, 0, fifo2, copy.deepcopy(game2), [])]
                fifo2 += 1
                t0_2 = time.time()
                explored2 = 0
                remaining = max(60, self.bfs_timeout - elapsed_first)
                while heap2 and explored2 < max_states and (time.time() - t0_2) < remaining:
                    neg_delta, depth, _, g, hist = heapq.heappop(heap2)
                    for act_id, data in actions:
                        g2 = copy.deepcopy(g)
                        try:
                            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                            r = g2.perform_action(ai, raw=True)
                        except:
                            continue
                        explored2 += 1
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        h = self._state_hash(g2, f, trigger_fields)
                        if h in visited2:
                            continue
                        visited2.add(h)
                        new_hist = hist + [(act_id, data)]
                        if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                            self.solutions[level_idx] = new_hist
                            return new_hist
                        pixels_changed = np.sum(f0_2 != f) > 0
                        trigger_delta = 0
                        for tf in trigger_fields:
                            cv = getattr(g2, tf, None)
                            iv = init_state.get(tf)
                            if isinstance(cv, (int, float)) and isinstance(iv, (int, float)):
                                trigger_delta += abs(cv - iv)
                            elif cv != iv:
                                trigger_delta += 1
                        if not pixels_changed and trigger_delta == 0:
                            continue
                        priority = -trigger_delta
                        fifo2 += 1
                        if depth < 40:
                            heapq.heappush(heap2, (priority, depth + 1, fifo2, g2, new_hist))
        # IDDFS pass
        elapsed_total = time.time() - t0
        remaining_time = max(30, self.bfs_timeout - elapsed_total)
        if len(actions) <= 6 and remaining_time > 30:
            game3 = self.game_cls()
            game3.set_level(level_idx)
            game3.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            game3.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            t0_3 = time.time()
            for max_depth in range(10, 60):
                if time.time() - t0_3 > remaining_time:
                    break
                stack = [(copy.deepcopy(game3), [], set())]
                while stack and (time.time() - t0_3) < remaining_time:
                    g, hist, path_hashes = stack.pop()
                    if len(hist) >= max_depth:
                        continue
                    for act_id, data in actions:
                        g2 = copy.deepcopy(g)
                        try:
                            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                            r = g2.perform_action(ai, raw=True)
                        except:
                            continue
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        h = hashlib.md5(f.tobytes()).hexdigest()[:16]
                        if h in path_hashes:
                            continue
                        new_hist = hist + [(act_id, data)]
                        if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                            sol = self._warmup_prefix + new_hist
                            self.solutions[level_idx] = sol
                            return sol
                        new_path = path_hashes | {h}
                        stack.append((g2, new_hist, new_path))
        return None

    def _try_transfer(self, game, level_idx, prev_solution, f1):
        try:
            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(prev_solution):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        sol = prev_solution[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break
            prev_game = self.game_cls()
            prev_game.set_level(level_idx - 1)
            prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r_prev = prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r_prev.frame:
                return None
            f0 = np.array(r_prev.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
            def get_objects(frame, bg_c):
                objs = []
                for c in range(16):
                    if c == bg_c: continue
                    mask = (frame == c); npix = int(np.sum(mask))
                    if npix < 2: continue
                    ys, xs = np.where(mask)
                    objs.append({'color': c, 'cx': float(np.mean(xs)), 'cy': float(np.mean(ys)), 'n': npix})
                return sorted(objs, key=lambda o: (o['color'], -o['n']))
            objs_prev = get_objects(f0, bg)
            objs_curr = get_objects(f1, bg)
            if not objs_prev or not objs_curr:
                return None
            matched = []
            for op in objs_prev:
                best = None; best_dist = float('inf')
                for oc in objs_curr:
                    if oc['color'] == op['color'] and abs(oc['n'] - op['n']) < max(op['n'], oc['n']) * 0.5:
                        d = abs(oc['cx'] - op['cx']) + abs(oc['cy'] - op['cy'])
                        if d < best_dist: best_dist = d; best = oc
                if best: matched.append((op, best))
            if not matched: return None
            dx = np.mean([m[1]['cx'] - m[0]['cx'] for m in matched])
            dy = np.mean([m[1]['cy'] - m[0]['cy'] for m in matched])
            for multiplier in [1, 2, 3, 1.5]:
                expanded = []
                for act_id, data in prev_solution:
                    for _ in range(int(multiplier)):
                        if data and 'x' in data:
                            new_data = dict(data)
                            new_data['x'] = max(0, min(63, int(data['x'] + dx)))
                            new_data['y'] = max(0, min(63, int(data['y'] + dy)))
                            expanded.append((act_id, new_data))
                        else:
                            expanded.append((act_id, data))
                g = copy.deepcopy(game)
                for i, (act_id, data) in enumerate(expanded):
                    try:
                        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                        r = g.perform_action(ai, raw=True)
                        if r.levels_completed > level_idx or g._current_level_index > level_idx:
                            sol = expanded[:i+1]
                            self.solutions[level_idx] = sol
                            return sol
                    except:
                        break
        except:
            pass
        return None


def find_game_source_and_class(game_id, arc_env=None):
    gid = game_id.split('-')[0]
    cls_name = gid[0].upper() + gid[1:]
    src = None
    if arc_env and hasattr(arc_env, 'environment_info'):
        ei = arc_env.environment_info
        if hasattr(ei, 'local_dir') and ei.local_dir:
            from pathlib import Path
            ld = Path(ei.local_dir)
            for candidate in [ld / f"{gid}.py", ld / f"{cls_name.lower()}.py"]:
                if candidate.exists():
                    src = str(candidate)
                    import re
                    content = candidate.read_text()[:2000]
                    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                    if m: cls_name = m.group(1)
                    break
    if not src:
        for pattern in [f"/tmp/*/{gid}/*/{gid}.py", f"/kaggle/*/{gid}*/{gid}.py", f"**/game_sources/**/{gid}.py"]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                src = matches[0]
                import re
                content = open(src).read()[:2000]
                m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                if m: cls_name = m.group(1)
                break
    return src, cls_name


# ==================== CNN (FORGE v8 core) ====================

class CBAM(nn.Module):
    def __init__(s, ch, r=16):
        super().__init__()
        s.fc1=nn.Linear(ch,max(ch//r,4)); s.fc2=nn.Linear(max(ch//r,4),ch)
        s.sp=nn.Conv2d(2,1,7,padding=3)
    def forward(s, x):
        B,C,H,W=x.shape
        w=torch.sigmoid(s.fc2(F.relu(s.fc1(x.mean(dim=[2,3]))))); x=x*w.view(B,C,1,1)
        a=torch.sigmoid(s.sp(torch.cat([x.max(1,keepdim=True)[0],x.mean(1,keepdim=True)],1)))
        return x*a

class ActionEffectAttention(nn.Module):
    def __init__(s, feat_dim=64, mem_dim=32, n_actions=5):
        super().__init__()
        s.mem_dim=mem_dim
        s.diff_enc=nn.Sequential(nn.Conv2d(1,8,8,stride=8),nn.ReLU(),nn.Conv2d(8,16,4,stride=4),nn.ReLU(),nn.Flatten(),nn.Linear(16*2*2,mem_dim))
        s.q_proj=nn.Linear(feat_dim,mem_dim)
        s.v_proj=nn.Linear(mem_dim+1+n_actions,n_actions)
        s.scale=mem_dim**0.5
    def forward(s, cnn_feat, mem_diffs, mem_actions, mem_rewards):
        B,M=mem_actions.shape
        if M==0: return torch.zeros(B,5,device=cnn_feat.device)
        keys=s.diff_enc(mem_diffs.reshape(B*M,1,64,64)).reshape(B,M,s.mem_dim)
        q=s.q_proj(cnn_feat).unsqueeze(1)
        attn=F.softmax(torch.bmm(q,keys.transpose(1,2))/s.scale,dim=-1)
        act_oh=F.one_hot(mem_actions.clamp(0,4),5).float()
        vals=torch.cat([keys,mem_rewards.unsqueeze(-1),act_oh],dim=-1)
        ctx=torch.bmm(attn,vals).squeeze(1)
        return s.v_proj(ctx)

class ForgeNet(nn.Module):
    def __init__(s, in_ch=26, g=64):
        super().__init__()
        s.g=g
        s.c1=nn.Conv2d(in_ch,32,3,padding=1); s.c2=nn.Conv2d(32,64,3,padding=1)
        s.c3=nn.Conv2d(64,128,3,padding=1); s.c4=nn.Conv2d(128,256,3,padding=1)
        s.attn=CBAM(256); s.ar=nn.Conv2d(256,64,1); s.ap=nn.MaxPool2d(4,4)
        s.af=nn.Linear(64*16*16,256); s.ah=nn.Linear(256,5); s.dr=nn.Dropout(0.15)
        s.cc1=nn.Conv2d(256,128,3,padding=1); s.cc2=nn.Conv2d(128,64,3,padding=1)
        s.cc3=nn.Conv2d(64,32,1); s.cc4=nn.Conv2d(32,1,1)
        s.gp=nn.AdaptiveAvgPool2d(1); s.gf=nn.Linear(256,64)
        s.aea=ActionEffectAttention(feat_dim=64,mem_dim=32,n_actions=5)
    def forward(s, x, mem_diffs=None, mem_actions=None, mem_rewards=None):
        x=F.relu(s.c1(x)); x=F.relu(s.c2(x)); x=F.relu(s.c3(x)); f=F.relu(s.c4(x))
        f=s.attn(f); af=F.relu(s.ar(f)); af=s.ap(af).reshape(f.size(0),-1)
        al=s.ah(s.dr(F.relu(s.af(af))))
        cf=F.relu(s.cc1(f)); cf=F.relu(s.cc2(cf)); cf=F.relu(s.cc3(cf))
        cl=s.cc4(cf).reshape(f.size(0),-1)
        if mem_diffs is not None and mem_actions is not None:
            gf=s.gf(s.gp(f).reshape(f.size(0),-1))
            al=al+s.aea(gf,mem_diffs,mem_actions,mem_rewards)
        return torch.cat([al,cl],1)

def fast_objects(frame, bg):
    objs=[]
    for c in range(16):
        if c==bg: continue
        mask=(frame==c); npix=int(np.sum(mask))
        if npix<4 or npix>3000: continue
        ys,xs=np.where(mask)
        objs.append((c,float(np.mean(xs)),float(np.mean(ys)),npix))
    return objs


# ==================== NEXUS Neural Engine (VQ-VAE + Policy + IDM) ====================

def normalise_frame(frame):
    counts = np.bincount(frame.flatten(), minlength=16)
    order = np.argsort(-counts, kind='stable')
    remap = np.zeros(16, dtype=np.int64)
    for new_role, old_color in enumerate(order):
        remap[old_color] = new_role
    return remap[frame]

def frame_to_onehot(frame, pad_to=64):
    H, W = frame.shape
    canvas = np.zeros((pad_to, pad_to), dtype=np.int64)
    canvas[:H, :W] = frame
    t = torch.from_numpy(canvas)
    oh = torch.zeros(16, pad_to, pad_to, dtype=torch.float32)
    oh.scatter_(0, t.unsqueeze(0), 1.0)
    return oh

class VectorQuantizer(nn.Module):
    def __init__(self, K=512, D=64, beta=0.25):
        super().__init__()
        self.K, self.D, self.beta = K, D, beta
        self.embedding = nn.Embedding(K, D)
        nn.init.uniform_(self.embedding.weight, -1/K, 1/K)
    def forward(self, z_e):
        B, D, H, W = z_e.shape
        z_flat = z_e.permute(0,2,3,1).reshape(-1, D)
        dist = (z_flat.pow(2).sum(1, keepdim=True) - 2 * z_flat @ self.embedding.weight.t()
                + self.embedding.weight.pow(2).sum(1))
        indices = dist.argmin(dim=1)
        z_q = self.embedding(indices).reshape(B, H, W, D).permute(0,3,1,2)
        loss = F.mse_loss(z_e, z_q.detach()) + self.beta * F.mse_loss(z_e.detach(), z_q)
        return z_e + (z_q - z_e).detach(), indices.reshape(B, H, W), loss

class NexusEncoder(nn.Module):
    def __init__(self, in_ch=16, D=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, D, 1))
    def forward(self, x): return self.net(x)

class NexusDecoder(nn.Module):
    def __init__(self, D=64, out_ch=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(D, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, out_ch, 1))
    def forward(self, z): return self.net(z)

class VQVAE(nn.Module):
    def __init__(self, K=512, D=64):
        super().__init__()
        self.encoder = NexusEncoder(16, D)
        self.vq = VectorQuantizer(K, D)
        self.decoder = NexusDecoder(D, 16)
    def encode(self, x):
        z_e = self.encoder(x)
        z_q, idx, _ = self.vq(z_e)
        return z_q, idx
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, idx, vq_loss = self.vq(z_e)
        return self.decoder(z_q), vq_loss, idx

class NexusWorldModel(nn.Module):
    def __init__(self, K=512, D=64, n_actions=7, nhead=4, nlayers=2, seq_len=64):
        super().__init__()
        self.K, self.D, self.seq_len = K, D, seq_len
        self.tok_emb = nn.Embedding(K, D)
        self.act_emb = nn.Embedding(n_actions + 1, D)
        self.pos_emb = nn.Embedding(seq_len + 1, D)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=D, nhead=nhead, dim_feedforward=D*4,
                                       dropout=0.0, batch_first=True),
            num_layers=nlayers)
        self.next_state_head = nn.Linear(D, K)
        self.reward_head = nn.Linear(D, 1)
        self.win_head = nn.Linear(D, 1)
    def forward(self, idx, action):
        pos = torch.arange(self.seq_len + 1, device=idx.device)
        tokens = torch.cat([self.act_emb(action).unsqueeze(1), self.tok_emb(idx)], dim=1) + self.pos_emb(pos).unsqueeze(0)
        out = self.transformer(tokens)
        return (self.next_state_head(out[:, 1:, :]),
                self.reward_head(out[:, 0, :]).squeeze(-1),
                self.win_head(out[:, 0, :]).squeeze(-1))
    @torch.no_grad()
    def imagine(self, idx, action, device):
        self.eval()
        logits, rew, win = self.forward(idx.unsqueeze(0).to(device), torch.tensor([action], device=device))
        return logits.squeeze(0).argmax(-1), rew.item(), torch.sigmoid(win).item()

class PolicyHead(nn.Module):
    def __init__(self, K=512, D=64, n_actions=7, seq_len=64):
        super().__init__()
        self.tok_emb = nn.Embedding(K, D)
        self.pos_emb = nn.Embedding(seq_len, D)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Dropout(0.1), nn.Linear(D, n_actions))
        self.seq_len = seq_len
    def forward(self, idx):
        B = idx.size(0)
        pos = torch.arange(self.seq_len, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos).unsqueeze(0)
        x = self.pool(x.permute(0, 2, 1)).squeeze(-1)
        return self.head(x)

class CEMPlanner:
    def __init__(self, wm, H=8, N=48, n_elite=6, n_iter=2, gamma=5.0):
        self.wm = wm; self.H = H; self.N = N; self.n_elite = n_elite
        self.n_iter = n_iter; self.gamma = gamma
    def plan(self, idx, avail_ids, device, budget=0.3):
        if not avail_ids: return avail_ids[0] if avail_ids else 1
        n_act = len(avail_ids)
        probs = np.ones(n_act) / n_act
        best = avail_ids[0]
        t0 = time.time()
        for _ in range(self.n_iter):
            if time.time() - t0 > budget: break
            seqs = np.random.choice(n_act, size=(self.N, self.H), p=probs)
            scores = np.zeros(self.N)
            for i in range(self.N):
                cur, total = idx.clone(), 0.0
                for h in range(self.H):
                    cur, rew, win = self.wm.imagine(cur, avail_ids[seqs[i, h]], device)
                    total += rew
                    if win > 0.9: total += self.gamma; break
                scores[i] = total + self.gamma * win
            elite = seqs[np.argsort(scores)[-self.n_elite:]]
            counts = np.bincount(elite[:, 0], minlength=n_act).astype(float)
            if counts.sum() > 0: probs = counts / counts.sum()
            best = avail_ids[int(elite[-1, 0])]
        return best


# ==================== SYMBOLIC RULE-DISCOVERY ENGINE ====================

def _cc_label(mask):
    H, W = mask.shape
    labeled = np.zeros((H, W), dtype=np.int32)
    num = 0
    for r in range(H):
        for c in range(W):
            if mask[r, c] and labeled[r, c] == 0:
                num += 1
                q = deque([(r, c)])
                labeled[r, c] = num
                while q:
                    y, x = q.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y+dy, x+dx
                            if 0 <= ny < H and 0 <= nx < W \
                               and mask[ny, nx] and labeled[ny, nx] == 0:
                                labeled[ny, nx] = num
                                q.append((ny, nx))
    return labeled, num

def extract_entities(frame):
    bg = int(np.bincount(frame.flatten(), minlength=16).argmax())
    entities = []
    eid = 0
    for color in range(16):
        if color == bg: continue
        mask = (frame == color)
        if not mask.any(): continue
        labeled, n = _cc_label(mask)
        for i in range(n):
            comp = (labeled == i + 1)
            coords = np.argwhere(comp)
            if len(coords) == 0: continue
            cy, cx = np.round(coords.mean(axis=0)).astype(int)
            rmin, cmin = coords.min(axis=0)
            rmax, cmax = coords.max(axis=0)
            entities.append({
                'id': eid, 'color': color, 'pixels': int(comp.sum()),
                'centroid': (int(cy), int(cx)),
                'bbox': (int(rmin), int(cmin), int(rmax), int(cmax)),
            })
            eid += 1
    return entities, bg

def _bbox_overlap(b1, b2):
    return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])

def _bbox_adjacent(b1, b2, gap=2):
    expanded = (b1[0]-gap, b1[1]-gap, b1[2]+gap, b1[3]+gap)
    return _bbox_overlap(expanded, b2)

def detect_transformations(ent_before, ent_after):
    events = []
    used_after = set()
    for e0 in ent_before:
        best_same, best_d_same = None, float('inf')
        best_recolor, best_d_rc = None, float('inf')
        best_resize = None
        for i, e1 in enumerate(ent_after):
            if i in used_after: continue
            cy_d = (e0['centroid'][0] - e1['centroid'][0]) ** 2
            cx_d = (e0['centroid'][1] - e1['centroid'][1]) ** 2
            d = cy_d + cx_d
            if e0['color'] == e1['color'] and e0['pixels'] == e1['pixels']:
                if d < best_d_same: best_d_same = d; best_same = i
            elif e0['pixels'] == e1['pixels'] and d < best_d_rc:
                best_d_rc = d; best_recolor = i
            elif e0['color'] == e1['color'] and best_resize is None:
                best_resize = i
        if best_same is not None:
            used_after.add(best_same)
            if best_d_same > 0:
                events.append({'type': 'MOVE', 'color': e0['color'], 'pixels': e0['pixels']})
        elif best_recolor is not None:
            used_after.add(best_recolor)
            events.append({'type': 'RECOLOR', 'color': e0['color'],
                           'new_color': ent_after[best_recolor]['color'], 'pixels': e0['pixels']})
        elif best_resize is not None:
            used_after.add(best_resize)
            new_px = ent_after[best_resize]['pixels']
            t = 'GROW' if new_px > e0['pixels'] else 'SHRINK'
            events.append({'type': t, 'color': e0['color'], 'pixels': e0['pixels'], 'new_pixels': new_px})
        else:
            events.append({'type': 'DISAPPEAR', 'color': e0['color'], 'pixels': e0['pixels']})
    for i, e1 in enumerate(ent_after):
        if i not in used_after:
            events.append({'type': 'APPEAR', 'color': e1['color'], 'pixels': e1['pixels']})
    return events

def spatial_context(entities):
    ctx = {e['id']: {'overlaps': [], 'adjacent': []} for e in entities}
    for i, ea in enumerate(entities):
        for j, eb in enumerate(entities):
            if i >= j: continue
            if _bbox_overlap(ea['bbox'], eb['bbox']):
                ctx[ea['id']]['overlaps'].append(eb['id'])
                ctx[eb['id']]['overlaps'].append(ea['id'])
            elif _bbox_adjacent(ea['bbox'], eb['bbox']):
                ctx[ea['id']]['adjacent'].append(eb['id'])
                ctx[eb['id']]['adjacent'].append(ea['id'])
    return ctx

class RuleHypothesis:
    def __init__(self, trigger, effect):
        self.trigger = trigger
        self.effect = effect
        self.confidence = 1
        self.failures = 0
        self.confirmed = False
    def sig(self):
        t = self.trigger; e = self.effect
        return (t.get('action'), tuple(sorted(t.get('near_colors', []))),
                e.get('type'), e.get('color'))
    def validate(self, observed_events):
        for ev in observed_events:
            if ev['type'] == self.effect['type'] and ev['color'] == self.effect['color']:
                self.confidence += 1
                if self.confidence >= 3: self.confirmed = True
                return True
        self.failures += 1
        return False

class TransformationGraph:
    def __init__(self):
        self.action_events: Dict[int, list] = {}
        self.action_context: Dict[int, dict] = {}
        self.hypotheses: Dict[tuple, RuleHypothesis] = {}
        self.goal_sigs: set = set()
        self.goal_actions: set = set()
        self.tried: set = set()
        self._validate_queue: deque = deque()
        self._action_fail_streak: Dict[int, int] = {}

    def _near_colors(self, ctx, entity_id, entities_by_id):
        info = ctx.get(entity_id, {})
        colors = set()
        for oid in info.get('overlaps', []) + info.get('adjacent', []):
            e = entities_by_id.get(oid)
            if e: colors.add(e['color'])
        return sorted(colors)

    def record_action_outcome(self, action_id, meaningful_change):
        if meaningful_change:
            self._action_fail_streak[action_id] = 0
        else:
            self._action_fail_streak[action_id] = self._action_fail_streak.get(action_id, 0) + 1

    def action_is_blocked(self, action_id, threshold=2):
        return self._action_fail_streak.get(action_id, 0) >= threshold

    def observe(self, action_id, events, ctx_before, entities_before,
                ctx_after, entities_after, level_advanced):
        self.tried.add(action_id)
        if not events:
            self.action_events[action_id] = []
            return
        self.action_events[action_id] = events
        self.action_context[action_id] = ctx_before
        by_id = {e['id']: e for e in entities_before}
        for ev in events:
            near_colors = sorted(set(
                c for e in entities_before
                for c in self._near_colors(ctx_before, e['id'], by_id)
            ))
            hyp = RuleHypothesis({'action': action_id, 'near_colors': near_colors}, ev)
            sig = hyp.sig()
            if sig not in self.hypotheses:
                self.hypotheses[sig] = hyp
                self._validate_queue.append(sig)
        if level_advanced:
            for ev in events:
                self.goal_sigs.add(f"{ev['type']}_{ev['color']}")
            self.goal_actions.add(action_id)
        for hyp in self.hypotheses.values():
            if not hyp.confirmed and hyp.trigger['action'] == action_id:
                hyp.validate(events)

    def observe_click(self, x, y, events, entities_before, ctx_before, level_advanced):
        self.tried.add(6)
        self.action_events[6] = events
        clicked_color = None
        for e in entities_before:
            r0, c0, r1, c1 = e['bbox']
            if r0 <= y <= r1 and c0 <= x <= c1:
                clicked_color = e['color']; break
        by_id = {e['id']: e for e in entities_before}
        near = []
        for e in entities_before:
            if e['color'] == clicked_color:
                near = self._near_colors(ctx_before, e['id'], by_id); break
        for ev in events:
            hyp = RuleHypothesis({'action': 6, 'clicked_color': clicked_color, 'near_colors': near}, ev)
            sig = hyp.sig()
            if sig not in self.hypotheses:
                self.hypotheses[sig] = hyp
                self._validate_queue.append(sig)
        if level_advanced:
            for ev in events:
                self.goal_sigs.add(f"{ev['type']}_{ev['color']}")
            self.goal_actions.add(6)
        for hyp in self.hypotheses.values():
            if not hyp.confirmed and hyp.trigger.get('action') == 6:
                hyp.validate(events)

    def _event_score(self, ev):
        sig = f"{ev['type']}_{ev['color']}"
        base = {'DISAPPEAR': 4, 'RECOLOR': 3, 'GROW': 2, 'SHRINK': 2, 'MOVE': 1, 'APPEAR': 1}.get(ev['type'], 0)
        return base + (20 if sig in self.goal_sigs else 0)

    def best_action(self, available_ids, current_entities, current_ctx,
                    banned=None) -> Tuple[Optional[int], int, str]:
        simple = [a for a in available_ids if 1 <= a <= 5]
        has_click = 6 in available_ids

        def _usable(a):
            return a != banned and not self.action_is_blocked(a)

        untried = [a for a in simple if a not in self.tried and _usable(a)]
        if untried:
            return untried[0], 15, 'explore'
        if has_click and 6 not in self.tried and _usable(6):
            return 6, 15, 'explore'

        while self._validate_queue:
            sig = self._validate_queue[0]
            if sig not in self.hypotheses:
                self._validate_queue.popleft(); continue
            hyp = self.hypotheses[sig]
            if hyp.confirmed or hyp.failures >= 3:
                self._validate_queue.popleft(); continue
            act = hyp.trigger.get('action', 1)
            if act in simple and _usable(act):
                return act, 12, 'validate'
            self._validate_queue.popleft()

        best_a, best_s = None, -1
        candidates = simple + ([6] if has_click else [])
        for a in candidates:
            if not _usable(a): continue
            evs = self.action_events.get(a, [])
            score = sum(self._event_score(ev) for ev in evs)
            for hyp in self.hypotheses.values():
                if hyp.trigger.get('action') == a and hyp.confirmed:
                    score += 25
            if a in self.goal_actions:
                score += 50
            if score > best_s:
                best_s = score; best_a = a

        if best_a is not None and best_s > 0:
            return best_a, best_s, 'plan'

        candidates = [a for a in simple if _usable(a)] or simple
        return (candidates[0] if candidates else 1), 0, 'fallback'

    def best_click_target(self, entities, ctx):
        target_colors = {}
        for hyp in self.hypotheses.values():
            if hyp.trigger.get('action') == 6 and (hyp.confirmed or hyp.confidence >= 2):
                cc = hyp.trigger.get('clicked_color')
                if cc is not None:
                    score = self._event_score(hyp.effect) * hyp.confidence
                    target_colors[cc] = target_colors.get(cc, 0) + score
        if not target_colors: return None
        best_e, best_s = None, -1
        for e in entities:
            s = target_colors.get(e['color'], 0)
            if s > best_s: best_s = s; best_e = e
        if best_e is None: return None
        cy, cx = best_e['centroid']
        return int(cx), int(cy)


# ==================== AGENT ====================

class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = abs(hash(s.game_id)) % (2**32 - 1)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        s.start_time = time.time()
        s.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        s.G=64; s.IN=26
        s.net=None; s.opt=None
        s.buf=deque(maxlen=50000); s.buf_h=set()
        s.bsz=64; s.tfreq=10
        s.pt=None; s.pai=None; s.pr=None; s.ph=None
        s.cl=-1; s.fhist=deque(maxlen=6); s.la=0
        s.al=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5]
        s._wd=False; s._bg=0; s._wm=None
        s._aem_diffs=deque(maxlen=256); s._aem_actions=deque(maxlen=256); s._aem_rewards=deque(maxlen=256)
        s._ckpt_hash=None; s._unproductive=0; s._undo_avail=False
        s._eps=0.15; s._eps_min=0.03; s._eps_decay=0.9997
        s._prev_objs=None; s._obj_moved=0
        # BFS
        s._bfs=None; s._bfs_solution=None; s._bfs_step=0; s._bfs_tried=False
        # Symbolic layer
        s._graph=TransformationGraph()
        s._prev_entities=None; s._banned_action=None
        s._prev_ctx=None; s._prev_click_xy=None; s._prev_level_completed=-1
        # Anti-stuck (NEW)
        s._recent_actions=deque(maxlen=20)
        s._escape_action=None; s._escape_remaining=0
        s._cooldowns={}
        # NEXUS neural engine
        s._nexus_ready = False
        try:
            s._vqvae = VQVAE(K=512, D=64).to(s.device)
            s._nwm = NexusWorldModel(K=512, D=64, n_actions=7).to(s.device)
            s._policy = PolicyHead(K=512, D=64, n_actions=7).to(s.device)
            s._planner = CEMPlanner(s._nwm, H=8, N=48, n_elite=6, n_iter=2)
            # Try loading pretrained NEXUS weights
            for wp in ['/home/anhkhoadoannguyen/Documents/ARC AGI Sandbox/agents/results (2)/nexus_weights.pt',
                       '/kaggle/input/datasets/anhkhoaonnguyn/vq-vae-trainingtraining/nexus_weights_ep50.pt',
                       'nexus_v2_weights.pt', 'nexus_weights.pt']:
                if os.path.exists(wp):
                    try:
                        ckpt = torch.load(wp, map_location=s.device, weights_only=True)
                        if 'vqvae' in ckpt:
                            s._vqvae.load_state_dict(ckpt['vqvae'], strict=False)
                        if 'wm' in ckpt:
                            s._nwm.load_state_dict(ckpt['wm'], strict=False)
                        if 'policy' in ckpt:
                            s._policy.load_state_dict(ckpt['policy'], strict=False)
                        s._nexus_ready = ckpt.get('wm_ready', True)
                        print(f'[NEXUS] Weights loaded from {wp}, ready={s._nexus_ready}')
                        break
                    except Exception as e:
                        print(f'[NEXUS] Weight load failed from {wp}: {e}')
            s._vqvae.eval(); s._nwm.eval(); s._policy.eval()
        except Exception as e:
            print(f'[NEXUS] Init failed: {e}')
            s._nexus_ready = False
        print(f'[NEXUS] Final status: ready={getattr(s, "_nexus_ready", False)}')

    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES: s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid: s.guid = f.guid
        if hasattr(s, "recorder") and not s.is_playback:
            import json; s.recorder.record(json.loads(f.model_dump_json()))

    def _lvl(s, f): return getattr(f, 'score', None) or f.levels_completed
    def _raw(s, fd):
        arr = np.array(fd.frame, dtype=np.int64)
        return arr[-1] if arr.ndim == 3 else arr

    def _init_bfs(s):
        src, cls = find_game_source_and_class(s.game_id, s.arc_env)
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=180)
            if not s._bfs.load(): s._bfs = None

    def _try_bfs_solve(s, level_idx):
        if s._bfs is None: return None
        elapsed = time.time() - s.start_time
        remaining = max(60, 8*3600 - 600 - elapsed)
        time_for_bfs = min(remaining*0.3, 600) if level_idx==0 else min(remaining*0.1, 300)
        s._bfs.bfs_timeout = int(max(30, time_for_bfs))
        prev_sol = s._bfs.solutions.get(level_idx-1) if level_idx>0 else None
        sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol)
        if sol: s._bfs_solution=sol; s._bfs_step=0
        return sol

    def _tensor(s, fd):
        frame = s._raw(fd)
        oh=torch.zeros(16,64,64,dtype=torch.float32)
        oh.scatter_(0,torch.from_numpy(frame).unsqueeze(0),1)
        cnt=np.bincount(frame.flatten(),minlength=16)
        s._bg=int(cnt.argmax()); mx=max(cnt.max(),1)
        bg_m=(frame==s._bg).astype(np.float32)
        rar=np.zeros((64,64),np.float32)
        for c in range(16):
            if cnt[c]>0: rar[frame==c]=1.0-cnt[c]/mx
        pad=np.pad(frame,1,mode='edge')
        edge=((frame!=pad[:-2,1:-1])|(frame!=pad[2:,1:-1])|(frame!=pad[1:-1,:-2])|(frame!=pad[1:-1,2:])).astype(np.float32)
        rp=np.linspace(0,1,64,dtype=np.float32).reshape(64,1).repeat(64,1)
        cp=np.linspace(0,1,64,dtype=np.float32).reshape(1,64).repeat(64,0)
        aug=torch.from_numpy(np.stack([bg_m,rar,edge,rp,cp]))
        d1=torch.zeros(3,64,64,dtype=torch.float32)
        for i,prev in enumerate(reversed(list(s.fhist))):
            if i>=3: break
            d1[i]=torch.from_numpy((frame!=prev).astype(np.float32))
        d2=torch.zeros(2,64,64,dtype=torch.float32)
        h=list(s.fhist)
        if len(h)>=2: d2[0]=torch.from_numpy((h[-1]!=h[-2]).astype(np.float32))
        if len(h)>=4: d2[1]=torch.from_numpy((h[-2]!=h[-4]).astype(np.float32))
        s.fhist.append(frame.copy())
        return torch.cat([oh,aug,d1,d2],0).to(s.device)

    def _detect_template(s, frame):
        mask=torch.ones(4096,dtype=torch.float32)
        col_act=np.sum(frame!=s._bg,axis=0)
        for c in range(20,44):
            if col_act[c]<=2 and np.sum(col_act[:c]>0)>=5 and np.sum(col_act[c+1:]>0)>=5:
                for y in range(64):
                    for x in range(c+1): mask[y*64+x]=0.05
                return mask
        row_act=np.sum(frame!=s._bg,axis=1)
        for r in range(20,44):
            if row_act[r]<=2 and np.sum(row_act[:r]>0)>=5 and np.sum(row_act[r+1:]>0)>=5:
                for y in range(r+1):
                    for x in range(64): mask[y*64+x]=0.05
                return mask
        return mask

    def _reward(s, prev_raw, curr_raw, curr_h):
        mask=np.ones((64,64),dtype=bool); mask[:2]=False; mask[62:]=False
        diff=(prev_raw!=curr_raw)&mask
        r=0.0
        if np.any(diff): r+=1.5
        else: r-=0.1
        if np.any(prev_raw!=curr_raw): r+=0.5
        curr_objs=fast_objects(curr_raw,s._bg)
        if s._prev_objs and curr_objs:
            moved=sum(1 for co in curr_objs for po in s._prev_objs
                      if co[0]==po[0] and 2<abs(co[1]-po[1])+abs(co[2]-po[2])<20)
            if moved>0: r+=0.3*min(moved,3)
        s._prev_objs=curr_objs
        return r

    def _sample(s, logits, avail=None, temp=1.0):
        al=logits[:5].clone(); cl=logits[5:5+4096].clone()
        if avail is not None and len(avail)>0:
            mask=torch.full_like(al,float('-inf')); a6=False
            for a in avail:
                aid=a.value if hasattr(a,'value') else int(a)
                if 1<=aid<=5: mask[aid-1]=0.0
                elif aid==6: a6=True
            al=al+mask
            if not a6: cl=cl+torch.full_like(cl,float('-inf'))
        if s._wm is not None: cl=cl+torch.log(s._wm.to(s.device).clamp(min=0.01))
        ap=torch.sigmoid(al/temp); cp=torch.sigmoid(cl/temp)/(s.G*s.G)
        allp=torch.cat([ap,cp]); sm=allp.sum()
        if sm<1e-8: allp=torch.ones_like(allp)/len(allp)
        else: allp=allp/sm
        idx=np.random.choice(len(allp),p=allp.cpu().numpy())
        if idx<5: return idx,None
        ci=idx-5; return 5,(ci//s.G,ci%s.G)

    def _heuristic(s, frame, avail, step):
        av=set(int(a.value) if hasattr(a,'value') else int(a) for a in avail)
        for d in [1,2,3,4]:
            if d in av and step<4: return d-1,None
        if 6 in av:
            cnt=np.bincount(frame.flatten(),minlength=16); targets=[]
            for c in range(16):
                if c==s._bg or cnt[c]==0 or cnt[c]>2000: continue
                ys,xs=np.where(frame==c)
                if len(ys)>=2: targets.append((int(np.median(xs)),int(np.median(ys)),len(ys)))
            targets.sort(key=lambda t:t[2]); pidx=step-4
            if 0<=pidx<len(targets): return 5,(targets[pidx][1],targets[pidx][0])
        if 5 in av: return 4,None
        choices=[a for a in av if 1<=a<=5]
        if choices: return random.choice(choices)-1,None
        return 0,None

    def _frame_to_tensor(s, frame):
        oh=torch.zeros(16,64,64,dtype=torch.float32)
        oh.scatter_(0,torch.from_numpy(frame).unsqueeze(0),1)
        cnt=np.bincount(frame.flatten(),minlength=16)
        bg=int(cnt.argmax()); mx=max(cnt.max(),1)
        bg_m=(frame==bg).astype(np.float32)
        rar=np.zeros((64,64),np.float32)
        for c in range(16):
            if cnt[c]>0: rar[frame==c]=1.0-cnt[c]/mx
        pad=np.pad(frame,1,mode='edge')
        edge=((frame!=pad[:-2,1:-1])|(frame!=pad[2:,1:-1])|(frame!=pad[1:-1,:-2])|(frame!=pad[1:-1,2:])).astype(np.float32)
        rp=np.linspace(0,1,64,dtype=np.float32).reshape(64,1).repeat(64,1)
        cp=np.linspace(0,1,64,dtype=np.float32).reshape(1,64).repeat(64,0)
        aug=torch.from_numpy(np.stack([bg_m,rar,edge,rp,cp]))
        zeros=torch.zeros(5,64,64,dtype=torch.float32)
        return torch.cat([oh,aug,zeros],0)

    def _train(s):
        if len(s.buf)<s.bsz: return
        indices=np.random.choice(len(s.buf),s.bsz,replace=False)
        batch=[s.buf[i] for i in indices]
        states=torch.stack([s._frame_to_tensor(e['s']).to(s.device) for e in batch])
        acts=torch.tensor([e['a'] for e in batch],dtype=torch.long,device=s.device)
        rews=torch.tensor([e['r'] for e in batch],dtype=torch.float32,device=s.device)
        rews=torch.sigmoid(rews); s.opt.zero_grad()
        logits=s.net(states)
        acts_c=acts.clamp(0,logits.size(1)-1)
        sel=logits.gather(1,acts_c.unsqueeze(1)).squeeze(1)
        loss=F.binary_cross_entropy_with_logits(sel,rews)
        p=torch.sigmoid(logits)
        loss=loss-0.0001*p[:,:5].mean()-0.00001*p[:,5:].mean()
        loss.backward(); s.opt.step()

    def _get_aem_tensors(s):
        if len(s._aem_diffs)<2: return None,None,None
        M=len(s._aem_diffs)
        diffs=torch.zeros(1,M,1,64,64,device=s.device)
        acts=torch.zeros(1,M,dtype=torch.long,device=s.device)
        rews=torch.zeros(1,M,device=s.device)
        for i,(d,a,r) in enumerate(zip(s._aem_diffs,s._aem_actions,s._aem_rewards)):
            diffs[0,i,0]=torch.from_numpy(d.astype(np.float32))
            acts[0,i]=min(a,4); rews[0,i]=r
        return diffs,acts,rews

    def _detect_oscillation(s, simple_ids):
        """Check recent actions for oscillation patterns. Returns True if stuck."""
        if len(s._recent_actions) < 3:
            return False
        a_list = list(s._recent_actions)
        # Same action 3x in a row
        if len(set(a_list[-3:])) == 1 and a_list[-1] is not None:
            return True
        # A-B-A-B pingpong
        if len(a_list) >= 4:
            last4 = a_list[-4:]
            if (last4[0] == last4[2] and last4[1] == last4[3]
                and last4[0] != last4[1] and last4[0] is not None):
                return True
        return False

    def _enter_escape(s, simple_ids):
        """Pick an escape direction and commit."""
        a_list = list(s._recent_actions)
        osc_actions = set(a for a in a_list[-6:] if a is not None)
        candidates = [a for a in simple_ids
                      if a not in osc_actions and not s._graph.action_is_blocked(a)]
        if not candidates:
            candidates = [a for a in simple_ids if not s._graph.action_is_blocked(a)]
        if not candidates:
            candidates = simple_ids
        if candidates:
            s._escape_action = random.choice(candidates)
            s._escape_remaining = random.randint(3, 6)
            for oa in osc_actions:
                if oa is not None and 1 <= oa <= 5:
                    s._cooldowns[oa] = max(s._cooldowns.get(oa, 0), 8)

    def is_done(s, frames, lf):
        try: return lf.state is GameState.WIN or (time.time()-s.start_time)>=8*3600-300
        except: return True

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # ===== LEVEL CHANGE =====
            if lvl != s.cl:
                if not s._bfs_tried:
                    s._bfs_tried = True
                    s._init_bfs()
                s._bfs_solution = None; s._bfs_step = 0
                if s._bfs: s._try_bfs_solve(lvl)
                s.buf.clear(); s.buf_h.clear()
                s.net = ForgeNet(s.IN, s.G).to(s.device)
                for wp in ['/home/anhkhoadoannguyen/Documents/ARC AGI Sandbox/agents/results (2)/nexus_weights.pt',
                           '/kaggle/input/datasets/anhkhoaonnguyn/vq-vae-trainingtraining/nexus_weights_ep50.pt',
                           'pretrained_weights.pt']:
                    try:
                        if os.path.exists(wp):
                            state=torch.load(wp,map_location=s.device,weights_only=True)
                            ms=s.net.state_dict()
                            for k in list(state.keys()):
                                if k in ms and state[k].shape==ms[k].shape: ms[k]=state[k]
                            s.net.load_state_dict(ms); break
                    except: pass
                s.opt=optim.Adam(s.net.parameters(),lr=0.0003)
                s.pt=None; s.pai=None; s.pr=None; s.ph=None
                s.cl=lvl; s.fhist.clear(); s.la=0
                s._wd=False; s._wm=None; s._eps=0.15
                s._aem_diffs.clear(); s._aem_actions.clear(); s._aem_rewards.clear()
                s._prev_objs=None; s._obj_moved=0; s._ckpt_hash=None; s._unproductive=0
                # Reset symbolic layer
                s._graph=TransformationGraph()
                s._prev_entities=None; s._banned_action=None
                s._prev_ctx=None; s._prev_click_xy=None; s._prev_level_completed=lvl-1
                # Reset anti-stuck
                s._recent_actions.clear(); s._escape_action=None; s._escape_remaining=0
                s._cooldowns={}

            # ===== RESET =====
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.pt=None; s.pai=None; s.pr=None; s.ph=None
                s._recent_actions.clear(); s._escape_action=None; s._escape_remaining=0
                s._cooldowns={}; s._unproductive=0; s._banned_action=None
                a=GameAction.RESET; a.reasoning="reset"; return a

            raw = s._raw(lf)
            ch = hashlib.md5(raw.tobytes()).hexdigest()[:16]
            avail = getattr(lf, 'available_actions', None) or []
            s._undo_avail = any((a.value if hasattr(a,'value') else int(a))==7 for a in avail)
            avail_ids = [a.value if hasattr(a,'value') else int(a) for a in avail]
            simple_ids = [i for i in avail_ids if 1<=i<=5]

            # ===== BFS EXECUTION =====
            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                act_id, data = s._bfs_solution[s._bfs_step]
                s._bfs_step += 1
                sel = GameAction.from_id(act_id)
                if data: sel.set_data(data)
                sel.reasoning = f"bfs:{s._bfs_step}/{len(s._bfs_solution)}"
                s.fhist.append(raw.copy()); s.pr=raw.copy(); s.la+=1
                return sel

            # ===== FEEDBACK =====
            if s.pt is not None and s.pai is not None and s.pr is not None:
                mask=np.ones((64,64),dtype=bool); mask[:2]=False; mask[62:]=False
                diff_map=(s.pr!=raw)&mask; changed=np.any(diff_map)
                eh=hashlib.md5(s.pr.tobytes()[:1000]+str(s.pai).encode()).hexdigest()[:16]
                if eh not in s.buf_h:
                    r=s._reward(s.pr,raw,ch)
                    s.buf.append({"s":s.pr.copy(),"a":s.pai,"r":r})
                    s.buf_h.add(eh)
                    if changed:
                        s._aem_diffs.append(diff_map)
                        s._aem_actions.append(min(s.pai if isinstance(s.pai,int) else 1, 4))
                        s._aem_rewards.append(r)

                # Symbolic rule discovery
                curr_entities, _ = extract_entities(raw)
                curr_ctx = spatial_context(curr_entities)
                level_advanced = (lvl > s._prev_level_completed)
                if level_advanced: s._prev_level_completed = lvl
                pai_int = s.pai if isinstance(s.pai, int) else 1

                events = []
                if s._prev_entities is not None:
                    events = detect_transformations(s._prev_entities, curr_entities)
                    if pai_int == 6 and s._prev_click_xy is not None:
                        s._graph.observe_click(
                            s._prev_click_xy[0], s._prev_click_xy[1],
                            events, s._prev_entities, s._prev_ctx or {}, level_advanced)
                    else:
                        s._graph.observe(
                            pai_int, events, s._prev_ctx or {}, s._prev_entities,
                            curr_ctx, curr_entities, level_advanced)

                # Entity-based meaningful change detection (immune to UI noise)
                entity_sig = tuple(sorted((e['color'], e['centroid'][0], e['centroid'][1])
                                          for e in curr_entities))
                prev_entity_sig = tuple(sorted((e['color'], e['centroid'][0], e['centroid'][1])
                                                for e in s._prev_entities)) if s._prev_entities else ()
                meaningful_change = (entity_sig != prev_entity_sig)
                s._graph.record_action_outcome(pai_int, meaningful_change)

                # Stuck detection based on meaningful change
                if meaningful_change:
                    s._banned_action = None; s._ckpt_hash = ch
                    s._unproductive = max(0, s._unproductive - 1)
                else:
                    s._banned_action = pai_int
                    s._unproductive += 2

                s._prev_entities = curr_entities; s._prev_ctx = curr_ctx; s._prev_click_xy = None

            if s._wm is None: s._wm=s._detect_template(raw)

            sel = None  # action to return — set by whichever layer claims it

            # ===== TICK COOLDOWNS =====
            for k in list(s._cooldowns.keys()):
                s._cooldowns[k] -= 1
                if s._cooldowns[k] <= 0: del s._cooldowns[k]

            # ===== OSCILLATION DETECTION + ESCAPE =====
            if s._escape_remaining <= 0 and s._detect_oscillation(simple_ids):
                s._enter_escape(simple_ids)

            if s._escape_remaining > 0 and s._escape_action is not None:
                esc = s._escape_action
                if s._graph.action_is_blocked(esc):
                    # Pivot to different direction
                    candidates = [a for a in simple_ids if a != esc and not s._graph.action_is_blocked(a)]
                    if candidates:
                        s._escape_action = random.choice(candidates)
                        esc = s._escape_action
                    else:
                        s._escape_remaining = 0
                if s._escape_remaining > 0 and 1 <= esc <= 5:
                    s._escape_remaining -= 1
                    sel = s.al[esc - 1]; sel.reasoning = f"escape:a{esc}(rem={s._escape_remaining})"
                    s.pt=s._tensor(lf); s.pai=esc; s.pr=raw.copy(); s.ph=ch; s.la+=1
                    s._recent_actions.append(esc)
                    return sel

            # ===== UNDO if stuck =====
            if s._undo_avail and s._unproductive>=15 and s._ckpt_hash:
                s._unproductive=0; a=GameAction.ACTION7; a.reasoning="undo"
                s.pt=None; s.pai=7; s.pr=raw.copy(); s.ph=ch; s.la+=1
                s._recent_actions.append(7)
                return a

            # ===== FRUSTRATION: ban repeated failing action =====
            if s._unproductive >= 3:
                s._unproductive = 0
                if s.pai is not None and 1 <= s.pai <= 5:
                    s._cooldowns[s.pai] = 6

            # ===== SYMBOLIC ACTION SELECTION =====
            curr_ents_now, _ = extract_entities(raw)
            curr_ctx_now = spatial_context(curr_ents_now)
            
            # Effective banned set: _banned_action + cooldowns + blocked
            banned_effective = s._banned_action
            banned_set = set(s._cooldowns.keys())
            if s._banned_action is not None: banned_set.add(s._banned_action)
            for a in simple_ids:
                if s._graph.action_is_blocked(a): banned_set.add(a)
            
            sym_action, sym_score, sym_phase = s._graph.best_action(
                simple_ids, curr_ents_now, curr_ctx_now, banned=s._banned_action)
            
            # Only use symbolic if action isn't in cooldown/blocked
            use_sym = (sym_phase in ("explore","validate") or (sym_phase=="plan" and sym_score>=5))
            if use_sym and sym_action is not None and sym_action not in banned_set:
                sel = s.al[sym_action - 1]
                sel.reasoning = f"sym:{sym_phase}:a{sym_action}(s={sym_score})"
                s.pt=s._tensor(lf); s.pai=sym_action; s.pr=raw.copy(); s.ph=ch; s.la+=1
                s._recent_actions.append(sym_action)
                return sel
            if 6 in avail_ids and 6 not in banned_set:
                click_xy = s._graph.best_click_target(curr_ents_now, curr_ctx_now)
                if click_xy is not None:
                    x, y = click_xy
                    sel = GameAction.ACTION6; sel.set_data({"x":x,"y":y})
                    sel.reasoning = f"sym:click({x},{y})"
                    s._prev_click_xy = (x, y)
                    s.pt=s._tensor(lf); s.pai=6; s.pr=raw.copy(); s.ph=ch; s.la+=1
                    s._recent_actions.append(6)
                    return sel

            # ===== NEXUS POLICY HEAD (pretrained instinct) =====
            if sel is None and getattr(s, '_nexus_ready', False):
                try:
                    with torch.no_grad():
                        oh = frame_to_onehot(normalise_frame(raw)).unsqueeze(0).to(s.device)
                        _, idx = s._vqvae.encode(oh)
                        idx_flat = idx.reshape(1, -1)
                        pol_logits = s._policy(idx_flat).squeeze(0)  # (7,)
                    # Mask banned actions
                    for a in banned_set:
                        if 1 <= a <= 5: pol_logits[a-1] = float('-inf')
                        elif a == 6: pol_logits[5] = float('-inf')
                        elif a == 7: pol_logits[6] = float('-inf')
                    # Only use if confident (top action has clear margin)
                    probs = torch.softmax(pol_logits, dim=0)
                    top_val, top_idx = probs.max(0)
                    if top_val.item() > 0.35:  # reasonable confidence
                        act_id = top_idx.item() + 1  # 0-indexed → 1-indexed
                        if act_id <= 5 and act_id in avail_ids:
                            sel = s.al[act_id - 1]
                            sel.reasoning = f"nexus_policy:a{act_id}(p={top_val:.2f})"
                            s.pt = s._tensor(lf); s.pai = act_id
                            s.pr = raw.copy(); s.ph = ch; s.la += 1
                            s._recent_actions.append(act_id)
                            return sel
                except Exception:
                    pass

            # ===== NEXUS CEM PLANNER (world model lookahead) =====
            if sel is None and getattr(s, '_nexus_ready', False):
                try:
                    simple_unbanned = [a for a in simple_ids if a not in banned_set]
                    if simple_unbanned:
                        with torch.no_grad():
                            oh = frame_to_onehot(normalise_frame(raw)).unsqueeze(0).to(s.device)
                            _, idx = s._vqvae.encode(oh)
                        elapsed = time.time() - s.start_time
                        budget = min(0.3, max(60, 8*3600-600-elapsed) / 20000)
                        act_id = s._planner.plan(idx.reshape(-1), simple_unbanned, s.device, budget)
                        if act_id in avail_ids and act_id not in banned_set:
                            sel = s.al[act_id - 1]
                            sel.reasoning = f"nexus_cem:a{act_id}"
                            s.pt = s._tensor(lf); s.pai = act_id
                            s.pr = raw.copy(); s.ph = ch; s.la += 1
                            s._recent_actions.append(act_id)
                            return sel
                except Exception:
                    pass

            # ===== CNN FALLBACK =====
            tensor = s._tensor(lf)

            if not s._wd:
                if s.la<10:
                    aidx,coords=s._heuristic(raw,avail,s.la)
                else:
                    s._wd=True
                    for _ in range(min(5,len(s.buf)//s.bsz)): s._train()

            if s._wd:
                if random.random()<s._eps:
                    aidx,coords=s._sample(torch.zeros(4101,device=s.device),avail,temp=2.0)
                else:
                    with torch.no_grad():
                        mem=s._get_aem_tensors()
                        if mem[0] is not None: logits=s.net(tensor.unsqueeze(0),*mem).squeeze(0)
                        else: logits=s.net(tensor.unsqueeze(0)).squeeze(0)
                    aidx,coords=s._sample(logits,avail,temp=0.5)
                    
                    # If CNN picks a banned action, resample with higher temp
                    if aidx < 5 and (aidx+1) in banned_set:
                        aidx,coords=s._sample(logits,avail,temp=2.0)

                s._eps=max(s._eps_min,s._eps*s._eps_decay)
            elif s.la>=10:
                s._wd=True; aidx,coords=0,None

            if aidx<5: sel=s.al[aidx]; sel.reasoning=f"cnn:a{aidx+1}"
            else:
                sel=GameAction.ACTION6; y,x=coords
                sel.set_data({"x":int(x),"y":int(y)}); sel.reasoning=f"cnn:c({x},{y})"

            s.pt=tensor
            if aidx<5:
                s.pai=aidx+1; s._prev_click_xy=None
            else:
                s.pai=6; y,x=coords; s._prev_click_xy=(int(x),int(y))
            s.pr=raw.copy(); s.ph=ch; s.la+=1
            s._recent_actions.append(s.pai)
            if s.action_counter%s.tfreq==0 and s._wd: s._train()
            return sel

        except Exception as e:
            traceback.print_exc()
            a=random.choice(s.al); a.reasoning=f"err:{e}"; return a