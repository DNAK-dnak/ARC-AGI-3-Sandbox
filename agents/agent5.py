#%%writefile /kaggle/working/my_agent.py
# ======================================================================
# SOVEREIGN v5.0 — BFS + TransformationGraph + Pathfinder Planner + CNN
#
# Decision priority:
#   1. BFS (offline exact solver)
#   2. TransformationGraph symbolic layer (rule discovery + goal actions)
#   3. Pathfinder Planner (entity navigation + interaction discovery)
#   4. Anti-stuck escape (oscillation detection + cooldowns)
#   5. CNN ForgeNet fallback
#
# Key improvements over 0.07 (agent5 / Planner-only):
#   - TransformationGraph restored for symbolic rule discovery
#   - Oscillation detection + escape mode restored
#   - Cooldown system restored
#   - UNDO on stuck restored
#   - Planner used for navigation when TGraph has no good action
#   - BFS warmup_prefix reset bug fixed
#   - Boolean win-condition detection
# ======================================================================
import copy
import glob
import hashlib
import heapq
import importlib.util
import logging
import os
import random
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

logger = logging.getLogger(__name__)


# ======================================================================
# PATHFINDER — Stage 1: Entity extraction (lightweight, no dataclass)
# ======================================================================

def pf_detect_bg(frame):
    return int(np.bincount(frame.flatten(), minlength=16).argmax())


def pf_extract_entities(frame, bg=-1):
    """Fast 4-connected flood-fill entity extraction. Returns list of dicts."""
    if bg < 0:
        bg = pf_detect_bg(frame)
    H, W = frame.shape
    visited = np.zeros((H, W), dtype=bool)
    entities = []
    eid = 0
    for row in range(H):
        for col in range(W):
            if visited[row, col]:
                continue
            c = int(frame[row, col])
            if c == bg:
                visited[row, col] = True
                continue
            pixels = []
            q = deque([(row, col)])
            visited[row, col] = True
            while q:
                r2, c2 = q.popleft()
                pixels.append((r2, c2))
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr, nc = r2+dr, c2+dc
                    if 0<=nr<H and 0<=nc<W and not visited[nr,nc] and frame[nr,nc]==c:
                        visited[nr,nc] = True
                        q.append((nr,nc))
            if pixels:
                rows = [p[0] for p in pixels]
                cols = [p[1] for p in pixels]
                cy = sum(rows)/len(rows)
                cx = sum(cols)/len(cols)
                entities.append({
                    'id': eid, 'color': c, 'pixels': len(pixels),
                    'centroid': (cy, cx),
                    'bbox': (min(rows), min(cols), max(rows), max(cols)),
                })
                eid += 1
    return entities, bg


def pf_compute_diff(prev_ents, curr_ents, prev_frame, curr_frame):
    """Compare entity lists, return list of change dicts."""
    if np.array_equal(prev_frame, curr_frame):
        return []
    changes = []
    prev_by_color = {}
    curr_by_color = {}
    for e in prev_ents:
        prev_by_color.setdefault(e['color'], []).append(e)
    for e in curr_ents:
        curr_by_color.setdefault(e['color'], []).append(e)

    matched_prev = set()
    matched_curr = set()
    for color in set(prev_by_color) | set(curr_by_color):
        for pe in prev_by_color.get(color, []):
            best_i, best_d = None, float('inf')
            for i, ce in enumerate(curr_by_color.get(color, [])):
                if (color, i) in matched_curr:
                    continue
                d = (abs(pe['centroid'][0]-ce['centroid'][0])
                     + abs(pe['centroid'][1]-ce['centroid'][1]))
                if d < best_d and d < 32:
                    best_d, best_i = d, i
            if best_i is not None:
                ce = curr_by_color[color][best_i]
                matched_prev.add(pe['id'])
                matched_curr.add((color, best_i))
                dr = ce['centroid'][0] - pe['centroid'][0]
                dc = ce['centroid'][1] - pe['centroid'][1]
                if abs(dr) > 0.5 or abs(dc) > 0.5:
                    changes.append({'type':'moved','eid':pe['id'],'color':color,
                                    'dr':round(dr),'dc':round(dc)})
        for pe in prev_by_color.get(color, []):
            if pe['id'] not in matched_prev:
                changes.append({'type':'vanished','eid':pe['id'],'color':color})
    for color, cl in curr_by_color.items():
        for i, ce in enumerate(cl):
            if (color, i) not in matched_curr:
                changes.append({'type':'appeared','eid':ce['id'],'color':color,
                                'centroid':ce['centroid']})
    return changes


def pf_get_agent_pos(entities, agent_color):
    for e in entities:
        if e['color'] == agent_color:
            return e['centroid']
    return None


def pf_navigate_toward(agent_pos, target_pos, move_map):
    """Return action_id that reduces distance to target. move_map: {aid:(dr,dc)}"""
    dr = target_pos[0] - agent_pos[0]
    dc = target_pos[1] - agent_pos[1]
    best_aid, best_red = None, 0
    for aid, (edr, edc) in move_map.items():
        red = (abs(dr)+abs(dc)) - (abs(dr-edr)+abs(dc-edc))
        if red > best_red:
            best_red, best_aid = red, aid
    return best_aid


# ======================================================================
# SYMBOLIC ENGINE — TransformationGraph (from 0.27 agent)
# ======================================================================

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
                    for dy in (-1,0,1):
                        for dx in (-1,0,1):
                            ny, nx = y+dy, x+dx
                            if (0<=ny<H and 0<=nx<W
                                    and mask[ny,nx] and labeled[ny,nx]==0):
                                labeled[ny,nx] = num
                                q.append((ny,nx))
    return labeled, num


def sym_extract_entities(frame):
    bg = int(np.bincount(frame.flatten(), minlength=16).argmax())
    entities = []
    eid = 0
    for color in range(16):
        if color == bg: continue
        mask = (frame == color)
        if not mask.any(): continue
        labeled, n = _cc_label(mask)
        for i in range(n):
            comp = (labeled == i+1)
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
    return not (b1[2]<b2[0] or b2[2]<b1[0] or b1[3]<b2[1] or b2[3]<b1[1])


def _bbox_adjacent(b1, b2, gap=2):
    return _bbox_overlap((b1[0]-gap,b1[1]-gap,b1[2]+gap,b1[3]+gap), b2)


def detect_transformations(ent_before, ent_after):
    events = []
    used_after = set()
    for e0 in ent_before:
        bs, bds = None, float('inf')
        br, bdr = None, float('inf')
        bz = None
        for i, e1 in enumerate(ent_after):
            if i in used_after: continue
            d = ((e0['centroid'][0]-e1['centroid'][0])**2
                 + (e0['centroid'][1]-e1['centroid'][1])**2)
            if e0['color']==e1['color'] and e0['pixels']==e1['pixels']:
                if d < bds: bds=d; bs=i
            elif e0['pixels']==e1['pixels'] and d < bdr:
                bdr=d; br=i
            elif e0['color']==e1['color'] and bz is None:
                bz=i
        if bs is not None:
            used_after.add(bs)
            if bds > 0:
                events.append({'type':'MOVE','color':e0['color'],'pixels':e0['pixels']})
        elif br is not None:
            used_after.add(br)
            events.append({'type':'RECOLOR','color':e0['color'],
                           'new_color':ent_after[br]['color'],'pixels':e0['pixels']})
        elif bz is not None:
            used_after.add(bz)
            np2 = ent_after[bz]['pixels']
            t = 'GROW' if np2 > e0['pixels'] else 'SHRINK'
            events.append({'type':t,'color':e0['color'],
                           'pixels':e0['pixels'],'new_pixels':np2})
        else:
            events.append({'type':'DISAPPEAR','color':e0['color'],'pixels':e0['pixels']})
    for i, e1 in enumerate(ent_after):
        if i not in used_after:
            events.append({'type':'APPEAR','color':e1['color'],'pixels':e1['pixels']})
    return events


def spatial_context(entities):
    ctx = {e['id']: {'overlaps':[],'adjacent':[]} for e in entities}
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
        return (t.get('action'), tuple(sorted(t.get('near_colors',[]))),
                e.get('type'), e.get('color'))

    def validate(self, observed_events):
        for ev in observed_events:
            if ev['type']==self.effect['type'] and ev['color']==self.effect['color']:
                self.confidence += 1
                if self.confidence >= 3: self.confirmed = True
                return True
        self.failures += 1
        return False


class TransformationGraph:
    def __init__(self):
        self.action_events: Dict[int,list] = {}
        self.action_context: Dict[int,dict] = {}
        self.hypotheses: Dict[tuple,RuleHypothesis] = {}
        self.goal_sigs: set = set()
        self.goal_actions: set = set()
        self.tried: set = set()
        self._validate_queue: deque = deque()
        self._action_fail_streak: Dict[int,int] = {}

    def _near_colors(self, ctx, entity_id, entities_by_id):
        info = ctx.get(entity_id, {})
        colors = set()
        for oid in info.get('overlaps',[]) + info.get('adjacent',[]):
            e = entities_by_id.get(oid)
            if e: colors.add(e['color'])
        return sorted(colors)

    def record_action_outcome(self, action_id, meaningful_change):
        if meaningful_change:
            self._action_fail_streak[action_id] = 0
        else:
            self._action_fail_streak[action_id] = (
                self._action_fail_streak.get(action_id, 0) + 1)

    def action_is_blocked(self, action_id, threshold=2):
        return self._action_fail_streak.get(action_id, 0) >= threshold

    def observe(self, action_id, events, ctx_before, entities_before,
                ctx_after, entities_after, level_advanced):
        self.tried.add(action_id)
        if not events:
            self.action_events[action_id] = []; return
        self.action_events[action_id] = events
        self.action_context[action_id] = ctx_before
        by_id = {e['id']:e for e in entities_before}
        for ev in events:
            nc = sorted(set(
                c for e in entities_before
                for c in self._near_colors(ctx_before, e['id'], by_id)))
            hyp = RuleHypothesis({'action':action_id,'near_colors':nc}, ev)
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
            r0,c0,r1,c1 = e['bbox']
            if r0<=y<=r1 and c0<=x<=c1:
                clicked_color = e['color']; break
        by_id = {e['id']:e for e in entities_before}
        near = []
        for e in entities_before:
            if e['color'] == clicked_color:
                near = self._near_colors(ctx_before, e['id'], by_id); break
        for ev in events:
            hyp = RuleHypothesis({'action':6,'clicked_color':clicked_color,
                                   'near_colors':near}, ev)
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
        base = {'DISAPPEAR':4,'RECOLOR':3,'GROW':2,'SHRINK':2,
                'MOVE':1,'APPEAR':1}.get(ev['type'], 0)
        return base + (20 if sig in self.goal_sigs else 0)

    def best_action(self, available_ids, current_entities, current_ctx,
                    banned=None):
        simple = [a for a in available_ids if 1<=a<=5]
        has_click = 6 in available_ids

        def _usable(a):
            return a != banned and not self.action_is_blocked(a)

        untried = [a for a in simple if a not in self.tried and _usable(a)]
        if untried: return untried[0], 15, 'explore'
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
        for a in simple + ([6] if has_click else []):
            if not _usable(a): continue
            score = sum(self._event_score(ev)
                        for ev in self.action_events.get(a, []))
            for hyp in self.hypotheses.values():
                if hyp.trigger.get('action')==a and hyp.confirmed:
                    score += 25
            if a in self.goal_actions: score += 50
            if score > best_s: best_s=score; best_a=a

        if best_a is not None and best_s > 0:
            return best_a, best_s, 'plan'

        candidates = [a for a in simple if _usable(a)] or simple
        return (candidates[0] if candidates else 1), 0, 'fallback'

    def best_click_target(self, entities, ctx):
        target_colors = {}
        for hyp in self.hypotheses.values():
            if hyp.trigger.get('action')==6 and (hyp.confirmed or hyp.confidence>=2):
                cc = hyp.trigger.get('clicked_color')
                if cc is not None:
                    target_colors[cc] = (target_colors.get(cc,0)
                                         + self._event_score(hyp.effect)*hyp.confidence)
        if not target_colors: return None
        best_e, best_s = None, -1
        for e in entities:
            s = target_colors.get(e['color'], 0)
            if s > best_s: best_s=s; best_e=e
        if best_e is None: return None
        cy, cx = best_e['centroid']
        return int(cx), int(cy)


# ======================================================================
# BFS SOLVER
# ======================================================================

class BFSSolver:
    def __init__(self, game_path, game_class_name, scan_timeout=3, bfs_timeout=120):
        self.game_path = game_path
        self.class_name = game_class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.game_cls = None
        self.solutions = {}

    def load(self):
        try:
            spec = importlib.util.spec_from_file_location('game_mod', self.game_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.game_cls = getattr(mod, self.class_name)
            return True
        except Exception as e:
            logger.warning(f"BFS load failed: {e}")
            return False

    def _state_hash(self, g, frame, hidden_fields=None):
        fh = hashlib.md5(frame.tobytes()).hexdigest()[:16]
        if hidden_fields:
            extras = [f"{f}={getattr(g,f,None)}" for f in hidden_fields
                      if getattr(g,f,None) is not None]
            if extras: return fh + "|" + "|".join(extras)
        return fh

    def _extract_win_field(self):
        try:
            import re
            source = open(self.game_path).read()
            lines = source.split('\n')
            for i, line in enumerate(lines):
                if 'self.next_level()' in line:
                    for j in range(i-1, max(0,i-8), -1):
                        s = lines[j].strip()
                        if s.startswith('if ') or s.startswith('elif '):
                            m = re.search(r'self\.(\w+)', s)
                            if m: return m.group(1)
                    break
        except: pass
        return None

    def _probe_hidden_fields(self, game, actions):
        if not actions: return []
        win_field = self._extract_win_field()
        initial = {k:v for k,v in game.__dict__.items()
                   if isinstance(v,(int,float,bool)) and not k.startswith('__')}
        changing = set()
        if win_field and win_field in initial:
            changing.add(win_field)
        for act_id, data in actions[:10]:
            g = copy.deepcopy(game)
            try:
                ai = (ActionInput(id=GameAction.from_id(act_id), data=data) if data
                      else ActionInput(id=GameAction.from_id(act_id)))
                g.perform_action(ai, raw=True)
            except: continue
            for k, v in g.__dict__.items():
                if isinstance(v,(int,float,bool)) and not k.startswith('__'):
                    if k in initial and v != initial[k]:
                        if k not in ('_action_count','_full_reset','_action_complete'):
                            changing.add(k)
        return sorted([f for f in changing
                       if not (f.startswith('_')
                               and f not in ('_current_level_index','_score'))])

    def _scan_actions(self, game, f0, bg):
        avail = game._available_actions
        actions = []
        for a in [a for a in avail if a <= 5]:
            g = copy.deepcopy(game)
            try:
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                if r.frame and np.sum(f0 != np.array(r.frame[-1])) > 0:
                    actions.append((a, None))
            except: pass
        if 6 in avail:
            t0 = time.time()
            seen = set()
            for y in range(0, 64, 2):
                if time.time()-t0 > self.scan_timeout: break
                for x in range(0, 64, 2):
                    if f0[y,x] == bg: continue
                    g = copy.deepcopy(game)
                    try:
                        r = g.perform_action(
                            ActionInput(id=GameAction.ACTION6,
                                        data={'x':x,'y':y,'game_id':'bfs'}), raw=True)
                        if not r.frame: continue
                        f = np.array(r.frame[-1])
                        if np.sum(f0!=f) > 0:
                            eh = hashlib.md5(f.tobytes()).hexdigest()[:12]
                            if eh not in seen:
                                seen.add(eh)
                                actions.append((6,{'x':x,'y':y,'game_id':'bfs'}))
                    except: pass
        return actions

    def solve_level(self, level_idx, max_states=500000, prev_solution=None):
        if not self.game_cls: return None
        # Reset warmup prefix each level
        warmup_prefix = []

        game = self.game_cls()
        game.set_level(level_idx)
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not r0.frame: return None
        f0 = np.array(r0.frame[-1])
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

        if prev_solution and level_idx > 0:
            tr = self._try_transfer(game, level_idx, prev_solution, f0)
            if tr: return tr

        actions = self._scan_actions(game, f0, bg)
        if not actions:
            for wid in [a for a in game._available_actions if a <= 4]:
                gw = copy.deepcopy(game)
                try:
                    gw.perform_action(ActionInput(id=GameAction.from_id(wid)), raw=True)
                    fa = np.array(gw.get_pixels(0,0,64,64))
                    wa = self._scan_actions(gw, fa, bg)
                    if wa:
                        game=gw; f0=fa; actions=wa
                        warmup_prefix=[(wid,None)]; break
                except: pass

        if not actions: return None

        # Phase 1: plain BFS
        visited = set()
        visited.add(self._state_hash(game, f0, None))
        base = copy.deepcopy(game)
        queue = deque([([], 0)])
        t0 = time.time()
        explored = 0

        while queue and explored < max_states and (time.time()-t0) < self.bfs_timeout:
            hist, depth = queue.popleft()
            g = copy.deepcopy(base)
            try:
                for a_id, a_data in hist:
                    ai = (ActionInput(id=GameAction.from_id(a_id),data=a_data) if a_data
                          else ActionInput(id=GameAction.from_id(a_id)))
                    g.perform_action(ai, raw=True)
            except: continue
            for act_id, data in actions:
                g2 = copy.deepcopy(g)
                try:
                    ai = (ActionInput(id=GameAction.from_id(act_id),data=data) if data
                          else ActionInput(id=GameAction.from_id(act_id)))
                    r = g2.perform_action(ai, raw=True)
                except: continue
                explored += 1
                if not r.frame: continue
                f = np.array(r.frame[-1])
                h = self._state_hash(g2, f, None)
                if h in visited: continue
                visited.add(h)
                nh = hist + [(act_id, data)]
                if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                    sol = warmup_prefix + nh
                    self.solutions[level_idx] = sol
                    return sol
                if depth < 30:
                    queue.append((nh, depth+1))

        elapsed = time.time() - t0

        # Phase 2: hidden-field A* retry
        if elapsed < self.bfs_timeout * 0.8:
            hf = self._probe_hidden_fields(game, actions)
            if hf:
                game2 = self.game_cls()
                game2.set_level(level_idx)
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                r02 = game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                if r02 and r02.frame:
                    f02 = np.array(r02.frame[-1])
                    init_st = {f: getattr(game2,f,None) for f in hf}
                    base2 = copy.deepcopy(game2)
                    v2 = {self._state_hash(game2, f02, hf)}
                    fifo = 0
                    heap = [(0, 0, fifo, [])]
                    t02 = time.time()
                    rem = max(30, self.bfs_timeout - elapsed)
                    while heap and (time.time()-t02) < rem:
                        _, dep, _, hist = heapq.heappop(heap)
                        g = copy.deepcopy(base2)
                        try:
                            for a_id, a_data in hist:
                                ai = (ActionInput(id=GameAction.from_id(a_id),data=a_data)
                                      if a_data else ActionInput(id=GameAction.from_id(a_id)))
                                g.perform_action(ai, raw=True)
                        except: continue
                        for act_id, data in actions:
                            g2 = copy.deepcopy(g)
                            try:
                                ai = (ActionInput(id=GameAction.from_id(act_id),data=data)
                                      if data else ActionInput(id=GameAction.from_id(act_id)))
                                r = g2.perform_action(ai, raw=True)
                            except: continue
                            if not r.frame: continue
                            f = np.array(r.frame[-1])
                            h = self._state_hash(g2, f, hf)
                            if h in v2: continue
                            v2.add(h)
                            nh = hist + [(act_id, data)]
                            if r.levels_completed>level_idx or g2._current_level_index>level_idx:
                                sol = warmup_prefix + nh
                                self.solutions[level_idx] = sol
                                return sol
                            # Score: sum of change in tracked fields
                            td = sum(
                                abs(getattr(g2,tf,0)-(init_st.get(tf,0) or 0))
                                if isinstance(getattr(g2,tf,None),(int,float))
                                and not isinstance(getattr(g2,tf,None),bool)
                                else (1 if getattr(g2,tf,None)!=init_st.get(tf) else 0)
                                for tf in hf)
                            if td == 0 and np.array_equal(f02, f): continue
                            fifo += 1
                            if dep < 40:
                                heapq.heappush(heap, (-td, dep+1, fifo, nh))
        return None

    def _try_transfer(self, game, level_idx, prev_solution, f1):
        try:
            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(prev_solution):
                try:
                    ai = (ActionInput(id=GameAction.from_id(act_id),data=data) if data
                          else ActionInput(id=GameAction.from_id(act_id)))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed>level_idx or g._current_level_index>level_idx:
                        sol = prev_solution[:i+1]
                        self.solutions[level_idx] = sol; return sol
                except: break

            prev_game = self.game_cls()
            prev_game.set_level(level_idx-1)
            prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r_prev = prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r_prev.frame: return None
            f0 = np.array(r_prev.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

            def get_objs(frame, bg_c):
                objs = []
                for c in range(16):
                    if c==bg_c: continue
                    mask=(frame==c); npix=int(np.sum(mask))
                    if npix<2: continue
                    ys,xs=np.where(mask)
                    objs.append({'color':c,'cx':float(np.mean(xs)),
                                 'cy':float(np.mean(ys)),'n':npix})
                return sorted(objs,key=lambda o:(o['color'],-o['n']))

            op=get_objs(f0,bg); oc=get_objs(f1,bg)
            if not op or not oc: return None
            matched=[]
            for o in op:
                best,bd=None,float('inf')
                for c in oc:
                    if c['color']==o['color'] and abs(c['n']-o['n'])<max(o['n'],c['n'])*0.5:
                        d=abs(c['cx']-o['cx'])+abs(c['cy']-o['cy'])
                        if d<bd: bd=d; best=c
                if best: matched.append((o,best))
            if not matched: return None
            dx=np.mean([m[1]['cx']-m[0]['cx'] for m in matched])
            dy=np.mean([m[1]['cy']-m[0]['cy'] for m in matched])
            transferred=[]
            for act_id,data in prev_solution:
                if data and 'x' in data:
                    nd=dict(data)
                    nd['x']=max(0,min(63,int(data['x']+dx)))
                    nd['y']=max(0,min(63,int(data['y']+dy)))
                    transferred.append((act_id,nd))
                else:
                    transferred.append((act_id,data))
            g=copy.deepcopy(game)
            for i,(act_id,data) in enumerate(transferred):
                try:
                    ai=(ActionInput(id=GameAction.from_id(act_id),data=data) if data
                        else ActionInput(id=GameAction.from_id(act_id)))
                    r=g.perform_action(ai,raw=True)
                    if r.levels_completed>level_idx or g._current_level_index>level_idx:
                        sol=transferred[:i+1]
                        self.solutions[level_idx]=sol; return sol
                except: break
        except: pass
        return None


def find_game_source_and_class(game_id, arc_env=None):
    import re
    gid = game_id.split('-')[0]
    cls_name = gid[0].upper() + gid[1:]
    src = None
    if arc_env and hasattr(arc_env,'environment_info'):
        ei = arc_env.environment_info
        if hasattr(ei,'local_dir') and ei.local_dir:
            from pathlib import Path
            ld = Path(ei.local_dir)
            for candidate in [ld/f"{gid}.py", ld/f"{cls_name.lower()}.py"]:
                if candidate.exists():
                    src = str(candidate)
                    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame',
                                  candidate.read_text()[:2000])
                    if m: cls_name = m.group(1)
                    break
    if not src:
        for pat in [f"/tmp/*/{gid}/*/{gid}.py",
                    f"/kaggle/*/{gid}*/{gid}.py",
                    f"**/game_sources/**/{gid}.py"]:
            matches = glob.glob(pat, recursive=True)
            if matches:
                src = matches[0]
                m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame',
                               open(src).read()[:2000])
                if m: cls_name = m.group(1)
                break
    return src, cls_name


# ======================================================================
# CNN
# ======================================================================

class CBAM(nn.Module):
    def __init__(s,ch,r=16):
        super().__init__()
        s.fc1=nn.Linear(ch,max(ch//r,4)); s.fc2=nn.Linear(max(ch//r,4),ch)
        s.sp=nn.Conv2d(2,1,7,padding=3)
    def forward(s,x):
        B,C,H,W=x.shape
        w=torch.sigmoid(s.fc2(F.relu(s.fc1(x.mean(dim=[2,3]))))); x=x*w.view(B,C,1,1)
        a=torch.sigmoid(s.sp(torch.cat([x.max(1,keepdim=True)[0],
                                         x.mean(1,keepdim=True)],1)))
        return x*a

class ActionEffectAttention(nn.Module):
    def __init__(s,feat_dim=64,mem_dim=32,n_actions=5):
        super().__init__()
        s.mem_dim=mem_dim
        s.diff_enc=nn.Sequential(
            nn.Conv2d(1,8,8,stride=8),nn.ReLU(),
            nn.Conv2d(8,16,4,stride=4),nn.ReLU(),
            nn.Flatten(),nn.Linear(16*2*2,mem_dim))
        s.q_proj=nn.Linear(feat_dim,mem_dim)
        s.v_proj=nn.Linear(mem_dim+1+n_actions,n_actions)
        s.scale=mem_dim**0.5
    def forward(s,cnn_feat,mem_diffs,mem_actions,mem_rewards):
        B,M=mem_actions.shape
        if M==0: return torch.zeros(B,5,device=cnn_feat.device)
        keys=s.diff_enc(mem_diffs.reshape(B*M,1,64,64)).reshape(B,M,s.mem_dim)
        q=s.q_proj(cnn_feat).unsqueeze(1)
        attn=F.softmax(torch.bmm(q,keys.transpose(1,2))/s.scale,dim=-1)
        act_oh=F.one_hot(mem_actions.clamp(0,4),5).float()
        vals=torch.cat([keys,mem_rewards.unsqueeze(-1),act_oh],dim=-1)
        return s.v_proj(torch.bmm(attn,vals).squeeze(1))

class ForgeNet(nn.Module):
    def __init__(s,in_ch=26,g=64):
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
    def forward(s,x,mem_diffs=None,mem_actions=None,mem_rewards=None):
        x=F.relu(s.c1(x)); x=F.relu(s.c2(x)); x=F.relu(s.c3(x)); f=F.relu(s.c4(x))
        f=s.attn(f); af=F.relu(s.ar(f)); af=s.ap(af).reshape(f.size(0),-1)
        al=s.ah(s.dr(F.relu(s.af(af))))
        cf=F.relu(s.cc1(f)); cf=F.relu(s.cc2(cf)); cf=F.relu(s.cc3(cf))
        cl=s.cc4(cf).reshape(f.size(0),-1)
        if mem_diffs is not None and mem_actions is not None:
            gf=s.gf(s.gp(f).reshape(f.size(0),-1))
            al=al+s.aea(gf,mem_diffs,mem_actions,mem_rewards)
        return torch.cat([al,cl],1)

def fast_objects(frame,bg):
    objs=[]
    for c in range(16):
        if c==bg: continue
        mask=(frame==c); npix=int(np.sum(mask))
        if npix<4 or npix>3000: continue
        ys,xs=np.where(mask)
        objs.append((c,float(np.mean(xs)),float(np.mean(ys)),npix))
    return objs


# ======================================================================
# AGENT
# ======================================================================

class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = abs(hash(s.game_id)) % (2**32-1)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        s.start_time = time.time()
        s.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            ('mps' if torch.backends.mps.is_available() else 'cpu'))
        s.G=64; s.IN=26
        s.net=None; s.opt=None
        s.buf=deque(maxlen=50000); s.buf_h=set()
        s.bsz=64; s.tfreq=10
        s.pt=None; s.pai=None; s.pr=None; s.ph=None
        s.cl=-1; s.fhist=deque(maxlen=6); s.la=0
        s.al=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,
              GameAction.ACTION4,GameAction.ACTION5]
        s._wd=False; s._bg=0; s._wm=None
        s._aem_diffs=deque(maxlen=256)
        s._aem_actions=deque(maxlen=256)
        s._aem_rewards=deque(maxlen=256)
        s._ckpt_hash=None; s._unproductive=0; s._undo_avail=False
        s._eps=0.15; s._eps_min=0.03; s._eps_decay=0.9997
        s._prev_objs=None; s._visited_hashes=set()
        # BFS
        s._bfs=None; s._bfs_solution=None; s._bfs_step=0; s._bfs_tried=False
        # Symbolic
        s._graph=TransformationGraph()
        s._prev_entities=None; s._banned_action=None
        s._prev_ctx=None; s._prev_click_xy=None; s._prev_level_completed=-1
        # Pathfinder nav state
        s._pf_agent_color=None     # detected from MOVE events
        s._pf_move_map={}          # {action_id: (dr,dc)}
        s._pf_pos_heatmap=defaultdict(int)
        s._pf_is_move_game=None
        s._pf_move_detect_count=0
        # Anti-stuck
        s._recent_actions=deque(maxlen=20)
        s._escape_action=None; s._escape_remaining=0
        s._cooldowns={}
        s._round_robin_idx=0
        # Click scan fallback
        s._click_grid_idx=0; s._click_targets=[]

    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES: s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid: s.guid = f.guid
        if hasattr(s,"recorder") and not s.is_playback:
            import json; s.recorder.record(json.loads(f.model_dump_json()))

    def _lvl(s, f):
        hint = getattr(f, '_level_hint', None)
        if hint is not None: return hint
        return getattr(f, 'levels_completed', 0) or 0

    def _raw(s, fd):
        arr = np.array(fd.frame, dtype=np.int64)
        return arr[-1] if arr.ndim == 3 else arr

    def _init_bfs(s):
        src, cls = find_game_source_and_class(s.game_id, s.arc_env)
        print(f"[BFS] game_id={s.game_id} src={src is not None} cls={cls}")
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=180)
            if not s._bfs.load(): s._bfs = None
            if s._bfs: s._analyze_source(src)
        else:
            print("[BFS] No source")

    def _analyze_source(s, src_path):
        import re as _re
        try: code = open(src_path, errors='ignore').read()
        except: return
        lines = code.split('\n')
        win_cond = None
        for j, line in enumerate(lines):
            if 'self.next_level()' in line:
                for k in range(j, max(0,j-8), -1):
                    stripped = lines[k].strip()
                    if stripped.startswith('if ') or stripped.startswith('elif '):
                        win_cond = stripped; break
                break
        if not win_cond: return
        m = _re.search(r'if self\.(\w+)\s*:', win_cond)
        if m:
            print(f"[SRC] Boolean win flag: {m.group(1)}")

    def _try_bfs_solve(s, level_idx):
        if s._bfs is None: return None
        elapsed = time.time() - s.start_time
        remaining = max(60, 8*3600-600-elapsed)
        tf = min(remaining*0.15, 30)
        s._bfs.bfs_timeout = int(max(30, tf))
        print(f"[BFS] Solving L{level_idx} timeout={s._bfs.bfs_timeout}s")
        t0 = time.time()
        prev = s._bfs.solutions.get(level_idx-1) if level_idx>0 else None
        try:
            sol = s._bfs.solve_level(level_idx, prev_solution=prev)
        except Exception as e:
            print(f"[BFS] CRASH: {e}"); traceback.print_exc(); return None
        print(f"[BFS] L{level_idx} {'SOLVED('+str(len(sol))+')' if sol else 'FAILED'} "
              f"in {time.time()-t0:.1f}s")
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
        edge=((frame!=pad[:-2,1:-1])|(frame!=pad[2:,1:-1])|
              (frame!=pad[1:-1,:-2])|(frame!=pad[1:-1,2:])).astype(np.float32)
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
        diff=(prev_raw!=curr_raw)&mask; r=0.0
        if curr_h not in s._visited_hashes:
            r+=1.5; s._visited_hashes.add(curr_h)
        else:
            r+=0.2 if np.any(diff) else -0.1
        curr_objs=fast_objects(curr_raw,s._bg)
        if s._prev_objs and curr_objs:
            moved=sum(1 for co in curr_objs for po in s._prev_objs
                      if co[0]==po[0] and 2<abs(co[1]-po[1])+abs(co[2]-po[2])<20)
            if moved>0: r+=0.3*min(moved,3)
        s._prev_objs=curr_objs; return r

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
                if len(ys)>=2:
                    targets.append((int(np.median(xs)),int(np.median(ys)),len(ys)))
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
        edge=((frame!=pad[:-2,1:-1])|(frame!=pad[2:,1:-1])|
              (frame!=pad[1:-1,:-2])|(frame!=pad[1:-1,2:])).astype(np.float32)
        rp=np.linspace(0,1,64,dtype=np.float32).reshape(64,1).repeat(64,1)
        cp=np.linspace(0,1,64,dtype=np.float32).reshape(1,64).repeat(64,0)
        aug=torch.from_numpy(np.stack([bg_m,rar,edge,rp,cp]))
        return torch.cat([oh,aug,torch.zeros(5,64,64)],0)

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
        if len(s._recent_actions) < 3: return False
        a_list = list(s._recent_actions)
        if len(set(a_list[-3:]))==1 and a_list[-1] is not None: return True
        if len(a_list)>=4:
            last4=a_list[-4:]
            if (last4[0]==last4[2] and last4[1]==last4[3]
                    and last4[0]!=last4[1] and last4[0] is not None):
                return True
        return False

    def _enter_escape(s, simple_ids):
        a_list = list(s._recent_actions)
        osc = set(a for a in a_list[-6:] if a is not None)
        candidates = [a for a in simple_ids
                      if a not in osc and not s._graph.action_is_blocked(a)]
        if not candidates:
            candidates = [a for a in simple_ids if not s._graph.action_is_blocked(a)]
        if not candidates: candidates = simple_ids
        if candidates:
            s._escape_action = random.choice(candidates)
            s._escape_remaining = random.randint(3,6)
            for oa in osc:
                if oa is not None and 1<=oa<=5:
                    s._cooldowns[oa] = max(s._cooldowns.get(oa,0), 8)

    def _pf_navigate(s, raw, simple_ids, banned_set):
        """Pathfinder navigation: use learned move_map to navigate toward nearest entity."""
        if not s._pf_move_map or s._pf_agent_color is None:
            return None
        ents, bg = pf_extract_entities(raw)
        agent_pos = pf_get_agent_pos(ents, s._pf_agent_color)
        if agent_pos is None: return None
        # Update heatmap
        rpos = (round(agent_pos[0]), round(agent_pos[1]))
        s._pf_pos_heatmap[rpos] += 1
        # Find nearest non-agent entity
        targets = [e for e in ents if e['color'] != s._pf_agent_color]
        if not targets: return None
        target = min(targets, key=lambda e: (
            abs(e['centroid'][0]-agent_pos[0])+abs(e['centroid'][1]-agent_pos[1])))
        # Navigate with heatmap penalty
        best_aid, best_score = None, float('inf')
        for aid, (dr, dc) in s._pf_move_map.items():
            if aid in banned_set: continue
            nr = round(agent_pos[0]+dr)
            nc = round(agent_pos[1]+dc)
            dist = (abs(target['centroid'][0]-nr)+abs(target['centroid'][1]-nc))
            heat = s._pf_pos_heatmap.get((nr,nc), 0)
            score = dist + heat*5 + random.random()*0.1
            if score < best_score:
                best_score=score; best_aid=aid
        return best_aid

    def is_done(s, frames, lf):
        try: return lf.state is GameState.WIN or (time.time()-s.start_time)>=8*3600-300
        except: return True

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # ── Level change ──────────────────────────────────────────
            if lvl != s.cl:
                if not s._bfs_tried:
                    s._bfs_tried=True; s._init_bfs()
                s._bfs_solution=None; s._bfs_step=0
                if s._bfs: s._try_bfs_solve(lvl)
                s.buf.clear(); s.buf_h.clear()
                s.net=ForgeNet(s.IN,s.G).to(s.device)
                for wp in ['/kaggle/input/forge-pretrained-weights/pretrained_weights.pt',
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
                s._wd=False; s._wm=None
                if not s._bfs_solution: s._eps=0.15
                s._aem_diffs.clear(); s._aem_actions.clear(); s._aem_rewards.clear()
                s._prev_objs=None; s._ckpt_hash=None; s._unproductive=0
                s._visited_hashes=set()
                # Reset symbolic
                s._graph=TransformationGraph()
                s._prev_entities=None; s._banned_action=None
                s._prev_ctx=None; s._prev_click_xy=None
                s._prev_level_completed=lvl-1
                # Reset pathfinder nav (keep _pf_move_map if same game type)
                s._pf_pos_heatmap=defaultdict(int)
                s._pf_move_detect_count=0
                s._pf_is_move_game=None
                # Reset anti-stuck
                s._recent_actions.clear()
                s._escape_action=None; s._escape_remaining=0
                s._cooldowns={}; s._round_robin_idx=0
                s._click_grid_idx=0; s._click_targets=[]
                # Inject BFS demos
                if lvl>0 and s._bfs and s._bfs.solutions.get(lvl-1):
                    prev_sol=s._bfs.solutions[lvl-1]
                    try:
                        rg=s._bfs.game_cls(); rg.set_level(lvl-1)
                        rg.perform_action(ActionInput(id=GameAction.RESET),raw=True)
                        r0=rg.perform_action(ActionInput(id=GameAction.RESET),raw=True)
                        if r0.frame:
                            pf=np.array(r0.frame[-1],dtype=np.int64)
                            for act_id,data in prev_sol:
                                ai=(ActionInput(id=GameAction.from_id(act_id),data=data)
                                    if data else ActionInput(id=GameAction.from_id(act_id)))
                                result=rg.perform_action(ai,raw=True)
                                aidx=((act_id-1) if act_id<=5 else
                                      (5+data.get('y',0)*64+data.get('x',0) if data else 0))
                                s.buf.append({'s':pf.copy(),'a':aidx,'r':2.0})
                                if result.frame:
                                    pf=np.array(result.frame[-1],dtype=np.int64)
                            if len(s.buf)>=s.bsz:
                                for _ in range(min(20,len(s.buf)//s.bsz)): s._train()
                    except: pass

            # ── Reset ─────────────────────────────────────────────────
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.pt=None; s.pai=None; s.pr=None; s.ph=None
                s._recent_actions.clear()
                s._escape_action=None; s._escape_remaining=0
                s._cooldowns={}; s._unproductive=0; s._banned_action=None
                a=GameAction.RESET; a.reasoning="reset"; return a

            # ── BFS execution ─────────────────────────────────────────
            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                act_id,data=s._bfs_solution[s._bfs_step]; s._bfs_step+=1
                sel=GameAction.from_id(act_id)
                if data: sel.set_data(data)
                sel.reasoning=f"bfs:{s._bfs_step}/{len(s._bfs_solution)}"
                raw=s._raw(lf)
                s.fhist.append(raw.copy()); s.pr=raw.copy(); s.la+=1
                return sel

            # ── Feedback + symbolic observation ───────────────────────
            raw=s._raw(lf)
            ch=hashlib.md5(raw.tobytes()).hexdigest()[:16]
            avail=getattr(lf,'available_actions',None) or []
            s._undo_avail=any((a.value if hasattr(a,'value') else int(a))==7 for a in avail)
            avail_ids=[a.value if hasattr(a,'value') else int(a) for a in avail]
            simple_ids=[i for i in avail_ids if 1<=i<=5]

            if s.pt is not None and s.pai is not None and s.pr is not None:
                mask_r=np.ones((64,64),dtype=bool); mask_r[:2]=False; mask_r[62:]=False
                diff_map=(s.pr!=raw)&mask_r; changed=np.any(diff_map)
                eh=hashlib.md5(s.pr.tobytes()[:1000]+str(s.pai).encode()).hexdigest()[:16]
                if eh not in s.buf_h:
                    r=s._reward(s.pr,raw,ch)
                    s.buf.append({"s":s.pr.copy(),"a":s.pai,"r":r})
                    s.buf_h.add(eh)
                    if changed:
                        s._aem_diffs.append(diff_map)
                        s._aem_actions.append(min(s.pai if isinstance(s.pai,int) else 1,4))
                        s._aem_rewards.append(r)

                # Symbolic entity tracking
                curr_entities,_=sym_extract_entities(raw)
                curr_ctx=spatial_context(curr_entities)
                level_advanced=(lvl>s._prev_level_completed)
                if level_advanced: s._prev_level_completed=lvl
                pai_int=s.pai if isinstance(s.pai,int) else 1

                events=[]
                if s._prev_entities is not None:
                    events=detect_transformations(s._prev_entities,curr_entities)
                    if pai_int==6 and s._prev_click_xy is not None:
                        s._graph.observe_click(
                            s._prev_click_xy[0],s._prev_click_xy[1],
                            events,s._prev_entities,s._prev_ctx or {},level_advanced)
                    else:
                        s._graph.observe(
                            pai_int,events,s._prev_ctx or {},s._prev_entities,
                            curr_ctx,curr_entities,level_advanced)

                    # Pathfinder: detect agent movement + build move_map
                    for ev in events:
                        if ev['type']=='MOVE':
                            s._pf_move_detect_count+=1
                            s._pf_agent_color=ev['color']
                            # Learn direction from diff
                            if s._prev_entities and curr_entities:
                                for e0 in s._prev_entities:
                                    if e0['color']==ev['color']:
                                        for e1 in curr_entities:
                                            if e1['color']==ev['color']:
                                                dr=e1['centroid'][0]-e0['centroid'][0]
                                                dc=e1['centroid'][1]-e0['centroid'][1]
                                                if dr!=0 or dc!=0:
                                                    s._pf_move_map[pai_int]=(
                                                        int(np.sign(dr)),int(np.sign(dc)))
                                                break
                                        break

                entity_sig=tuple(sorted(
                    (e['color'],e['centroid'][0],e['centroid'][1])
                    for e in curr_entities))
                prev_sig=tuple(sorted(
                    (e['color'],e['centroid'][0],e['centroid'][1])
                    for e in s._prev_entities)) if s._prev_entities else ()
                meaningful=entity_sig!=prev_sig
                s._graph.record_action_outcome(pai_int,meaningful)

                if meaningful:
                    s._banned_action=None; s._ckpt_hash=ch; s._unproductive=0
                else:
                    s._banned_action=pai_int; s._unproductive+=1

                s._prev_entities=curr_entities
                s._prev_ctx=curr_ctx; s._prev_click_xy=None

            if s.la>=8 and s._pf_is_move_game is None:
                s._pf_is_move_game=(s._pf_move_detect_count>=2)

            if s._wm is None: s._wm=s._detect_template(raw)

            # ── Tick cooldowns ────────────────────────────────────────
            for k in list(s._cooldowns.keys()):
                s._cooldowns[k]-=1
                if s._cooldowns[k]<=0: del s._cooldowns[k]

            banned_set=set(s._cooldowns.keys())
            if s._banned_action is not None: banned_set.add(s._banned_action)
            for a in simple_ids:
                if s._graph.action_is_blocked(a): banned_set.add(a)

            # ── Oscillation detection + escape ────────────────────────
            if s._escape_remaining<=0 and s._detect_oscillation(simple_ids):
                s._enter_escape(simple_ids)

            if s._escape_remaining>0 and s._escape_action is not None:
                esc=s._escape_action
                if s._graph.action_is_blocked(esc):
                    cands=[a for a in simple_ids
                           if a!=esc and not s._graph.action_is_blocked(a)]
                    if cands: s._escape_action=random.choice(cands); esc=s._escape_action
                    else: s._escape_remaining=0
                if s._escape_remaining>0 and 1<=esc<=5:
                    s._escape_remaining-=1
                    sel=s.al[esc-1]; sel.reasoning=f"escape:a{esc}(rem={s._escape_remaining})"
                    s.pt=s._tensor(lf); s.pai=esc; s.pr=raw.copy(); s.ph=ch; s.la+=1
                    s._recent_actions.append(esc); return sel

            # ── UNDO if very stuck ────────────────────────────────────
            if s._undo_avail and s._unproductive>=15 and s._ckpt_hash:
                s._unproductive=0
                a=GameAction.ACTION7; a.reasoning="undo"
                s.pt=None; s.pai=7; s.pr=raw.copy(); s.ph=ch; s.la+=1
                s._recent_actions.append(7); return a

            # ── Cooldown frustration ──────────────────────────────────
            if s._unproductive>=3:
                s._unproductive=0
                if s.pai is not None and 1<=s.pai<=5:
                    s._cooldowns[s.pai]=6; banned_set.add(s.pai)

            # ── Symbolic action ───────────────────────────────────────
            curr_ents_now,_=sym_extract_entities(raw)
            curr_ctx_now=spatial_context(curr_ents_now)

            sym_action,sym_score,sym_phase=s._graph.best_action(
                simple_ids,curr_ents_now,curr_ctx_now,banned=s._banned_action)

            use_sym=(sym_phase in ("explore","validate")
                     or (sym_phase=="plan" and sym_score>=5))
            if use_sym and sym_action is not None and sym_action not in banned_set:
                sel=s.al[sym_action-1]
                sel.reasoning=f"sym:{sym_phase}:a{sym_action}(s={sym_score})"
                s.pt=s._tensor(lf); s.pai=sym_action; s.pr=raw.copy(); s.ph=ch; s.la+=1
                s._recent_actions.append(sym_action); return sel

            # Symbolic click
            if 6 in avail_ids and 6 not in banned_set:
                click_xy=s._graph.best_click_target(curr_ents_now,curr_ctx_now)
                if click_xy is not None:
                    x,y=click_xy
                    sel=GameAction.ACTION6; sel.set_data({"x":x,"y":y})
                    sel.reasoning=f"sym:click({x},{y})"
                    s._prev_click_xy=(x,y)
                    s.pt=s._tensor(lf); s.pai=6; s.pr=raw.copy(); s.ph=ch; s.la+=1
                    s._recent_actions.append(6); return sel

            # Goal actions (confirmed by level advance)
            if s._graph.goal_actions:
                for ga in s._graph.goal_actions:
                    if ga in simple_ids and ga not in banned_set:
                        sel=s.al[ga-1]; sel.reasoning=f"goal:a{ga}"
                        s.pt=s._tensor(lf); s.pai=ga; s.pr=raw.copy(); s.ph=ch; s.la+=1
                        s._recent_actions.append(ga); return sel

            # ── Pathfinder navigation ─────────────────────────────────
            if s._pf_is_move_game and s._pf_move_map:
                nav=s._pf_navigate(raw,simple_ids,banned_set)
                if nav is not None:
                    sel=s.al[nav-1]; sel.reasoning=f"pf:nav(a{nav})"
                    s.pt=s._tensor(lf); s.pai=nav; s.pr=raw.copy(); s.ph=ch; s.la+=1
                    s._recent_actions.append(nav); return sel

            # ── Click scan (for click games) ──────────────────────────
            if 6 in avail_ids and (not simple_ids or s._pf_is_move_game is False):
                if not s._click_targets:
                    bg2=int(np.bincount(raw.flatten(),minlength=16).argmax())
                    tgts=[]
                    for y in range(0,64,3):
                        for x in range(0,64,3):
                            if raw[y,x]!=bg2: tgts.append((x,y))
                    if not tgts: tgts=[(x,y) for y in range(0,64,8) for x in range(0,64,8)]
                    s._click_targets=tgts
                if s._click_targets:
                    idx=s._click_grid_idx%len(s._click_targets)
                    x,y=s._click_targets[idx]; s._click_grid_idx+=1
                    sel=GameAction.ACTION6; sel.set_data({"x":x,"y":y})
                    sel.reasoning=f"cscan:({x},{y})"
                    s._prev_click_xy=(x,y)
                    s.pt=s._tensor(lf); s.pai=6; s.pr=raw.copy(); s.ph=ch; s.la+=1
                    s._recent_actions.append(6); return sel

            # ── Round-robin over unbanned simple actions ───────────────
            if simple_ids:
                unbanned=[a for a in simple_ids if a not in banned_set]
                pool=unbanned if unbanned else simple_ids
                for offset in range(len(pool)):
                    idx=(s._round_robin_idx+offset)%len(pool)
                    act=pool[idx]
                    if act not in banned_set:
                        s._round_robin_idx=(idx+1)%len(pool)
                        sel=s.al[act-1]; sel.reasoning=f"rr:a{act}"
                        s.pt=s._tensor(lf); s.pai=act; s.pr=raw.copy(); s.ph=ch; s.la+=1
                        s._recent_actions.append(act); return sel
                # All banned — clear and pick first
                s._cooldowns.clear(); s._banned_action=None
                act=simple_ids[0]; sel=s.al[act-1]; sel.reasoning=f"rr0:a{act}"
                s.pt=s._tensor(lf); s.pai=act; s.pr=raw.copy(); s.ph=ch; s.la+=1
                s._recent_actions.append(act); return sel

            # ── CNN fallback ──────────────────────────────────────────
            tensor=s._tensor(lf)
            if not s._wd:
                if s.la<10: aidx,coords=s._heuristic(raw,avail,s.la)
                else:
                    s._wd=True
                    for _ in range(min(5,len(s.buf)//s.bsz)): s._train()

            if s._wd:
                if random.random()<s._eps:
                    aidx,coords=s._sample(torch.zeros(4101,device=s.device),avail,temp=2.0)
                else:
                    with torch.no_grad():
                        mem=s._get_aem_tensors()
                        if mem[0] is not None:
                            logits=s.net(tensor.unsqueeze(0),*mem).squeeze(0)
                        else:
                            logits=s.net(tensor.unsqueeze(0)).squeeze(0)
                    aidx,coords=s._sample(logits,avail,temp=0.5)
                s._eps=max(s._eps_min,s._eps*s._eps_decay)
            elif s.la>=10:
                s._wd=True; aidx,coords=0,None

            if aidx<5:
                sel=s.al[aidx]; sel.reasoning=f"cnn:a{aidx+1}"
            else:
                if coords is None:
                    cnt=np.bincount(raw.flatten(),minlength=16); bg=int(cnt.argmax())
                    for c in range(16):
                        if c!=bg and cnt[c]>2:
                            ys,xs=np.where(raw==c)
                            coords=(int(np.median(ys)),int(np.median(xs))); break
                    if coords is None: coords=(32,32)
                sel=GameAction.ACTION6; y,x=coords
                sel.set_data({"x":int(x),"y":int(y)}); sel.reasoning=f"cnn:c({x},{y})"

            s.pt=tensor
            if aidx<5: s.pai=aidx+1; s._prev_click_xy=None
            else: s.pai=6; y,x=coords; s._prev_click_xy=(int(x),int(y))
            s.pr=raw.copy(); s.ph=ch; s.la+=1
            s._recent_actions.append(s.pai)
            if s.action_counter%s.tfreq==0 and s._wd: s._train()
            return sel

        except Exception as e:
            traceback.print_exc()
            a=random.choice(s.al); a.reasoning=f"err:{e}"; return a