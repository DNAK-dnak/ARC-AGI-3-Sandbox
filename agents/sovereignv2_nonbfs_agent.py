# ======================================================================
# SOVEREIGN v2 — FORGE v15 + Symbolic Transformation Layer
#
# FORGE v15 is the base (proven, unchanged structure).
# Added: symbolic entity tracking + transformation graph
# to improve action selection in the CNN fallback phase.
#
# Changes from FORGE v15:
#   1. Fixed seed: abs(hash(game_id)) % (2**32-1), no time()
#   2. Symbolic entity extraction (pure numpy, no scipy)
#   3. TransformationGraph scores actions by causal history
#   4. Graph consulted BEFORE epsilon-greedy in CNN fallback
#   5. Graph resets per level, persists within level
# ======================================================================
import pickle
import copy
import glob
import hashlib
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
    
# Add these classes before the ForgeNet class
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelAttention(gate_channels, reduction_ratio)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialAttention()
            
    def forward(self, x):
        x_out = self.ChannelGate(x) * x
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out) * x_out
        return x_out

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


# ==================== SYMBOLIC RULE-DISCOVERY ENGINE ====================
# Universal Online System Identification — no hardcoded game concepts.
# Works by tracking generic geometric events and forming/validating
# rule hypotheses about cause→effect relationships.
# =========================================================================

def _cc_label(mask):
    """8-connected component labeling via BFS. Pure numpy, no scipy."""
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
    """
    Decompose frame into agnostic geometric entities.
    Returns (entities, bg_color).
    Each entity: {id, color, pixels, centroid, bbox}
    """
    bg = int(np.bincount(frame.flatten(), minlength=16).argmax())
    entities = []
    eid = 0
    for color in range(16):
        if color == bg:
            continue
        mask = (frame == color)
        if not mask.any():
            continue
        labeled, n = _cc_label(mask)
        for i in range(n):
            comp = (labeled == i + 1)
            coords = np.argwhere(comp)
            if len(coords) == 0:
                continue
            cy, cx = np.round(coords.mean(axis=0)).astype(int)
            rmin, cmin = coords.min(axis=0)
            rmax, cmax = coords.max(axis=0)
            entities.append({
                'id':       eid,
                'color':    color,
                'pixels':   int(comp.sum()),
                'centroid': (int(cy), int(cx)),
                'bbox':     (int(rmin), int(cmin), int(rmax), int(cmax)),
            })
            eid += 1
    return entities, bg


def _bbox_overlap(b1, b2):
    """True if two bounding boxes overlap."""
    return not (b1[2] < b2[0] or b2[2] < b1[0] or
                b1[3] < b2[1] or b2[3] < b1[1])


def _bbox_adjacent(b1, b2, gap=2):
    """True if two bounding boxes are within `gap` pixels of each other."""
    expanded = (b1[0]-gap, b1[1]-gap, b1[2]+gap, b1[3]+gap)
    return _bbox_overlap(expanded, b2)


def detect_transformations(ent_before, ent_after):
    """
    Universal Interaction Grammar — detects generic events:
      MOVE      — entity moved (same color+mass, different centroid)
      DISAPPEAR — entity no longer exists
      APPEAR    — new entity with no prior match
      RECOLOR   — entity same position+mass, different color
      GROW      — entity same color, larger mass
      SHRINK    — entity same color, smaller mass
    Also records spatial context: which entities were OVERLAPPING or
    ADJACENT to the acting entity before the event.
    Returns list of event dicts.
    """
    events = []
    used_after = set()

    for e0 in ent_before:
        best_same, best_d_same = None, float('inf')   # same color+mass → MOVE
        best_recolor, best_d_rc = None, float('inf')  # same mass, diff color → RECOLOR
        best_resize = None                             # same color, diff mass

        for i, e1 in enumerate(ent_after):
            if i in used_after:
                continue
            cy_d = (e0['centroid'][0] - e1['centroid'][0]) ** 2
            cx_d = (e0['centroid'][1] - e1['centroid'][1]) ** 2
            d = cy_d + cx_d

            if e0['color'] == e1['color'] and e0['pixels'] == e1['pixels']:
                if d < best_d_same:
                    best_d_same = d; best_same = i
            elif e0['pixels'] == e1['pixels'] and d < best_d_rc:
                best_d_rc = d; best_recolor = i
            elif e0['color'] == e1['color'] and best_resize is None:
                best_resize = i

        if best_same is not None:
            used_after.add(best_same)
            if best_d_same > 0:
                events.append({'type': 'MOVE', 'color': e0['color'],
                               'pixels': e0['pixels']})
        elif best_recolor is not None:
            used_after.add(best_recolor)
            new_color = ent_after[best_recolor]['color']
            events.append({'type': 'RECOLOR', 'color': e0['color'],
                           'new_color': new_color, 'pixels': e0['pixels']})
        elif best_resize is not None:
            used_after.add(best_resize)
            new_px = ent_after[best_resize]['pixels']
            t = 'GROW' if new_px > e0['pixels'] else 'SHRINK'
            events.append({'type': t, 'color': e0['color'],
                           'pixels': e0['pixels'], 'new_pixels': new_px})
        else:
            events.append({'type': 'DISAPPEAR', 'color': e0['color'],
                           'pixels': e0['pixels']})

    for i, e1 in enumerate(ent_after):
        if i not in used_after:
            events.append({'type': 'APPEAR', 'color': e1['color'],
                           'pixels': e1['pixels']})

    return events


def spatial_context(entities):
    """
    Build spatial relationship map for current entities.
    Returns dict: entity_id → {'overlaps': [ids], 'adjacent': [ids]}
    """
    ctx = {e['id']: {'overlaps': [], 'adjacent': []} for e in entities}
    for i, ea in enumerate(entities):
        for j, eb in enumerate(entities):
            if i >= j:
                continue
            if _bbox_overlap(ea['bbox'], eb['bbox']):
                ctx[ea['id']]['overlaps'].append(eb['id'])
                ctx[eb['id']]['overlaps'].append(ea['id'])
            elif _bbox_adjacent(ea['bbox'], eb['bbox']):
                ctx[ea['id']]['adjacent'].append(eb['id'])
                ctx[eb['id']]['adjacent'].append(ea['id'])
    return ctx


class RuleHypothesis:
    """
    A single cause→effect rule hypothesis.
    trigger: dict describing what action + spatial context caused it
    effect:  the transformation event observed
    confidence: increments on successful validation
    failures: increments when predicted effect doesn't occur
    """
    def __init__(self, trigger, effect):
        self.trigger    = trigger    # {action, near_colors, overlap_colors}
        self.effect     = effect     # {type, color, ...}
        self.confidence = 1
        self.failures   = 0
        self.confirmed  = False

    def sig(self):
        t = self.trigger
        e = self.effect
        return (t.get('action'), tuple(sorted(t.get('near_colors', []))),
                e.get('type'), e.get('color'))

    def validate(self, observed_events):
        """Check if this hypothesis's predicted effect is in observed events."""
        for ev in observed_events:
            if ev['type'] == self.effect['type'] and ev['color'] == self.effect['color']:
                self.confidence += 1
                if self.confidence >= 3:
                    self.confirmed = True
                return True
        self.failures += 1
        return False


class TransformationGraph:
    """
    Universal rule-discovery engine.

    Phase 1 — Exploration: try each action once, record events + spatial context.
    Phase 2 — Hypothesis: form RuleHypothesis objects from observed cause→effect.
    Phase 3 — Validation: actively seek similar spatial conditions to test hypotheses.
    Phase 4 — Planning: use confirmed rules to select goal-directed actions.
    """

    def __init__(self):
        # action_id → list of observed event dicts
        self.action_events: Dict[int, list] = {}
        # action_id → spatial context at time of action
        self.action_context: Dict[int, dict] = {}
        # all hypotheses, keyed by sig tuple
        self.hypotheses: Dict[tuple, RuleHypothesis] = {}
        # goal-relevant transformation signatures (set when level advances)
        self.goal_sigs: set = set()
        # actions confirmed to produce goal-relevant effects
        self.goal_actions: set = set()
        # track which actions were tried this level
        self.tried: set = set()
        # validation queue: (hypothesis_sig, target_entity_color)
        self._validate_queue: deque = deque()

    def _near_colors(self, ctx, entity_id, entities_by_id):
        """Colors of entities adjacent or overlapping with entity_id."""
        info = ctx.get(entity_id, {})
        colors = set()
        for oid in info.get('overlaps', []) + info.get('adjacent', []):
            e = entities_by_id.get(oid)
            if e:
                colors.add(e['color'])
        return sorted(colors)

    def observe(self, action_id: int, events: list,
                 ctx_before: dict, entities_before: list,
                 ctx_after: dict, entities_after: list,
                 level_advanced: bool):
        """
        Called after every action. Updates hypotheses and goal signals.
        """
        self.tried.add(action_id)

        if not events:
            self.action_events[action_id] = []
            return

        # Store raw observations
        self.action_events[action_id] = events
        self.action_context[action_id] = ctx_before

        by_id_before = {e['id']: e for e in entities_before}

        # Form hypotheses from each event
        for ev in events:
            # Find the most likely "actor" entity — the one closest to action target
            # (for simple actions we use the smallest entity that MOVEd)
            near_colors = []
            if entities_before:
                for e in entities_before:
                    nc = self._near_colors(ctx_before, e['id'], by_id_before)
                    near_colors.extend(nc)
            near_colors = sorted(set(near_colors))

            trigger = {
                'action':        action_id,
                'near_colors':   near_colors,
            }
            hyp = RuleHypothesis(trigger, ev)
            sig = hyp.sig()
            if sig not in self.hypotheses:
                self.hypotheses[sig] = hyp
                # Queue for validation if not yet confirmed
                self._validate_queue.append(sig)

        # Mark goal-relevant rules when level advances
        if level_advanced:
            for ev in events:
                sig_key = f"{ev['type']}_{ev['color']}"
                self.goal_sigs.add(sig_key)
            self.goal_actions.add(action_id)

        # Validate pending hypotheses against this observation
        for hyp in self.hypotheses.values():
            if not hyp.confirmed and hyp.trigger['action'] == action_id:
                hyp.validate(events)

    def observe_click(self, x: int, y: int, events: list,
                      entities_before: list, ctx_before: dict,
                      level_advanced: bool):
        """Specialised observation for ACTION6 clicks."""
        self.tried.add(6)
        self.action_events[6] = events  # record raw events for scoring
        # Find which entity was clicked
        clicked_color = None
        for e in entities_before:
            r0, c0, r1, c1 = e['bbox']
            if r0 <= y <= r1 and c0 <= x <= c1:
                clicked_color = e['color']
                break

        by_id = {e['id']: e for e in entities_before}
        near = []
        for e in entities_before:
            if e['color'] == clicked_color:
                near = self._near_colors(ctx_before, e['id'], by_id)
                break

        for ev in events:
            trigger = {
                'action':        6,
                'clicked_color': clicked_color,
                'near_colors':   near,
            }
            hyp = RuleHypothesis(trigger, ev)
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
        """Score an event by how goal-relevant it is."""
        sig = f"{ev['type']}_{ev['color']}"
        base = {'DISAPPEAR': 4, 'RECOLOR': 3, 'GROW': 2,
                'SHRINK': 2, 'MOVE': 1, 'APPEAR': 1}.get(ev['type'], 0)
        return base + (20 if sig in self.goal_sigs else 0)

    def best_action(self, available_ids: list,
                    current_entities: list, current_ctx: dict,
                    banned=None) -> Tuple[Optional[int], int, str]:
        """
        Returns (action_id, score, phase) for the best next action.
        Phases: 'explore' | 'validate' | 'plan' | 'fallback'
        """
        simple = [a for a in available_ids if 1 <= a <= 5]
        has_click = 6 in available_ids

        # Phase 1 — Exploration: try every untried simple action once
        # Also trigger click exploration if ACTION6 not yet tried
        untried = [a for a in simple if a not in self.tried and a != banned]
        if untried:
            return untried[0], 15, 'explore'
        if has_click and 6 not in self.tried:
            return 6, 15, 'explore'  # signal caller to pick a click target

        # Phase 3 — Validation: test pending hypotheses
        while self._validate_queue:
            sig = self._validate_queue[0]
            if sig not in self.hypotheses:
                self._validate_queue.popleft()
                continue
            hyp = self.hypotheses[sig]
            if hyp.confirmed or hyp.failures >= 3:
                self._validate_queue.popleft()
                continue
            act = hyp.trigger.get('action', 1)
            if act in simple and act != banned:
                return act, 12, 'validate'
            self._validate_queue.popleft()

        # Phase 4 — Planning: use highest-scoring confirmed/strong hypothesis
        best_a, best_s = None, -1
        candidates = simple + ([6] if has_click else [])
        for a in candidates:
            if a == banned:
                continue
            evs = self.action_events.get(a, [])
            score = sum(self._event_score(ev) for ev in evs)
            # Bonus for confirmed hypotheses
            for hyp in self.hypotheses.values():
                if hyp.trigger.get('action') == a and hyp.confirmed:
                    score += 25
            # Bonus for known goal actions
            if a in self.goal_actions:
                score += 50
            if score > best_s:
                best_s = score; best_a = a

        if best_a is not None and best_s > 0:
            return best_a, best_s, 'plan'

        # Fallback
        candidates = [a for a in simple if a != banned] or simple
        return (candidates[0] if candidates else 1), 0, 'fallback'

    def best_click_target(self, entities: list, ctx: dict) -> Optional[Tuple[int, int]]:
        """
        Returns (x, y) of the most promising click target based on
        confirmed ACTION6 hypotheses, or None if no signal.
        """
        # Collect colors that confirmed click hypotheses predict effects on
        target_colors = {}
        for hyp in self.hypotheses.values():
            if hyp.trigger.get('action') == 6 and (hyp.confirmed or hyp.confidence >= 2):
                cc = hyp.trigger.get('clicked_color')
                if cc is not None:
                    score = self._event_score(hyp.effect) * hyp.confidence
                    target_colors[cc] = target_colors.get(cc, 0) + score

        if not target_colors:
            return None

        # Find entity with highest-scoring color
        best_e, best_s = None, -1
        for e in entities:
            s = target_colors.get(e['color'], 0)
            if s > best_s:
                best_s = s; best_e = e

        if best_e is None:
            return None
        cy, cx = best_e['centroid']
        return int(cx), int(cy)  # (x, y) for ACTION6


    def plan_sequence(self, available_ids: list,
                       entities: list, ctx: dict,
                       depth: int = 3) -> List[int]:
        """
        Subgoal chaining: return a short action sequence (len<=depth)
        by chaining transformations that enable each other.

        Strategy:
          1. Find all goal-relevant transformations (goal_sigs).
          2. For each, find which action produces it.
          3. Find prerequisite transformations (actions whose effects
             are OVERLAP/ADJACENT preconditions for the goal action).
          4. Return the enabling sequence first, then the goal action.
        """
        simple = [a for a in available_ids if 1 <= a <= 5]
        has_click = 6 in available_ids

        # Map: transformation_sig → action that produces it
        sig_to_action: Dict[str, int] = {}
        for a, evs in self.action_events.items():
            for ev in evs:
                sig = f"{ev['type']}_{ev['color']}"
                if sig not in sig_to_action:
                    sig_to_action[sig] = a

        # Find goal actions (produce goal sigs)
        goal_acts = []
        for sig in self.goal_sigs:
            a = sig_to_action.get(sig)
            if a is not None and (a in simple or (has_click and a == 6)):
                goal_acts.append(a)

        if not goal_acts:
            # No known goal action yet — return best single action
            a, _, _ = self.best_action(available_ids, entities, ctx)
            return [a] if a else []

        # For each goal action, find enabling prerequisites
        # An enabling action is one that produces MOVE/APPEAR of an entity
        # that is needed near the goal target
        sequence = []
        for goal_act in goal_acts[:1]:  # chain from first goal action
            # Look for actions that produce MOVE or APPEAR of colors
            # that appear in the goal action's spatial context
            goal_ctx_colors = set()
            for hyp in self.hypotheses.values():
                if hyp.trigger.get('action') == goal_act and hyp.confirmed:
                    goal_ctx_colors.update(hyp.trigger.get('near_colors', []))

            # Find enabling action that moves/creates those colors
            enabler = None
            for a in simple:
                if a == goal_act:
                    continue
                evs = self.action_events.get(a, [])
                for ev in evs:
                    if ev['type'] in ('MOVE', 'APPEAR') and ev['color'] in goal_ctx_colors:
                        enabler = a
                        break
                if enabler:
                    break

            if enabler and len(sequence) < depth - 1:
                sequence.append(enabler)
            sequence.append(goal_act)
            if len(sequence) >= depth:
                break

        return sequence[:depth] if sequence else []


# ==================== AGENT ====================

class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        # FIX: deterministic seed per game, no time()
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
        # Subgoal chain (replaces BFS solution replay)
        s._subgoal_chain=[]; s._chain_step=0
        s._stagnant_steps=0; s._stagnant_threshold=15
        # Symbolic layer
        s._graph=TransformationGraph()
        s._prev_entities=None
        s._banned_action=None
        s._prev_ctx=None
        s._prev_click_xy=None
        s._prev_level_completed=-1

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

    def is_done(s, frames, lf):
        try: return lf.state is GameState.WIN or (time.time()-s.start_time)>=8*3600-300
        except: return True

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # ===== LEVEL CHANGE & INITIALIZATION =====
            if lvl != s.cl:
                s._subgoal_chain = []; s._chain_step = 0
                s._stagnant_steps = 0
                s.buf.clear(); s.buf_h.clear()
                
                # ForgeNet initialization with CBAM fix
                s.net = ForgeNet(s.IN, s.G).to(s.device)
                s.opt = optim.Adam(s.net.parameters(), lr=0.0003)
                
                s.pt = None; s.pai = None; s.pr = None; s.ph = None
                s.cl = lvl; s.fhist.clear(); s.la = 0
                s._wd = False; s._wm = None; s._eps = 0.15
                s._aem_diffs.clear(); s._aem_actions.clear(); s._aem_rewards.clear()
                s._prev_objs = None; s._ckpt_hash = None; s._unproductive = 0
                
                s._graph = TransformationGraph()
                s._prev_entities = None; s._banned_action = None
                s._prev_ctx = None; s._prev_click_xy = None
                s._prev_level_completed = lvl - 1

            # ===== RESET LOGIC =====
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.pt = None; s.pai = None; s.pr = None; s.ph = None
                a = GameAction.RESET; a.reasoning = "reset"; return a

            raw = s._raw(lf)
            ch = hashlib.md5(raw.tobytes()).hexdigest()[:16]
            avail = getattr(lf, 'available_actions', None) or []
            s._undo_avail = any((a.value if hasattr(a,'value') else int(a)) == 7 for a in avail)
            avail_ids = [a.value if hasattr(a,'value') else int(a) for a in avail]
            simple_ids = [i for i in avail_ids if 1 <= i <= 5]

            # ===== STAGNATION / STUCK DETECTION =====
            # Check if the world actually changed after our last action
            if s.pr is not None:
                if np.array_equal(s.pr, raw):
                    s._unproductive += 1
                else:
                    s._unproductive = 0 # Reset if we moved or changed something
                    s._ckpt_hash = ch    # Save this state as a "good" checkpoint

            # ===== ESCAPE MECHANISM: If stuck, force variety =====
            if s._unproductive >= 3:
                # Filter out the action that is currently failing
                escape_pool = [a for a in simple_ids if a != s.pai]
                if not escape_pool: escape_pool = simple_ids
                
                act_id = random.choice(escape_pool)
                sel = s.al[act_id - 1]
                sel.reasoning = f"escape:stuck_for_{s._unproductive}_steps:a{act_id}"
                
                # Update state and return immediately to break the loop
                s.pt = s._tensor(lf); s.pai = act_id; s.pr = raw.copy(); s.ph = ch; s.la += 1
                return sel

            # ===== FEEDBACK: Update Symbolic Graph =====
            if s.pt is not None and s.pai is not None and s.pr is not None:
                curr_entities, _ = extract_entities(raw)
                curr_ctx = spatial_context(curr_entities)
                level_advanced = (lvl > s._prev_level_completed)
                
                if s._prev_entities is not None:
                    events = detect_transformations(s._prev_entities, curr_entities)
                    pai_int = s.pai if isinstance(s.pai, int) else 1
                    if pai_int == 6 and s._prev_click_xy is not None:
                        s._graph.observe_click(s._prev_click_xy[0], s._prev_click_xy[1],
                                               events, s._prev_entities, s._prev_ctx or {}, level_advanced)
                    else:
                        s._graph.observe(pai_int, events, s._prev_ctx or {}, s._prev_entities,
                                         curr_ctx, curr_entities, level_advanced)
                
                s._prev_entities = curr_entities; s._prev_ctx = curr_ctx; s._prev_click_xy = None

            # ===== ACTION SELECTION: Symbolic Layer =====
            curr_ents_now, _ = extract_entities(raw)
            curr_ctx_now = spatial_context(curr_ents_now)
            sym_action, sym_score, sym_phase = s._graph.best_action(
                simple_ids, curr_ents_now, curr_ctx_now, banned=s._banned_action)
            
            # Prioritize Rule Discovery
            use_sym = sym_phase in ("explore", "validate") or (sym_phase == "plan" and sym_score >= 5)
            if use_sym and sym_action is not None:
                sel = s.al[sym_action - 1]
                sel.reasoning = f"sym:{sym_phase}:a{sym_action}(s={sym_score})"
                s.pt = s._tensor(lf); s.pai = sym_action; s.pr = raw.copy(); s.ph = ch; s.la += 1
                return sel

            # ===== UNDO: Last resort if heavily stuck =====
            if s._undo_avail and s._unproductive >= 10 and s._ckpt_hash:
                s._unproductive = 0
                a = GameAction.ACTION7; a.reasoning = "stuck:undo_to_checkpoint"
                s.pt = None; s.pai = 7; s.pr = raw.copy(); s.ph = ch; s.la += 1
                return a

            # ===== CNN FALLBACK (Standard FORGE logic) =====
            tensor = s._tensor(lf)
            if not s._wd:
                if s.la < 10: aidx, coords = s._heuristic(raw, avail, s.la)
                else: 
                    s._wd = True
                    for _ in range(min(5, len(s.buf)//s.bsz)): s._train()

            if s._wd:
                if random.random() < s._eps:
                    aidx, coords = s._sample(torch.zeros(4101, device=s.device), avail, temp=2.0)
                else:
                    with torch.no_grad():
                        mem = s._get_aem_tensors()
                        if mem[0] is not None: logits = s.net(tensor.unsqueeze(0), *mem).squeeze(0)
                        else: logits = s.net(tensor.unsqueeze(0)).squeeze(0)
                    aidx, coords = s._sample(logits, avail, temp=0.5)
                s._eps = max(s._eps_min, s._eps * s._eps_decay)
            
            # Final Return
            if aidx < 5: 
                sel = s.al[aidx]; sel.reasoning = f"cnn:a{aidx+1}"
                s.pai = aidx + 1; s._prev_click_xy = None
            else:
                sel = GameAction.ACTION6; y, x = coords
                sel.set_data({"x": int(x), "y": int(y)}); sel.reasoning = f"cnn:c({x},{y})"
                s.pai = 6; s._prev_click_xy = (int(x), int(y))
            
            s.pt = tensor; s.pr = raw.copy(); s.ph = ch; s.la += 1
            if s.la % s.tfreq == 0 and s._wd: s._train()
            return sel

        except Exception as e:
            traceback.print_exc()
            a = random.choice(s.al); a.reasoning = f"err:{e}"; return a