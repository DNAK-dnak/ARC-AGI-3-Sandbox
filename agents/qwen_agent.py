# ======================================================================
# QWEN-VL AGENT v1.0 — Vision-Language Reasoning for ARC-AGI-3
#
# Architecture:
#   1. Qwen2.5-VL-7B-Instruct loaded once (class-level singleton)
#   2. Each step: render grid → image, build prompt, ask VLM
#   3. Parse VLM response to pick action
#   4. Heuristic fallbacks: stuck detection, oscillation escape,
#      random exploration for first few steps
# ======================================================================
import hashlib
import io
import logging
import os
import random
import re
import time
import traceback
from collections import defaultdict, deque
from typing import List, Optional

import numpy as np
import torch

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logger = logging.getLogger(__name__)

# ======================================================================
# MODEL SINGLETON — load once across all game instances
# ======================================================================

_MODEL = None
_PROCESSOR = None
_DEVICE = None

def _load_model():
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is not None:
        return

    from transformers import AutoProcessor, AutoModelForMultimodalLM

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

    print(f"[QWEN] Loading processor from {MODEL_ID}...")
    _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"[QWEN] Loading model (float16 on {_DEVICE})...")
    _MODEL = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    _MODEL.eval()
    print(f"[QWEN] Model loaded.")


def _ask_vlm(messages, max_tokens=80):
    """Send chat messages (with images) to Qwen2.5-VL, return text response."""
    global _MODEL, _PROCESSOR
    _load_model()

    inputs = _PROCESSOR.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_MODEL.device)

    with torch.no_grad():
        outputs = _MODEL.generate(**inputs, max_new_tokens=max_tokens)
    
    gen = outputs[0][inputs["input_ids"].shape[-1]:]
    return _PROCESSOR.decode(gen, skip_special_tokens=True).strip()


# ======================================================================
# GRID RENDERING
# ======================================================================

from PIL import Image, ImageDraw

ARC_COLORS = {
    0: "#1A1A2E", 1: "#4A4A5A", 2: "#7A7A8A", 3: "#AAAAAA",
    4: "#D0D0D0", 5: "#FFFFFF", 6: "#E53AA3", 7: "#FF7BCC",
    8: "#F93C31", 9: "#1E93FF", 10: "#88D8F1", 11: "#FFDC00",
    12: "#FF851B", 13: "#921231", 14: "#4FCC30", 15: "#A356D6",
}

def _render_grid(grid, cell_size=8):
    """Render a 2D numpy array as a colored PIL Image."""
    h, w = grid.shape
    img = Image.new("RGB", (w * cell_size, h * cell_size), "black")
    draw = ImageDraw.Draw(img)
    for y in range(h):
        for x in range(w):
            c = ARC_COLORS.get(int(grid[y, x]), "#888")
            draw.rectangle(
                [x*cell_size, y*cell_size, (x+1)*cell_size-1, (y+1)*cell_size-1],
                fill=c,
            )
    return img


def _grid_to_text(grid, max_rows=24):
    """Compact text representation of a grid."""
    lines = []
    for y in range(min(len(grid), max_rows)):
        row = grid[y]
        lines.append(f"r{y:02d}: " + " ".join(f"{int(v):2d}" for v in row))
    if len(grid) > max_rows:
        lines.append(f"... ({len(grid) - max_rows} more rows)")
    return "\n".join(lines)


# ======================================================================
# ACTION PARSING
# ======================================================================

def _parse_action(text, available_actions, avail_ids):
    """Extract a GameAction from VLM text output."""
    t = text.upper()

    # Match explicit action names
    m = re.search(r"(RESET|ACTION[1-7])", t)
    if m:
        name = m.group(0)
        act = getattr(GameAction, name, None)
        if act is not None and act in available_actions:
            if act == GameAction.ACTION6:
                cm = re.search(r"ACTION6\D*(\d+)\D+(\d+)", t)
                if cm:
                    x, y = int(cm.group(1)), int(cm.group(2))
                    act.set_data({"x": min(x,63), "y": min(y,63)})
                else:
                    act.set_data({"x": 32, "y": 32})
            return act

    # Match directional words
    dirs = {"UP": "ACTION1", "DOWN": "ACTION2", "LEFT": "ACTION3", "RIGHT": "ACTION4"}
    for word, act_name in dirs.items():
        if word in t:
            act = getattr(GameAction, act_name, None)
            if act and act in available_actions:
                return act

    # Fallback: random non-RESET action
    valid = [a for a in available_actions if a != GameAction.RESET]
    return random.choice(valid) if valid else GameAction.RESET


# ======================================================================
# AGENT
# ======================================================================

class QwenAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = abs(hash(s.game_id)) % (2**32 - 1)
        random.seed(seed); np.random.seed(seed)
        s.start_time = time.time()

        # Per-level state
        s.cl = -1          # current level
        s.la = 0           # actions this level
        s.pr = None        # previous raw frame
        s.pai = None       # previous action id
        s.stuck = 0        # consecutive no-change steps
        s.findings = []    # VLM learnings
        s.fhist = deque(maxlen=6)

        # Anti-stuck
        s._recent_actions = deque(maxlen=20)
        s._cooldowns = {}
        s._escape_action = None
        s._escape_remaining = 0
        s._round_robin_idx = 0

        # Click scan
        s._click_targets = []
        s._click_grid_idx = 0

        # Exploration budget
        s.EXPLORE_STEPS = 5
        s.VLM_INTERVAL = 3  # call VLM every N steps (to save time)
        s._last_vlm_action = None

    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES:
            s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid:
            s.guid = f.guid
        if hasattr(s, "recorder") and not s.is_playback:
            import json
            s.recorder.record(json.loads(f.model_dump_json()))

    def _raw(s, fd):
        arr = np.array(fd.frame, dtype=np.int64)
        return arr[-1] if arr.ndim == 3 else arr

    def _lvl(s, f):
        hint = getattr(f, '_level_hint', None)
        if hint is not None:
            return hint
        return getattr(f, 'levels_completed', 0) or 0

    def _grids_eq(s, a, b):
        if a is None or b is None:
            return False
        return np.array_equal(a, b)

    def _detect_oscillation(s):
        if len(s._recent_actions) < 4:
            return False
        a = list(s._recent_actions)
        if len(set(a[-3:])) == 1:
            return True
        if a[-4] == a[-2] and a[-3] == a[-1] and a[-4] != a[-3]:
            return True
        return False

    def is_done(s, frames, lf):
        try:
            return lf.state is GameState.WIN or (time.time() - s.start_time) >= 8*3600 - 300
        except:
            return True

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # ── Level change ──────────────────────────────────────────
            if lvl != s.cl:
                s.cl = lvl
                s.la = 0
                s.pr = None
                s.pai = None
                s.stuck = 0
                s.findings = []
                s.fhist.clear()
                s._recent_actions.clear()
                s._cooldowns = {}
                s._escape_action = None
                s._escape_remaining = 0
                s._round_robin_idx = 0
                s._click_targets = []
                s._click_grid_idx = 0
                s._last_vlm_action = None
                print(f"[QWEN] New level: {lvl}")

            # ── Reset state ───────────────────────────────────────────
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.pr = None; s.pai = None
                s._recent_actions.clear()
                s._escape_action = None; s._escape_remaining = 0
                s._cooldowns = {}; s.stuck = 0
                a = GameAction.RESET
                a.reasoning = "reset"
                return a

            raw = s._raw(lf)
            s.fhist.append(raw.copy())
            avail = getattr(lf, 'available_actions', None) or []
            avail_ids = [a.value if hasattr(a, 'value') else int(a) for a in avail]
            simple_ids = [i for i in avail_ids if 1 <= i <= 5]
            s.la += 1

            # ── Stuck detection ───────────────────────────────────────
            if s._grids_eq(s.pr, raw):
                s.stuck += 1
            else:
                s.stuck = 0

            # ── Tick cooldowns ────────────────────────────────────────
            for k in list(s._cooldowns.keys()):
                s._cooldowns[k] -= 1
                if s._cooldowns[k] <= 0:
                    del s._cooldowns[k]
            banned_set = set(s._cooldowns.keys())

            # ── Oscillation escape ────────────────────────────────────
            if s._escape_remaining <= 0 and s._detect_oscillation():
                osc = set(a for a in list(s._recent_actions)[-6:] if a is not None)
                cands = [a for a in simple_ids if a not in osc]
                if not cands:
                    cands = list(simple_ids)
                if cands:
                    s._escape_action = random.choice(cands)
                    s._escape_remaining = random.randint(3, 6)
                    for oa in osc:
                        if oa is not None and 1 <= oa <= 5:
                            s._cooldowns[oa] = max(s._cooldowns.get(oa, 0), 8)

            if s._escape_remaining > 0 and s._escape_action is not None:
                esc = s._escape_action
                if 1 <= esc <= 5:
                    s._escape_remaining -= 1
                    al = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                          GameAction.ACTION4, GameAction.ACTION5]
                    sel = al[esc-1]
                    sel.reasoning = f"escape:a{esc}(rem={s._escape_remaining})"
                    s.pr = raw.copy(); s.pai = esc
                    s._recent_actions.append(esc)
                    return sel

            # ── Very stuck: reset ─────────────────────────────────────
            if s.stuck >= 12:
                s.stuck = 0
                a = GameAction.RESET
                a.reasoning = "stuck_reset"
                s.pr = raw.copy()
                return a

            # ── Early random exploration ──────────────────────────────
            if s.la <= s.EXPLORE_STEPS and simple_ids:
                unbanned = [a for a in simple_ids if a not in banned_set]
                pool = unbanned if unbanned else simple_ids
                act_id = pool[s.la % len(pool)]
                al = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                      GameAction.ACTION4, GameAction.ACTION5]
                sel = al[act_id - 1]
                sel.reasoning = f"explore:a{act_id}"
                s.pr = raw.copy(); s.pai = act_id
                s._recent_actions.append(act_id)
                return sel

            # ── VLM reasoning (every VLM_INTERVAL steps) ──────────────
            use_vlm = (s.la % s.VLM_INTERVAL == 0) or s._last_vlm_action is None

            if use_vlm:
                try:
                    img = _render_grid(raw, cell_size=8)
                    action_names = [f"ACTION{i}" for i in simple_ids]
                    if 6 in avail_ids:
                        action_names.append("ACTION6(x,y)")

                    ctx = ""
                    if s.pai is not None:
                        changed = "NO change" if s._grids_eq(s.pr, raw) else "grid CHANGED"
                        ctx = f"\nLast action: ACTION{s.pai} -> {changed}"

                    findings_text = ""
                    if s.findings:
                        findings_text = "\nLearnings: " + "; ".join(s.findings[-3:])

                    prompt = (
                        f"You play an interactive 2D puzzle game (ARC-AGI-3).\n"
                        f"Available: {', '.join(action_names)}\n"
                        f"ACTION1=Up ACTION2=Down ACTION3=Left ACTION4=Right "
                        f"ACTION5=Interact ACTION6=Click(x,y) ACTION7=Undo\n"
                        f"{ctx}{findings_text}\n\n"
                        f"Grid ({raw.shape[0]}x{raw.shape[1]}, values=colors 0-15):\n"
                        f"{_grid_to_text(raw)}\n\n"
                        f"Look at the image. Find your player, walls, doors, goals. "
                        f"Choose ONE action to progress.\n"
                        f"Reply with ONLY the action (e.g. ACTION2)."
                    )

                    msgs = [{"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ]}]

                    t0 = time.time()
                    resp = _ask_vlm(msgs, max_tokens=60)
                    elapsed = time.time() - t0
                    print(f"[QWEN] L{lvl} step {s.la}: '{resp[:80]}' ({elapsed:.1f}s)")

                    sel = _parse_action(resp, avail, avail_ids)
                    sel.reasoning = f"vlm:{resp[:40]}"
                    s._last_vlm_action = sel

                    # Extract learning
                    if len(resp) > 30:
                        finding = resp[:120].replace("\n", " ").strip()
                        if finding not in s.findings:
                            s.findings.append(finding)
                            if len(s.findings) > 8:
                                s.findings.pop(0)

                    act_id = sel.value if hasattr(sel, 'value') else 1
                    s.pr = raw.copy(); s.pai = act_id
                    s._recent_actions.append(act_id)
                    return sel

                except Exception as e:
                    print(f"[QWEN] VLM error: {e}")
                    traceback.print_exc()

            # ── Repeat last VLM action or round-robin ─────────────────
            al = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                  GameAction.ACTION4, GameAction.ACTION5]

            if s._last_vlm_action is not None and s._last_vlm_action in avail:
                sel = s._last_vlm_action
                sel.reasoning = "vlm_repeat"
                act_id = sel.value if hasattr(sel, 'value') else 1
                s.pr = raw.copy(); s.pai = act_id
                s._recent_actions.append(act_id)
                return sel

            # ── Click scan (for click-only games) ─────────────────────
            if 6 in avail_ids and (not simple_ids):
                if not s._click_targets:
                    bg = int(np.bincount(raw.flatten(), minlength=16).argmax())
                    tgts = []
                    for y in range(0, 64, 3):
                        for x in range(0, 64, 3):
                            if raw[y, x] != bg:
                                tgts.append((x, y))
                    if not tgts:
                        tgts = [(x, y) for y in range(0, 64, 8) for x in range(0, 64, 8)]
                    s._click_targets = tgts
                if s._click_targets:
                    idx = s._click_grid_idx % len(s._click_targets)
                    x, y = s._click_targets[idx]
                    s._click_grid_idx += 1
                    sel = GameAction.ACTION6
                    sel.set_data({"x": x, "y": y})
                    sel.reasoning = f"cscan:({x},{y})"
                    s.pr = raw.copy(); s.pai = 6
                    s._recent_actions.append(6)
                    return sel

            # ── Round-robin simple actions ────────────────────────────
            if simple_ids:
                unbanned = [a for a in simple_ids if a not in banned_set]
                pool = unbanned if unbanned else simple_ids
                idx = s._round_robin_idx % len(pool)
                act_id = pool[idx]
                s._round_robin_idx = (idx + 1) % len(pool)
                sel = al[act_id - 1]
                sel.reasoning = f"rr:a{act_id}"
                s.pr = raw.copy(); s.pai = act_id
                s._recent_actions.append(act_id)
                return sel

            # ── Absolute fallback ─────────────────────────────────────
            sel = random.choice(avail) if avail else GameAction.RESET
            sel.reasoning = "fallback"
            s.pr = raw.copy()
            return sel

        except Exception as e:
            traceback.print_exc()
            al = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                  GameAction.ACTION4, GameAction.ACTION5]
            a = random.choice(al)
            a.reasoning = f"err:{e}"
            return a
