# Combined Project Code

## sandbox.py
```python
"""
Local ARC-AGI-3 sandbox — low memory, no Kaggle needed.
Usage: python sandbox_local.py --game ls20 --level 0 --steps 30
"""
import argparse, glob, importlib.util, inspect, os, re, sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # change to 'Qt5Agg' or 'Agg' if TkAgg unavailable
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#AGENT_FILE   = './agents/agent.py'
AGENT_FILE   = './agents/nexus_agent.py'
#AGENT_FILE   = './agents/sovereignv2_nonbfs_agent.py'
GAMES_DIR    = './arc-prize-2026-arc-agi-3/environment_files'

arc_colors = ['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00',
              '#AAAAAA','#F012BE','#FF851B','#7FDBFF','#870C25'] + ['#333333']*6
cmap = ListedColormap(arc_colors)

# ── Loaders ────────────────────────────────────────────────────────────
def load_game(game_id):
    pattern = os.path.join(GAMES_DIR, '**', f'{game_id}*.py')
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"No game file for '{game_id}' in {GAMES_DIR}")
    path = matches[0]
    spec = importlib.util.spec_from_file_location('_game', path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src  = open(path, errors='ignore').read()
    m    = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', src)
    if not m: raise ValueError(f"No ARCBaseGame subclass in {path}")
    return getattr(mod, m.group(1))(), m.group(1)

def load_agent(agent_file):
    # Stub out agents.agent if not installed locally
    import types
    stub = types.ModuleType('agents')
    stub.agent = types.ModuleType('agents.agent')

    class _Agent:
        MAX_ACTIONS = float('inf')
        _MAX_FRAMES = 10
        def __init__(self, **kw):
            for k, v in kw.items(): setattr(self, k, v)
            self.frames = []; self.guid = None
            self.is_playback = False; self.arc_env = None
            self.action_counter = 0
        def append_frame(self, f): pass

    stub.agent.Agent    = _Agent
    stub.agent.Playback = object
    sys.modules['agents']       = stub
    sys.modules['agents.agent'] = stub.agent

    spec = importlib.util.spec_from_file_location('_myagent', agent_file)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules['_myagent'] = mod
    spec.loader.exec_module(mod)
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        try:
            if issubclass(obj, _Agent) and obj is not _Agent:
                return obj, name
        except: pass
    raise ValueError("No Agent subclass found in agent file")

# ── Helpers ────────────────────────────────────────────────────────────
def get_frame(res):
    if res and res.frame:
        return np.array(res.frame[-1])
    return np.zeros((8,8), dtype=int)

def get_action_data(action):
    for attr in ('data', '_data'):
        v = getattr(action, attr, None)
        if isinstance(v, dict): return v
    try: return action.get_data()
    except: return None

def graph_summary(agent):
    g = getattr(agent, '_graph', None)
    if g is None: return "no graph"
    lines = [f"tried={sorted(g.tried)}"]
    for a, evs in sorted(g.action_events.items()):
        s = ', '.join(f"{e['type']}(c{e['color']})" for e in evs) or 'no effect'
        lines.append(f"  A{a}: {s}")
    nc = sum(1 for h in g.hypotheses.values() if h.confirmed)
    np_ = sum(1 for h in g.hypotheses.values() if not h.confirmed)
    lines.append(f"hyp: {nc} confirmed, {np_} pending")
    if g.goal_sigs: lines.append(f"goal_sigs: {sorted(g.goal_sigs)}")
    return '\n'.join(lines)

# ── Main loop ──────────────────────────────────────────────────────────
def run(game_id, level, steps, headless):
    from arcengine import ActionInput, GameAction, GameState

    game, cls_name = load_game(game_id)
    print(f"Game: {cls_name}")

    AgentClass, agent_name = load_agent(AGENT_FILE)
    agent = AgentClass(
        game_id=game_id, card_id='local', agent_name=agent_name,
        ROOT_URL='http://localhost', record=False, arc_env=None
    )
    print(f"Agent: {agent_name}")

    game.set_level(level)
    res = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    res = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)

    if not headless:
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.patch.set_facecolor('#0f0f1a')

    for step in range(1, steps + 1):
        frame = get_frame(res)

        try:
            action = agent.choose_action([], res)
            agent.action_counter += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{step:02d}] choose_action crashed: {e}")
            break

        act_data    = get_action_data(action)
        action_name = action.name if hasattr(action, 'name') else str(action)
        reasoning   = str(getattr(agent, 'pai', 'n/a'))
        state       = getattr(res, 'state', '?')
        state_str   = state.name if hasattr(state, 'name') else str(state)

        print(f"[{step:02d}] {action_name:12s} | {reasoning} | {state_str}")

        if not headless:
            ax1.clear(); ax2.clear()
            ax1.imshow(frame, cmap=cmap, vmin=0, vmax=15, interpolation='nearest')
            if act_data and 'x' in act_data:
                ax1.plot(act_data['x'], act_data['y'], 'rx', markersize=18, markeredgewidth=3)
            ax1.set_title(f"Step {step}: {action_name}\n{reasoning}", color='white', fontsize=10)
            ax1.axis('off'); ax1.set_facecolor('#1a1a2e')

            ax2.axis('off'); ax2.set_facecolor('#1a1a2e')
            ax2.text(0.02, 0.98, 'Rule-Discovery', color='#7fdbff',
                     transform=ax2.transAxes, va='top', fontsize=10, fontweight='bold',
                     fontfamily='monospace')
            ax2.text(0.02, 0.88, graph_summary(agent), color='#2ecc40',
                     transform=ax2.transAxes, va='top', fontsize=8,
                     fontfamily='monospace', wrap=True)

            plt.tight_layout(); plt.pause(0.3)  # 0.3s per step — adjust for speed

        try:
            ai  = ActionInput(id=action, data=act_data) if act_data else ActionInput(id=action)
            res = game.perform_action(ai, raw=True)
        except Exception as e:
            print(f"[{step:02d}] perform_action crashed: {e}"); break

        if not res or not res.frame:
            print("No frame — stopping."); break
        if getattr(res, 'state', None) == GameState.WIN:
            print("WIN!"); break
        elif getattr(res, 'state', None) == GameState.GAME_OVER:
            print("GAME OVER."); break

    if not headless:
        plt.ioff(); plt.show()
    print("\n── Rule-discovery state ──")
    print(graph_summary(agent))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--game',     default='bp35')
    p.add_argument('--level',    type=int, default=0)
    p.add_argument('--steps',    type=int, default=500)
    p.add_argument('--headless', action='store_true',
                   help='No display — print only (lowest memory)')
    args = p.parse_args()
    run(args.game, args.level, args.steps, args.headless)
```

## sandbox_test.py
```python
import arcengine
import os

# This prints the exact folder where the arcengine package is installed
engine_path = arcengine.__path__[0]
print("ARC Engine is installed at:", engine_path)

# You can list the files inside it to see where the games are bundled
print("Contents:", os.listdir(engine_path))

```

## tempCodeRunnerFile.py
```python
import argparse, glob, importlib.util, inspect, os, re, sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # change to 'Qt5Agg' or 'Agg' if TkAgg unavailable
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
```

## batch_sandbox.py
```python
"""
batch_sandbox.py — Run agent on ALL games, log results.
Usage: python batch_sandbox.py --steps 200 --levels 3
       python batch_sandbox.py --games ls20,ar25 --steps 500
"""
import argparse, glob, importlib.util, inspect, os, re, sys, time, traceback
import numpy as np
import types

AGENT_FILE = './agents/nexus_agent.py'
GAMES_DIR  = './arc-prize-2026-arc-agi-3/environment_files'

# ── Loaders ────────────────────────────────────────────────────────────
def load_game(game_id):
    pattern = os.path.join(GAMES_DIR, '**', f'{game_id}*.py')
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None, None
    path = matches[0]
    spec = importlib.util.spec_from_file_location('_game', path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src  = open(path, errors='ignore').read()
    m    = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', src)
    if not m: return None, None
    return getattr(mod, m.group(1))(), m.group(1)

_agent_class = None
def load_agent_class(agent_file):
    global _agent_class
    if _agent_class is not None:
        return _agent_class

    stub = types.ModuleType('agents')
    stub.agent = types.ModuleType('agents.agent')

    class _Agent:
        MAX_ACTIONS = float('inf')
        _MAX_FRAMES = 10
        def __init__(self, **kw):
            for k, v in kw.items(): setattr(self, k, v)
            self.frames = []; self.guid = None
            self.is_playback = False; self.arc_env = None
            self.action_counter = 0
        def append_frame(self, f): pass

    stub.agent.Agent    = _Agent
    stub.agent.Playback = object
    sys.modules['agents']       = stub
    sys.modules['agents.agent'] = stub.agent

    spec = importlib.util.spec_from_file_location('_myagent', agent_file)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules['_myagent'] = mod
    spec.loader.exec_module(mod)
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        try:
            if issubclass(obj, _Agent) and obj is not _Agent:
                _agent_class = (obj, name)
                return _agent_class
        except: pass
    raise ValueError("No Agent subclass found")

# ── Helpers ────────────────────────────────────────────────────────────
def get_frame(res):
    if res and res.frame:
        return np.array(res.frame[-1])
    return np.zeros((8,8), dtype=int)

def get_action_data(action):
    for attr in ('data', '_data'):
        v = getattr(action, attr, None)
        if isinstance(v, dict): return v
    try: return action.get_data()
    except: return None

def get_reasoning(action):
    return getattr(action, 'reasoning', '') or ''

# ── Run one game ───────────────────────────────────────────────────────
def run_game(game_id, max_levels, max_steps, verbose):
    from arcengine import ActionInput, GameAction, GameState

    game, cls_name = load_game(game_id)
    if game is None:
        return {'game': game_id, 'error': 'load_failed', 'levels': []}

    AgentClass, agent_name = load_agent_class(AGENT_FILE)
    agent = AgentClass(
        game_id=game_id, card_id='local', agent_name=agent_name,
        ROOT_URL='http://localhost', record=False, arc_env=None
    )

    result = {
        'game': game_id,
        'cls': cls_name,
        'levels': [],
        'error': None,
    }

    for level in range(max_levels):
        level_result = {
            'level': level,
            'steps': 0,
            'solved': False,
            'game_over': False,
            'error': None,
            'reasoning_counts': {},
            'last_actions': [],
        }

        try:
            game.set_level(level)
            res = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            res = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        except Exception as e:
            level_result['error'] = f'reset:{e}'
            result['levels'].append(level_result)
            continue

        for step in range(1, max_steps + 1):
            try:
                action = agent.choose_action([], res)
                agent.action_counter += 1
            except Exception as e:
                level_result['error'] = f'choose:{e}'
                break

            act_data = get_action_data(action)
            action_name = action.name if hasattr(action, 'name') else str(action)
            reasoning = get_reasoning(action)

            # Track reasoning source distribution
            src = reasoning.split(':')[0] if reasoning else 'unknown'
            level_result['reasoning_counts'][src] = level_result['reasoning_counts'].get(src, 0) + 1

            # Keep last 10 actions for loop detection logging
            level_result['last_actions'].append(action_name)
            if len(level_result['last_actions']) > 10:
                level_result['last_actions'] = level_result['last_actions'][-10:]

            if verbose:
                state_str = getattr(res, 'state', '?')
                if hasattr(state_str, 'name'): state_str = state_str.name
                print(f"  [{step:03d}] {action_name:12s} | {reasoning:40s} | {state_str}")

            try:
                ai = ActionInput(id=action, data=act_data) if act_data else ActionInput(id=action)
                res = game.perform_action(ai, raw=True)
            except Exception as e:
                level_result['error'] = f'perform:{e}'
                break

            if not res or not res.frame:
                level_result['error'] = 'no_frame'
                break

            state = getattr(res, 'state', None)
            if state == GameState.WIN:
                level_result['solved'] = True
                level_result['steps'] = step
                break
            elif state == GameState.GAME_OVER:
                level_result['game_over'] = True
                level_result['steps'] = step
                break

            level_result['steps'] = step

        result['levels'].append(level_result)

        status = '✓ SOLVED' if level_result['solved'] else ('✗ GAME_OVER' if level_result['game_over'] else f'… timeout({level_result["steps"]})')
        if level_result['error']:
            status += f' ERR:{level_result["error"]}'
        print(f"  L{level}: {status}  sources={level_result['reasoning_counts']}")

        # If we can't even start this level, skip higher ones
        if level_result['error'] and 'reset' in level_result['error']:
            break

    return result

# ── Main ───────────────────────────────────────────────────────────────
def main(args):
    # Find all games
    if args.games:
        game_ids = [g.strip() for g in args.games.split(',')]
    else:
        game_files = sorted(glob.glob(os.path.join(GAMES_DIR, '**', '*.py'), recursive=True))
        game_ids = []
        for gf in game_files:
            name = os.path.splitext(os.path.basename(gf))[0]
            if not name.startswith('_'):
                game_ids.append(name)
        game_ids = sorted(set(game_ids))

    print(f"{'='*60}")
    print(f"  Batch sandbox: {len(game_ids)} games, {args.levels} levels, {args.steps} steps max")
    print(f"  Agent: {AGENT_FILE}")
    print(f"{'='*60}\n")

    all_results = []
    total_solved = 0
    total_levels = 0
    t_start = time.time()

    for gi, gid in enumerate(game_ids):
        print(f"[{gi+1}/{len(game_ids)}] {gid}")
        t0 = time.time()

        try:
            result = run_game(gid, args.levels, args.steps, args.verbose)
        except Exception as e:
            traceback.print_exc()
            result = {'game': gid, 'error': str(e), 'levels': []}

        elapsed = time.time() - t0
        n_solved = sum(1 for l in result.get('levels', []) if l.get('solved'))
        n_levels = len(result.get('levels', []))
        total_solved += n_solved
        total_levels += n_levels

        print(f"  → {n_solved}/{n_levels} solved in {elapsed:.1f}s\n")
        all_results.append(result)

    total_time = time.time() - t_start

    # ── Summary report ──
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total: {total_solved}/{total_levels} levels solved across {len(game_ids)} games")
    print(f"  Time:  {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Score: {total_solved / max(total_levels, 1):.4f}")
    print()

    # Per-game breakdown
    print(f"  {'Game':<12} {'L0':>4} {'L1':>4} {'L2':>4} {'Total':>6}  Sources")
    print(f"  {'-'*60}")
    for r in all_results:
        cols = []
        all_sources = {}
        for i in range(args.levels):
            if i < len(r.get('levels', [])):
                l = r['levels'][i]
                if l.get('solved'):
                    cols.append(f"✓{l['steps']:>3}")
                elif l.get('game_over'):
                    cols.append('  GO')
                elif l.get('error'):
                    cols.append(' ERR')
                else:
                    cols.append(f"…{l['steps']:>3}")
                for src, cnt in l.get('reasoning_counts', {}).items():
                    all_sources[src] = all_sources.get(src, 0) + cnt
            else:
                cols.append('   -')

        n_solved = sum(1 for l in r.get('levels', []) if l.get('solved'))
        n_total = len(r.get('levels', []))
        top_sources = sorted(all_sources.items(), key=lambda x: -x[1])[:3]
        src_str = ' '.join(f"{s}:{c}" for s, c in top_sources)

        while len(cols) < args.levels:
            cols.append('   -')

        print(f"  {r['game']:<12} {cols[0]:>4} {cols[1]:>4} {cols[2] if args.levels > 2 else '':>4} {n_solved:>2}/{n_total:<2}   {src_str}")

    # Reasoning source totals
    print(f"\n  {'─'*40}")
    print(f"  Reasoning source distribution:")
    grand_sources = {}
    for r in all_results:
        for l in r.get('levels', []):
            for src, cnt in l.get('reasoning_counts', {}).items():
                grand_sources[src] = grand_sources.get(src, 0) + cnt
    for src, cnt in sorted(grand_sources.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / max(sum(grand_sources.values()), 1)
        bar = '█' * int(pct / 2)
        print(f"    {src:<20} {cnt:>6} ({pct:>5.1f}%) {bar}")

    print()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--games',   default='', help='Comma-separated game IDs, or empty for all')
    p.add_argument('--levels',  type=int, default=3)
    p.add_argument('--steps',   type=int, default=200)
    p.add_argument('--verbose', action='store_true', help='Print every action')
    p.add_argument('--agent',   default='', help='Override agent file path')
    args = p.parse_args()
    if args.agent:
        AGENT_FILE = args.agent
    main(args)
```

