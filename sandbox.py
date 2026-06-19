"""
Local ARC-AGI-3 sandbox — low memory, no Kaggle needed.
Usage: python sandbox_local.py --game ls20 --level 0 --steps 30
"""
import argparse, glob, importlib.util, inspect, os, re, sys
os.environ["ONLY_RESET_LEVELS"] = "true"
import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # change to 'Qt5Agg' or 'Agg' if TkAgg unavailable
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#AGENT_FILE   = './agents/agent.py'
AGENT_FILE   = './agents/qwen_agent.py'
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
        lvl_now = getattr(res, 'levels_completed', 0) or 0
        if lvl_now > level or getattr(res, 'state', None) == GameState.WIN:
            print("WIN!"); break
        elif getattr(res, 'state', None) == GameState.GAME_OVER:
            print("GAME OVER."); break

    if not headless:
        plt.ioff(); plt.show()
    print("\n── Rule-discovery state ──")
    print(graph_summary(agent))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--game',     default='ls20')
    p.add_argument('--level',    type=int, default=0)
    p.add_argument('--steps',    type=int, default=500)
    p.add_argument('--headless', action='store_true',
                   help='No display — print only (lowest memory)')
    args = p.parse_args()
    run(args.game, args.level, args.steps, args.headless)