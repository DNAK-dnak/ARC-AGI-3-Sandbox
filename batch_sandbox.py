"""
batch_sandbox.py — Run agent on ALL games with BFS support.
Usage: python batch_sandbox.py --steps 200 --levels 3
       python batch_sandbox.py --games ls20,ar25 --steps 500
       python batch_sandbox.py --nobfs  (disable BFS for online-only testing)
"""
import argparse, glob, importlib.util, inspect, json, os, re, sys, time, traceback
import numpy as np
import types
os.environ["ONLY_RESET_LEVELS"] = "true"

AGENT_FILE = './agents/agent4.py'
GAMES_DIR  = './arc-prize-2026-arc-agi-3/environment_files'
LOG_DIR    = './logs'

# ── Logging ────────────────────────────────────────────────────────────
def get_next_log_path():
    os.makedirs(LOG_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(LOG_DIR, 'run_*.txt'))
    nums = []
    for f in existing:
        m = re.search(r'run_(\d+)', os.path.basename(f))
        if m: nums.append(int(m.group(1)))
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(LOG_DIR, f'run_{next_num:04d}.txt')

class Tee:
    def __init__(self, log_path):
        self.log_path = log_path
        self.file = open(log_path, 'w', buffering=1, encoding='utf-8')
        self.stdout = sys.stdout
    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()

# ── Arc env stub for BFS ──────────────────────────────────────────────
class _EnvInfo:
    def __init__(self, local_dir):
        self.local_dir = local_dir

class _ArcEnv:
    def __init__(self, local_dir):
        self.environment_info = _EnvInfo(local_dir)

def find_game_local_dir(game_id):
    """Find the game's local_dir by reading metadata.json or searching."""
    # Strategy 1: Walk GAMES_DIR looking for metadata.json
    if os.path.isdir(GAMES_DIR):
        for root, dirs, files in os.walk(GAMES_DIR):
            if 'metadata.json' in files:
                meta_path = os.path.join(root, 'metadata.json')
                try:
                    meta = json.load(open(meta_path))
                    gid = meta.get('game_id', '').split('-')[0]
                    if gid == game_id:
                        return root
                except:
                    pass
            # Also check if folder name matches game_id
            if os.path.basename(os.path.dirname(root)) == game_id:
                if any(f.endswith('.py') and not f.startswith('_') for f in files):
                    return root
    
    # Strategy 2: Direct glob for game_id folder
    for pattern in [
        os.path.join(GAMES_DIR, game_id, '**', f'{game_id}.py'),
        os.path.join(GAMES_DIR, '**', f'{game_id}.py'),
    ]:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return os.path.dirname(matches[0])
    
    return None

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
            self.frames = []; self.guid = None
            self.is_playback = False; self.arc_env = None
            self.action_counter = 0
            # Apply kwargs LAST so they override defaults
            for k, v in kw.items(): setattr(self, k, v)
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
def get_action_data(action):
    for attr in ('data', '_data'):
        v = getattr(action, attr, None)
        if isinstance(v, dict): return v
    try: return action.get_data()
    except: return None

def get_reasoning(action):
    return getattr(action, 'reasoning', '') or ''

def load_baseline_actions(game_id):
    """Load baseline_actions from metadata.json for comparison."""
    local_dir = find_game_local_dir(game_id)
    if not local_dir:
        return None
    meta_path = os.path.join(local_dir, 'metadata.json')
    if os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
            return meta.get('baseline_actions')
        except:
            pass
    return None

# ── Run one game ───────────────────────────────────────────────────────
def run_game(game_id, max_levels, max_steps, verbose, enable_bfs):
    from arcengine import ActionInput, GameAction, GameState

    game, cls_name = load_game(game_id)
    if game is None:
        return {'game': game_id, 'error': 'load_failed', 'levels': []}

    # Create arc_env stub so BFS can find game source
    arc_env = None
    if enable_bfs:
        local_dir = find_game_local_dir(game_id)
        if local_dir:
            arc_env = _ArcEnv(local_dir)
            print(f"  BFS:ON local_dir={local_dir}")
        else:
            print(f"  BFS:FAIL — could not find local_dir for {game_id} in {GAMES_DIR}")

    AgentClass, agent_name = load_agent_class(AGENT_FILE)
    agent = AgentClass(
        game_id=game_id, card_id='local', agent_name=agent_name,
        ROOT_URL='http://localhost', record=False, arc_env=arc_env
    )

    baseline = load_baseline_actions(game_id)
    bfs_status = 'BFS:ON' if arc_env else 'BFS:OFF'
    if baseline:
        print(f"  {bfs_status} | baseline={baseline[:max_levels]}")

    result = {
        'game': game_id,
        'cls': cls_name,
        'levels': [],
        'error': None,
        'baseline': baseline,
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

        # Only set_level for L0. After that, let levels advance naturally via WIN.
        if level == 0:
            try:
                game.set_level(level)
                res = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                res = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            except Exception as e:
                level_result['error'] = f'reset:{e}'
                result['levels'].append(level_result)
                continue
        else:
            # Level should have advanced naturally from previous WIN
            # If previous level wasn't solved, we can't continue
            if not result['levels'] or not result['levels'][-1].get('solved'):
                level_result['error'] = 'prev_not_solved'
                result['levels'].append(level_result)
                continue
            # After a WIN, the game needs a RESET to start the new level
            try:
                res = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                res.levels_completed = level  # force agent to see correct level index
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

            src = reasoning.split(':')[0] if reasoning else 'unknown'
            level_result['reasoning_counts'][src] = level_result['reasoning_counts'].get(src, 0) + 1

            level_result['last_actions'].append(action_name)
            if len(level_result['last_actions']) > 10:
                level_result['last_actions'] = level_result['last_actions'][-10:]

            if verbose:
                state_str = getattr(res, 'state', '?')
                if hasattr(state_str, 'name'): state_str = state_str.name
                lvl_completed = getattr(res, 'levels_completed', '?')
                print(f"  [{step:03d}] {action_name:12s} | {reasoning:40s} | {state_str} lvl={lvl_completed}")

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
            lvl_now = getattr(res, 'levels_completed', 0) or 0
            if state == GameState.WIN or lvl_now > level:
                level_result['solved'] = True
                level_result['steps'] = step
                if verbose:
                    print(f"  >>> LEVEL COMPLETE at step {step} (state={state} lvl_completed={lvl_now})")
                break
            elif state == GameState.GAME_OVER:
                level_result['game_over'] = True
                level_result['steps'] = step
                break

            level_result['steps'] = step

        result['levels'].append(level_result)

        bl_str = ''
        if baseline and level < len(baseline):
            bl_str = f' (baseline:{baseline[level]})'
        status = '✓ SOLVED' if level_result['solved'] else ('✗ GAME_OVER' if level_result['game_over'] else f'… timeout({level_result["steps"]})')
        if level_result['error']:
            status += f' ERR:{level_result["error"]}'
        print(f"  L{level}: {status}{bl_str}  sources={level_result['reasoning_counts']}")

        if level_result['error'] and 'reset' in level_result['error']:
            break
        if level_result['solved'] and getattr(res, 'state', None) == GameState.WIN:
            break


    return result

# ── Main ───────────────────────────────────────────────────────────────
def main(args):
    log_path = get_next_log_path()
    tee = Tee(log_path)
    sys.stdout = tee

    print(f"Log: {log_path}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

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

    bfs_label = 'BFS enabled' if not args.nobfs else 'BFS disabled (online only)'
    print(f"{'='*60}")
    print(f"  Batch sandbox: {len(game_ids)} games, {args.levels} levels, {args.steps} steps max")
    print(f"  Agent: {AGENT_FILE}")
    print(f"  {bfs_label}")
    print(f"{'='*60}\n")

    all_results = []
    total_solved = 0
    total_levels = 0
    bfs_solved = 0
    online_solved = 0
    t_start = time.time()

    for gi, gid in enumerate(game_ids):
        print(f"[{gi+1}/{len(game_ids)}] {gid}")
        t0 = time.time()

        try:
            result = run_game(gid, args.levels, args.steps, args.verbose, not args.nobfs)
        except Exception as e:
            traceback.print_exc()
            result = {'game': gid, 'error': str(e), 'levels': []}

        elapsed = time.time() - t0
        n_solved = sum(1 for l in result.get('levels', []) if l.get('solved'))
        n_levels = len(result.get('levels', []))
        total_solved += n_solved
        total_levels += n_levels

        # Count BFS vs online solves
        for l in result.get('levels', []):
            if l.get('solved'):
                sources = l.get('reasoning_counts', {})
                if 'bfs' in sources and sources.get('bfs', 0) > 0:
                    bfs_solved += 1
                else:
                    online_solved += 1

        print(f"  → {n_solved}/{n_levels} solved in {elapsed:.1f}s\n")
        all_results.append(result)

    total_time = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Total: {total_solved}/{total_levels} levels solved across {len(game_ids)} games")
    print(f"  BFS solves: {bfs_solved} | Online solves: {online_solved}")
    print(f"  Time:  {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Score: {total_solved / max(total_levels, 1):.4f}")
    print()

    # Build header
    lvl_headers = ' '.join(f"{'L'+str(i):>5}" for i in range(args.levels))
    print(f"  {'Game':<12} {lvl_headers} {'Total':>6}  Sources")
    print(f"  {'-'*70}")
    for r in all_results:
        cols = []
        all_sources = {}
        for i in range(args.levels):
            if i < len(r.get('levels', [])):
                l = r['levels'][i]
                if l.get('solved'):
                    cols.append(f"✓{l['steps']:>3}")
                elif l.get('game_over'):
                    cols.append('   GO')
                elif l.get('error'):
                    cols.append('  ERR')
                else:
                    cols.append(f"…{l['steps']:>3}")
                for src, cnt in l.get('reasoning_counts', {}).items():
                    all_sources[src] = all_sources.get(src, 0) + cnt
            else:
                cols.append('    -')

        n_solved = sum(1 for l in r.get('levels', []) if l.get('solved'))
        n_total = len(r.get('levels', []))
        top_sources = sorted(all_sources.items(), key=lambda x: -x[1])[:3]
        src_str = ' '.join(f"{s}:{c}" for s, c in top_sources)

        col_str = ' '.join(f"{c:>5}" for c in cols)
        print(f"  {r['game']:<12} {col_str} {n_solved:>2}/{n_total:<2}   {src_str}")

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

    sys.stdout = tee.stdout
    tee.close()
    print(f"Log saved: {log_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--games',   default='', help='Comma-separated game IDs, or empty for all')
    p.add_argument('--levels',  type=int, default=3)
    p.add_argument('--steps',   type=int, default=200)
    p.add_argument('--verbose', action='store_true', help='Print every action')
    p.add_argument('--agent',   default='', help='Override agent file path')
    p.add_argument('--nobfs',   action='store_true', help='Disable BFS (online agent only)')
    args = p.parse_args()
    if args.agent:
        AGENT_FILE = args.agent
    main(args)