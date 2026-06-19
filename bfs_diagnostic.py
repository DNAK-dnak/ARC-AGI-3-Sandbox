"""
bfs_diagnostic.py — Test if BFS can find and load game sources.
Usage: python bfs_diagnostic.py
"""
import glob, json, os, re, sys, importlib.util, types
from pathlib import Path

GAMES_DIR = './arc-prize-2026-arc-agi-3/environment_files'

# Stub agents module
stub = types.ModuleType('agents')
stub.agent = types.ModuleType('agents.agent')
class _Agent:
    MAX_ACTIONS = float('inf'); _MAX_FRAMES = 10
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
        self.frames = []; self.guid = None; self.is_playback = False
        self.arc_env = None; self.action_counter = 0
    def append_frame(self, f): pass
stub.agent.Agent = _Agent; stub.agent.Playback = object
sys.modules['agents'] = stub; sys.modules['agents.agent'] = stub.agent

# Find all games
for root, dirs, files in os.walk(GAMES_DIR):
    if 'metadata.json' not in files:
        continue
    meta = json.load(open(os.path.join(root, 'metadata.json')))
    game_id_full = meta.get('game_id', '')
    gid = game_id_full.split('-')[0]
    
    # Check if game .py exists in this folder
    py_files = [f for f in files if f.endswith('.py') and not f.startswith('_')]
    game_py = os.path.join(root, f'{gid}.py') if os.path.exists(os.path.join(root, f'{gid}.py')) else None
    
    # Simulate find_game_source_and_class
    ld = Path(root)
    cls_name = gid[0].upper() + gid[1:]
    found_src = None
    for candidate in [ld / f"{gid}.py", ld / f"{cls_name.lower()}.py"]:
        if candidate.exists():
            found_src = str(candidate)
            content = candidate.read_text()[:2000]
            m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
            if m: cls_name = m.group(1)
            break
    
    # Try to actually load the class
    load_ok = False
    if found_src:
        try:
            spec = importlib.util.spec_from_file_location('game_mod', found_src)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            game_cls = getattr(mod, cls_name, None)
            if game_cls:
                load_ok = True
        except Exception as e:
            load_ok = f"ERR:{e}"
    
    status = '✓' if load_ok is True else ('✗' if not found_src else f'✗ {load_ok}')
    print(f"  {status} {gid:<8} src={found_src is not None}  cls={cls_name:<15} load={load_ok}  dir={root}")