#%%writefile /kaggle/working/my_agent.py
# ======================================================================
# SOVEREIGN v4-clean — Exact 0.27 logic, deduplicated boilerplate
# Only change: _commit() helper, _nexus_encode() cache, use_tg toggle
# ======================================================================
import heapq, pickle, copy, glob, hashlib, importlib.util, logging, os
import random, time, traceback
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

logger = logging.getLogger(__name__)

# ==================== BFS SOLVER (EXACT 0.27) ====================
class BFSSolver:
    def __init__(s, game_path, game_class_name, scan_timeout=3, bfs_timeout=120):
        s.game_path=game_path; s.class_name=game_class_name
        s.scan_timeout=scan_timeout; s.bfs_timeout=bfs_timeout
        s.game_cls=None; s.solutions={}; s._warmup_prefix=[]
    def load(s):
        try:
            spec=importlib.util.spec_from_file_location('game_mod',s.game_path)
            mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            s.game_cls=getattr(mod,s.class_name); return True
        except Exception as e: logger.warning(f"BFS load fail: {e}"); return False
    def _extract_win_field(s):
        try:
            import re; lines=open(s.game_path).read().split('\n')
            for i,line in enumerate(lines):
                if 'self.next_level()' in line:
                    for j in range(i-1,max(0,i-8),-1):
                        st=lines[j].strip()
                        if st.startswith('if ') or st.startswith('elif '):
                            m=re.search(r'self\.(\w+)',st)
                            if m: return m.group(1)
                    break
        except: pass
        return None
    def _probe_hidden_fields(s, game, actions):
        if not actions: return []
        wf=s._extract_win_field(); initial={}
        for k,v in game.__dict__.items():
            if isinstance(v,(int,float,bool)) and not k.startswith('__'): initial[k]=v
        changing=set()
        if wf and wf in initial: changing.add(wf)
        for aid,data in actions[:10]:
            g=copy.deepcopy(game)
            try:
                ai=ActionInput(id=GameAction.from_id(aid),data=data) if data else ActionInput(id=GameAction.from_id(aid))
                g.perform_action(ai,raw=True)
            except: continue
            for k,v in g.__dict__.items():
                if isinstance(v,(int,float,bool)) and not k.startswith('__'):
                    if k in initial and v!=initial[k] and k not in ('_action_count','_full_reset','_action_complete'):
                        changing.add(k)
        return sorted([f for f in changing if not(f.startswith('_') and f not in ('_current_level_index','_score'))])
    def _scan_actions(s, game, f0, bg):
        avail=game._available_actions; actions=[]
        for a in [a for a in avail if a<=5]:
            g=copy.deepcopy(game)
            try:
                r=g.perform_action(ActionInput(id=GameAction.from_id(a)),raw=True)
                if r.frame and np.sum(f0!=np.array(r.frame[-1]))>0: actions.append((a,None))
            except: pass
        if 6 in avail:
            t0=time.time()
            for y in range(0,64,2):
                if time.time()-t0>s.scan_timeout: break
                for x in range(0,64,2):
                    if f0[y,x]==bg: continue
                    g=copy.deepcopy(game)
                    try:
                        r=g.perform_action(ActionInput(id=GameAction.ACTION6,data={'x':x,'y':y,'game_id':'bfs'}),raw=True)
                        if r.frame and np.sum(f0!=np.array(r.frame[-1]))>0:
                            actions.append((6,{'x':x,'y':y,'game_id':'bfs'}))
                    except: pass
        return actions
    def solve_level(s, level_idx, max_states=500000, prev_solution=None):
        if not s.game_cls: return None
        game=s.game_cls(); game.set_level(level_idx)
        game.perform_action(ActionInput(id=GameAction.RESET),raw=True)
        r0=game.perform_action(ActionInput(id=GameAction.RESET),raw=True)
        if not r0.frame: return None
        f0=np.array(r0.frame[-1]); bg=int(np.bincount(f0.flatten(),minlength=16).argmax())
        if prev_solution and level_idx>0:
            tr=s._try_transfer(game,level_idx,prev_solution,f0)
            if tr: return tr
        actions=s._scan_actions(game,f0,bg)
        if not actions:
            for wid in [a for a in game._available_actions if a<=4]:
                gw=copy.deepcopy(game)
                try:
                    gw.perform_action(ActionInput(id=GameAction.from_id(wid)),raw=True)
                    fa=np.array(gw.get_pixels(0,0,64,64)); wa=s._scan_actions(gw,fa,bg)
                    if wa: game=gw; f0=fa; actions=wa; s._warmup_prefix=[(wid,None)]; break
                except: pass
        if not actions: return None
        visited=set(); visited.add(hashlib.md5(f0.tobytes()).hexdigest()[:16])
        queue=deque([(copy.deepcopy(game),[],0)]); t0=time.time(); explored=0
        while queue and explored<max_states and (time.time()-t0)<s.bfs_timeout:
            g,hist,depth=queue.popleft()
            for aid,data in actions:
                g2=copy.deepcopy(g)
                try:
                    ai=ActionInput(id=GameAction.from_id(aid),data=data) if data else ActionInput(id=GameAction.from_id(aid))
                    r=g2.perform_action(ai,raw=True)
                except: continue
                explored+=1
                if not r.frame: continue
                f=np.array(r.frame[-1]); h=hashlib.md5(f.tobytes()).hexdigest()[:16]
                if h in visited: continue
                visited.add(h); nh=hist+[(aid,data)]
                if r.levels_completed>level_idx or g2._current_level_index>level_idx:
                    s.solutions[level_idx]=nh; return nh
                if depth<30: queue.append((g2,nh,depth+1))
        ef=time.time()-t0
        if explored<20 and ef>10: return None
        # ACMD
        if len(visited)<100 and ef<s.bfs_timeout*0.7:
            hf=s._probe_hidden_fields(game,actions)
            if hf:
                cf=set()
                try:
                    if actions:
                        g2=copy.deepcopy(game); ai=ActionInput(id=GameAction.from_id(actions[0][0]),data=actions[0][1]) if actions[0][1] else ActionInput(id=GameAction.from_id(actions[0][0]))
                        g2.perform_action(ai,raw=True); g3=copy.deepcopy(g2); g3.perform_action(ai,raw=True)
                        for fn in hf:
                            if getattr(g2,fn,None)!=getattr(g3,fn,None): cf.add(fn)
                except: pass
                tf=[f for f in hf if f not in cf] or hf
                gm2=s.game_cls(); gm2.set_level(level_idx)
                gm2.perform_action(ActionInput(id=GameAction.RESET),raw=True)
                gm2.perform_action(ActionInput(id=GameAction.RESET),raw=True)
                r02=gm2.perform_action(ActionInput(id=GameAction.RESET),raw=True)
                if r02 and r02.frame:
                    f02=np.array(r02.frame[-1]); ist={fn:getattr(gm2,fn,None) for fn in tf}
                    v2=set(); fh2=hashlib.md5(f02.tobytes()).hexdigest()[:16]
                    for fn in tf:
                        cv=getattr(gm2,fn,None)
                        if cv is not None: fh2+=f"|{fn}={cv}"
                    v2.add(fh2); fifo=0
                    heap=[(0,0,fifo,copy.deepcopy(gm2),[])]; fifo+=1
                    t02=time.time(); exp2=0; rem=max(60,s.bfs_timeout-ef)
                    while heap and exp2<max_states and (time.time()-t02)<rem:
                        _,dep,_,g,hist=heapq.heappop(heap)
                        for aid,data in actions:
                            g2=copy.deepcopy(g)
                            try:
                                ai=ActionInput(id=GameAction.from_id(aid),data=data) if data else ActionInput(id=GameAction.from_id(aid))
                                r=g2.perform_action(ai,raw=True)
                            except: continue
                            exp2+=1
                            if not r.frame: continue
                            f=np.array(r.frame[-1]); fh=hashlib.md5(f.tobytes()).hexdigest()[:16]
                            for fn in tf:
                                cv=getattr(g2,fn,None)
                                if cv is not None: fh+=f"|{fn}={cv}"
                            if fh in v2: continue
                            v2.add(fh); nh=hist+[(aid,data)]
                            if r.levels_completed>level_idx or g2._current_level_index>level_idx:
                                s.solutions[level_idx]=nh; return nh
                            td=sum(abs(getattr(g2,fn,0)-(ist.get(fn,0) or 0)) if isinstance(getattr(g2,fn,None),(int,float)) else (1 if getattr(g2,fn,None)!=ist.get(fn) else 0) for fn in tf)
                            pc=np.sum(f02!=f)>0
                            if not pc and td==0: continue
                            fifo+=1
                            if dep<40: heapq.heappush(heap,(-td,dep+1,fifo,g2,nh))
        # IDDFS
        et=time.time()-t0; rt=max(30,s.bfs_timeout-et)
        if len(actions)<=6 and rt>30:
            gm3=s.game_cls(); gm3.set_level(level_idx)
            gm3.perform_action(ActionInput(id=GameAction.RESET),raw=True)
            gm3.perform_action(ActionInput(id=GameAction.RESET),raw=True)
            t03=time.time()
            for md in range(10,60):
                if time.time()-t03>rt: break
                stack=[(copy.deepcopy(gm3),[],set())]
                while stack and (time.time()-t03)<rt:
                    g,hist,ph=stack.pop()
                    if len(hist)>=md: continue
                    for aid,data in actions:
                        g2=copy.deepcopy(g)
                        try:
                            ai=ActionInput(id=GameAction.from_id(aid),data=data) if data else ActionInput(id=GameAction.from_id(aid))
                            r=g2.perform_action(ai,raw=True)
                        except: continue
                        if not r.frame: continue
                        f=np.array(r.frame[-1]); h=hashlib.md5(f.tobytes()).hexdigest()[:16]
                        if h in ph: continue
                        nh=hist+[(aid,data)]
                        if r.levels_completed>level_idx or g2._current_level_index>level_idx:
                            sol=s._warmup_prefix+nh; s.solutions[level_idx]=sol; return sol
                        stack.append((g2,nh,ph|{h}))
        return None
    def _try_transfer(s, game, level_idx, prev_sol, f1):
        try:
            g=copy.deepcopy(game)
            for i,(aid,data) in enumerate(prev_sol):
                try:
                    ai=ActionInput(id=GameAction.from_id(aid),data=data) if data else ActionInput(id=GameAction.from_id(aid))
                    r=g.perform_action(ai,raw=True)
                    if r.levels_completed>level_idx or g._current_level_index>level_idx:
                        s.solutions[level_idx]=prev_sol[:i+1]; return prev_sol[:i+1]
                except: break
            pg=s.game_cls(); pg.set_level(level_idx-1)
            pg.perform_action(ActionInput(id=GameAction.RESET),raw=True)
            rp=pg.perform_action(ActionInput(id=GameAction.RESET),raw=True)
            if not rp.frame: return None
            f0=np.array(rp.frame[-1]); bg=int(np.bincount(f0.flatten(),minlength=16).argmax())
            def gobj(fr,bg):
                o=[]
                for c in range(16):
                    if c==bg: continue
                    m=(fr==c); n=int(np.sum(m))
                    if n<2: continue
                    ys,xs=np.where(m); o.append({'color':c,'cx':float(np.mean(xs)),'cy':float(np.mean(ys)),'n':n})
                return sorted(o,key=lambda x:(x['color'],-x['n']))
            op=gobj(f0,bg); oc=gobj(f1,bg)
            if not op or not oc: return None
            matched=[]
            for o in op:
                b=None; bd=float('inf')
                for c in oc:
                    if c['color']==o['color'] and abs(c['n']-o['n'])<max(o['n'],c['n'])*0.5:
                        d=abs(c['cx']-o['cx'])+abs(c['cy']-o['cy'])
                        if d<bd: bd=d; b=c
                if b: matched.append((o,b))
            if not matched: return None
            dx=np.mean([m[1]['cx']-m[0]['cx'] for m in matched])
            dy=np.mean([m[1]['cy']-m[0]['cy'] for m in matched])
            for mult in [1,2,3,1.5]:
                exp=[]
                for aid,data in prev_sol:
                    for _ in range(int(mult)):
                        if data and 'x' in data:
                            nd=dict(data); nd['x']=max(0,min(63,int(data['x']+dx))); nd['y']=max(0,min(63,int(data['y']+dy)))
                            exp.append((aid,nd))
                        else: exp.append((aid,data))
                g=copy.deepcopy(game)
                for i,(aid,data) in enumerate(exp):
                    try:
                        ai=ActionInput(id=GameAction.from_id(aid),data=data) if data else ActionInput(id=GameAction.from_id(aid))
                        r=g.perform_action(ai,raw=True)
                        if r.levels_completed>level_idx or g._current_level_index>level_idx:
                            s.solutions[level_idx]=exp[:i+1]; return exp[:i+1]
                    except: break
        except: pass
        return None

def find_game_source_and_class(game_id, arc_env=None):
    gid=game_id.split('-')[0]; cn=gid[0].upper()+gid[1:]; src=None
    if arc_env and hasattr(arc_env,'environment_info'):
        ei=arc_env.environment_info
        if hasattr(ei,'local_dir') and ei.local_dir:
            from pathlib import Path; import re; ld=Path(ei.local_dir)
            for c in [ld/f"{gid}.py",ld/f"{cn.lower()}.py"]:
                if c.exists():
                    src=str(c); m=re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame',c.read_text()[:2000])
                    if m: cn=m.group(1); break
    if not src:
        import re
        for p in [f"/tmp/*/{gid}/*/{gid}.py",f"/kaggle/*/{gid}*/{gid}.py",f"**/game_sources/**/{gid}.py"]:
            ms=glob.glob(p,recursive=True)
            if ms:
                src=ms[0]; m=re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame',open(src).read()[:2000])
                if m: cn=m.group(1); break
    return src,cn

# ==================== CNN ====================
class CBAM(nn.Module):
    def __init__(s,ch,r=16):
        super().__init__(); s.fc1=nn.Linear(ch,max(ch//r,4)); s.fc2=nn.Linear(max(ch//r,4),ch); s.sp=nn.Conv2d(2,1,7,padding=3)
    def forward(s,x):
        B,C,H,W=x.shape; w=torch.sigmoid(s.fc2(F.relu(s.fc1(x.mean(dim=[2,3]))))); x=x*w.view(B,C,1,1)
        return x*torch.sigmoid(s.sp(torch.cat([x.max(1,keepdim=True)[0],x.mean(1,keepdim=True)],1)))

class ActionEffectAttention(nn.Module):
    def __init__(s,fd=64,md=32,na=5):
        super().__init__(); s.md=md
        s.de=nn.Sequential(nn.Conv2d(1,8,8,stride=8),nn.ReLU(),nn.Conv2d(8,16,4,stride=4),nn.ReLU(),nn.Flatten(),nn.Linear(16*2*2,md))
        s.qp=nn.Linear(fd,md); s.vp=nn.Linear(md+1+na,na); s.sc=md**0.5
    def forward(s,cf,md,ma,mr):
        B,M=ma.shape
        if M==0: return torch.zeros(B,5,device=cf.device)
        k=s.de(md.reshape(B*M,1,64,64)).reshape(B,M,s.md); q=s.qp(cf).unsqueeze(1)
        a=F.softmax(torch.bmm(q,k.transpose(1,2))/s.sc,dim=-1)
        v=torch.cat([k,mr.unsqueeze(-1),F.one_hot(ma.clamp(0,4),5).float()],dim=-1)
        return s.vp(torch.bmm(a,v).squeeze(1))

class ForgeNet(nn.Module):
    def __init__(s,ic=26,g=64):
        super().__init__(); s.g=g
        s.c1=nn.Conv2d(ic,32,3,padding=1); s.c2=nn.Conv2d(32,64,3,padding=1)
        s.c3=nn.Conv2d(64,128,3,padding=1); s.c4=nn.Conv2d(128,256,3,padding=1)
        s.at=CBAM(256); s.ar=nn.Conv2d(256,64,1); s.ap=nn.MaxPool2d(4,4)
        s.af=nn.Linear(64*16*16,256); s.ah=nn.Linear(256,5); s.dr=nn.Dropout(0.15)
        s.cc1=nn.Conv2d(256,128,3,padding=1); s.cc2=nn.Conv2d(128,64,3,padding=1)
        s.cc3=nn.Conv2d(64,32,1); s.cc4=nn.Conv2d(32,1,1)
        s.gp=nn.AdaptiveAvgPool2d(1); s.gf=nn.Linear(256,64); s.aea=ActionEffectAttention()
    def forward(s,x,md=None,ma=None,mr=None):
        x=F.relu(s.c1(x)); x=F.relu(s.c2(x)); x=F.relu(s.c3(x)); f=F.relu(s.c4(x))
        f=s.at(f); af=F.relu(s.ar(f)); af=s.ap(af).reshape(f.size(0),-1)
        al=s.ah(s.dr(F.relu(s.af(af))))
        cf=F.relu(s.cc1(f)); cf=F.relu(s.cc2(cf)); cf=F.relu(s.cc3(cf)); cl=s.cc4(cf).reshape(f.size(0),-1)
        if md is not None and ma is not None: al=al+s.aea(s.gf(s.gp(f).reshape(f.size(0),-1)),md,ma,mr)
        return torch.cat([al,cl],1)

def fast_objects(fr,bg):
    o=[]
    for c in range(16):
        if c==bg: continue
        m=(fr==c); n=int(np.sum(m))
        if n<4 or n>3000: continue
        ys,xs=np.where(m); o.append((c,float(np.mean(xs)),float(np.mean(ys)),n))
    return o

# ==================== NEXUS ====================
def normalise_frame(fr):
    c=np.bincount(fr.flatten(),minlength=16); o=np.argsort(-c,kind='stable')
    r=np.zeros(16,dtype=np.int64)
    for i,v in enumerate(o): r[v]=i
    return r[fr]

def frame_to_onehot(fr,p=64):
    cv=np.zeros((p,p),dtype=np.int64); H,W=fr.shape; cv[:H,:W]=fr
    oh=torch.zeros(16,p,p,dtype=torch.float32); oh.scatter_(0,torch.from_numpy(cv).unsqueeze(0),1.0); return oh

class VQ(nn.Module):
    def __init__(s,K=512,D=64,b=0.25):
        super().__init__(); s.K=K; s.D=D; s.b=b; s.emb=nn.Embedding(K,D); nn.init.uniform_(s.emb.weight,-1/K,1/K)
    def forward(s,ze):
        B,D,H,W=ze.shape; zf=ze.permute(0,2,3,1).reshape(-1,D)
        d=zf.pow(2).sum(1,keepdim=True)-2*zf@s.emb.weight.t()+s.emb.weight.pow(2).sum(1)
        ix=d.argmin(1); zq=s.emb(ix).reshape(B,H,W,D).permute(0,3,1,2)
        l=F.mse_loss(ze,zq.detach())+s.b*F.mse_loss(ze.detach(),zq)
        return ze+(zq-ze).detach(),ix.reshape(B,H,W),l

class VQVAE(nn.Module):
    def __init__(s,K=512,D=64):
        super().__init__()
        s.enc=nn.Sequential(nn.Conv2d(16,32,4,2,1),nn.ReLU(),nn.Conv2d(32,64,4,2,1),nn.ReLU(),nn.Conv2d(64,128,4,2,1),nn.ReLU(),nn.Conv2d(128,D,1))
        s.vq=VQ(K,D)
        s.dec=nn.Sequential(nn.ConvTranspose2d(D,128,4,2,1),nn.ReLU(),nn.ConvTranspose2d(128,64,4,2,1),nn.ReLU(),nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(),nn.Conv2d(32,16,1))
    def encode(s,x): ze=s.enc(x); zq,ix,_=s.vq(ze); return zq,ix

class NWM(nn.Module):
    def __init__(s,K=512,D=64,na=7,nh=4,nl=2,sl=64):
        super().__init__(); s.K=K; s.sl=sl
        s.te=nn.Embedding(K,D); s.ae=nn.Embedding(na+1,D); s.pe=nn.Embedding(sl+1,D)
        s.tf=nn.TransformerEncoder(nn.TransformerEncoderLayer(D,nh,D*4,0.0,batch_first=True),nl)
        s.ns=nn.Linear(D,K); s.rh=nn.Linear(D,1); s.wh=nn.Linear(D,1)
    def forward(s,ix,act):
        p=torch.arange(s.sl+1,device=ix.device)
        t=torch.cat([s.ae(act).unsqueeze(1),s.te(ix)],1)+s.pe(p).unsqueeze(0)
        o=s.tf(t); return s.ns(o[:,1:,:]),s.rh(o[:,0,:]).squeeze(-1),s.wh(o[:,0,:]).squeeze(-1)
    @torch.no_grad()
    def imagine(s,ix,act,dev):
        s.eval(); l,r,w=s.forward(ix.unsqueeze(0).to(dev),torch.tensor([act],device=dev))
        return l.squeeze(0).argmax(-1),r.item(),torch.sigmoid(w).item()

class PolicyHead(nn.Module):
    def __init__(s,K=512,D=64,na=7,sl=64):
        super().__init__(); s.te=nn.Embedding(K,D); s.pe=nn.Embedding(sl,D)
        s.pool=nn.AdaptiveAvgPool1d(1); s.hd=nn.Sequential(nn.Linear(D,D),nn.ReLU(),nn.Dropout(0.1),nn.Linear(D,na)); s.sl=sl
    def forward(s,ix):
        p=torch.arange(s.sl,device=ix.device); x=s.te(ix)+s.pe(p).unsqueeze(0)
        return s.hd(s.pool(x.permute(0,2,1)).squeeze(-1))

class CEMP:
    def __init__(s,wm,H=8,N=48,ne=6,ni=2,g=5.0):
        s.wm=wm; s.H=H; s.N=N; s.ne=ne; s.ni=ni; s.g=g
    def plan(s,ix,av,dev,bud=0.3):
        if not av: return 1
        na=len(av); pr=np.ones(na)/na; best=av[0]; t0=time.time()
        for _ in range(s.ni):
            if time.time()-t0>bud: break
            sq=np.random.choice(na,size=(s.N,s.H),p=pr); sc=np.zeros(s.N)
            for i in range(s.N):
                cur,tot=ix.clone(),0.0
                for h in range(s.H):
                    cur,rw,wn=s.wm.imagine(cur,av[sq[i,h]],dev); tot+=rw
                    if wn>0.9: tot+=s.g; break
                sc[i]=tot+s.g*wn
            el=sq[np.argsort(sc)[-s.ne:]]; cn=np.bincount(el[:,0],minlength=na).astype(float)
            if cn.sum()>0: pr=cn/cn.sum()
            best=av[int(el[-1,0])]
        return best

# ==================== SYMBOLIC (lightweight) ====================
def _ccl(mask):
    H,W=mask.shape; lb=np.zeros((H,W),dtype=np.int32); n=0
    for r in range(H):
        for c in range(W):
            if mask[r,c] and lb[r,c]==0:
                n+=1; q=deque([(r,c)]); lb[r,c]=n
                while q:
                    y,x=q.popleft()
                    for dy in(-1,0,1):
                        for dx in(-1,0,1):
                            ny,nx=y+dy,x+dx
                            if 0<=ny<H and 0<=nx<W and mask[ny,nx] and lb[ny,nx]==0: lb[ny,nx]=n; q.append((ny,nx))
    return lb,n

def extract_entities(fr):
    bg=int(np.bincount(fr.flatten(),minlength=16).argmax()); ents=[]; eid=0
    for col in range(16):
        if col==bg: continue
        m=(fr==col)
        if not m.any(): continue
        lb,n=_ccl(m)
        for i in range(n):
            comp=(lb==i+1); coords=np.argwhere(comp)
            if len(coords)==0: continue
            cy,cx=np.round(coords.mean(axis=0)).astype(int)
            rmin,cmin=coords.min(axis=0); rmax,cmax=coords.max(axis=0)
            ents.append({'id':eid,'color':col,'pixels':int(comp.sum()),'centroid':(int(cy),int(cx)),'bbox':(int(rmin),int(cmin),int(rmax),int(cmax))}); eid+=1
    return ents,bg

def detect_xforms(eb,ea):
    evs=[]; used=set()
    for e0 in eb:
        bs,bds=None,float('inf'); br,bdr=None,float('inf'); bz=None
        for i,e1 in enumerate(ea):
            if i in used: continue
            d=(e0['centroid'][0]-e1['centroid'][0])**2+(e0['centroid'][1]-e1['centroid'][1])**2
            if e0['color']==e1['color'] and e0['pixels']==e1['pixels']:
                if d<bds: bds=d; bs=i
            elif e0['pixels']==e1['pixels'] and d<bdr: bdr=d; br=i
            elif e0['color']==e1['color'] and bz is None: bz=i
        if bs is not None: used.add(bs); (bds>0 and evs.append({'type':'MOVE','color':e0['color'],'pixels':e0['pixels']}))
        elif br is not None: used.add(br); evs.append({'type':'RECOLOR','color':e0['color'],'new_color':ea[br]['color'],'pixels':e0['pixels']})
        elif bz is not None: used.add(bz); evs.append({'type':'GROW' if ea[bz]['pixels']>e0['pixels'] else 'SHRINK','color':e0['color'],'pixels':e0['pixels']})
        else: evs.append({'type':'DISAPPEAR','color':e0['color'],'pixels':e0['pixels']})
    for i,e1 in enumerate(ea):
        if i not in used: evs.append({'type':'APPEAR','color':e1['color'],'pixels':e1['pixels']})
    return evs

def spatial_ctx(ents):
    ctx={e['id']:{'ov':[],'adj':[]} for e in ents}
    for i,a in enumerate(ents):
        for j,b in enumerate(ents):
            if i>=j: continue
            ba,bb=a['bbox'],b['bbox']
            if not(ba[2]<bb[0] or bb[2]<ba[0] or ba[3]<bb[1] or bb[3]<ba[1]):
                ctx[a['id']]['ov'].append(b['id']); ctx[b['id']]['ov'].append(a['id'])
            else:
                e=(ba[0]-2,ba[1]-2,ba[2]+2,ba[3]+2)
                if not(e[2]<bb[0] or bb[2]<e[0] or e[3]<bb[1] or bb[3]<e[1]):
                    ctx[a['id']]['adj'].append(b['id']); ctx[b['id']]['adj'].append(a['id'])
    return ctx

class RH:
    def __init__(s,trig,eff): s.trig=trig; s.eff=eff; s.conf=1; s.fail=0; s.ok=False
    def sig(s): return (s.trig.get('action'),tuple(sorted(s.trig.get('nc',[]))),s.eff.get('type'),s.eff.get('color'))
    def validate(s,evs):
        for e in evs:
            if e['type']==s.eff['type'] and e['color']==s.eff['color']:
                s.conf+=1
                if s.conf>=3: s.ok=True
                return True
        s.fail+=1; return False

class TGraph:
    def __init__(s): s.ae={}; s.hyps={}; s.gs=set(); s.ga=set(); s.tried=set(); s._vq=deque(); s._afs={}
    def _nc(s,ctx,eid,bi):
        info=ctx.get(eid,{}); c=set()
        for o in info.get('ov',[])+info.get('adj',[]):
            e=bi.get(o)
            if e: c.add(e['color'])
        return sorted(c)
    def rao(s,a,mc):
        if mc: s._afs[a]=0
        else: s._afs[a]=s._afs.get(a,0)+1
    def aib(s,a,th=2): return s._afs.get(a,0)>=th
    def observe(s,aid,evs,cb,eb,ca,ea,la):
        s.tried.add(aid)
        if not evs: s.ae[aid]=[]; return
        s.ae[aid]=evs; bi={e['id']:e for e in eb}
        for ev in evs:
            nc=sorted(set(c for e in eb for c in s._nc(cb,e['id'],bi)))
            h=RH({'action':aid,'nc':nc},ev); sg=h.sig()
            if sg not in s.hyps: s.hyps[sg]=h; s._vq.append(sg)
        if la:
            for ev in evs: s.gs.add(f"{ev['type']}_{ev['color']}"); s.ga.add(aid)
        for h in s.hyps.values():
            if not h.ok and h.trig.get('action')==aid: h.validate(evs)
    def observe_click(s,x,y,evs,eb,cb,la):
        s.tried.add(6); s.ae[6]=evs; cc=None
        for e in eb:
            r0,c0,r1,c1=e['bbox']
            if r0<=y<=r1 and c0<=x<=c1: cc=e['color']; break
        bi={e['id']:e for e in eb}; nc=[]
        for e in eb:
            if e['color']==cc: nc=s._nc(cb,e['id'],bi); break
        for ev in evs:
            h=RH({'action':6,'clicked_color':cc,'nc':nc},ev); sg=h.sig()
            if sg not in s.hyps: s.hyps[sg]=h; s._vq.append(sg)
        if la:
            for ev in evs: s.gs.add(f"{ev['type']}_{ev['color']}"); s.ga.add(6)
        for h in s.hyps.values():
            if not h.ok and h.trig.get('action')==6: h.validate(evs)
    def _es(s,ev):
        sg=f"{ev['type']}_{ev['color']}"
        return {'DISAPPEAR':4,'RECOLOR':3,'GROW':2,'SHRINK':2,'MOVE':1,'APPEAR':1}.get(ev['type'],0)+(20 if sg in s.gs else 0)
    def best_action(s,aids,ents,ctx,banned=None):
        si=[a for a in aids if 1<=a<=5]; hc=6 in aids
        def _u(a): return a!=banned and not s.aib(a)
        ut=[a for a in si if a not in s.tried and _u(a)]
        if ut: return ut[0],15,'explore'
        if hc and 6 not in s.tried and _u(6): return 6,15,'explore'
        while s._vq:
            sg=s._vq[0]
            if sg not in s.hyps: s._vq.popleft(); continue
            h=s.hyps[sg]
            if h.ok or h.fail>=3: s._vq.popleft(); continue
            a=h.trig.get('action',1)
            if a in si and _u(a): return a,12,'validate'
            s._vq.popleft()
        ba,bs=None,-1
        for a in si+([6] if hc else []):
            if not _u(a): continue
            sc=sum(s._es(ev) for ev in s.ae.get(a,[]))
            for h in s.hyps.values():
                if h.trig.get('action')==a and h.ok: sc+=25
            if a in s.ga: sc+=50
            if sc>bs: bs=sc; ba=a
        if ba is not None and bs>0: return ba,bs,'plan'
        ca=[a for a in si if _u(a)] or si
        return (ca[0] if ca else 1),0,'fallback'
    def best_click(s,ents,ctx):
        tc={}
        for h in s.hyps.values():
            if h.trig.get('action')==6 and (h.ok or h.conf>=2):
                cc=h.trig.get('clicked_color')
                if cc is not None: tc[cc]=tc.get(cc,0)+s._es(h.eff)*h.conf
        if not tc: return None
        be,bs=None,-1
        for e in ents:
            sc=tc.get(e['color'],0)
            if sc>bs: bs=sc; be=e
        if be is None: return None
        return int(be['centroid'][1]),int(be['centroid'][0])

# ==================== AGENT ====================
class MyAgent(Agent):
    MAX_ACTIONS=float('inf'); _MAX_FRAMES=10
    use_tg=False  # Toggle transformation graph

    def __init__(s,*a,**kw):
        super().__init__(*a,**kw)
        seed=abs(hash(s.game_id))%(2**32-1); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        s.t0=time.time(); s.dev=torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        s.G=64; s.IN=26; s.net=None; s.opt=None
        s.buf=deque(maxlen=50000); s.bh=set(); s.bsz=64; s.tf=10
        s.pt=None; s.pai=None; s.pr=None; s.ph=None
        s.cl=-1; s.fh=deque(maxlen=6); s.la=0
        s.al=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5]
        s._wd=False; s._bg=0; s._wm=None
        s._ad=deque(maxlen=256); s._aa=deque(maxlen=256); s._ar=deque(maxlen=256)
        s._ch=None; s._up=0; s._ua=False; s._eps=0.15; s._po=None
        s._bfs=None; s._bs=None; s._bi=0; s._bt=False
        s._gr=TGraph() if s.use_tg else None; s._pe=None; s._ba=None; s._pc=None; s._pxy=None; s._plc=-1
        s._ra=deque(maxlen=20); s._ea=None; s._er=0; s._cd={}
        # NEXUS
        s._nr=False; s._nc=None
        try:
            s._vq=VQVAE(512,64).to(s.dev); s._nw=NWM(512,64,7).to(s.dev)
            s._ph2=PolicyHead(512,64,7).to(s.dev); s._pl=CEMP(s._nw)
            for wp in ['/kaggle/input/datasets/anhkhoaonnguyn/nexus-training-timed-out/results (2)/nexus_weights.pt','nexus_weights.pt']:
                if os.path.exists(wp):
                    try:
                        ck=torch.load(wp,map_location=s.dev,weights_only=True)
                        if 'vqvae' in ck: s._vq.load_state_dict(ck['vqvae'],strict=False)
                        if 'wm' in ck: s._nw.load_state_dict(ck['wm'],strict=False)
                        if 'policy' in ck: s._ph2.load_state_dict(ck['policy'],strict=False)
                        s._nr=ck.get('wm_ready',True); break
                    except: pass
            s._vq.eval(); s._nw.eval(); s._ph2.eval()
        except: s._nr=False

    # --- Helpers ---
    def append_frame(s,f):
        s.frames.append(f)
        if len(s.frames)>s._MAX_FRAMES: s.frames=s.frames[-s._MAX_FRAMES:]
        if f.guid: s.guid=f.guid
        if hasattr(s,"recorder") and not s.is_playback:
            import json; s.recorder.record(json.loads(f.model_dump_json()))
    def _lvl(s,f): return getattr(f,'score',None) or f.levels_completed
    def _raw(s,fd): arr=np.array(fd.frame,dtype=np.int64); return arr[-1] if arr.ndim==3 else arr
    def _ib(s): src,cls=find_game_source_and_class(s.game_id,s.arc_env); return BFSSolver(src,cls,5,180) if src else None
    def _sb(s,li):
        if not s._bfs: return None
        el=time.time()-s.t0; rm=max(60,8*3600-600-el)
        s._bfs.bfs_timeout=int(max(30,min(rm*0.3,600) if li==0 else min(rm*0.1,300)))
        sol=s._bfs.solve_level(li,prev_solution=s._bfs.solutions.get(li-1) if li>0 else None)
        if sol: s._bs=sol; s._bi=0
        return sol
    def _ne(s,raw):
        if s._nc is not None: return s._nc
        try:
            with torch.no_grad():
                oh=frame_to_onehot(normalise_frame(raw)).unsqueeze(0).to(s.dev); _,ix=s._vq.encode(oh)
            s._nc=ix.reshape(1,-1); return s._nc
        except: return None
    def _ret(s,sel,raw,ch,aid):
        """Common commit: set state, record, return action."""
        s.pt=s._tensor_from_raw(raw); s.pai=aid; s.pr=raw.copy(); s.ph=ch; s.la+=1; s._ra.append(aid); return sel
    def _tensor_from_raw(s,raw):
        oh=torch.zeros(16,64,64,dtype=torch.float32); oh.scatter_(0,torch.from_numpy(raw).unsqueeze(0),1)
        cnt=np.bincount(raw.flatten(),minlength=16); s._bg=int(cnt.argmax()); mx=max(cnt.max(),1)
        bg_m=(raw==s._bg).astype(np.float32); rar=np.zeros((64,64),np.float32)
        for c in range(16):
            if cnt[c]>0: rar[raw==c]=1.0-cnt[c]/mx
        pad=np.pad(raw,1,mode='edge')
        edge=((raw!=pad[:-2,1:-1])|(raw!=pad[2:,1:-1])|(raw!=pad[1:-1,:-2])|(raw!=pad[1:-1,2:])).astype(np.float32)
        rp=np.linspace(0,1,64,dtype=np.float32).reshape(64,1).repeat(64,1)
        cp=np.linspace(0,1,64,dtype=np.float32).reshape(1,64).repeat(64,0)
        aug=torch.from_numpy(np.stack([bg_m,rar,edge,rp,cp]))
        d1=torch.zeros(3,64,64,dtype=torch.float32)
        for i,prev in enumerate(reversed(list(s.fh))):
            if i>=3: break
            d1[i]=torch.from_numpy((raw!=prev).astype(np.float32))
        d2=torch.zeros(2,64,64,dtype=torch.float32); h=list(s.fh)
        if len(h)>=2: d2[0]=torch.from_numpy((h[-1]!=h[-2]).astype(np.float32))
        if len(h)>=4: d2[1]=torch.from_numpy((h[-2]!=h[-4]).astype(np.float32))
        s.fh.append(raw.copy())
        return torch.cat([oh,aug,d1,d2],0).to(s.dev)
    def _detect_template(s,fr):
        mask=torch.ones(4096,dtype=torch.float32)
        ca=np.sum(fr!=s._bg,axis=0)
        for c in range(20,44):
            if ca[c]<=2 and np.sum(ca[:c]>0)>=5 and np.sum(ca[c+1:]>0)>=5:
                for y in range(64):
                    for x in range(c+1): mask[y*64+x]=0.05
                return mask
        ra=np.sum(fr!=s._bg,axis=1)
        for r in range(20,44):
            if ra[r]<=2 and np.sum(ra[:r]>0)>=5 and np.sum(ra[r+1:]>0)>=5:
                for y in range(r+1):
                    for x in range(64): mask[y*64+x]=0.05
                return mask
        return mask
    def _reward(s,p,c,ch):
        m=np.ones((64,64),dtype=bool); m[:2]=False; m[62:]=False; d=(p!=c)&m; r=0.0
        if np.any(d): r+=1.5
        else: r-=0.1
        if np.any(p!=c): r+=0.5
        co=fast_objects(c,s._bg)
        if s._po and co:
            mv=sum(1 for a in co for b in s._po if a[0]==b[0] and 2<abs(a[1]-b[1])+abs(a[2]-b[2])<20)
            if mv>0: r+=0.3*min(mv,3)
        s._po=co; return r
    def _sample(s,logits,avail=None,temp=1.0):
        al=logits[:5].clone(); cl=logits[5:5+4096].clone()
        if avail:
            m=torch.full_like(al,float('-inf')); a6=False
            for a in avail:
                aid=a.value if hasattr(a,'value') else int(a)
                if 1<=aid<=5: m[aid-1]=0.0
                elif aid==6: a6=True
            al=al+m
            if not a6: cl=cl+torch.full_like(cl,float('-inf'))
        if s._wm is not None: cl=cl+torch.log(s._wm.to(s.dev).clamp(min=0.01))
        ap=torch.sigmoid(al/temp); cp=torch.sigmoid(cl/temp)/(s.G*s.G)
        allp=torch.cat([ap,cp]); sm=allp.sum()
        if sm<1e-8: allp=torch.ones_like(allp)/len(allp)
        else: allp=allp/sm
        idx=np.random.choice(len(allp),p=allp.cpu().numpy())
        if idx<5: return idx,None
        ci=idx-5; return 5,(ci//s.G,ci%s.G)
    def _heuristic(s,fr,avail,step):
        av=set(int(a.value) if hasattr(a,'value') else int(a) for a in avail)
        for d in [1,2,3,4]:
            if d in av and step<4: return d-1,None
        if 6 in av:
            cnt=np.bincount(fr.flatten(),minlength=16); tg=[]
            for c in range(16):
                if c==s._bg or cnt[c]==0 or cnt[c]>2000: continue
                ys,xs=np.where(fr==c)
                if len(ys)>=2: tg.append((int(np.median(xs)),int(np.median(ys)),len(ys)))
            tg.sort(key=lambda t:t[2]); pi=step-4
            if 0<=pi<len(tg): return 5,(tg[pi][1],tg[pi][0])
        if 5 in av: return 4,None
        ch=[a for a in av if 1<=a<=5]
        if ch: return random.choice(ch)-1,None
        return 0,None
    def _ftt(s,fr):
        oh=torch.zeros(16,64,64,dtype=torch.float32); oh.scatter_(0,torch.from_numpy(fr).unsqueeze(0),1)
        cnt=np.bincount(fr.flatten(),minlength=16); bg=int(cnt.argmax()); mx=max(cnt.max(),1)
        bm=(fr==bg).astype(np.float32); ra=np.zeros((64,64),np.float32)
        for c in range(16):
            if cnt[c]>0: ra[fr==c]=1.0-cnt[c]/mx
        pad=np.pad(fr,1,mode='edge')
        edge=((fr!=pad[:-2,1:-1])|(fr!=pad[2:,1:-1])|(fr!=pad[1:-1,:-2])|(fr!=pad[1:-1,2:])).astype(np.float32)
        rp=np.linspace(0,1,64,dtype=np.float32).reshape(64,1).repeat(64,1)
        cp=np.linspace(0,1,64,dtype=np.float32).reshape(1,64).repeat(64,0)
        return torch.cat([oh,torch.from_numpy(np.stack([bm,ra,edge,rp,cp])),torch.zeros(5,64,64)],0)
    def _train(s):
        if len(s.buf)<s.bsz: return
        ix=np.random.choice(len(s.buf),s.bsz,replace=False); b=[s.buf[i] for i in ix]
        st=torch.stack([s._ftt(e['s']).to(s.dev) for e in b])
        ac=torch.tensor([e['a'] for e in b],dtype=torch.long,device=s.dev)
        rw=torch.sigmoid(torch.tensor([e['r'] for e in b],dtype=torch.float32,device=s.dev))
        s.opt.zero_grad(); lg=s.net(st); sel=lg.gather(1,ac.clamp(0,lg.size(1)-1).unsqueeze(1)).squeeze(1)
        loss=F.binary_cross_entropy_with_logits(sel,rw); p=torch.sigmoid(lg)
        loss=loss-0.0001*p[:,:5].mean()-0.00001*p[:,5:].mean()
        loss.backward(); s.opt.step()
    def _gaem(s):
        if len(s._ad)<2: return None,None,None
        M=len(s._ad); d=torch.zeros(1,M,1,64,64,device=s.dev)
        a=torch.zeros(1,M,dtype=torch.long,device=s.dev); r=torch.zeros(1,M,device=s.dev)
        for i,(dd,aa,rr) in enumerate(zip(s._ad,s._aa,s._ar)):
            d[0,i,0]=torch.from_numpy(dd.astype(np.float32)); a[0,i]=min(aa,4); r[0,i]=rr
        return d,a,r
    def _dosc(s,si):
        if len(s._ra)<3: return False
        a=list(s._ra)
        if len(set(a[-3:]))==1 and a[-1] is not None: return True
        if len(a)>=4:
            l=a[-4:]
            if l[0]==l[2] and l[1]==l[3] and l[0]!=l[1] and l[0] is not None: return True
        return False
    def _esc(s,si):
        a=list(s._ra); osc=set(x for x in a[-6:] if x is not None)
        ca=[x for x in si if x not in osc and not(s._gr and s._gr.aib(x))]
        if not ca: ca=[x for x in si if not(s._gr and s._gr.aib(x))]
        if not ca: ca=si
        if ca: s._ea=random.choice(ca); s._er=random.randint(3,6)
        for o in osc:
            if o is not None and 1<=o<=5: s._cd[o]=max(s._cd.get(o,0),8)

    def is_done(s,frames,lf):
        try: return lf.state is GameState.WIN or (time.time()-s.t0)>=8*3600-300
        except: return True

    def choose_action(s,frames,lf):
        try:
            lvl=s._lvl(lf)
            # === LEVEL CHANGE ===
            if lvl!=s.cl:
                if not s._bt:
                    s._bt=True; bfs=s._ib()
                    if bfs and bfs.load(): s._bfs=bfs
                s._bs=None; s._bi=0
                if s._bfs: s._sb(lvl)
                s.buf.clear(); s.bh.clear()
                s.net=ForgeNet(s.IN,s.G).to(s.dev)
                for wp in ['/kaggle/input/forge-pretrained-weights/pretrained_weights.pt','pretrained_weights.pt']:
                    try:
                        if os.path.exists(wp):
                            st=torch.load(wp,map_location=s.dev,weights_only=True); ms=s.net.state_dict()
                            for k in list(st.keys()):
                                if k in ms and st[k].shape==ms[k].shape: ms[k]=st[k]
                            s.net.load_state_dict(ms); break
                    except: pass
                s.opt=optim.Adam(s.net.parameters(),lr=3e-4)
                s.pt=None; s.pai=None; s.pr=None; s.ph=None; s.cl=lvl; s.fh.clear(); s.la=0
                s._wd=False; s._wm=None; s._eps=0.15
                s._ad.clear(); s._aa.clear(); s._ar.clear(); s._po=None; s._ch=None; s._up=0
                if s.use_tg: s._gr=TGraph()
                s._pe=None; s._ba=None; s._pc=None; s._pxy=None; s._plc=lvl-1
                s._ra.clear(); s._ea=None; s._er=0; s._cd={}
            # === RESET ===
            if lf.state in [GameState.NOT_PLAYED,GameState.GAME_OVER]:
                s.pt=None; s.pai=None; s.pr=None; s.ph=None
                s._ra.clear(); s._ea=None; s._er=0; s._cd={}; s._up=0; s._ba=None
                a=GameAction.RESET; a.reasoning="reset"; return a

            raw=s._raw(lf); ch=hashlib.md5(raw.tobytes()).hexdigest()[:16]
            avail=getattr(lf,'available_actions',None) or []
            s._ua=any((a.value if hasattr(a,'value') else int(a))==7 for a in avail)
            aids=[a.value if hasattr(a,'value') else int(a) for a in avail]
            si=[i for i in aids if 1<=i<=5]
            s._nc=None  # reset nexus cache per step

            # === BFS ===
            if s._bs and s._bi<len(s._bs):
                aid,data=s._bs[s._bi]; s._bi+=1
                sel=GameAction.from_id(aid)
                if data: sel.set_data(data)
                sel.reasoning=f"bfs:{s._bi}/{len(s._bs)}"
                s.fh.append(raw.copy()); s.pr=raw.copy(); s.la+=1; return sel

            # === FEEDBACK ===
            if s.pt is not None and s.pai is not None and s.pr is not None:
                mr=np.ones((64,64),dtype=bool); mr[:2]=False; mr[62:]=False
                dm=(s.pr!=raw)&mr; chg=np.any(dm)
                eh=hashlib.md5(s.pr.tobytes()[:1000]+str(s.pai).encode()).hexdigest()[:16]
                if eh not in s.bh:
                    r=s._reward(s.pr,raw,ch); s.buf.append({"s":s.pr.copy(),"a":s.pai,"r":r}); s.bh.add(eh)
                    if chg: s._ad.append(dm); s._aa.append(min(s.pai if isinstance(s.pai,int) else 1,4)); s._ar.append(r)
                if s.use_tg and s._gr:
                    ce,_=extract_entities(raw); cc=spatial_ctx(ce)
                    la=(lvl>s._plc)
                    if la: s._plc=lvl
                    pi=s.pai if isinstance(s.pai,int) else 1
                    if s._pe is not None:
                        evs=detect_xforms(s._pe,ce)
                        if pi==6 and s._pxy: s._gr.observe_click(s._pxy[0],s._pxy[1],evs,s._pe,s._pc or {},la)
                        else: s._gr.observe(pi,evs,s._pc or {},s._pe,cc,ce,la)
                        es=tuple(sorted((e['color'],e['centroid'][0],e['centroid'][1]) for e in ce))
                        ps=tuple(sorted((e['color'],e['centroid'][0],e['centroid'][1]) for e in s._pe)) if s._pe else ()
                        s._gr.rao(pi,es!=ps)
                    s._pe=ce; s._pc=cc
                s._pxy=None
                if chg: s._ba=None; s._ch=ch; s._up=max(0,s._up-1)
                else: s._ba=(s.pai if isinstance(s.pai,int) else 1); s._up+=2

            if s._wm is None: s._wm=s._detect_template(raw)
            for k in list(s._cd.keys()):
                s._cd[k]-=1
                if s._cd[k]<=0: del s._cd[k]
            if s._er<=0 and s._dosc(si): s._esc(si)
            if s._er>0 and s._ea is not None:
                esc=s._ea
                if s._gr and s._gr.aib(esc):
                    ca=[a for a in si if a!=esc and not s._gr.aib(a)]
                    if ca: s._ea=random.choice(ca); esc=s._ea
                    else: s._er=0
                if s._er>0 and 1<=esc<=5:
                    s._er-=1; sel=s.al[esc-1]; sel.reasoning=f"esc:a{esc}"
                    return s._ret(sel,raw,ch,esc)
            if s._ua and s._up>=15:
                s._up=0; a=GameAction.ACTION7; a.reasoning="undo"
                s.pt=None; s.pai=7; s.pr=raw.copy(); s.ph=ch; s.la+=1; return a
            if s._up>=3:
                s._up=0
                if s.pai is not None and 1<=s.pai<=5: s._cd[s.pai]=6
            bs=set(s._cd.keys())
            if s._ba is not None: bs.add(s._ba)

            # === SYMBOLIC ===
            if s.use_tg and s._gr:
                ce,_=extract_entities(raw); cc=spatial_ctx(ce)
                sa,ss,sp=s._gr.best_action(si,ce,cc,banned=s._ba)
                use=(sp in ("explore","validate") or (sp=="plan" and ss>=5))
                if use and sa is not None and sa not in bs:
                    sel=s.al[sa-1]; sel.reasoning=f"sym:{sp}:a{sa}"
                    return s._ret(sel,raw,ch,sa)
                if 6 in aids and 6 not in bs:
                    ct=s._gr.best_click(ce,cc)
                    if ct:
                        x,y=ct; sel=GameAction.ACTION6; sel.set_data({"x":x,"y":y}); sel.reasoning=f"sym:c({x},{y})"
                        s._pxy=(x,y); return s._ret(sel,raw,ch,6)

            # === NEXUS POLICY ===
            if s._nr:
                ix=s._ne(raw)
                if ix is not None:
                    try:
                        with torch.no_grad(): pl=s._ph2(ix).squeeze(0)
                        for a in bs:
                            if 1<=a<=5: pl[a-1]=float('-inf')
                        pr=torch.softmax(pl,0); tv,ti=pr.max(0)
                        if tv.item()>0.35:
                            aid=ti.item()+1
                            if aid<=5 and aid in aids:
                                sel=s.al[aid-1]; sel.reasoning=f"nxp:a{aid}"
                                return s._ret(sel,raw,ch,aid)
                    except: pass

            # === NEXUS CEM ===
            if s._nr:
                ix=s._ne(raw)
                if ix is not None:
                    try:
                        su=[a for a in si if a not in bs]
                        if su:
                            el=time.time()-s.t0; bud=min(0.3,max(60,8*3600-600-el)/20000)
                            aid=s._pl.plan(ix.squeeze(0),su,s.dev,bud)
                            if aid in aids and aid not in bs:
                                sel=s.al[aid-1]; sel.reasoning=f"nxc:a{aid}"
                                return s._ret(sel,raw,ch,aid)
                    except: pass

            # === CNN ===
            tensor=s._tensor_from_raw(raw)
            if not s._wd:
                if s.la<10: aidx,coords=s._heuristic(raw,avail,s.la)
                else: s._wd=True; [s._train() for _ in range(min(5,len(s.buf)//s.bsz))]
            if s._wd:
                if random.random()<s._eps: aidx,coords=s._sample(torch.zeros(4101,device=s.dev),avail,temp=2.0)
                else:
                    with torch.no_grad():
                        mem=s._gaem()
                        lg=s.net(tensor.unsqueeze(0),*mem).squeeze(0) if mem[0] is not None else s.net(tensor.unsqueeze(0)).squeeze(0)
                    aidx,coords=s._sample(lg,avail,temp=0.5)
                    if aidx<5 and (aidx+1) in bs: aidx,coords=s._sample(lg,avail,temp=2.0)
                s._eps=max(s._eps_min,s._eps*s._eps_decay)
            elif s.la>=10: s._wd=True; aidx,coords=0,None
            if aidx<5: sel=s.al[aidx]; sel.reasoning=f"cnn:a{aidx+1}"
            else:
                if coords is None:
                    cnt=np.bincount(raw.flatten(),minlength=16); bg=int(cnt.argmax())
                    for c in range(16):
                        if c!=bg and cnt[c]>2: ys,xs=np.where(raw==c); coords=(int(np.median(ys)),int(np.median(xs))); break
                    if coords is None: coords=(32,32)
                sel=GameAction.ACTION6; y,x=coords; sel.set_data({"x":int(x),"y":int(y)}); sel.reasoning=f"cnn:c({x},{y})"
            s.pt=tensor
            if aidx<5: s.pai=aidx+1; s._pxy=None
            else: s.pai=6; y,x=coords; s._pxy=(int(x),int(y))
            s.pr=raw.copy(); s.ph=ch; s.la+=1; s._ra.append(s.pai)
            if s.action_counter%s.tf==0 and s._wd: s._train()
            return sel
        except Exception as e:
            traceback.print_exc(); a=random.choice(s.al); a.reasoning=f"err:{e}"; return a