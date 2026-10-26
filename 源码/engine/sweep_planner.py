"""Pack an enclosed pocket, then cross cells vacated by the measured tail.

Every search state contains the whole moving body, growth debt and consumed
items. A diamond on an escape route is a real five-step tail stall, even when
the controller's primary objective is survival.
"""
from functools import lru_cache
from collections import deque
from causal_planner import DIRS,OPP,inside


def pocket(head,body,hard,limit):
    solid=set(body[1:])|set(hard)
    seen={head};q=deque([head])
    while q and len(seen)<=limit:
        c=q.popleft()
        for d in DIRS:
            n=(c[0]+d[0],c[1]+d[1])
            if inside(n) and n not in solid and n not in seen:
                seen.add(n);q.append(n)
    return frozenset(seen) if len(seen)<=limit else None


def walk(body,path,growth=0,food=None):
    """Simulate all actually crossed items once, including incidental food."""
    body=tuple(map(tuple,body));eaten=set();food=food or {}
    if not path or not body or tuple(path[0])!=body[0]:
        return None
    for c in path[1:]:
        if c in (body if growth else body[:-1]):
            return None
        body=(c,)+body if growth else (c,)+body[:-1]
        growth=max(0,growth-1)
        if c in food and c not in eaten:
            growth+=food[c][0];eaten.add(c)
    return body,growth,frozenset(eaten)


def plan_sweep(body,heading,hard,region,food=None,growth=0,segment=1,initial=0):
    food=tuple(sorted((tuple(c),int(v[0]),float(v[1]),str(v[2])) for c,v in (food or {}).items()))
    allowance=sum(sorted((g for _c,g,_v,_k in food),reverse=True)[:3])
    horizon=min(88,max(16,len(body)+growth+allowance+8))
    return _plan(tuple(map(tuple,body)),heading,frozenset(hard),frozenset(region),
                 food,int(growth),int(segment),int(initial),horizon)


@lru_cache(maxsize=32)
def _plan(body,heading,hard,region,food,growth,segment,initial,horizon):
    foods={c:(g,v,k,i) for i,(c,g,v,k) in enumerate(food)}
    major=0 if heading and heading[0] else 1;minor=1-major
    tail=set(body[-max(3,len(body)//2):])
    # body, heading, straight run, growth, eaten bits, sweep sign, penalty,
    # reward, exited pocket, turn spacing, path, number of real tail advances
    states=[(body,heading,segment-initial,growth,0,0,0.,0.,False,segment,(body[0],),0)]
    for depth in range(1,horizon+1):
        candidates=[];seen=set()
        for b,h,run,g,eaten,sign,cost,reward,escaped,spacing,path,released in states:
            occupied=set(b if g else b[:-1])
            for d in DIRS:
                turn=h is not None and d!=h
                if d==OPP.get(h) or (turn and run<spacing):
                    continue
                n=(b[0][0]+d[0],b[0][1]+d[1])
                if not inside(n) or n in hard or n in occupied:
                    continue
                nb=(n,)+b if g else (n,)+b[:-1];ng=max(0,g-1)
                bits=eaten;gain=reward;ns=spacing
                item=foods.get(n)
                if item and not eaten&(1<<item[3]):
                    ng+=item[0];gain+=item[1];bits|=1<<item[3]
                    if item[2]=='grn': ns=min(3,spacing+1)
                nr=1 if turn else min(ns,run+1)
                sweep=d[minor] or sign
                penalty=cost+3.*bool(d[minor] and sign and d[minor]!=sign)+.03*turn
                out=escaped or n not in region
                key=(nb,d,nr,ng,bits,sweep,ns)
                if key in seen: continue
                seen.add(key)
                trail=set(nb[2:])
                contact=sum((n[0]+v[0],n[1]+v[1]) in trail for v in DIRS)
                score=12.*out+1.7*gain+2.*contact+3.*(n in tail)+.3*(d==h)-penalty
                candidates.append((score,(nb,d,nr,ng,bits,sweep,penalty,gain,out,ns,
                                          path+(n,),released+int(g==0))))
        if not candidates:
            return None
        candidates.sort(key=lambda x:x[0],reverse=True)
        states=[s for _,s in candidates[:40]]
    # Surviving beside the head is insufficient: actually leave the ORIGINAL
    # pocket through an occupied body cell only after its tail has vacated it.
    finalists=[]
    for s in states:
        b,h,run,g,_e,_sign,_cost,_reward,escaped,spacing,_path,released=s
        occupied=set(b if g else b[:-1])
        exit_exists=any(d!=OPP.get(h) and (d==h or run>=spacing) and
            inside((b[0][0]+d[0],b[0][1]+d[1])) and
            (b[0][0]+d[0],b[0][1]+d[1]) not in hard and
            (b[0][0]+d[0],b[0][1]+d[1]) not in occupied for d in DIRS)
        if escaped and released>=len(body)+max(2,2*spacing) and exit_exists:
            finalists.append(s)
    if not finalists:
        return None
    return list(max(finalists,key=lambda s:1.7*s[7]-s[6])[10])
