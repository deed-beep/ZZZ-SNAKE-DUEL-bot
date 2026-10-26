"""Grid routes with executable turn spacing and explicit moving-body witnesses."""
from heapq import heappush, heappop
from itertools import count
import math
from functools import lru_cache
from config import CELL, COLS, ROWS

DIRS=((1,0),(-1,0),(0,1),(0,-1))
OPP={d:(-d[0],-d[1]) for d in DIRS}


def inside(c):
    return 0<=c[0]<COLS and 0<=c[1]<ROWS


def advance(body, path, growth=0):
    """Simulate an existing growth debt, including every newly occupied cell."""
    body=tuple(map(tuple,body))
    for c in path[1:]:
        if c in (body if growth else body[:-1]):
            return None
        body=(c,)+body if growth else (c,)+body[:-1]
        growth=max(0,growth-1)
    return body,growth


def search(start, heading, goal, hard, own, *, initial=0, segment=1,
           length=5, limit=1400, risk=None):
    """Bounded A*: direction/run length are state, own releases are timed.

    No obstacle padding. A current leg must cover `initial` cells before its
    first turn, later legs `segment` cells. Parent visits are checked as real
    body occupancy, not discarded after assigning release times.
    """
    if not inside(goal) or goal in hard:
        return None
    if start==goal:
        return [start]
    segment=max(1,min(3,int(segment)))
    initial=max(0,min(4,int(initial)))
    serial=count()
    first=(start,heading,0)
    # Run count is capped once turning is possible; the first leg has its
    # own threshold, encoded with a negative initial run when needed.
    first=(start,heading,segment-initial)
    q=[]
    heappush(q,(abs(start[0]-goal[0])+abs(start[1]-goal[1]),0.,next(serial),first,0))
    best={first:0.}
    parent={first:None}
    expanded=0
    while q and expanded<limit:
        _,cost,_,key,depth=heappop(q)
        if cost>best.get(key,float('inf')):
            continue
        expanded+=1
        c,h,run=key
        # Reconstruct a short visited prefix to check the body's new trail.
        trail=[]
        k=key
        while k is not None:
            trail.append(k[0]);k=parent[k]
        trail.reverse()
        if c==goal:
            return trail
        if depth>=70:
            continue
        recent=set(trail[-max(1,length-1):])
        for d in DIRS:
            if h in OPP and d==OPP[h]:
                continue
            turn=h in OPP and d!=h
            if turn and run<segment:
                continue
            n=(c[0]+d[0],c[1]+d[1])
            if not inside(n) or n in hard or own.get(n,-1)>=depth+1 or n in recent:
                continue
            nr=1 if turn else min(segment,run+1)
            nk=(n,d,nr)
            value=cost+1.+(.12 if turn else 0.)+3.*(risk or {}).get(n,0.)
            if value>=best.get(nk,float('inf')):
                continue
            best[nk]=value;parent[nk]=key
            heuristic=abs(n[0]-goal[0])+abs(n[1]-goal[1])
            heappush(q,(value+heuristic,value,next(serial),nk,depth+1))
    return None


class RoutePlans:
    def __init__(self):
        self.cache={}

    def approach(self,key,head,px,heading,hard,own,speed,delay,length,risk=None,pending=None,geometry_speed=None):
        travel=max(0.,speed*delay)
        spacing_speed=max(speed,geometry_speed or 0.)
        segment=max(1,min(3,math.ceil(max(0.,spacing_speed*delay)/CELL)))
        initial=0
        if heading in OPP:
            axis=0 if heading[0] else 1
            to_center=((head[axis]+.5)*CELL-px[axis])*heading[axis]
            initial=max(0,math.ceil((travel-8-to_center)/CELL))
        plan=self.cache.get(key)
        if plan and head in plan and (segment<=self.cache.get(('segment',key),1) or pending is not None):
            plan=plan[plan.index(head):]
            first=(plan[1][0]-head[0],plan[1][1]-head[1]) if len(plan)>1 else heading
            recent=[]
            lead_run=0
            for a,b in zip(plan,plan[1:]):
                if (b[0]-a[0],b[1]-a[1])!=heading:break
                lead_run+=1
            # A previously reachable corner can already be behind the response
            # horizon later in the SAME tile. Only a turn already issued to the
            # controller can bypass the fresh initial-run requirement.
            queued=False
            if pending and tuple(pending.get('cell',())) in plan:
                corner_index=plan.index(tuple(pending['cell']))
                queued=(corner_index<len(plan)-1 and lead_run==corner_index
                        and (plan[corner_index+1][0]-plan[corner_index][0],
                             plan[corner_index+1][1]-plan[corner_index][1])==pending.get('dir'))
            elif pending and pending.get('cell') is None:
                queued=pending.get('dir')==first and first!=heading
            # The acceptance/issue overlap must span a control sample. An
            # 8px overlap was narrower than a normal 12--24px observation step:
            # the next sample invalidated the bend precisely when it should be
            # issued, repeatedly moving it farther toward the wall (r056).
            if lead_run<len(plan)-1:
                corner=plan[lead_run]
                axis=0 if heading and heading[0] else 1
                remaining=((corner[axis]+.5)*CELL-px[axis])*(heading[axis] if heading else 0)
                tolerance=max(.5*CELL,min(CELL,speed*.10))
                reachable=remaining>=max(0.,travel-tolerance)
            else:reachable=True
            valid=(first!=OPP.get(heading) and (queued or reachable)
                   and (queued or segment<=self.cache.get(('segment',key),1)))
            for depth,c in enumerate(plan[1:],1):
                if c in hard or own.get(c,-1)>=depth or c in recent[-max(1,length-1):]:
                    valid=False;break
                recent.append(c)
            if valid:
                self.cache[key]=plan
                return plan
        initial=0
        if heading in OPP:
            axis=0 if heading[0] else 1
            to_center=((head[axis]+.5)*CELL-px[axis])*heading[axis]
            initial=max(0,math.ceil((travel-8-to_center)/CELL))
        plan=search(head,heading,key[1],hard,own,initial=initial,segment=segment,
                    length=length, risk=risk)
        if plan:
            self.cache[key]=plan
            self.cache[('segment',key)]=segment
        return plan


def tail_witness(path, body, heading, hard, growth=0, segment=1, limit=650, initial=0):
    """After the approach and pickup, prove a route to the actual old tail.

    The tail is blocked for the full growth debt. Its cell is never treated as
    already free just because it is visually near the head.
    """
    moved=advance(body,path)
    if moved is None:
        return None
    b,_=moved
    h=(path[-1][0]-path[-2][0],path[-1][1]-path[-2][1]) if len(path)>1 else heading
    own={c:len(b)-1-i+growth for i,c in enumerate(b)}
    escape=search(b[0],h,b[-1],hard,own,initial=initial,segment=segment,
                  length=len(b)+growth,limit=limit)
    if escape is None or advance(b,escape,growth) is None:
        return None
    return escape


def packing_escape(body,heading,hard,growth=0,segment=1,initial=0,depth=None,partial=False):
    """Return an actual moving-body sweep, favoring parallel rows over spirals.

    A full witness survives a body turnover, including growth. `partial` is
    only for an emergency with unknown tail order: it maximizes survival time
    without claiming an unseen tail has vacated any cell.
    """
    b=tuple(map(tuple,body))
    horizon=depth if depth is not None else min(128,max(12,len(b)+growth+3))
    return _packing(b,heading,frozenset(hard),int(growth),int(segment),int(initial),int(horizon),partial)


@lru_cache(maxsize=64)
def _packing(body,heading,hard,growth,segment,initial,horizon,partial):
    major=0 if heading and heading[0] else 1
    minor=1-major
    original_tail=set(body[-max(4,len(body)//2):])
    # body, direction, run, growth, minor direction, penalty, path
    states=[(body,heading,segment-initial,growth,0,0.,(body[0],))]
    last=states[0][-1]
    for _ in range(horizon):
        options=[];seen=set()
        for b,h,run,g,minor_sign,penalty,path in states:
            occupied=set(b if g else b[:-1])
            for d in DIRS:
                if d==OPP.get(h) or (h and d!=h and run<segment):
                    continue
                n=(b[0][0]+d[0],b[0][1]+d[1])
                if not inside(n) or n in hard or n in occupied:
                    continue
                nb=(n,)+b if g else (n,)+b[:-1]
                ng=max(0,g-1)
                nr=min(segment,run+1) if d==h else 1
                sign=d[minor] or minor_sign
                flipped=bool(d[minor] and minor_sign and d[minor]!=minor_sign)
                cost=penalty+4*flipped+.04*bool(d!=h)
                key=(nb,d,nr,ng,sign)
                if key in seen:
                    continue
                seen.add(key)
                solid=set(nb if ng else nb[:-1])
                exits=sum(v!=OPP[d] and inside((n[0]+v[0],n[1]+v[1])) and
                          (n[0]+v[0],n[1]+v[1]) not in hard and
                          (n[0]+v[0],n[1]+v[1]) not in solid for v in DIRS)
                if not exits:
                    continue
                # Parallel packed lanes gain body contact without changing
                # the minor sweep direction, unlike a tightening inward spiral.
                parallel=set(nb[2:])
                contact=sum((n[0]+v[0],n[1]+v[1]) in parallel for v in DIRS)
                score=5*int(n in original_tail)+1.5*contact+.4*int(d==h)+.15*exits-cost
                state=(nb,d,nr,ng,sign,cost,path+(n,))
                options.append((score,state))
        if not options:
            return list(last) if partial and len(last)>=6 else None
        options.sort(key=lambda x:x[0],reverse=True)
        states=[s for _,s in options[:24]]
        last=states[0][-1]
    return list(last)
