"""Bounded topology and native head crossing checks, with no obstacle halo."""
from collections import deque
from config import CELL,COLS,ROWS
from sweep_planner import walk
DIRS=((1,0),(-1,0),(0,1),(0,-1))


def chamber(path,red,hard,growth,food,limit=80):
    ordered=red.path_trust=='ordered' and red.path and tuple(red.path[0])==tuple(path[0])
    tail=set()
    if ordered:
        moved=walk(red.path,path,growth,food)
        if moved is None:return 0,0,False
        body,debt,_=moved
        occupied=set(body if debt else body[:-1])
        tail=set(body[-min(6,len(body)):])
    else:
        occupied=set(red.cells)|set(path[:-1])
    blocked=set(hard)|occupied
    start=tuple(path[-1]);blocked.discard(start)
    heading=(path[-1][0]-path[-2][0],path[-1][1]-path[-2][1])
    reverse=(-heading[0],-heading[1])
    valid=lambda c:0<=c[0]<COLS and 0<=c[1]<ROWS and c not in blocked
    exits=[(start[0]+d[0],start[1]+d[1])for d in DIRS if d!=reverse and valid((start[0]+d[0],start[1]+d[1]))]
    seen={start};queue=deque(exits)
    while queue and len(seen)<limit:
        c=queue.popleft()
        if c in seen:continue
        seen.add(c)
        for d in DIRS:
            n=(c[0]+d[0],c[1]+d[1])
            if valid(n)and n not in seen:queue.append(n)
    near_tail=bool(tail and any((c[0]+d[0],c[1]+d[1])in tail for c in seen for d in DIRS))
    return len(seen),len(exits),near_tail


def head_crossing(path,px,speed,heads,now,horizon=.45):
    predictions=[]
    for kind,h in heads.items():
        if kind not in ('blu','pur'):continue  # warm ownership is less reliable
        age=now-float(h['at'])
        if not 0<=age<=.18:continue
        predictions.append((h,age))
    if not predictions:return False
    speed=max(100.,float(speed));previous=px;distance=0.
    for cell in path[1:]:
        point=tuple((v+.5)*CELL for v in cell)
        length=sum(abs(point[i]-previous[i])for i in (0,1))
        n=max(1,int(length/8.)+1)
        for k in range(1,n+1):
            eta=(distance+length*k/n)/speed
            if eta>horizon:return False
            q=tuple(previous[i]+(point[i]-previous[i])*k/n for i in (0,1))
            for h,age in predictions:
                center=tuple((h['cell'][i]+.5)*CELL+h['direction'][i]*h['speed']*(age+eta)for i in (0,1))
                # Two rendered heads overlap; a parallel adjacent lane stays
                # legal. This is not an extra empty grid-cell reserve.
                if abs(q[0]-center[0])<31. and abs(q[1]-center[1])<31.:
                    return True
        distance+=length;previous=point
    return False
