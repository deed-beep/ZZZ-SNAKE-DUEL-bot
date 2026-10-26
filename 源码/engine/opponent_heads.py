"""Head hypotheses from newly occupied native body tiles, with short expiry."""
import math

DIRS = ((1,0),(-1,0),(0,1),(0,-1))


class NativeOpponentHeads:
    def __init__(self):
        self.previous = {}
        self.heads = {}

    def update(self, snakes, now):
        now = float(now)
        for kind in ('org', 'pur', 'blu'):
            snake = snakes.get(kind)
            current = {tuple(map(int,c)) for c in snake.cells} if snake else set()
            previous = self.previous.get(kind, set())
            born = current - previous
            shared = current & previous
            if previous and len(shared) < .65 * len(previous):
                self.heads.pop(kind, None)
            # A new native head tile may hide the old one under another
            # snake or an item. Shared-neighbor-only matching then freezes
            # the old direction through a real corner (r060 f592). Continue
            # from the recent witnessed head, rejecting reverse restorations.
            old=self.heads.get(kind)
            continued=False
            if old and 0<now-old['at']<=.35 and len(shared)>=max(2,.65*len(previous)):
                options=[]
                for cell in born:
                    delta=tuple(cell[i]-old['cell'][i] for i in (0,1))
                    # A flashing interior tile can reappear beside the old
                    # head. Reject an internal T/cross junction with three body
                    # neighbors (r069 f509 incorrectly turned UP into LEFT).
                    # Two neighbors can be a real head touching another own
                    # lane while turning; round60 f588/593 must retain that.
                    degree=sum((cell[0]+d[0],cell[1]+d[1]) in current for d in DIRS)
                    if (sum(abs(v) for v in delta)==1 and degree<=2 and
                            delta!=tuple(-v for v in old['direction'])):
                        options.append((cell,delta))
                if len(options)==1:
                    cell,direction=options[0]
                    speed=max(140.,min(480.,32/(now-old['at'])))
                    self.heads[kind]=dict(cell=cell,direction=direction,speed=speed,at=now)
                    continued=True
            if not continued and len(born) == 1 and len(shared) >= max(2, .65 * len(previous)):
                cell = next(iter(born))
                neighbors = [c for c in shared if sum(abs(c[i]-cell[i]) for i in (0,1)) == 1]
                if len(neighbors) == 1:
                    direction = tuple(cell[i]-neighbors[0][i] for i in (0,1))
                    old = self.heads.get(kind)
                    distance = sum(abs(cell[i]-old['cell'][i]) for i in (0,1)) if old else 0
                    dt = now-old['at'] if old else 0
                    if old is None or (0 < dt <= .45 and distance <= 3):
                        speed = (max(140., min(480., 32*distance/dt)) if old and distance
                                 else 240.)
                        self.heads[kind] = dict(cell=cell, direction=direction, speed=speed, at=now)
            self.previous[kind] = current
        self.heads = {k:v for k,v in self.heads.items() if 0 <= now-v['at'] <= .35}
        return dict(self.heads)


def possible_crossings(head, occupied, hard):
    """Short reachable head sweep allowing one bend, not a body-wide halo."""
    start = tuple(head['cell'])
    direction = tuple(head['direction'])
    steps = max(2, min(4, math.ceil(head['speed']*.45/32)))
    frontier = {(start, direction, 0)}
    risk = {}
    for depth in range(1, steps+1):
        following = set()
        for cell, facing, turns in frontier:
            for d in DIRS:
                if d == (-facing[0],-facing[1]):
                    continue
                bend = d != facing
                if turns+bend > 1:
                    continue
                n = (cell[0]+d[0],cell[1]+d[1])
                if not (0 <= n[0] < 29 and 0 <= n[1] < 25):
                    continue
                if n in hard or n in occupied:
                    continue
                value=(1.8-.22*depth if turns+bend==0 else .95 if depth==1 else .65-.06*(depth-1))
                risk[n] = max(risk.get(n,0.), value)
                following.add((n,d,turns+bend))
        frontier = following
    return risk
