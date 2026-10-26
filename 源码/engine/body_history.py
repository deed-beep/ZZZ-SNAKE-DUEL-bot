"""Conservative head-to-tail history, separate from visual observations.

Only measured eyes or explicitly marked native-pixel heads extend the trail.
No body observation is edited.  The caller may use an ordered result to schedule
release of matched cells; unmatched visible cells must remain solid.  When
release_padding > 0, add it to ALL matched release delays: an incomplete matched
prefix is not proof that its last point is the actual tail.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from config import CELL, COLS, ROWS


@dataclass(frozen=True)
class BodyHistoryResult:
    path: tuple = ()
    path_trust: str = 'uncertain'
    coverage: float = 0.0
    reason: str = 'no_history'
    uncovered: frozenset = frozenset()
    tail_confirmed: bool = False
    release_padding: int = 0
    head_added: bool = False
    trace_cells: int = 0
    epoch: int = 0

    @property
    def ordered(self):
        return self.path_trust == 'ordered'


class OwnBodyHistory:
    """Recover visits newest-first, never infer topology from a BFS diameter.

    update(red, st=None, *, now=None, head_cell=None, head_px=None,
           verified=None, whiteout=False, respawn=False)

    Optional head overrides must still match the measured pixel tile.  Pass
    verified=False if a belief corrected the tile without native pixel evidence.
    Old pickled SnakeObs instances need no new fields: every optional field is
    read with getattr.  Zero verified history does NOT reorder an uncertain raw
    path.  A complete, already-ordered, 4-connected raw path can seed history.
    """

    def __init__(self, min_coverage=.90, max_gap=.35, max_trace=None):
        self.min_coverage = max(.90, float(min_coverage))
        self.max_gap = float(max_gap)
        self.max_trace = max_trace or COLS * ROWS
        self.trace = []
        self.measured_cells=0
        self.epoch = 0
        self._last_head = None
        self._last_px = None
        self._last_time = None
        self._tick = 0
        self._trusted_size = None
        self.last = BodyHistoryResult()

    def reset(self, reason='reset'):
        self.trace.clear()
        self.measured_cells=0
        self._last_head = self._last_px = self._last_time = None
        self._trusted_size = None
        self.epoch += 1
        self.last = BodyHistoryResult(reason=reason, epoch=self.epoch)
        return self.last

    def occupied(self, length=0):
        """Recent measured head trail still inside the last witnessed body.

        A missing middle pixel does not vacate the body. Old visits fall out
        after the head has actually advanced a body's length, rather than a
        timeout. This set grants no tail-release schedule by itself.
        """
        if self._trusted_size is None:
            return frozenset()
        count=max(self._trusted_size,int(length))
        return frozenset(self.trace[:count])

    @staticmethod
    def _cell(value):
        if value is None or len(value) != 2:
            return None
        try:
            x, y = value
            if not math.isfinite(float(x)) or not math.isfinite(float(y)):
                return None
            ix, iy = int(x), int(y)
            if float(x) != ix or float(y) != iy or not (0 <= ix < COLS and 0 <= iy < ROWS):
                return None
            return ix, iy
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def verified_head(red):
        if red is None:
            return False
        try:
            age = float(getattr(red, 'head_age', 0))
        except (TypeError, ValueError):
            return False
        eyes = (bool(getattr(red, 'head_from_eyes', False))
                and not getattr(red, 'eye_extrapolated', False)
                and age == 0 and getattr(red, 'eye_state', None) in ('normal', 'item'))
        native = bool(getattr(red, 'head_from_pixels', False)) and 0 <= age <= 3
        return bool(eyes or native)

    @staticmethod
    def _distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _finish(self, head, visible, reason=None):
        # A freshly measured eye/native head can prove the leading tile even
        # when quantisation omitted it. This exception never covers intermediates.
        path = []
        for i, c in enumerate(self.trace):
            if (i == 0 and c == head) or c in visible:
                path.append(c)
            else:
                break
        covered = set(path) & visible
        uncovered = visible - covered
        coverage = len(covered) / len(visible) if visible else 0.0
        hidden_gap = bool(uncovered & set(self.trace[len(path):]))
        coherent = (bool(path) and path[0] == head and len(set(path)) == len(path)
                    and all(self._distance(a, b) == 1 for a, b in zip(path, path[1:])))
        ordered = coherent and coverage >= self.min_coverage and not hidden_gap and reason is None
        # A mask hiding an old body does not prove that its neck became a tail.
        # Keep the last confirmed size through continuous frames until visibility
        # recovers; a timer must not eventually turn persistent occlusion into proof.
        if (coherent and self._trusted_size is not None
                and self._trusted_size - (len(path) + len(uncovered)) > 3):
            ordered = False
            reason = 'body_collapse_or_occlusion'
        if ordered and not uncovered:
            self._trusted_size = len(path)
        if reason is None:
            reason = ('history' if ordered else
                      'visible_gap' if hidden_gap else 'insufficient_coverage')
        self.last = BodyHistoryResult(
            path=tuple(path), path_trust='ordered' if ordered else 'uncertain',
            coverage=coverage, reason=reason, uncovered=frozenset(uncovered),
            tail_confirmed=ordered and not uncovered,
            release_padding=len(uncovered) if ordered else 0,
            head_added=bool(path and path[0] == head and head not in visible
                            and reason not in ('unverified_head', 'head_pixel_mismatch')),
            trace_cells=len(self.trace), epoch=self.epoch)
        return self.last

    def _trusted_raw(self, red, head, visible):
        # Never reinterpret an uncertain BFS diameter as a body ordering.
        raw = [self._cell(c) for c in (getattr(red, 'path', None) or ())]
        if (getattr(red, 'path_trust', '') == 'ordered' and raw and raw[0] == head
                and all(c is not None for c in raw) and len(set(raw)) == len(raw)
                and set(raw) - {head} <= visible and visible <= set(raw)
                and all(self._distance(a, b) == 1 for a, b in zip(raw, raw[1:]))):
            return raw
        return []

    def _refresh_tail(self, red, head, visible):
        raw = self._trusted_raw(red, head, visible)
        shared = min(len(raw), len(self.trace))
        if raw and raw[:shared] == self.trace[:shared] and len(raw) > len(self.trace):
            # An independently trusted complete body can add a never-visited
            # spawn tail, but cannot contradict any measured historical visit.
            self.trace = raw
            return True
        return False

    def _seed(self, red, head, visible):
        self.trace = self._trusted_raw(red, head, visible) or [head]
        self.measured_cells=1  # a seeded body tail is not an actual head visit
        return 'seed_verified_raw' if len(self.trace) > 1 else None

    def update(self, red, st=None, *, now=None, head_cell=None, head_px=None,
               verified=None, whiteout=False, respawn=False):
        st = st or {}
        self._tick += 1
        if now is None:
            now = st.get('t', self._tick / 15.0)
        now = float(now)
        dead = (red is None or not getattr(red, 'alive', False)
                or whiteout or st.get('whiteout', False)
                or respawn or st.get('player_respawn_pending', False)
                or st.get('player_alive', True) is False)
        if dead:
            return self.reset('dead_or_absent')
        visible = {c for value in (getattr(red, 'cells', None) or ())
                   if (c := self._cell(value)) is not None}
        if not visible:
            return self.reset('no_visible_body')
        if getattr(red,'eye_state',None)=='buffer':
            # A brief collision pause is not a new life. Preserve measured
            # history, but do not append a jittering buffer head to the trail.
            return self._finish(getattr(red,'head_cell',None),visible,'collision_buffer')
        head = self._cell(head_cell if head_cell is not None else getattr(red, 'head_cell', None))
        point = head_px if head_px is not None else getattr(red, 'head_px', None)
        try:
            point = tuple(float(v) for v in point) if point is not None else None
        except (ValueError, TypeError):
            point = None
        fresh = self.verified_head(red) if verified is None else bool(verified)
        if (head is None or point is None or len(point) != 2
                or not all(math.isfinite(v) for v in point)
                or (math.floor(point[0] / CELL), math.floor(point[1] / CELL)) != head):
            return self._finish(head, visible, 'head_pixel_mismatch')
        if not fresh:
            if self._last_time is not None and now - self._last_time > self.max_gap:
                self.reset('unverified_gap')
            return self._finish(head, visible, 'unverified_head')
        if self._last_time is not None and (now < self._last_time or now - self._last_time > self.max_gap):
            self.reset('time_discontinuity')
        if not self.trace:
            reason = self._seed(red, head, visible)
            self._last_head, self._last_px, self._last_time = head, point, now
            result = self._finish(head, visible)
            if reason and result.ordered:
                self.last = BodyHistoryResult(**{**result.__dict__, 'reason': reason})
            return self.last
        if now == self._last_time:
            return self._finish(head, visible, None if head == self._last_head else 'duplicate_time')
        previous = self._last_head
        distance = self._distance(head, previous)
        dx, dy = point[0] - self._last_px[0], point[1] - self._last_px[1]
        if distance == 0:
            self._last_px, self._last_time = point, now
            self._refresh_tail(red, head, visible)
            return self._finish(head, visible)
        native_order=self._trusted_raw(red,head,visible)
        fresh_facing=(bool(native_order) and getattr(red,'head_from_eyes',False) and getattr(red,'head_age',99)==0
                      and not getattr(red,'eye_extrapolated',False)
                      and getattr(red,'facing',None) in ((0,-1),(0,1),(-1,0),(1,0)))
        eye_axis=getattr(red,'eye_travel_axis',None)
        fresh_axis=(eye_axis in ((1,0),(0,1)) and getattr(red,'head_from_eyes',False)
                    and getattr(red,'head_age',99)==0 and not getattr(red,'eye_extrapolated',False))
        gap=now-self._last_time
        step_axis=0 if abs(dx)>=abs(dy) else 1
        measured_straight=(getattr(red,'head_from_eyes',False) and getattr(red,'head_age',99)==0
            and not getattr(red,'eye_extrapolated',False) and min(abs(dx),abs(dy))<3
            and head[1-step_axis]==previous[1-step_axis] and gap>=.10)
        # The limit applies per elapsed observation time. A fresh head seen
        # again after a brief flash can have travelled several cells. Only a
        # unique, CURRENTLY VISIBLE straight bridge may extend this allowance.
        extra=max(2,min(6,math.ceil(600*gap/CELL))) if measured_straight else 0
        max_steps=max(3 if fresh_facing or fresh_axis else 2,extra)
        max_pixels=max(CELL*2.25,min(6*CELL,600*gap)) if measured_straight else CELL*2.25
        if (distance > max_steps or math.hypot(dx, dy) > max_pixels):
            self.reset('head_jump')
            self.trace = [head]
            self.measured_cells=1
            self._last_head, self._last_px, self._last_time = head, point, now
            return self._finish(head, visible, 'head_jump')
        step_dx, step_dy = head[0] - previous[0], head[1] - previous[1]
        if math.hypot(dx, dy) < 2 or dx * step_dx + dy * step_dy <= 0:
            return self._finish(head, visible, 'head_jitter')
        if len(self.trace) >= 2 and head == self.trace[1]:
            # A/B/A boundary noise is not two new pieces of body history.
            return self._finish(head, visible, 'reverse_jitter')
        added = [head]
        if distance>=2:
            # Every intermediate tile must be CURRENTLY visible. Direction
            # must come from an independently ordered native body at a fresh
            # eye position, not carried facing from an uncertain fragment.
            candidates=[(previous,)]
            for depth in range(distance):
                extended=[]
                for path in candidates:
                    last=path[-1]
                    for x,y in ((0,-1),(0,1),(-1,0),(1,0)):
                        cell=(last[0]+x,last[1]+y)
                        if (cell in visible and cell not in path
                                and self._distance(cell,head)==distance-depth-1):
                            extended.append(path+(cell,))
                candidates=extended
            if (fresh_facing or fresh_axis) and (len(candidates)!=1 or distance==3):
                facing=tuple(red.facing) if fresh_facing else None
                incoming=(self.trace[0][0]-self.trace[1][0],self.trace[0][1]-self.trace[1][1]) if len(self.trace)>=2 else None
                def coherent(path):
                    ds=[(b[0]-a[0],b[1]-a[1]) for a,b in zip(path,path[1:])]
                    aligned=(ds[-1]==facing if facing is not None else
                             abs(ds[-1][0]*eye_axis[0]+ds[-1][1]*eye_axis[1])==1)
                    return (aligned and (incoming is None or ds[0]!=(-incoming[0],-incoming[1]))
                            and sum(a!=b for a,b in zip(ds,ds[1:]))<=1)
                candidates=[path for path in candidates if coherent(path)]
            if len(candidates) != 1:
                # Ambiguous L-corners and missing native middle tiles cannot be
                # resolved by choosing the prettiest shortest path.
                self.reset('ambiguous_or_missing_middle')
                self.trace = [head]
                self.measured_cells=1
                self._last_head, self._last_px, self._last_time = head, point, now
                return self._finish(head, visible, 'ambiguous_or_missing_middle')
            added=list(reversed(candidates[0][1:]))
        proposed = added + self.trace
        seen, unique = set(), []
        for c in proposed:
            if c in seen:
                # Only an old visit beyond the current visible body's capacity
                # can be discarded safely. A recent revisit is a self-contact.
                if len(unique) < max(3, len(visible)):
                    self.reset('repeat_inside_body')
                    self.trace = [head]
                    self.measured_cells=1
                    self._last_head, self._last_px, self._last_time = head, point, now
                    return self._finish(head, visible, 'repeat_inside_body')
                break
            seen.add(c)
            unique.append(c)
        self.trace = unique[:self.max_trace]
        self.measured_cells+=len(added)
        self._last_head, self._last_px, self._last_time = head, point, now
        self._refresh_tail(red, head, visible)
        return self._finish(head, visible)
