"""The attention coordinator: decides where each frame's effort is spent.

Order of work per frame, highest priority first, which is also the order the budget is spent:

  1. the player's head       - always, because everything else is positioned relative to it
  2. the 7x7 window on it    - diamonds, then obstacles, then beans, all at full resolution
  3. the periphery           - only on the frames the scheduler says so, and its result is
                               marked with its age so nothing downstream treats it as current

After the spike balls appear the window's obstacle pass runs BEFORE its diamond pass for that
frame, because a ball is fatal on contact while a gem is only points - that is the ordering the
user specified and it is encoded in `Attention.scan`.

The tiers are not just labels.  Each one carries a trust value or an age, and the state analysis
uses them: a window gem is accepted on the spot, a peripheral gem has to survive the mechanics
check, and the head is never taken from the periphery when the window can see it.
"""
from __future__ import annotations

import time

import numpy as np
import cv2

from config import (CELL, FRAME_W, FRAME_H, HEAD_WIN_CELLS, HEAD_WIN_WEIGHT,
                    PERIPHERAL_WEIGHT, WINDOW_FIRST, PERIPHERAL_EVERY)
from attention import (AttentionFrame, WindowDetector, PeripheralVision,
                       FixedGemPair)
from temporal import HeadWindow


def known_gem_position_empty(image,center,occupied):
    x,y=map(lambda v:int(round(v)),center)
    cell=(x//CELL,y//CELL)
    if cell in occupied or x<12 or y<12 or x+12>=image.shape[1] or y+12>=image.shape[0]:
        return False
    patch=image[y-12:y+13,x-12:x+13].astype(np.int16)
    r,g,b=(patch[...,i] for i in (0,1,2))
    hue=cv2.cvtColor(patch.astype(np.uint8),cv2.COLOR_RGB2HSV)[...,0]
    floor=(r<180)&(g>=150)&(b>=180)&(g-r>80)&(b-g>=0)&(b-g<=60)&(hue>=87)&(hue<=97)
    return float(floor.mean())>=.90


class Attention:
    """Owns the window, the periphery, and the per-frame priority allocation."""

    def __init__(self, perception, temporal=None):
        self.per = perception
        self.win_det = WindowDetector(perception)
        self.global_gem_det = WindowDetector(perception)
        self.periph = PeripheralVision()
        self.temporal = temporal
        self.head_win = HeadWindow("red")
        # the fixed diamond pair, from the user's mechanics: exactly two fixed gems,
        # which is a constraint a per-frame detector cannot express
        self.gem_pair = FixedGemPair()
        self.frame = 0
        self.last = AttentionFrame()
        self.scan_times = []          # ms of the peripheral scans, for the report
        self._item_centers = {}
        self._near_missing = {}
        self._empty_since = {}
        self._gem_passed = {}
        self._retired_gems = {}
        self._verified_gems = {}
        self._previous_head_px = None

    # ------------------------------------------------------------------ helpers
    def box_for(self, head_px):
        """The 7x7-cell box centred on the head, in full-resolution pixels.

        Clamped to the frame; when the head is unknown the box is None and the caller falls back
        to the peripheral reading for that frame rather than guessing a position.
        """
        if head_px is None:
            return None
        half = (HEAD_WIN_CELLS * CELL) // 2
        x0 = int(max(0, min(FRAME_W - 1, head_px[0] - half)))
        y0 = int(max(0, min(FRAME_H - 1, head_px[1] - half)))
        x1 = int(max(x0 + 1, min(FRAME_W, head_px[0] + half)))
        y1 = int(max(y0 + 1, min(FRAME_H, head_px[1] + half)))
        return (x0, y0, x1, y1)

    # ------------------------------------------------------------------ per frame
    def scan(self, img_full, snakes, items, flags, red_cells=None, family_masks=None,
             match_t=None):
        """-> AttentionFrame.  `snakes`/`items` come from Perception.run this frame."""
        t0 = time.perf_counter()
        self.frame += 1
        af = AttentionFrame(frame=self.frame, match_t=match_t, sample_t=flags.get("sample_t"))
        af.scan_ms = flags.get("ms", 0.0)
        self._item_centers.update(flags.get("item_centers", {}))
        af.item_centers = dict(self._item_centers)
        # board totals, for the mechanics cross-checks in state_analysis
        # ---- the fixed gem pair, from the full-frame gem list rather than the window alone:
        # the pair is a board-level fact, and the window only sees 7x7 cells of it
        # New fixed slots need strong full-resolution shape evidence. Raw cyan
        # floor/body fragments such as luping3 f605/f877/f883 must not become a
        # persistent goal just because they were first in connected-component order.
        global_gems=[]
        if self.periph.due(self.frame):
            global_gems=self.global_gem_det.diamonds(img_full,(0,0,FRAME_W,FRAME_H),sample_t=af.sample_t)
            self._item_centers.update(self.global_gem_det.centers)
            for e in global_gems:
                self._verified_gems[(int(e[0]),int(e[1]))]=self.frame
        for c in list(self._verified_gems):
            if self.frame-self._verified_gems[c]>12:
                self._verified_gems.pop(c,None)
        af.board_beans = len(items.get("bean", []))
        af.board_balls = len(items.get("ball", []))
        red = (snakes or {}).get("red")

        # ---- tier 1: the head.  Everything else is positioned relative to it, so it is never
        # skipped and the peripheral reading is never allowed to override the window.
        if red is not None and red.head_px is not None:
            af.head_px = tuple(float(v) for v in red.head_px)
            af.head_cell = tuple(int(v) for v in red.head_cell) if red.head_cell else None
            af.head_cell_raw = af.head_cell
            af.facing = red.facing
            af.red_length = int(red.length)
            af.head_source = red.orientation or "perception"
            af.head_trust = 1.0 if red.orientation == "eyes" else 0.7
            self.head_win.observe(af.head_px)
        else:
            pred = self.head_win.predict()
            if pred is not None and self.head_win.miss <= HEAD_WIN_CELLS:
                af.head_px = (float(pred[0]), float(pred[1]))
                af.head_cell = (int(pred[0] // CELL), int(pred[1] // CELL))
                af.head_source = "predicted"
                af.head_trust = max(0.1, 0.6 - 0.05 * self.head_win.miss)
                af.red_length = int(red.length) if red is not None else 0
                af.notes.append("head-predicted")
            self.head_win.miss += 1

        af.window_box = self.box_for(af.head_px)
        own_cells = set()
        # the body itself, for the state analysis to repair a bad head against
        af.red_cells = sorted((int(c[0]), int(c[1])) for c in (red.cells if red is not None and red.cells else set()))
        # the ORDERED path and its trust, which is what "the two ends of the snake" needs
        af.red_path = [(int(c[0]), int(c[1])) for c in (red.path if red is not None and red.path else [])]
        af.red_path_trust = getattr(red, "path_trust", "absent") if red is not None else "absent"
        if red is not None and red.cells:
            own_cells = {(int(c[0]), int(c[1])) for c in red.cells}

        # balls_live decides whether obstacles or diamonds come first inside the window
        balls_all = list(items.get("ball", []))
        af.balls_live = bool(match_t is not None and match_t >= 42.0) or bool(balls_all)

        # ---- tier 2/3/4: the window, fully re-measured this frame
        if af.window_box is not None:
            box = af.window_box
            if af.balls_live:
                # obstacles first: a ball on the head is fatal now, a gem is points later
                af.win_obstacles = self.win_det.obstacles(
                    img_full, box, family_masks or {}, own_cells, balls_all,
                    items.get("slow", []))
                af.win_diamonds = self.win_det.diamonds(img_full, box, sample_t=af.sample_t)
            else:
                af.win_diamonds = self.win_det.diamonds(img_full, box, sample_t=af.sample_t)
                af.win_obstacles = self.win_det.obstacles(
                    img_full, box, family_masks or {}, own_cells, balls_all,
                    items.get("slow", []))
            # beans in the window: cheap, taken from the already-decoded item list
            for e in items.get("bean", []):
                cx, cy = int(e[0]), int(e[1])
                px, py = cx * CELL + CELL // 2, cy * CELL + CELL // 2
                if box[0] <= px < box[2] and box[1] <= py < box[3]:
                    af.win_beans.append((cx, cy, int(e[2]) if len(e) > 2 else 0))
            for e in items.get("slow", []):
                cx, cy = int(e[0]), int(e[1])
                px, py = cx * CELL + CELL // 2, cy * CELL + CELL // 2
                if box[0] <= px < box[2] and box[1] <= py < box[3]:
                    af.win_slow.append((cx, cy, int(e[2]) if len(e) > 2 else 0))
            for e in items.get("grn", []):
                cx,cy=int(e[0]),int(e[1])
                if box[0] <= (cx+.5)*CELL < box[2] and box[1] <= (cy+.5)*CELL < box[3]:
                    af.win_speed.append((cx,cy,int(e[2])))

        strict_cells={(int(e[0]),int(e[1])) for e in af.win_diamonds}
        strict_cells.update(self._verified_gems)
        existing=[tuple(s['cell']) for s in self.gem_pair.slots]
        qualified=[e for e in items.get('dia',[])
                   if tuple(map(int,e[:2])) in strict_cells
                   or any(max(abs(int(e[0])-c[0]),abs(int(e[1])-c[1]))<=1 for c in existing)]
        qualified_cells={tuple(map(int,e[:2])) for e in qualified}
        qualified.extend(e for e in global_gems if tuple(map(int,e[:2])) not in qualified_cells)
        # Preserve the full-resolution stationarity verdict even when a raw
        # detector supplied the same cell without an identity tag.
        confirmed={tuple(map(int,e[:2])):e for e in global_gems if len(e)>4 and e[4]=='fixed'}
        qualified=[confirmed.get(tuple(map(int,e[:2])),e) for e in qualified]
        fixed,extra=self.gem_pair.observe(qualified,self.frame)
        qualified_cells={tuple(map(int,e[:2])) for e in qualified}
        rejected={tuple(map(int,e[:2])) for e in items.get('dia',[])
                  if tuple(map(int,e[:2])) not in qualified_cells}
        af.gem_fixed=fixed
        af.gem_extra=list(set(extra)|rejected)
        af.gem_respawning=self.gem_pair.respawning()

        # ---- tier 5/6: the periphery, only when due
        if self.periph.due(self.frame):
            self.periph.observe(items)
            af.per_fresh = True
            self.scan_times.append(flags.get("ms", 0.0))
        else:
            self.periph.tick()
        af.per_age = self.periph.age
        d, b, bl, sl = self.periph.as_lists()
        # Populate BEFORE merging; the old order always merged an empty periphery.
        af.per_diamonds = [tuple(x) for x in d]
        af.per_beans = [tuple(x) for x in b]
        af.per_balls = [tuple(x) for x in bl]
        af.per_slow = [tuple(x) for x in sl]
        af.per_speed = [tuple(e)+(self.periph.age,) for e in self.periph.speed] if self.periph.fresh_enough() else []
        self._item_centers.update(self.win_det.centers)
        af.item_centers = dict(self._item_centers)
        af.gem_last_seen = {tuple(s["cell"]): s["last"] for s in self.gem_pair.slots}
        # Missing pixels are not collection. Require an actual head-centre
        # crossing followed by repeated empty observations. Merely circling
        # within two cells must keep the stationary goal alive.
        seen_gems = {(int(e[0]),int(e[1])) for e in af.win_diamonds}
        seen_gems.update((int(k[1]), int(k[2])) for k in self.win_det.centers
                         if k[0] == 'dia')
        for c in list(self._retired_gems):
            if self.frame-self._retired_gems[c]>45 or c in seen_gems:
                self._retired_gems.pop(c,None)
        for c in list(af.gem_fixed):
            center = af.item_centers.get(("dia",*c),((c[0]+.5)*CELL,(c[1]+.5)*CELL))
            crossed = False
            if af.head_px is not None:
                end = np.asarray(af.head_px, dtype=float)
                start = np.asarray(self._previous_head_px or af.head_px, dtype=float)
                delta = end-start
                if np.linalg.norm(delta) <= 2*CELL:
                    u = np.clip(np.dot(np.asarray(center)-start, delta) / max(1., np.dot(delta, delta)), 0., 1.)
                    crossed = np.linalg.norm(np.asarray(center)-(start+u*delta)) <= .35*CELL
            if crossed:
                self._gem_passed[c] = self.frame
            recently_passed = self.frame-self._gem_passed.get(c,-1000) <= 10
            self._near_missing[c] = self._near_missing.get(c,0)+1 if recently_passed and c not in seen_gems else 0
            if self._near_missing[c] >= 3:
                self.gem_pair.slots = [s for s in self.gem_pair.slots if tuple(s["cell"]) != tuple(c)]
                af.gem_fixed.remove(c)
                self._retired_gems[c] = self.frame
                af.gem_collected.append(c)
                af.notes.append(f"gem-crossed-and-absent {c}")
        # Other snakes can collect a fixed gem too. Repeated unobstructed
        # floor at its measured old position is independent removal evidence;
        # do not fabricate our own pickup/growth from this observation.
        stamp=af.sample_t if af.sample_t is not None else self.frame/15.
        occupied={tuple(map(int,c)) for snake in (snakes or {}).values() for c in snake.cells}
        for c in list(af.gem_fixed):
            center=af.item_centers.get(('dia',*c),((c[0]+.5)*CELL,(c[1]+.5)*CELL))
            empty=c not in seen_gems and known_gem_position_empty(img_full,center,occupied)
            if empty:
                since,hits=self._empty_since.get(c,(stamp,0))
                self._empty_since[c]=(since,hits+1)
                if hits+1>=4 and stamp-since>=.4:
                    self.gem_pair.slots=[slot for slot in self.gem_pair.slots if tuple(slot['cell'])!=c]
                    af.gem_fixed.remove(c)
                    self._retired_gems[c]=self.frame
                    af.notes.append('gem-verified-empty '+str(c))
                    self._empty_since.pop(c,None)
            else:self._empty_since.pop(c,None)
        self._previous_head_px = af.head_px
        # Window obstacles must use the actual enemy occupied CELLS. The live
        # caller never supplied family masks; previously the body list was empty.
        if af.window_box is not None:
            x0,y0,x1,y1=af.window_box
            for kind, snake in (snakes or {}).items():
                if kind == "red":
                    continue
                for cx,cy in snake.cells:
                    if x0 <= (cx+.5)*CELL < x1 and y0 <= (cy+.5)*CELL < y1:
                        af.win_obstacles.append((int(cx),int(cy),"body",0))

        # ---- the merged gem pipeline (see the module docstring for the precedence order)
        best = {}
        in_win = set()
        if af.window_box is not None:
            x0, y0, x1, y1 = af.window_box
            for e in af.win_diamonds:
                cx, cy = int(e[0]), int(e[1])
                ring = float(e[3]) if len(e) > 3 else 0.0
                area = int(e[2]) if len(e) > 2 else 0
                best[(cx, cy)] = (cx, cy, area, ring, "window", True)
                in_win.add((cx, cy))
        for c in af.gem_fixed:
            cell = (int(c[0]), int(c[1]))
            if cell not in best and cell not in self._retired_gems:
                best[cell] = (cell[0], cell[1], 0, 0.0, "pair", False)
        # A current full-frame detection corroborated by the recent strict
        # shape pass is collectable even when both fixed-identity slots still
        # refer to old, occluded locations. Keep identity and pickup eligibility
        # separate: a third real gem must not become an ignored "extra".
        for e in items.get('dia',[]):
            cell=tuple(map(int,e[:2]))
            if (self.frame-self._verified_gems.get(cell,-999)<=PERIPHERAL_EVERY+1 and cell not in self._retired_gems
                    and cell not in best):
                best[cell]=(cell[0],cell[1],int(e[2]),float(e[3]) if len(e)>3 else 0.,'native',False)
        for e in af.per_diamonds:
            cell = (int(e[0]), int(e[1]))
            # Preserve source exclusion: an extra must not enter via periphery.
            if cell in af.gem_extra or cell in self._retired_gems:
                continue
            if cell not in best:
                best[cell] = (cell[0], cell[1], int(e[2]) if len(e) > 2 else 0,
                              float(e[3]) if len(e) > 3 else 0.0, "peripheral", False)
        for c in af.gem_extra:
            cell = (int(c[0]), int(c[1]))
            if cell not in best:
                best[cell] = (cell[0], cell[1], 0, 0.0, "extra", False)
        af.gem_best = list(best.values())
        af.per_diamonds = [tuple(x) for x in d]
        af.per_beans = [tuple(x) for x in b]
        af.per_balls = [tuple(x) for x in bl]
        af.per_slow = [tuple(x) for x in sl]

        af.scan_ms = (time.perf_counter() - t0) * 1000 + flags.get("ms", 0.0)
        self.last = af
        return af
