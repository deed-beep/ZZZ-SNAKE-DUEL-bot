"""Window mode: track the red head with a 13x13-cell window instead of a global board decode.

WHAT THE MEASUREMENTS SAY
-------------------------
Profiled with `probe_window.py` on recorded frames (luping3 f200+), per frame:

    Perception.run(), global            59.90 ms
      diamonds_full()                   21.19 ms   scales with AREA  -> 0.75 ms in the window
      dark_full()                        1.34 ms   scales with AREA  -> 0.22 ms in the window
      families()                         2.98 ms   FIXED cost, barely shrinks (3.02 on a crop)
      connectedComponents on 7 masks      5.76 ms   FIXED cost, barely shrinks (4.37 on a crop)
      eye search + 4-family grouping + grid_path + tracking   ~28 ms

Two honest conclusions follow, and they shape this module:

 1. **A window does not make the colour masks cheap.** `families` and the connected-component
    sweep are dominated by per-call overhead, not pixel count, so cropping them buys ~nothing.  A
    "13x13 window" that still runs the full colour pipeline will save ~20 ms of 60, not 55.
 2. **The real saving is not running the board-level decode at all.**  The ~28 ms of eye search,
    four-family grouping and path extraction exists to answer "where is every snake and what shape
    is it" across 29x25 cells.  The policy needs the head, our own body, and what is close - and
    `OwnBodyHistory` already reconstructs the body from its recorded trail rather than from this
    frame's pixels.

So this module implements the honest version of the user's instruction: the FAST path gives the
head, our body reach and the immediate surroundings from a 13x13 window, and a LOW-RATE global
sweep supplies the board-level facts a window cannot see (fixed diamonds across the board, respawn,
whiteout, match end).  Where a windowed reading is weaker than the global one, that is recorded in
`meta` instead of being hidden.

WHAT IS REUSED RATHER THAN REIMPLEMENTED
----------------------------------------
`Perception.diamonds_full` does not reference the frame size (checked: no FRAME_*/WORK_* and no
`img_full.shape`), so it runs correctly on a crop and its cells are translated back to the board by
adding the crop's own origin.  Re-deriving its tuned gates here would risk silently regressing a
detector that took several rounds to calibrate.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

import config as C
from config import (CELL, COLS, ROWS, WORK_SCALE, FRAME_W, FRAME_H,
                    WIN_CELLS, SWEEP_EVERY, WHITEOUT_SAT, WHITEOUT_VAL)
from head_solver import HeadSolver


def window_box(head_px, cells=WIN_CELLS, w=FRAME_W, h=FRAME_H):
    """(x0, y0, x1, y1) in full-resolution pixels, centred on `head_px`, clipped to the frame."""
    half = (cells * CELL) // 2
    cx, cy = (w // 2, h // 2) if head_px is None else (int(head_px[0]), int(head_px[1]))
    x0 = max(0, min(w - 1, cx - half))
    y0 = max(0, min(h - 1, cy - half))
    x1 = max(x0 + CELL, min(w, cx + half))
    y1 = max(y0 + CELL, min(h, cy + half))
    return (x0, y0, x1, y1)


def work_box(box):
    """The same box in working-resolution pixels."""
    x0, y0, x1, y1 = box
    return (x0 // WORK_SCALE, y0 // WORK_SCALE, x1 // WORK_SCALE, y1 // WORK_SCALE)


class WindowSight:
    """One frame's windowed observation, in board coordinates."""

    __slots__ = ("box", "mode", "reason", "head_cell", "head_px", "cells", "items",
                 "opponent_cells", "balls", "whiteout", "mean_val", "mean_sat", "ms")

    def __init__(self):
        self.box = None
        self.mode = "none"
        self.reason = ""
        self.head_cell = None
        self.head_px = None
        self.cells = set()            # our own body cells inside the window
        self.items = {"bean": [], "grn": [], "ball": [], "slow": [], "dia": []}
        self.opponent_cells = set()
        self.balls = []               # (cell_x, cell_y, area) inside the window
        self.whiteout = False
        self.mean_val = 0.0
        self.mean_sat = 0.0
        self.ms = 0.0

    def as_items(self):
        return {k: list(v) for k, v in self.items.items()}


class WindowPerception:
    """The window loop, with a low-rate global sweep for the facts a window cannot see."""

    def __init__(self, perception, sweep_every=None):
        self.per = perception
        self.sweep_every = SWEEP_EVERY if sweep_every is None else sweep_every
        self.frame = 0
        self.head_px = None
        self.head_cell = None
        self.last_sweep = -10 ** 9
        self.sight = WindowSight()
        # the head is SOLVED inside the window each frame, not carried over from a sweep
        self.solver = HeadSolver(perception)
        self.stats = {"window": 0, "sweep": 0}
        self.sweep_items = {"bean": [], "grn": [], "ball": [], "slow": [], "dia": []}
        self.sweep_snakes = {}
        self.last_sweep_ms = 0.0
        self.last_window_ms = 0.0

    # ------------------------------------------------------------------ public
    def observe(self, img_full, t=None, force_full=None):
        """-> (WindowSight, snakes, items, flags).

        On a windowed frame `snakes`/`items` are the WINDOW-LOCAL view and the caller is expected to
        use `sight` for steering; on a sweep frame they are the global decode, cached for the slow
        facts.  `sight.mode` says which happened, so nothing has to guess.
        """
        t0 = time.perf_counter()
        self.frame += 1
        reason = force_full or self._full_reason()
        if reason is None:
            sight = self._windowed(img_full)
            snakes = self.sweep_snakes      # the last global decode, for body/path continuity
            items = self.sweep_items
            flags = dict(whiteout=sight.whiteout, mean_val=sight.mean_val,
                         mean_sat=sight.mean_sat)
            self.stats["window"] += 1
            self.last_window_ms = (time.perf_counter() - t0) * 1000.0
        else:
            snakes, items, flags = self.per.run(img_full, t=t)
            self._absorb_sweep(snakes, items)
            sight = WindowSight()
            sight.mode = "full"
            sight.reason = reason
            sight.whiteout = bool(flags.get("whiteout"))
            sight.mean_val = float(flags.get("mean_val", 0.0))
            sight.mean_sat = float(flags.get("mean_sat", 0.0))
            sight.box = window_box(self.head_px)
            red = snakes.get("red")
            if red is not None and red.head_px is not None:
                sight.head_px = red.head_px
                sight.head_cell = red.head_cell
                sight.cells = {(int(c[0]), int(c[1])) for c in red.cells}
                # re-anchor: the sweep decoded THIS frame, so the offset between its head
                # and ours is our accumulated drift, and applying it stops the drift from
                # compounding between sweeps (measured: 10 of 13 head disagreements had the
                # true head outside the window because nothing ever corrected the drift).
                if C.WIN_ANCHOR_TO_SWEEP:
                    self.solver.anchor(red.head_px)
                else:
                    # still seed the position when the solver has none, which is a one-off
                    # initialisation rather than a running correction
                    if self.solver.px is None and red.head_px is not None:
                        self.solver.reset(red.head_px)
            sight.items = {k: list(v) for k, v in items.items()}
            sight.balls = list(items.get("ball", []))
            self.stats["sweep"] += 1
            self.last_sweep_ms = (time.perf_counter() - t0) * 1000.0

        sight.ms = (time.perf_counter() - t0) * 1000.0
        self.sight = sight
        if sight.head_px is not None:
            self.head_px = sight.head_px
            self.head_cell = sight.head_cell
        return sight, snakes, items, flags

    # ------------------------------------------------------------------ internals
    def _full_reason(self):
        if self.head_px is None:
            return "no_head"
        if self.frame - self.last_sweep >= self.sweep_every:
            return "sweep_due"
        return None

    def _absorb_sweep(self, snakes, items):
        """Cache the global decode so a windowed frame still has a body and a path."""
        self.last_sweep = self.frame
        self.sweep_snakes = snakes
        self.sweep_items = items

    def _windowed(self, img_full):
        """The fast path: head, own cells, items and balls inside the 13x13 window."""
        sight = WindowSight()
        sight.mode = "window"
        # centre the window on the PREDICTION, so a frame the eyes cannot read still keeps
        # the true head inside the window for the next readable frame
        box = window_box(self.solver.predicted() or self.head_px)
        sight.box = box
        x0, y0, x1, y1 = box
        crop = img_full[y0:y1, x0:x1]

        wx0, wy0, wx1, wy1 = work_box(box)
        work = self.per.work(crop)
        fam, _hsv, S, V = self.per.families(work)
        sight.mean_val = float(V.mean() / 255.0)
        sight.mean_sat = float(S.mean() / 255.0)
        sight.whiteout = (sight.mean_sat < WHITEOUT_SAT and sight.mean_val > WHITEOUT_VAL)

        # ---- our body and the opponents, from the window's own masks, in board cells.
        # `to_grid` maps 16 px blocks, and the window's work crop starts at (wx0, wy0), so the
        # block indices are board-aligned: cell = block index.  The crop's work width is not a
        # multiple of 16 in general, so the mapping is done from the crop's own origin.
        own = self._cells_from_mask(fam["red"], x0, y0, crop.shape)
        sight.cells = own
        opp = set()
        for k in ("org", "pur", "blu"):
            opp |= self._cells_from_mask(fam[k], x0, y0, crop.shape)
        sight.opponent_cells = opp - own

        # ---- items: reuse the tuned detectors on the crop
        sight.items["dia"] = self._diamonds_in(crop, x0, y0)
        self._small_items(fam, crop, sight, x0, y0)

        # ---- the head, solved inside the window
        px, cell, source = self.solver.solve(img_full, box)
        sight.head_px = px
        sight.head_cell = cell
        sight.reason = source
        return sight

    def _cells_from_mask(self, mask, x0, y0, crop_shape):
        """Boolean work-resolution mask crop -> board cells, via fixed 16 px blocks.

        The global `to_grid` divides the WORK image into 16 px blocks and calls a block occupied at
        30% coverage.  The crop's blocks are the same size and aligned to the same absolute grid
        only when the crop's origin is a multiple of the block size, which it is not in general - so
        the coverage is computed against the ABSOLUTE block grid instead of the crop's.
        """
        out = set()
        if not mask.any():
            return out
        # mask is in work pixels of the crop; its full-resolution origin is (x0, y0)
        ys, xs = np.nonzero(mask)
        # absolute full-resolution pixel coordinates of the occupied pixels
        fx = xs * WORK_SCALE + x0
        fy = ys * WORK_SCALE + y0
        cx = fx // CELL
        cy = fy // CELL
        # coverage: count pixels per absolute cell and compare with 30% of the cell's work area
        block_area = (CELL // WORK_SCALE) ** 2
        thresh = max(1, int(0.30 * block_area))
        counts = {}
        for a, b in zip(cx, cy):
            counts[(int(a), int(b))] = counts.get((int(a), int(b)), 0) + 1
        for (a, b), n in counts.items():
            if n >= thresh and 0 <= a < COLS and 0 <= b < ROWS:
                out.add((a, b))
        return out

    def _diamonds_in(self, crop, x0, y0):
        """`diamonds_full` on the crop, cells translated back to the board."""
        out = []
        for e in self.per.diamonds_full(crop):
            cx = int(e[0]) + x0 // CELL
            cy = int(e[1]) + y0 // CELL
            if 0 <= cx < COLS and 0 <= cy < ROWS:
                out.append((cx, cy) + tuple(e[2:]))
        return out

    def _small_items(self, fam, crop, sight, x0, y0):
        """Beans, the speed item and the slowdown item inside the window."""
        ids = np.zeros(fam["bean"].shape, np.uint8)
        ids[fam["bean"]] = 3
        ids[fam["grn"]] = 2
        if "ball" in fam:
            ids[fam["ball"]] = 1
        n, lab, st, ce = cv2.connectedComponentsWithStats(ids, 8)
        for i in range(1, n):
            a = int(st[i, cv2.CC_STAT_AREA])
            kind = {1: "ball", 2: "grn", 3: "bean"}.get(int(lab[int(ce[i][1]),
                                                                int(ce[i][0])]))
            if kind is None or a < 40:
                continue
            if kind in C.ITEM_MAX_PX_OF and a > C.ITEM_MAX_PX_OF[kind]:
                continue
            cx = (int(ce[i][0]) * WORK_SCALE + x0) // CELL
            cy = (int(ce[i][1]) * WORK_SCALE + y0) // CELL
            if not (0 <= cx < COLS and 0 <= cy < ROWS):
                continue
            if kind == "ball":
                if not (C.BALL_MIN_PX <= a <= C.BALL_MAX_PX):
                    sight.items["slow"].append((cx, cy, a))
                    continue
                sight.items["ball"].append((cx, cy, a))
                sight.balls.append((cx, cy, a))
            else:
                sight.items[kind].append((cx, cy, a))
