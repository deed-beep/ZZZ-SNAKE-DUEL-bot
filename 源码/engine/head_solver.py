"""Find the head INSIDE the window: eyes first, motion as the fallback.

The first version of the window path had no head finder at all - it carried over the head from the
last global sweep, which `validate_window.py` exposed as a contradiction: "median error 0 but only
12% exact", because 221 of 251 frames simply had no window head and the 0 error came from reusing
the previous sweep's answer.

This adds the real solver.  Two signals, in the order the game makes them reliable:

 1. **THE EYES.**  The red snake is the only one with two white slits, so an eye pair found inside
    the window identifies the head without needing the body mask at all.  `Perception` already has
    the tuned detector for this (`white_slits` + `eye_pair_candidates`), and both take an arbitrary
    image - checked: neither references the frame size - so they run correctly on the window crop.
    The one catch is that the pair's midpoint comes back in CROP coordinates, so the crop origin has
    to be added; that is done in one place (`_to_board`).

 2. **MOTION, as the fallback.**  When the eyes are not readable (the acceleration flash washes them
    out, or the head is turned away) the head is predicted from its own recent travel.  This is safe
    here in a way it is not globally: the window is centred on that prediction, so a prediction that
    is wrong by a cell or two still keeps the true head inside the window, and the next readable
    frame corrects it.  The user's stated limit - the head moves at most about 2 cells per frame -
    bounds how far the prediction can be off after one unreadable frame.
"""
from __future__ import annotations

import math

import numpy as np

import config as C
from config import CELL
from perception import _EyePair


class HeadSolver:
    """Locate the red head inside a window, from the eyes or from its own motion."""

    def __init__(self, perception, alpha=None):
        self.per = perception
        self.alpha = 0.5 if alpha is None else alpha
        self.px = None            # last solved head, full-resolution pixels
        self.vel = (0.0, 0.0)     # px per frame, smoothed
        self.source = "none"
        self.miss = 0
        self.hits = 0
        self.eye_frames = 0
        self.motion_frames = 0
        self.streak = 0          # consecutive frames the EYES were read
        self.streak_cell = None  # the cell those frames agreed on
        self.anchor_err = (0.0, 0.0)   # sweep head minus solver head at the last sweep
        self.anchor_used = 0

    def reset(self, px=None):
        self.px = px
        self.vel = (0.0, 0.0)
        self.source = "none"
        self.miss = 0
        self.streak = 0
        self.streak_cell = None
        self.anchor_err = (0.0, 0.0)

    # ------------------------------------------------------------------ prediction
    def predicted(self, ahead=1.0):
        """Where the head should be, with the last sweep's correction applied.

        `anchor_err` is (sweep head - solver head) measured on the most recent sweep frame, i.e. the
        solver's accumulated error at that moment.  Adding it drags the window back onto the true head
        every SWEEP_EVERY frames, which is what stops the drift from compounding over a whole match.
        """
        if self.px is None:
            return None
        ax, ay = self.anchor_err
        return (self.px[0] + self.vel[0] * ahead + ax,
                self.px[1] + self.vel[1] * ahead + ay)

    def anchor(self, sweep_head_px):
        """Record the offset between the sweep's head and ours, from the SAME frame."""
        if sweep_head_px is None:
            return
        if self.px is None:
            self.px = (float(sweep_head_px[0]), float(sweep_head_px[1]))
            self.anchor_err = (0.0, 0.0)
            return
        # the sweep decoded this same frame, so the difference is our error, not motion - but the
        # sweep is itself a decode and can be wrong, so the correction is clamped.  Without the clamp
        # a single bad sweep injects an offset that compounds through the prediction.
        ax = float(sweep_head_px[0]) - self.px[0]
        ay = float(sweep_head_px[1]) - self.px[1]
        lim = C.WIN_ANCHOR_MAX_PX
        self.anchor_err = (max(-lim, min(lim, ax)), max(-lim, min(lim, ay)))
        self.anchor_used += 1

    # ------------------------------------------------------------------ per frame
    def _snap_check(self):
        """Has the velocity drifted away from a place the eyes keep confirming?"""
        if self.px is None or self.streak < C.WIN_SOLVER_SNAP_N or self.streak_cell is None:
            return None
        # distance from the PUBLISHED head to the repeated observation.  The published head is the
        # thing that can lag; the streak is the repeated evidence about where the head is.  (The
        # earlier version measured the inverse and therefore read ~0 exactly when the head had fallen
        # behind, so it never fired: measured 0 snaps over 251 frames with streaks up to 6.)
        gap = max(abs(self.px[0] - (self.streak_cell[0] * CELL + CELL / 2.0)),
                  abs(self.px[1] - (self.streak_cell[1] * CELL + CELL / 2.0)))
        if gap <= C.WIN_SOLVER_SNAP_CELLS * CELL:
            return None
        # snap: put the head where the eyes keep saying it is, and rebuild the velocity from the
        # streak direction rather than from the drift
        cx = self.streak_cell[0] * CELL + CELL / 2.0
        cy = self.streak_cell[1] * CELL + CELL / 2.0
        return (cx, cy)

    def solve(self, img_full, box):
        """-> (head_px, head_cell, source) in BOARD coordinates, or (None, None, 'none').

        `box` is the window in full-resolution pixels.  The eyes are searched inside it; the motion
        prediction is centred by the caller, which builds `box` from `predicted()`.
        """
        x0, y0, x1, y1 = box
        crop = img_full[y0:y1, x0:x1]
        head = None
        snap = self._snap_check()
        if snap is not None:
            # the streak wins: the velocity had drifted, so follow the repeated observation and
            # let the caller re-centre the window on it next frame
            self.px = snap
            self.vel = (0.0, 0.0)
            self.source = "snap"
            self.streak = 0
            self.streak_cell = None
            return self.px, (int(snap[0] // CELL), int(snap[1] // CELL)), "snap"

        # ---- 1. the eyes, inside the window only
        try:
            slits = self.per.white_slits(crop)
            pairs = (self.per.eye_pair_candidates(slits, crop,
                                                 self._warm_in(crop))
                     if len(slits) >= 2 else [])
        except Exception:
            slits, pairs = [], []
        if pairs:
            best = pairs[0]
            mid = best[1]
            head = (float(mid[0]) + x0, float(mid[1]) + y0)
            self.eye_frames += 1
            self.source = "eyes"
        elif slits:
            # A lone slit is much weaker evidence than a pair: the pair search has validated spacing,
            # elongation and a ring of surrounding body, whereas a lone slit is only "a bright neutral
            # blob".  The gate is TEMPORAL, from the user's stated mechanics - the head moves at most
            # about 2 cells per frame - so a lone blob many cells from the previous head is not the
            # head however bright it looks.  That is the measured failure being fixed: on luping3
            # f216/f220 a diamond's sparkle 25 cells away was taken as the head.
            #
            # The previous gate tested proximity to our own body mask instead, and FAILED: a 2-cell
            # radius around a point on a board where our body is usually within 2 cells made it almost
            # always true (exact rate 94.0% -> 93.6%), while costing an extra families() call a frame.
            cand = slits[0]
            cxp = float(cand["c"][0]) + x0
            cyp = float(cand["c"][1]) + y0
            near = True
            if self.px is not None:
                d = max(abs(cxp - self.px[0]), abs(cyp - self.px[1])) / CELL
                near = d <= C.WIN_SLIT_MAX_STEP_CELLS
            if near:
                head = (cxp, cyp)
                self.eye_frames += 1
                self.source = "slit"
            else:
                self.source = "slit-too-far"

        # ---- absolute sanity FIRST: a head has to be on the board.  Every earlier guard was
        # relative ("far from the previous head", "near our body") and a runaway satisfies all of
        # them, because once the position is millions of cells out the next bogus midpoint near it
        # looks "close".  Measured on luping3 f248+ the solved head reached cell 539,779,889.
        if head is not None:
            if not (0.0 <= head[0] < C.FRAME_W and 0.0 <= head[1] < C.FRAME_H):
                self.source = "off-board"
                head = None

        if head is not None and self.px is not None:
            step = max(abs(head[0] - self.px[0]), abs(head[1] - self.px[1]))
            if step > C.WIN_HEAD_MAX_STEP_PX:
                # a jump this large is the detector latching onto something else, whatever the
                # source: a pair out of nowhere is as impossible as a slit out of nowhere
                head = None
                self.source = "rejected"

        # ---- 2. motion, when the eyes were not usable
        if head is None:
            pred = self.predicted()
            if pred is None:
                self.miss += 1
                return None, None, "none"
            head = (float(pred[0]), float(pred[1]))
            self.motion_frames += 1
            self.miss += 1
            if self.source != "rejected":
                self.source = "motion"
        else:
            self.hits += 1

        # ---- 2b. maintain the eye streak that authorises a snap-back
        if self.source in ("eyes", "slit"):
            cell_now = (int(head[0] // CELL), int(head[1] // CELL))
            if self.streak_cell is not None and cell_now == self.streak_cell:
                self.streak += 1
            else:
                self.streak_cell = cell_now
                self.streak = 1
        else:
            self.streak = 0
            self.streak_cell = None

        # ---- 3. smooth the velocity, so the prediction tracks turns without overshooting
        if self.px is not None:
            vx = head[0] - self.px[0]
            vy = head[1] - self.px[1]
            if math.hypot(vx, vy) <= 3.0 * CELL:
                a = self.alpha
                self.vel = (a * vx + (1 - a) * self.vel[0],
                            a * vy + (1 - a) * self.vel[1])
        self.px = (float(head[0]), float(head[1]))
        cell = (int(head[0] // CELL), int(head[1] // CELL))
        return self.px, cell, self.source

    # ------------------------------------------------------------------ helpers
    def _warm_in(self, crop):
        """The warm (red-or-own-body) mask for the crop, which the pair ring test needs.

        The ring test asks whether a candidate pair is surrounded by OUR body, so the mask has to
        cover both families the player's body spans: the body's hue slides from red at the head to
        orange at the tail, so a warm mask is red OR orange.
        """
        try:
            fam, _h, _s, _v = self.per.families(self.per.work(crop))
            return self.per.clean(fam["red"]) | self.per.clean(fam["org"])
        except Exception:
            return None
