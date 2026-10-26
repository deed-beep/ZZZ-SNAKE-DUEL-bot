"""Motion model: what the moving things may do, and what that implies for danger.

The user's rules, which this encodes:

  * EVERY item breathes - its brightness cycles like a breathing lamp - and the fixed diamonds and
    beans also wobble slightly in place.  So an item's measured position and mask area fluctuate
    every frame and NONE of that is motion.
  * a moving gem moves ONLY vertically or horizontally.  Never diagonally.
  * the three spike balls always move one horizontally, one vertically, and one at 45 degrees -
    and the 45-degree one is the most dangerous, because it crosses the other two's lanes and its
    threat cone is widest.

The point of writing the constraints down is that they are strong enough to FILTER noise: a
one-frame "movement" of a bean or a fixed gem is the breathing/wobble, not travel; a gem that
appears to move diagonally is a misreading, because the game does not move gems that way; and a
ball whose track is inconsistent with any of the three modes is suspect.

Threat model for the diagonal ball, which is the part with real consequences: a horizontal or
vertical ball sweeps one row or column, so a snake can wait one cell off that line and be safe.  A
45-degree ball sweeps a diagonal, so its reach is not a line in cell space - and it closes on a
snake that thinks it is safe.  So the model ranks it highest and, when a choice exists, steers
away from the diagonal ball's lane first.
"""
from __future__ import annotations

import math
from collections import deque

from config import (CELL, COLS, ROWS, BALL_SPEED_CPS, GAME_HZ,
                    BALL_DIAG_THREAT, BALL_AXIS_THREAT,
                    ITEM_MOVE_TOL_CELLS, MOTION_MIN_CPS, WOBBLE_TOL_CELLS, BALL_STILL_THREAT)


class Track:
    """Position history for one moving thing, with a motion mode fitted to it."""

    def __init__(self, cell, kind, maxlen=10):
        self.kind = kind
        self.cells = deque([(int(cell[0]), int(cell[1]))], maxlen=maxlen)
        self.frames = deque([0], maxlen=maxlen)
        self.mode = "unknown"          # horizontal | vertical | diagonal | still | unknown
        self.threat = 0.0
        self._has_observation = False

    def observe(self, cell, frame):
        if not self._has_observation:
            # Constructor's placeholder is not a sighting at match time zero.
            self.cells.clear()
            self.frames.clear()
            self._has_observation = True
        if self.frames and frame == self.frames[-1]:
            self.cells[-1] = (int(cell[0]),int(cell[1]))
        else:
            self.cells.append((int(cell[0]), int(cell[1])))
            self.frames.append(frame)
        self._fit()

    def _fit(self):
        """Classify the recent motion, using the game's constraint on what is possible."""
        if len(self.cells) < 2:
            return
        # use the whole visible track, not just the last step: one step is inside the wobble
        d = (self.cells[-1][0] - self.cells[0][0], self.cells[-1][1] - self.cells[0][1])
        span = max(1, self.frames[-1] - self.frames[0])
        cps = math.hypot(*d) / span * GAME_HZ
        if abs(d[0]) == 0 and abs(d[1]) == 0:
            self.mode = "still"
            self.threat = 0.0
            return
        if cps < MOTION_MIN_CPS:
            # below the noise floor: this is the breathing / wobble, not travel
            self.mode = "still"
            self.threat = 0.0
            return
        ax, ay = abs(d[0]), abs(d[1])
        if ay == 0:
            self.mode = "horizontal"
        elif ax == 0:
            self.mode = "vertical"
        else:
            # both axes moved.  For a BALL that is the 45-degree case (and the most dangerous);
            # for a GEM it contradicts the rules, so it is reported as a misreading.
            ratio = min(ax, ay) / float(max(ax, ay))
            if ratio >= 0.6:
                self.mode = "diagonal"
            else:
                self.mode = "axis-noisy"
        # A STILL ball keeps a real threat.  It used to be 0.0, and `danger()` skips tracks with
        # `threat <= 0`, so a ball that had not moved yet was invisible to the forecast - measured, on 16 of
        # the 42 frames where a ball was within 3 cells of the head the forecast was empty, including frames
        # with the ball at distance 0.  Contact kills with no buffer, so a still ball is as lethal as a
        # moving one; the only thing it lacks is a prediction of where it will be.
        self.threat = {"diagonal": BALL_DIAG_THREAT,
                       "horizontal": BALL_AXIS_THREAT,
                       "vertical": BALL_AXIS_THREAT,
                       "axis-noisy": BALL_AXIS_THREAT * 0.5,
                       "unknown": 0.0,
                       "still": BALL_STILL_THREAT}.get(self.mode, 0.0)

    def velocity(self):
        """cells per frame, from the fitted track."""
        if len(self.cells) < 2:
            return (0.0, 0.0)
        span = max(1, self.frames[-1] - self.frames[0])
        return ((self.cells[-1][0] - self.cells[0][0]) / span,
                (self.cells[-1][1] - self.cells[0][1]) / span)

    def ahead(self, steps=1):
        """Where this thing will be in `steps` frames, snapped to cells."""
        if not self.cells:
            return None
        vx, vy = self.velocity()
        c = self.cells[-1]
        return (int(round(c[0] + vx * steps)), int(round(c[1] + vy * steps)))

    def swept(self, steps=8):
        """The cells this thing will sweep over the next `steps` frames.

        A horizontal or vertical mover sweeps a line; a diagonal mover sweeps a diagonal, which
        is why it is worth far more attention - a snake one cell off a horizontal ball's row is
        safe, but one cell off a diagonal ball's path is not.
        """
        out = set()
        vx, vy = self.velocity()
        if vx == 0 and vy == 0:
            return out
        c = self.cells[-1]
        for k in range(1, steps + 1):
            cx = int(round(c[0] + vx * k))
            cy = int(round(c[1] + vy * k))
            if 0 <= cx < COLS and 0 <= cy < ROWS:
                out.add((cx, cy))
        return out


class MotionModel:
    """Keeps a track per moving object and answers "what is dangerous next".

    Balls are matched to their previous positions greedily by nearest cell, which is enough
    because they drift slowly (0.7 cells/s, measured, i.e. well under a cell per frame) and
    because a wrong match only lasts until the next frame re-fits the track.
    """

    def __init__(self):
        self.balls = []            # list[Track]
        self.ball_frames = 0
        self.notes = []

    # ------------------------------------------------------------------ balls
    def update_balls(self, items, frame):
        from config import BALL_CELLS
        cells = []
        for e in items.get("ball", []):
            # Perception returns the centroid cell ALREADY, not top-left.
            cells.append((int(e[0]), int(e[1])))
        self.ball_frames = frame
        used = set()
        for t in self.balls:
            if not t.cells:
                continue
            last = t.cells[-1]
            best, bestd = None, 1e9
            for i, c in enumerate(cells):
                if i in used:
                    continue
                d = max(abs(c[0] - last[0]), abs(c[1] - last[1]))
                if d < bestd:
                    best, bestd = i, d
            if best is not None and bestd <= 3:
                t.observe(cells[best], frame)
                used.add(best)
        for i, c in enumerate(cells):
            if i not in used:
                self.balls.append(Track(c, "ball"))
                self.balls[-1].observe(c, frame)
        # drop stale tracks
        self.balls = [t for t in self.balls
                      if t.frames and frame - t.frames[-1] <= 12]
        self.notes = [t.mode for t in self.balls]

    def ball_modes(self):
        return [t.mode for t in self.balls]

    def check_ball_modes(self):
        """The user's rule: among the three balls there is one of each mode.

        Returns (ok, message).  It is a diagnostic, not a filter: the modes are fitted from
        noisy tracks, so a mismatch is worth logging but not worth rejecting detections over.
        """
        modes = [m for m in self.ball_modes() if m not in ("unknown", "still", "axis-noisy")]
        if len(modes) < 3:
            return True, "fewer than three moving balls fitted"
        has_h = modes.count("horizontal")
        has_v = modes.count("vertical")
        has_d = modes.count("diagonal")
        ok = has_h >= 1 and has_v >= 1 and has_d >= 1
        return ok, f"horizontal {has_h}, vertical {has_v}, diagonal {has_d}"

    def danger(self, head, steps=8):
        """Cells in the head's neighbourhood that a ball will sweep, ranked by threat.

        Returns [(cell, threat, mode)].  The diagonal ball is ranked highest because its sweep
        is not a single row or column: a snake can stand off a horizontal ball's lane and be
        safe, but the diagonal's lane cuts across the space it would retreat into.
        """
        from config import BALL_DANGER_MARGIN
        out = {}
        m = max(0, int(BALL_DANGER_MARGIN))
        for t in self.balls:
            if t.threat <= 0:
                continue
            for c in t.swept(steps):
                if head is not None:
                    if max(abs(c[0] - head[0]), abs(c[1] - head[1])) > steps:
                        continue
                # Inflate by the margin: the ball is a 2x2 object that also drifts between vision
                # samples, so the swept line is not the collision zone.  Without this the forecast
                # listed nothing at a Chebyshev distance of 3, where the ball's body is one cell away.
                for dx in range(-m, m + 1):
                    for dy in range(-m, m + 1):
                        n = (c[0] + dx, c[1] + dy)
                        if not (0 <= n[0] < COLS and 0 <= n[1] < ROWS):
                            continue
                        # the margin ring is a little less certain than the swept line itself
                        w = t.threat if (dx == 0 and dy == 0) else t.threat * 0.6
                        if w > out.get(n, (0.0, ""))[0]:
                            out[n] = (w, t.mode)
        return [(c, v[0], v[1]) for c, v in
                sorted(out.items(), key=lambda kv: -kv[1][0])]

    def avoid_hint(self, head, steps=8):
        """The single cell the policy should treat as most-to-avoid, or None.

        Used to bias the choice of step: the diagonal ball's lane first, then the axis balls.
        """
        d = self.danger(head, steps)
        return d[0] if d else None


# ============================================================ stationary items
class StillnessFilter:
    """Pins items the game does not move, so a breathing highlight cannot walk the target.

    Beans, the speed item and the slowdown item never move at all (user-stated), and a fixed gem
    only wobbles in place.  Their mask position therefore jitters by up to WOBBLE_TOL_CELLS as
    the brightness cycles, and if that jitter is taken at face value the policy's chosen target
    moves with it - the head chases a cell the item is no longer drawn on.

    So for those kinds the reported cell is the one where the item has been SEEN MOST OFTEN
    recently, not the one measured this frame.  That is the same "trust the record, not the
    single frame" principle that the diamonds needed, applied to the kinds where motion is
    impossible by the rules.
    """

    KINDS = ("bean", "slow")

    def __init__(self):
        # kind -> cell -> list of recent frames it was reported at
        self.hist = {k: {} for k in self.KINDS}

    def observe(self, items, frame):
        """Record this frame's readings and return the pinned position per item."""
        out = {}
        for kind in self.KINDS:
            hist = self.hist[kind]
            seen_here = set()
            for e in items.get(kind, []):
                c = (int(e[0]), int(e[1]))
                seen_here.add(c)
                hist.setdefault(c, []).append(frame)
            # age out
            for c in list(hist):
                hist[c] = [fr for fr in hist[c] if frame - fr <= 20]
                if not hist[c]:
                    del hist[c]
            out[kind] = seen_here
        return out

    def pinned_cell(self, kind, cell, frame):
        """Where this item really is: the cell with the most sightings near `cell`.

        A gem or bean wobbling between two adjacent cells has sightings on both; the more
        frequent one is where it is actually drawn.
        """
        hist = self.hist.get(kind, {})
        best, bestn = (int(cell[0]), int(cell[1])), -1
        for c, frames in hist.items():
            if max(abs(c[0] - cell[0]), abs(c[1] - cell[1])) > WOBBLE_TOL_CELLS:
                continue
            n = len(frames)
            if n > bestn or (n == bestn and c < best):
                best, bestn = c, n
        return best


def classify_gem_motion(track, kind="gem"):
    """Is this gem's apparent motion legal?

    A moving gem may travel ONLY vertically or horizontally (user-stated).  So a gem whose fitted
    track has both axes moving is a misreading - a wobble, two different gems being treated as
    one, or the breathing highlight - and it must not be handed to the policy as a moving target.
    """
    if track.mode in ("horizontal", "vertical", "still", "unknown"):
        return True, track.mode
    if track.mode == "diagonal":
        return False, "diagonal is impossible for a gem"
    return False, f"both axes moving ({track.mode}) is impossible for a gem"
