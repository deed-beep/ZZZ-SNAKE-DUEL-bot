"""The vision3 temporal layer: a persistent high-priority head window plus length
continuity.

Both come straight from the user's frame report:

  * "建议给红蛇蛇头增加高优先级的7乘7的小窗口进行持续追踪，当其余识别出错时确保红蛇蛇头
    位置且确保附近的障碍与生路" - keep a 7x7 CELL window on the head, predicted from
    the previous frames, so the head position survives frames where the wide segmentation
    fails (the long snake glued to itself, the covered head).

  * "蛇的长度不会突变，吃到黄豆和钻石后会在接下来几帧边长1格或5格左右，按键加速时会缓慢
    从蛇尾掉黄豆减少蛇的长度，仅在死亡时会快速消融" - length is a physics quantity:
    +1 per bean, +5 per diamond, gradual loss while the key is held, and only DEATH makes
    it collapse.  So a single-frame jump is a measurement error, not an event.

Design: this sits ON TOP of Perception and only corrects its output; it never invents a
body.  Everything it decides is recorded in `last` so a disagreement can be traced.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import cv2

from config import (CELL, COLS, ROWS, HEAD_WIN_CELLS,
                    HEAD_WIN_WEIGHT, LEN_MAX_JUMP, LEN_MAX_DROP_PER_FRAME, LEN_REJECT_MAX,
                    HEAD_REJECT_MAX, ITEM_TOL_FRAMES, DIA_RING_TRUST, BEAN_MIN_TRUST_PX,
                    DIA_PERSIST_FRAMES, DIA_CONFIRM_N, DIA_CONFIRM_M,
                    DIA_RING_STRONG, DEATH_AREA_FRAC, DEATH_MIN_PX, DEATH_CONFIRM,
                    EYE_EXTEND_MAX, EYE_STABLE_N, EYE_STABLE_M, DEATH_AREA_FRAC,
                    DEATH_MIN_PX, DEATH_DECLARE)


class HeadWindow:
    """A small full-resolution window kept on a snake's predicted head.

    The window is the only thing the bot looks at when the wide decode fails, so it has
    to be both cheap and independent: it re-runs the eye test and a small warm-blob
    search inside 7x7 cells and nothing else.
    """

    def __init__(self, family, head_px=None, facing=None):
        self.family = family
        self.head_px = head_px            # (x, y) full resolution, last confirmed
        self.facing = facing              # (dx, dy) unit-ish cell step
        self.vel = (0.0, 0.0)             # px per frame, smoothed
        self.miss = 0                     # consecutive frames without confirmation

    # ------------------------------------------------------------------ predict
    def predict(self, dt_frames=1.0):
        if self.head_px is None:
            return None
        x = self.head_px[0] + self.vel[0] * dt_frames
        y = self.head_px[1] + self.vel[1] * dt_frames
        return (x, y)

    def observe(self, head_px, alpha=0.5):
        """Fold a confirmed head position in, updating the velocity estimate."""
        if self.head_px is not None and head_px is not None:
            vx = head_px[0] - self.head_px[0]
            vy = head_px[1] - self.head_px[1]
            # reject absurd motion (a swap to the other end of the snake shows up as a
            # jump the length of the body) instead of letting it poison the estimate
            if math.hypot(vx, vy) <= 3.0 * CELL:
                self.vel = (alpha * vx + (1 - alpha) * self.vel[0],
                            alpha * vy + (1 - alpha) * self.vel[1])
        self.head_px = head_px
        self.miss = 0

    def window(self, shape):
        """Pixel box for the 7x7 cell search around the predicted head, or None."""
        p = self.predict()
        if p is None:
            return None
        half = (HEAD_WIN_CELLS * CELL) // 2
        h, w = shape[:2]
        x0 = int(max(0, min(w - 1, p[0] - half)))
        y0 = int(max(0, min(h - 1, p[1] - half)))
        x1 = int(max(x0 + 1, min(w, p[0] + half)))
        y1 = int(max(y0 + 1, min(h, p[1] + half)))
        return (x0, y0, x1, y1)


class TemporalSense:
    """Corrects Perception's per-frame output using the physics of the game.

    Nothing here is allowed to create a snake that was not seen; it can only choose
    between what perception reported, or fall back to the head window's own reading.
    """

    def __init__(self):
        self.win = {k: HeadWindow(k) for k in ("red", "org", "pur", "blu")}
        self.len_hist = {k: deque(maxlen=LEN_MAX_JUMP + 1)
                         for k in ("red", "org", "pur", "blu")}
        self.reject = {k: 0 for k in ("red", "org", "pur", "blu")}
        self.death = {k: 0 for k in ("red", "org", "pur", "blu")}
        self.gone = {}
        self.px_hist = {k: deque(maxlen=5)
                        for k in ("red", "org", "pur", "blu")}
        # what the snake typically IS - the reference the death test compares against
        self.px_long = {k: deque(maxlen=60)
                        for k in ("red", "org", "pur", "blu")}
        self.frame = 0
        self.item_hist = {"dia": [], "bean": []}
        self.dia_confirmed = []
        self.eye_hist = {}          # family -> list of recent DECODED states
        self.eye_gap = {}           # family -> consecutive frames without a state
        self.eye_extended = {}      # family -> frames filled by extension
        self.last = {}

    # ------------------------------------------------------------------ length
    @staticmethod
    def length_ok(prev_len, new_len, alive_before):
        """Is this length a plausible continuation of the previous one?

        +1 bean / +5 diamond are instant, and the key-held tail drops at roughly one bean
        per 5.7 s (so at most a couple of cells between two frames).  A large jump is
        therefore a misread, unless the snake died - and death is the caller's business
        (it clears the history instead of asking this).
        """
        if not alive_before or prev_len <= 0:
            return True
        d = new_len - prev_len
        if d >= 0:
            return d <= LEN_MAX_JUMP
        return (-d) <= LEN_MAX_DROP_PER_FRAME

    def _reorient_from_window(self, k, s):
        """If the head is not eye-confirmed, put it at the end inside the window.

        Returns True when the path was reversed.  This is the whole point of keeping the
        window: without it a body that lies against itself makes `grid_path`'s two ends
        indistinguishable and the published head alternates between them (measured on
        luping1 f580-f598: a 19-cell swing per frame).
        """
        w = self.win[k]
        if (w.head_px is None or s.head_from_eyes or getattr(s,"head_from_pixels",False)
                or getattr(s,"path_trust","absent") != "ordered" or len(s.path)<2):
            return False
        pred = w.predict()
        if pred is None:
            return False

        def d2(c):
            return ((c[0] * CELL + CELL / 2.0 - pred[0]) ** 2 +
                    (c[1] * CELL + CELL / 2.0 - pred[1]) ** 2)

        if d2(s.path[-1]) < d2(s.path[0]):
            s.path.reverse()
            s.head_cell, s.tail_cell = s.path[0], s.path[-1]
            s.head_px = (s.head_cell[0]*CELL+CELL/2,s.head_cell[1]*CELL+CELL/2)
            s.head_age = max(getattr(s,"head_age",0),1)
            s.facing = (s.path[0][0] - s.path[1][0], s.path[0][1] - s.path[1][1])
            return True
        return False

    def update(self, snakes, items, flags, whiteout):
        """-> (corrected snakes, notes).  Mutates the SnakeObs in place where needed."""
        self.frame += 1
        notes = {}
        for k, s in snakes.items():
            w = self.win[k]
            # ---- the area history is recorded for EVERY frame the family is present, before
            # any head reasoning: it is the evidence the death test runs on, and the earlier
            # version only appended it in one branch, so the list stayed empty and the
            # detector could never fire (measured: purple on luping1 had gone=12 and an empty
            # px_hist at f640).
            if s is not None and s.n_px > 0:
                self.px_hist[k].append(s.n_px)
                self.px_long[k].append(s.n_px)
            if s is None or s.head_cell is None:
                w.miss += 1
                # A snake whose head cannot be resolved for many consecutive frames is dead.
                # This is the purple case: measured on luping1 its mask decays
                # 1588 -> 1250 -> 952 -> 448 -> 170 through f635-f639 and is then absent for
                # 60 frames, and through all of it the head is unresolvable.  The collapse
                # test below cannot see that, because it needs a published head to run.
                self.gone[k] = self.gone.get(k, 0) + 1
                if (DEATH_DECLARE and self.gone[k] >= DEATH_CONFIRM
                        and len(self.px_long[k]) >= 5):
                    # compare the CURRENT area against what the snake typically is; an
                    # earlier version compared the recent median against itself, so the
                    # condition was unsatisfiable (measured: purple's f638 asked 952 < 333)
                    cur = float(s.n_px) if s is not None else 0.0
                    typical = float(np.median(list(self.px_long[k])))
                    if (typical >= DEATH_MIN_PX and
                            cur < DEATH_AREA_FRAC * typical):
                        notes[k] = (f"dead (no head {self.gone[k]} frames, "
                                    f"px {cur:.0f} vs typical {typical:.0f})")
                        self.px_hist[k].clear()
                        self.len_hist[k].clear()
                        self.win[k].head_px = None
                        self.win[k].vel = (0.0, 0.0)
                        self.reject[k] = 0
                        self.gone[k] = 0
                continue
            self.gone[k] = 0
            # ---- orientation first: the window decides which end is the head when the
            # eyes could not, BEFORE the jump test below judges the resulting position
            if self._reorient_from_window(k, s):
                notes[k] = "head-from-window"
            head_px = s.head_px
            if head_px is None:
                head_px = (s.head_cell[0] * CELL + CELL / 2.0,
                           s.head_cell[1] * CELL + CELL / 2.0)
            # A head that jumped the length of the snake is a swap, not motion.  Same
            # resync guard as the length: after HEAD_REJECT_MAX consecutive rejections the
            # measurement wins, because a window whose prediction has stopped matching
            # reality is worse than no window at all.  Measured on luping1 f186-f200 the
            # window sat at (18,1) while the snake was moving, so every frame was
            # "rejected" and the published head was frozen in place.
            if w.head_px is not None:
                jump = math.hypot(head_px[0] - w.head_px[0], head_px[1] - w.head_px[1])
                if jump > 4.0 * CELL and not whiteout:
                    # A jump is only a SWAP if the snake's own other end is the one that
                    # is near the prediction.  If both ends are far away the snake really
                    # moved (or the window is stale), and rejecting would freeze the head:
                    # measured on luping1 f580 the eyes were lost, the body lay against
                    # itself, and the window - anchored at the old head - rejected every
                    # frame for 19 frames while the snake travelled.
                    other_far = True
                    if len(s.path) >= 2:
                        tail = s.path[-1]
                        dt = math.hypot(tail[0] * CELL + CELL / 2.0 - w.head_px[0],
                                        tail[1] * CELL + CELL / 2.0 - w.head_px[1])
                        other_far = dt > 4.0 * CELL
                    if (w.miss < HEAD_REJECT_MAX and not other_far
                            and not s.head_from_eyes and not getattr(s,"head_from_pixels",False)
                            and getattr(s,"path_trust","absent") == "ordered"):
                        notes[k] = "head-jump-rejected"
                        s.head_cell = (int(w.head_px[0] // CELL),
                                       int(w.head_px[1] // CELL))
                        s.head_px = w.head_px
                        s.head_age = w.miss + 1
                        w.miss += 1
                        continue
                    notes[k] = "head-resync"
                    w.miss = 0
            w.observe(head_px)
            # ---- length continuity.  A rejection must not be able to latch: after
            # LEN_REJECT_MAX consecutive disagreements the MEASUREMENT is accepted as the
            # new reference, because a persistent reading is far more likely to be the
            # truth than a 40-frame-old memory.  Measured on luping1 f83-f110 the
            # reference stuck at 17-19 while the snake was really 11-15, so every frame
            # after the first was "corrected" against a wrong number.
            hist = self.len_hist[k]
            # ---- DEATH: a snake whose body has largely gone is dissolving, not moving.
            # The user reports the orange snake in its death effect over f629-f712, and the
            # measurement shows why this needs saying explicitly: on luping1 the orange area
            # oscillates 12231 <-> 1952 px and its cell count 51 <-> 9 between consecutive
            # frames, so the continuity rule kept re-adopting a fragment and the length test
            # fired on every frame.  A collapse below DEATH_AREA_FRAC of the recent typical
            # length is not a measurement, it is the end of a life: publish the small body
            # for what it is, forget the history, and let the caller treat it as gone.
            px_hist = self.px_hist[k]
            px_typ = (float(np.median(list(self.px_long[k])))
                      if len(self.px_long[k]) >= 5 else 0.0)
            collapsed = (px_typ > DEATH_MIN_PX and
                         s.n_px < DEATH_AREA_FRAC * px_typ)
            if collapsed and DEATH_DECLARE:
                self.death[k] += 1
                if self.death[k] >= DEATH_CONFIRM:
                    # SUSTAINED collapse: a fragment recovers within a frame or two, a
                    # dying snake does not (see DEATH_AREA_FRAC for both measurements)
                    notes[k] = f"dead (px {s.n_px} << typical {px_typ:.0f})"
                    px_hist.clear()
                    self.px_long[k].clear()
                    hist.clear()
                    self.reject[k] = 0
                    self.win[k].head_px = None
                    self.win[k].vel = (0.0, 0.0)
                    self.win[k].miss = 0
                    continue
            else:
                self.death[k] = 0
            prev_len = hist[-1] if hist else 0
            if self.reject[k] >= LEN_REJECT_MAX or self.length_ok(prev_len, s.length,
                                                                  bool(hist)):
                if self.reject[k]:
                    notes[k] = f"length-resync {prev_len}->{s.length}"
                self.reject[k] = 0
                hist.append(s.length)
            else:
                notes[k] = f"length-jump {prev_len}->{s.length}"
                self.reject[k] += 1
                hist.append(prev_len)
        # ---- extend the eye state across short unreadable gaps (red only: it is the one
        # snake that carries the marker).  Done last so the extension sees this frame's
        # decode and the stability history is built from DECODED frames only.
        red = snakes.get("red")
        if red is not None:
            state, ext = self.eye_state("red", red)
            if state is not None and state != getattr(red, "eye_state", None):
                red.eye_state = state
            red.eye_extrapolated = bool(ext)
            if not ext and state is None:
                red.eye_extrapolated = False
        # Reorientation/rejection may change head without changing body. Recheck
        # trust AFTER those mutations; an old 'ordered' flag is not evidence.
        from perception import Perception
        for s in snakes.values():
            s.path_trust = Perception.path_trust(s.path, s.head_cell, s.cells)
        self.last = notes
        return snakes, notes

    # ------------------------------------------------------------------ eye state
    def eye_state(self, family, obs):
        """-> (state, extrapolated?) for a snake, extending across short gaps.

        Only DECODED states enter the history; only a stable decoded state is extended, and
        only across EYE_EXTEND_MAX frames.  When nothing can be extended the honest answer
        is None, which the policy already treats as "unknown" - inventing a state would be
        worse than admitting the eyes are not readable.
        """
        hist = self.eye_hist.setdefault(family, [])
        gap = self.eye_gap.get(family, 0)
        state = getattr(obs, "eye_state", None)
        if state is not None:
            hist.append(state)
            del hist[:-EYE_STABLE_M]
            self.eye_gap[family] = 0
            self.eye_extended[family] = 0
            return state, False
        # undecoded: extend, if allowed
        self.eye_gap[family] = gap + 1
        if not hist or gap + 1 > EYE_EXTEND_MAX:
            return None, False
        last = hist[-1]
        if hist.count(last) < EYE_STABLE_N:
            return None, False
        self.eye_extended[family] = self.eye_extended.get(family, 0) + 1
        return last, True

    # ------------------------------------------------------------------ items
    def _confirmed(self, kind, cell):
        """Was this kind seen at this cell within ITEM_TOL_FRAMES?"""
        tol = ITEM_TOL_FRAMES
        for (f, c) in reversed(self.item_hist[kind]):
            if self.frame - f > tol:
                break
            if c == cell:
                return True
        return False

    def filter_items(self, items):
        """-> (items, dropped).  Drop item flicker the mechanics do not support.

        Only BEANS and the small categories are filtered here.  Gems used to be filtered here too,
        with a "strong ring this frame, or seen before" test whose threshold `DIA_CONFIRM_N` was
        configured to 0 - and `seen_count >= 0` is always true, so the weak-edge branch accepted every
        candidate outright and the confirmation path could never run.  The comment claimed the
        opposite of what the code did, which is how a filter comes to do nothing while looking strict.

        Rather than pick a new threshold by guess - the user was explicit that one digit is not a fix -
        the branch is REMOVED.  Gems now have a single authoritative gate: the 7x7 window test plus the
        fixed-pair model, merged into `gem_best`.  Measured over luping1 the old branch dropped 0 gems
        anyway (`items dropped by the tracker: {'dia': 0, ...}`), so removing it changes no decision
        and deletes a mechanism that was not doing what its comment said.

        The return is a 2-tuple because every caller unpacks it that way.
        """
        out = {}
        dropped = {}
        for kind in ("bean", "grn", "ball", "slow"):
            keep = []
            n_drop = 0
            for entry in items.get(kind, []):
                area = int(entry[2]) if len(entry) > 2 else 0
                if kind == "bean" and area < BEAN_MIN_TRUST_PX:
                    # a very small bean must be confirmed by a neighbouring frame: a one-frame
                    # highlight on a snake's body is the size of a small bean and does not persist
                    cell = (int(entry[0]), int(entry[1]))
                    if not self._confirmed(kind, cell):
                        n_drop += 1
                        continue
                keep.append(entry)
            out[kind] = keep
            if n_drop:
                dropped[kind] = n_drop
        # Record RAW small-bean sightings for the next-frame confirmation. This
        # history used to be queried but never populated after the filter rewrite.
        self.item_hist["bean"].extend((self.frame,(int(e[0]),int(e[1]))) for e in items.get("bean",[]))
        self.item_hist["bean"] = [(f,c) for f,c in self.item_hist["bean"] if self.frame-f <= ITEM_TOL_FRAMES]
        # gems pass through: the window test and the pair model own that decision now
        if "dia" in items:
            out["dia"] = list(items["dia"])
        return out, dropped

    def _recent_gem(self, cell):
        """Was a gem CONFIRMED at this cell within DIA_PERSIST_FRAMES?"""
        return any(c == cell for _f, c in self.dia_confirmed)

    def _seen_count(self, kind, cell):
        """How many of the last DIA_CONFIRM_M frames saw this cell (excluding now)."""
        lo = self.frame - DIA_CONFIRM_M
        return sum(1 for f, c in self.item_hist[kind]
                   if c == cell and lo <= f < self.frame)

    def head_hint(self, family):
        """Predicted head pixel and window for a family, for the caller to search."""
        w = self.win[family]
        return w.predict(), w.window((800, 928, 3))
