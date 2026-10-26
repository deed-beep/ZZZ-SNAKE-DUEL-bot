"""State analysis: the step between perception and control that asks "can this be true?".

THE PROBLEM IT SOLVES
---------------------
Perception answers "what do the pixels look like".  Control answers "what key do I press".  There
was nothing in between, so every misreading went straight into a decision: a phantom gem steered
the head, a one-frame length spike reset the length reference, an item that had already been
eaten kept attracting the bot for six frames of peripheral staleness.

This layer sits in the middle and uses the GAME'S OWN RULES to decide which readings can be
believed, which are the perception being wrong, and what the world most likely actually is.

THE RULES USED, all measured or stated by the user
--------------------------------------------------
1. A snake's length grows by exactly +1 per bean and +5 per diamond, and NEVER shrinks - except
   when the accelerate key is held, which drops one tail cell at a time (about one per 5.7 s),
   and on death, where the whole body dissolves fast.  So a length that falls and recovers is a
   misreading, not a tail drop; a length that rises by an amount no item explains is a missed
   item, and the correct response is to look for what was eaten rather than to believe the jump.
2. The head moves at most a few cells per frame; it cannot teleport.  A 7-cell jump is the
   tracker having swapped ends or lost the snake.
3. An item does not move.  Beans and gems sit still until eaten, so a reading that jumps cells
   is noise, and an item that disappears while the head was adjacent is a collect - which also
   predicts a length increase next frame.
4. Spike balls are exactly three, 2x2, and drift slowly; the slowdown item is one cell and
   harmless.  A "fourth ball" or a one-cell "ball" is a misclassification, not a new object.
5. The arena border and the snakes' own bodies are the only permanent obstacles; anything the
   window reports as an obstacle that has not persisted for a frame is suspect during a whiteout.

WHAT IT OUTPUTS
---------------
A `Belief` per frame: the corrected head, the corrected length with the reason for any change,
the item set with a per-item disposition (fresh window reading / confirmed / carried / eaten /
phantom), plus what it predicts for the next frame.  The policy reads the belief, not the raw
perception, so that a single bad frame cannot flip a decision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from config import (CELL, GAME_HZ, BEAN_GROWTH, DIA_GROWTH, EAT_ADJACENT_CELLS,
                    ITEM_MOVE_TOL_CELLS, BALL_EXPECTED_COUNT, PERIPHERAL_WEIGHT,
                    HEAD_MAX_CELLS_PER_FRAME, HEAD_STALL_MAX_FRAMES,
                    HEAD_EST_ALPHA,
                    TAIL_DROP_MIN_INTERVAL, DIAMOND_TAIL_LAG_CELLS, DEATH_CONFIRM)


@dataclass
class Belief:
    """The corrected world model the policy acts on."""

    frame: int = 0
    head_cell: tuple | None = None
    facing: tuple | None = None
    length: int = 0
    length_reason: str = ""
    items: dict = field(default_factory=dict)       # kind -> [(cx, cy, area, status)]
    obstacles: list = field(default_factory=list)   # [(cx, cy, kind)]
    ball_tracks: list = field(default_factory=list)   # [(mode, threat, cell, ahead)]
    danger_cells: list = field(default_factory=list)  # [(cell, threat, mode)] worst first
    predicted_head: tuple | None = None
    predicted_growth: int = 0                       # length to expect next frame
    anomalies: list = field(default_factory=list)   # human-readable, for the log
    notes: list = field(default_factory=list)


class StateAnalysis:
    """Turns an AttentionFrame into a Belief, using the game's mechanics as the constraint."""

    def __init__(self):
        self.frame = 0
        self.items = {}            # (kind, cx, cy) -> dict(area, last_seen, first_seen, status)
        self.length = 0
        self.len_hist = []
        self.head_hist = []
        self.eat_events = []       # (frame, kind, cell) collected but not yet explained
        self.pending_growth = 0
        self.tail_fill_until = 0
        self.last_drop_frame = -10**6
        self.dissolve_run = 0
        self.stall = 0
        self._body = []            # the player's body CELLS - a set, not an ordering
        self._path = []            # the ORDERED body path
        self._path_trust = "absent"  # ordered | uncertain | absent
        # bean accounting for the tail-drop cross-check: a press drops exactly one bean, so
        # the beans that appeared between frames must account for the length that fell
        self.bean_count = 0
        self.prev_bean_count = 0
        self.bean_hist = []        # recent board bean counts, for the look-back window
        self.deaths = {k: 0 for k in ("red", "org", "pur", "blu")}
        from motion import MotionModel, StillnessFilter
        self.motion = MotionModel()
        self.stillness = StillnessFilter()

    # ------------------------------------------------------------------ items
    def _age_items(self, af):
        """Mark every known item with how long since it was last actually measured."""
        for key, rec in self.items.items():
            rec["age"] = self.frame - rec["last_seen"]

    def _reconcile_items(self, af):
        """Merge window readings, peripheral readings and memory into one consistent set.

        Rules applied here:
          * a window reading is authoritative for its cell and resets the item's age;
          * a peripheral reading is accepted but weaker, and its staleness is recorded;
          * an item that MOVED is a problem: items are stationary, so a moved reading is
            treated as a new candidate with low confidence rather than as the same item;
          * an item that vanished while the head was next to it was EATEN, which is recorded
            as a growth claim to check against the length series.
        """
        seen = {}
        # the merged pipeline is authoritative; the individual lists are diagnostics
        merged = getattr(af, "gem_best", None)
        if merged:
            for entry in merged:
                cx, cy, area, ring, src, inw = entry
                if src == "extra":
                    continue
                age = max(0, self.frame-getattr(af,"gem_last_seen",{}).get((int(cx),int(cy)),self.frame)) if src == "pair" else 0
                seen[("dia", int(cx), int(cy))] = dict(area=int(area), ring=ring,
                                                       source=src, in_window=bool(inw), age=age)
        for entry in af.win_diamonds:
            cx, cy, area, ring = entry[0], entry[1], entry[2], entry[3]
            kind = entry[4] if len(entry) > 4 else "candidate"
            k = ("dia", int(cx), int(cy))
            if k not in seen:
                seen[k] = dict(area=int(area), ring=ring, source="window", dia_kind=kind)
        for entry in af.per_diamonds:
            cx, cy, area, ring, age = entry[0], entry[1], entry[2], entry[3], entry[4]
            kind = entry[5] if len(entry) > 5 else "peripheral"
            k = ("dia", int(cx), int(cy))
            if merged is not None:
                break
            if k not in seen:
                seen[k] = dict(area=int(area), ring=ring, source="peripheral", age=age,
                               dia_kind=kind)
        for cx, cy, area in af.win_beans:
            seen[("bean", int(cx), int(cy))] = dict(area=int(area), source="window")
        for cx, cy, area, age in af.per_beans:
            k = ("bean", int(cx), int(cy))
            if k not in seen:
                seen[k] = dict(area=int(area), source="peripheral", age=age)
        for cx, cy, area in af.win_slow:
            seen[("slow", int(cx), int(cy))] = dict(area=int(area), source="window")
        for cx, cy, area, age in af.per_slow:
            k = ("slow", int(cx), int(cy))
            if k not in seen:
                seen[k] = dict(area=int(area), source="peripheral", age=age)

        for cx,cy,a in getattr(af,"win_speed",[]):
            seen[("grn",int(cx),int(cy))] = dict(area=int(a),source="window")
        for cx,cy,a,age in getattr(af,"per_speed",[]):
            seen.setdefault(("grn",int(cx),int(cy)),dict(area=int(a),source="peripheral",age=age))

        # 1) update or create
        for k, rec in seen.items():
            kind, cx, cy = k
            if k in self.items:
                old = self.items[k]
                # Carrying the same cached peripheral record is not a new sighting.
                old["last_seen"] = max(old["last_seen"], self.frame-int(rec.get("age",0)))
                old["age"] = self.frame-old["last_seen"]
                old["area"] = rec["area"]
                if rec.get("ring") is not None:
                    old["ring"] = rec["ring"]
                old["source"] = rec["source"]
                if rec.get("dia_kind"):
                    old["dia_kind"] = rec["dia_kind"]
                old["status"] = "confirmed"
            else:
                age = int(rec.get("age",0))
                rec.update(last_seen=self.frame-age, first_seen=self.frame-age,
                           status="new", age=age)
                self.items[k] = rec

        # 2) an item that vanished next to the head was eaten
        head = af.head_cell
        for k in list(self.items):
            rec = self.items[k]
            if rec["last_seen"] == self.frame:
                continue
            kind, cx, cy = k
            if head is not None:
                d = max(abs(cx - head[0]), abs(cy - head[1]))
                center = getattr(af,"item_centers",{}).get((kind,cx,cy),((cx+.5)*CELL,(cy+.5)*CELL))
                close = af.head_px is not None and max(abs(center[0]-af.head_px[0]),abs(center[1]-af.head_px[1])) <= .65*CELL
                collected = ((cx,cy) in getattr(af,'gem_collected',[]) if kind == 'dia'
                             else close and (kind,cx,cy) not in seen and rec["age"] <= 3)
                if collected and rec["status"] != "eaten":
                    self.eat_events.append((self.frame, kind, (cx, cy)))
                    rec["status"] = "eaten"
                    continue
            if rec["age"] > 30 and not (kind == 'dia' and (cx,cy) in af.gem_fixed):
                rec["status"] = "forgotten"

    # ------------------------------------------------------------------ length
    def _reconcile_length(self, af):
        """Apply the user's length mechanics, which are about the TAIL lagging, not the number.

        Head and tail travel together, so the length is normally CONSTANT.  A change is a growth
        event and looks like this:
          * bean    - the tail pauses briefly while the head moves on, so the body gains a cell
                      and fills it within a frame or two;
          * diamond - the tail pauses until the head has travelled about DIAMOND_TAIL_LAG_CELLS,
                      so the body keeps filling in over those frames;
          * accelerate - the tail drops a cell, about once per 5.7 s, so at most one cell every
                      ~85 frames;
          * death   - the tail stops and the body dissolves head-to-tail, taking longer the
                      longer the snake was.

        So the old "snap to the nearest legal growth" behaviour was wrong in both directions: it
        invented growth that no item explained, and it treated a fast COLLAPSE as a tail drop when
        a tail drop cannot be fast.  What replaces it:
          * a rise is attributed to an item, and a pending fill-in is recorded so the following
            frames are expected to keep the larger length rather than being called anomalies;
          * a one-cell fall is a tail drop only if enough frames have passed since the last one;
          * a sustained fall is the death dissolve, which is worth naming because it changes what
            the policy should do;
          * a steady length is the normal case and is recorded as such, not as a non-event.
        """
        cur = int(af.red_length)
        prev = self.length
        if prev == 0:
            self.length = cur
            af.notes.append("length-init")
            self.len_hist.append(self.length)
            return
        delta = cur - prev
        if delta == 0:
            self.length = cur
            af.notes.append("length-stable")
        elif delta > 0:
            # attribute it: 1 -> bean, 5 -> diamond, more -> a missed item plus growth
            if delta == BEAN_GROWTH:
                kind = "bean"
            elif delta == DIA_GROWTH:
                kind = "diamond"
            elif delta > DIA_GROWTH:
                kind = "diamond+"
            else:
                kind = "bean+"
            self.length = cur
            self.pending_growth = max(0, self.pending_growth) + 0
            af.notes.append(f"length+{delta} ({kind})")
            if delta not in (BEAN_GROWTH, DIA_GROWTH):
                af.notes.append(f"anomaly:unexplained-growth {delta}")
            # remember that the tail is still filling in, so the next frames are not surprises
            self.tail_fill_until = self.frame + (
                DIAMOND_TAIL_LAG_CELLS if kind.startswith("diamond") else 2)
        else:
            # A FALL is a claim, not a fact.  The only way the length goes down is the accelerate
            # key: one press drops exactly one cell as a bean on the tail cell, and holding drops
            # at KEY_LEAK_BEANS_PER_S.  So the fall is cross-checked against the beans that
            # APPEARED since the previous frame (see TAIL_DROP_BEAN_CHECK).
            # the baseline is the bean population just before this frame's rise
            new_beans = self.bean_count - min(self.bean_hist[:-1] or [self.bean_count])
            if delta == -1 and new_beans > 1:
                # MORE beans than the length lost: the stacking exploit's exact signature.  The
                # length is the trustworthy quantity - a snake cannot shed fewer cells than the
                # beans it left behind - so the length is believed and the surplus is reported.
                # Checked BEFORE the ordinary press case, because one press accounts for exactly
                # one bean and anything more is not a press.
                self.length = cur
                self.last_drop_frame = self.frame
                af.notes.append(f"anomaly:beans-exceed-drop beans {new_beans} > drop {-delta}")
            elif delta == -1 and new_beans == 1:
                self.length = cur
                self.last_drop_frame = self.frame
                af.notes.append("length-1 (tail drop, 1 bean appeared)")
            elif delta == -1 and new_beans == 0:
                # the tail cell was not a bean, so this is not a press: the tail cell simply was
                # not seen, or the body fragmented and re-formed shorter
                self.length = cur
                af.notes.append("length-1 (no bean appeared: not a press)")
                self.dissolve_run += 1
            else:
                # a fast fall is impossible for presses; it is the death dissolve (sustained) or
                # the body fragmenting (which recovers)
                self.dissolve_run += 1
                self.length = cur
                af.notes.append(f"length{delta} (dissolve? run {self.dissolve_run})")
                if self.dissolve_run >= DEATH_CONFIRM:
                    af.notes.append("anomaly:dissolve-sustained")
        if delta <= 0:
            self.dissolve_run = self.dissolve_run if delta < 0 else 0
        self.len_hist.append(self.length)
        del self.len_hist[:-8]

    # ------------------------------------------------------------------ head
    def _reconcile_head(self, af):
        """Correct the head only when the reading is both implausible AND the body contradicts it.

        See the module note above _local_body_end for the five designs this replaces and why each
        failed.  The rule kept is the narrowest one that can neither teleport nor freeze:

            the reading is more than the per-frame limit from the previous head, AND a body END is
            within that limit of the reading -> the reader named the wrong end, so use that end.

        The first guard makes every correction local, so no teleport is possible.  The second makes
        the correction fire only when the body supplies the answer, so no freeze is possible: when
        neither end is near the reading, the reading is kept and an anomaly is recorded, which puts
        the doubt in the per-frame document rather than hiding it behind a held value.
        """
        cell = af.head_cell
        if cell is None:
            af.notes.append("head-none")
            self.stall += 1
            if self.stall > HEAD_STALL_MAX_FRAMES:
                af.notes.append(f"anomaly:head-stall {self.stall}")
            return
        cell = (int(cell[0]), int(cell[1]))
        if self.head_hist:
            pc, _pf = self.head_hist[-1]
            d = max(abs(cell[0] - pc[0]), abs(cell[1] - pc[1]))
            if d > HEAD_MAX_CELLS_PER_FRAME:
                # Correct only with independent body continuity and no fresh eye
                # evidence. Creeping through empty space invented a head that was
                # on neither end, making its path and collection pixels disagree.
                alt = self._end_near(pc) if af.head_source != "eyes" else None
                if alt is not None and alt != cell and alt in self._body:
                    af.notes.append(f"head-end-swap {cell}->{alt}")
                    cell = alt
                    d = max(abs(cell[0]-pc[0]),abs(cell[1]-pc[1]))
                else:
                    af.notes.append(f"anomaly:head-resync {d} cells (measured, no imaginary interpolation)")
            self.stall = self.stall + 1 if d == 0 else 0
            if self.stall > HEAD_STALL_MAX_FRAMES:
                af.notes.append(f"anomaly:head-stall {self.stall}")
        af.head_cell = cell
        self.head_hist.append((cell, af.facing))
        del self.head_hist[:-8]

    def _end_near(self, cell):
        """A body END within the per-frame limit of `cell`, or None.

        "End" means an end of the ORDERED path, and only a path whose order is known counts.  The
        previous version took the first and last elements of the coordinate-sorted CELL SET, which on
        a bent snake are the topmost-left and bottommost-right cells - not ends - so the arbitration
        was comparing the reading against arbitrary body cells.  That is the bug the user identified.

        When the order is unknown the honest answer is None: no arbitration, the reading stands, and
        the frame records that the head could not be checked.  Guessing an end here is exactly how
        "the body was reported as the head" happened.
        """
        if self._path_trust != "ordered":
            return None
        path = self._path or []
        if len(path) < 2:
            return None
        best, bestd = None, 1e9
        for e in (path[0], path[-1]):
            dist = max(abs(e[0] - cell[0]), abs(e[1] - cell[1]))
            if dist <= HEAD_MAX_CELLS_PER_FRAME and dist < bestd:
                best, bestd = e, dist
        return (int(best[0]), int(best[1])) if best is not None else None

    # ------------------------------------------------------------------ main
    def analyse(self, af):
        """-> Belief for this frame."""
        self.frame += 1
        b = Belief(frame=self.frame)
        # ---- the frame's own facts, gathered before any judgement is made.
        # The body, so a head reading that breaks the motion limit can be snapped onto it.
        self._body = [(int(c[0]), int(c[1]))
                      for c in (getattr(af, "red_cells", None) or [])]
        self._path = [tuple(int(v) for v in c) for c in getattr(af, "red_path", [])]
        self._path_trust = getattr(af, "red_path_trust", "absent")
        # The BOARD bean count, for the tail-drop cross-check: a press drops exactly one bean, so
        # the beans that appeared between frames have to account for the length that fell.  This
        # must be the perception's board total, not len(items()["bean"]) - that list is only the
        # window plus whatever the last peripheral scan carried, and using it made the check see
        # "no bean appeared" on every fall (measured: all three synthetic cases in taildrop_test.py
        # returned the same verdict).
        self.bean_count = int(getattr(af, "board_beans", 0))
        # Record the bean population EVERY frame, not only on the frames the length falls: the
        # previous version appended only inside the fall branch, so the first fall had nothing to
        # compare against and always read as "no bean appeared" (measured: the exploit case in
        # taildrop_test.py reported a normal fall).  bean_hist[:-1] is then the recent baseline.
        self.bean_hist.append(self.bean_count)
        del self.bean_hist[:-4]
        self._age_items(af)
        self._reconcile_items(af)
        self._reconcile_length(af)
        self._reconcile_head(af)

        b.head_cell = af.head_cell
        b.facing = af.facing
        b.length = self.length
        b.length_reason = af.notes[-1] if af.notes else ""
        b.anomalies = [n for n in af.notes if n.startswith("anomaly:")]

        # ---- present the items with their disposition, window first
        out = {"dia": [], "bean": [], "ball": [], "slow": [], "grn": []}
        for (kind, cx, cy), rec in self.items.items():
            if rec["status"] == "forgotten":
                continue
            status = rec["status"]
            if status != "eaten" and rec.get("age", 0) > 0:
                status = f"carried({rec['age']})"
            out.setdefault(kind, []).append((cx, cy, rec.get("area", 0), status))
        b.items = out

        # ---- motion: fit each ball's trajectory, then rank the danger it creates.  The
        # user's rule is that the three balls move one horizontally, one vertically and one at
        # 45 degrees, with the 45-degree one worst: a horizontal ball sweeps a row, so standing
        # one cell off that row is safe, while the diagonal's lane cuts across the space a snake
        # would retreat into.  The modes come from the fitted track, and the deliberate
        # constraint is that a gem may only move on an axis - so a gem whose track has both axes
        # moving is reported as a misreading rather than obeyed.
        # Fit in actual GAME_HZ ticks, not control-loop iterations (30 Hz vs
        # the game's 15 Hz). Cached duplicate observations do not create travel.
        sample_t = getattr(af,"sample_t",None)
        motion_frame = sample_t*GAME_HZ if sample_t is not None else (af.match_t*GAME_HZ if getattr(af,"match_t",None) is not None else self.frame)
        self.motion.update_balls(af.items(), motion_frame)
        b.ball_tracks = [(t.mode, t.threat, t.cells[-1] if t.cells else None, t.ahead(2))
                         for t in self.motion.balls]
        ok, diag = self.motion.check_ball_modes()
        if not ok:
            b.anomalies.append(f"anomaly:ball-modes {diag}")
        b.danger_cells = self.motion.danger(af.head_cell, steps=8)
        if b.danger_cells:
            _c, _th, _m = b.danger_cells[0]
            b.notes.append(f"top-threat {_m} at {_c}")

        # ---- obstacles: the window's reading, plus the ball rule from the mechanics
        obs = []
        for cx, cy, kind, a in af.win_obstacles:
            obs.append((int(cx), int(cy), kind))
        # the user's rule: exactly BALL_EXPECTED_COUNT spike balls, 2x2.  More than that means
        # something else was classified as a ball, so the extras are dropped rather than believed.
        balls = [o for o in obs if o[2] == "ball"]
        if len(balls) > BALL_EXPECTED_COUNT:
            keep = sorted(balls, key=lambda o: -self._ball_area(af, o))[:BALL_EXPECTED_COUNT]
            obs = [o for o in obs if o[2] != "ball"] + keep
            b.anomalies.append(f"anomaly:too-many-balls {len(balls)}")
        b.obstacles = obs

        # ---- prediction: where the head goes, and what growth to expect
        if self.head_hist and af.facing is not None:
            hx, hy = self.head_hist[-1][0]
            b.predicted_head = (hx + int(af.facing[0]), hy + int(af.facing[1]))
        b.predicted_growth = self.pending_growth
        if self.eat_events:
            for _fr, kind, _cell in self.eat_events:
                if kind in ('dia','bean'):
                    b.predicted_growth += DIA_GROWTH if kind == "dia" else BEAN_GROWTH
            self.eat_events = []
            self.pending_growth = 0
        # These are cross-frame histories, NOT scratch variables. Clearing them
        # here prevented every sustained collapse/stall rule from ever firing.
        b.notes = list(af.notes)
        return b

    def _ball_area(self, af, o):
        for cx, cy, kind, a in af.win_obstacles:
            if (int(cx), int(cy)) == (o[0], o[1]) and kind == "ball":
                return int(a)
        return 0
