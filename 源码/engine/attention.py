"""Attention: a 7x7 window on the player's head, plus deliberately cheaper peripheral vision.

WHY THIS EXISTS
---------------
`Perception.run` treats the whole 928x800 frame as equally interesting: it scans every pixel at
full resolution for diamonds, builds every family mask, and clusters every family, every frame.
That spends the budget uniformly, so the place where a mistake actually kills the snake - the
cells in front of its own head - gets no more attention than the far corner of the arena.

This module turns that inside out to match how the user asked for it:

    priority   what                     how it is looked at
    --------   ----------------------   -----------------------------------------------
    1          the player's head        every frame, full resolution, 7x7 cells
    2          diamonds IN the window   every frame, full resolution, strict acceptance
    3          obstacles IN the window  every frame (outranks diamonds once balls exist)
    4          beans / speed IN window  every frame, lower acceptance effort
    5          everything outside       PERIPHERAL: every PERIPHERAL_EVERY frames, lower
                                        weight, carried forward in between
    6          slowdown items           lowest; only reported when confidence is high

So the window is the "fovea": fully re-measured each frame.  The periphery is the "corner of the
eye": sampled less often, and its readings are marked as such so the state analysis and the
policy can discount them instead of treating a 6-frame-old bean position as current.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not re-implement segmentation.  `Perception` already provides the family masks, the
snake decode, the eye test and the full-resolution diamond detector; this module decides WHERE
accuracy is spent and how long a reading stays trusted, and it exposes the result as one
`AttentionFrame` per frame.  The point is to change the allocation of effort, not to build a
second detector that would drift out of sync with the first.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import (CELL, FRAME_W, FRAME_H, COLS, ROWS, HEAD_WIN_CELLS,
                    PERIPHERAL_EVERY, PERIPHERAL_WEIGHT, PERIPHERAL_MAX_AGE,
                    WIN_DIA_DARK_MIN, WIN_DIA_MIN_PX, WIN_DIA_MAX_PX,
                    WIN_DIA_MIN_FILL, WIN_DIA_MAX_ELONG,
                    WIN_OBSTACLE_MARGIN, WINDOW_FIRST)

# ============================================================ what one frame reports
@dataclass
class AttentionFrame:
    """Everything the policy is allowed to know this frame, with the trust attached.

    Every field carries where it came from, because "a diamond at (12,12)" means something very
    different when it was just measured inside the window and when it is a peripheral reading
    from six frames ago.  The policy and the state analysis both need that distinction, and
    collapsing it into a bare list is what made the old item list untrustworthy.
    """

    frame: int = 0
    match_t: float | None = None
    sample_t: float | None = None
    item_centers: dict = field(default_factory=dict)  # (kind,cx,cy) -> measured full-res pixels
    gem_last_seen: dict = field(default_factory=dict) # cell -> actual observation frame
    head_cell: tuple | None = None          # tier 1
    head_cell_raw: tuple | None = None      # as reported, before the motion rules
    head_px: tuple | None = None
    facing: tuple | None = None
    head_trust: float = 0.0                 # 1.0 = eyes, lower = extrapolated
    head_source: str = "none"               # eyes | window | peripheral | predicted | none
    red_length: int = 0
    red_cells: list = field(default_factory=list)   # the body cell set (unordered!)
    # The ORDERED body path and how much its order can be trusted.  A sorted cell set is
    # not an ordering: its first and last elements are the topmost-left and bottommost-right
    # cells, which on a bent snake are not the ends.  Consumers that need "the two ends of
    # the snake" must use this, and must check the trust first.
    red_path: list = field(default_factory=list)
    red_path_trust: str = "absent"

    # tier 2/3/4: measured INSIDE the 7x7 window THIS frame
    win_diamonds: list = field(default_factory=list)     # [(cx, cy, area, ring, kind)]
    win_obstacles: list = field(default_factory=list)    # [(cx, cy, kind, area)]
    win_beans: list = field(default_factory=list)        # [(cx, cy, area)]
    win_slow: list = field(default_factory=list)
    win_speed: list = field(default_factory=list)
    per_speed: list = field(default_factory=list)
    window_box: tuple | None = None                      # (x0,y0,x1,y1) full res

    # tier 5/6: periphery, possibly carried forward
    per_diamonds: list = field(default_factory=list)     # [(cx, cy, area, ring, age)]
    per_beans: list = field(default_factory=list)
    per_balls: list = field(default_factory=list)
    per_slow: list = field(default_factory=list)
    per_age: int = 0                        # frames since the last peripheral scan
    per_fresh: bool = False                 # True on the frames a scan actually ran

    # The decoded bean count for the whole board this frame.  Not the same as
    # len(items()['bean']), which is only the window plus whatever the last peripheral
    # scan carried - and the tail-drop cross-check needs the board total.
    # the fixed diamond pair, tracked across frames (see FixedGemPair).  `gem_fixed` is the
    # persistent model of the two stationary gems; `gem_extra` is every other gem reading,
    # i.e. the moving gem or a false positive, which must NOT displace a fixed one.
    # The merged gem view: [(cx, cy, area, ring, source, in_window)].  This is the ONE list
    # the state analysis and the policy should read - `win_diamonds`, `per_diamonds` and
    # `gem_fixed` remain for diagnostics only.
    gem_best: list = field(default_factory=list)
    gem_fixed: list = field(default_factory=list)
    gem_extra: list = field(default_factory=list)
    gem_respawning: list = field(default_factory=list)
    gem_collected: list = field(default_factory=list)  # head crossed hitbox, then absent repeatedly
    board_beans: int = 0
    board_balls: int = 0
    balls_live: bool = False                # spike balls exist -> obstacles outrank diamonds
    scan_ms: float = 0.0
    notes: list = field(default_factory=list)

    # ------------------------------------------------------------------ views
    def items(self):
        """The merged item view, in priority order, with window readings first.

        Both lists keep their provenance as a trailing tag so the state analysis can weigh a
        fresh window reading above a stale peripheral one.  The window copy always wins for the
        same cell - it was measured later and more strictly.
        """
        out = {"dia": [], "bean": [], "ball": [], "slow": [], "grn": []}
        seen = set()
        for entry in self.win_diamonds:
            cx, cy, a, ring = entry[0], entry[1], entry[2], entry[3]
            kind = entry[4] if len(entry) > 4 else "candidate"
            out["dia"].append((cx, cy, a, ring, kind))
            seen.add(("dia", cx, cy))
        for cx, cy, a, ring, _age in self.per_diamonds:
            if ("dia", cx, cy) not in seen:
                out["dia"].append((cx, cy, a, ring, "peripheral"))
        for cx, cy, a in self.win_beans:
            out["bean"].append((cx, cy, a))
            seen.add(("bean", cx, cy))
        for cx, cy, a, _age in self.per_beans:
            if ("bean", cx, cy) not in seen:
                out["bean"].append((cx, cy, a))
        for cx, cy, kind, a in self.win_obstacles:
            if kind == "ball" and ("ball",cx,cy) not in seen:
                out["ball"].append((cx, cy, a))
                seen.add(("ball",cx,cy))
        for cx, cy, a, _age in self.per_balls:
            if ("ball", cx, cy) not in seen:
                out["ball"].append((cx, cy, a))
        for cx, cy, a in self.win_slow:
            out["slow"].append((cx, cy, a))
            seen.add(("slow",cx,cy))
        for cx, cy, a, _age in self.per_slow:
            if ("slow", cx, cy) not in seen:
                out["slow"].append((cx, cy, a))
        for cx,cy,a in self.win_speed:
            out["grn"].append((cx,cy,a))
            seen.add(("grn",cx,cy))
        for cx,cy,a,_age in self.per_speed:
            if ("grn",cx,cy) not in seen:
                out["grn"].append((cx,cy,a))
        return out

    def window_diamond_cells(self):
        return {(int(c[0]), int(c[1])) for c in self.win_diamonds}


# ============================================================ the fovea
class WindowDetector:
    """Re-measures diamonds and obstacles inside the 7x7 cells around the head.

    Diamonds here get a STRICTER test than the global scan, not a looser one, because the
    window is where a wrong answer costs the snake: a false gem in front of the head steers it
    into a wall, while a missed gem three cells away only loses points.  So a window gem must
    satisfy the colour, the size, the shape AND the dark hexagonal edge:

      * colour    cyan-dominant (B-R >= 40, G-R >= 20) in the gem's hue window
      * size      60..2500 px at full resolution (gems measure 126-611)
      * shape     fill ratio and elongation of the blob's own pixels, i.e. the hexagonal face
      * edge      fraction of dark pixels in the ring just outside the blob (the hexagon's
                  dark border) must clear WIN_DIA_DARK_MIN

    The hexagon's dark border is the single most discriminating feature: the arena floor's cyan
    texture has no such border, which is why the dark-ring test is what lets this be strict
    without losing real gems.
    """

    def __init__(self, perception):
        self.per = perception
        self.frame = 0          # the window's own clock, for the stationarity record
        self._k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        # Persistence bookkeeping, because the dark-ring measurement alone cannot carry the
        # decision.  Measured on luping1: cell (17,1) holds a REAL gem from about f101 to f166,
        # and across those frames its ring fraction wanders 0.029 - 0.139 with no pattern.  The
        # gem does not change; the measurement does (the white spawn/collect sparkle, the
        # highlight crossing a grid line).  So a strict ring floor rejects real gems (0.18 lost
        # 16 of the user's 17 gems) and a loose one admits floor patches (the global 0.02 admits
        # a bright background patch on f124).  What separates the two is TIME: a gem sits at one
        # cell for tens of frames, a background patch does not.
        # (cx, cy) -> list of recent frames at which a candidate occupied this cell.  A list
        # rather than a counter, because the detector's gaps mean consecutive-run counting
        # understates a gem that is genuinely sitting still (see DIA_FIXED_MEMORY).
        self.seen = {}
        self.centers = {}
        self.positions = {}
        self.strong_seen = {}

    # ------------------------------------------------------------------ diamonds
    def diamonds(self, img_full, box, sample_t=None):
        """Diamonds inside `box` = (x0,y0,x1,y1) full resolution."""
        self.frame += 1
        stamp=float(sample_t) if sample_t is not None else self.frame/15.0
        self.centers = {}
        x0, y0, x1, y1 = box
        if x1 - x0 < 8 or y1 - y0 < 8:
            return []
        sub = img_full[y0:y1, x0:x1]
        hsv = cv2.cvtColor(sub, cv2.COLOR_RGB2HSV)
        R, G, B = sub[..., 0], sub[..., 1], sub[..., 2]
        from config import (DIA_RMAX, DIA_DARK_VMAX, DIA_FIXED_MIN_RUN,
                            DIA_MOVING_MIN_DARK, DIA_FIXED_MEMORY)
        from perception import _DIA
        m = cv2.inRange(hsv, (_DIA[0], _DIA[1], _DIA[2]), (_DIA[3], 255, 255))
        m = cv2.bitwise_and(m, cv2.inRange(B, 161, 255))
        m = cv2.bitwise_and(m, cv2.inRange(G, 141, 255))
        m = cv2.bitwise_and(m, cv2.inRange(R, 0, DIA_RMAX))
        m = cv2.bitwise_and(m, cv2.threshold(cv2.subtract(B, R), 40, 255,
                                             cv2.THRESH_BINARY)[1])
        m = cv2.bitwise_and(m, cv2.threshold(cv2.subtract(G, R), 20, 255,
                                             cv2.THRESH_BINARY)[1])
        dark = (sub.max(axis=2) < DIA_DARK_VMAX)
        n, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
        found = {}
        for i in range(1, n):
            a = int(st[i, cv2.CC_STAT_AREA])
            if not (WIN_DIA_MIN_PX <= a <= min(800,WIN_DIA_MAX_PX)):
                continue
            bx, by = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
            w, h = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
            if w <= 0 or h <= 0 or max(w,h)>44:
                continue
            fill = a / float(w * h)
            if fill < WIN_DIA_MIN_FILL or fill>.85:
                continue
            if max(w, h) / float(min(w, h)) > WIN_DIA_MAX_ELONG:
                continue
            # dark hexagonal border, measured in the ring just outside the blob
            rx0, ry0 = max(0, bx - 5), max(0, by - 5)
            rx1, ry1 = min(x1 - x0, bx + w + 5), min(y1 - y0, by + h + 5)
            blob = (lab[ry0:ry1, rx0:rx1] == i)
            grown = cv2.dilate(blob.view(np.uint8), self._k_ring) > 0
            ring = grown & ~blob
            ring_n = int(ring.sum())
            frac = 0.0
            if ring_n:
                frac = float((dark[ry0:ry1, rx0:rx1] & ring).sum()) / ring_n
            cx = int(ce[i][0] + x0) // CELL
            cy = int(ce[i][1] + y0) // CELL
            if (cx, cy) not in found or a > found[(cx, cy)][0]:
                found[(cx, cy)] = (a, frac)
                self.centers[("dia", cx, cy)] = (float(ce[i][0] + x0), float(ce[i][1] + y0))

        # ---- classify each candidate as FIXED or MOVING, and accept accordingly.
        #
        # The game's rules, from the user:
        #   * a match opens with exactly DIA_FIXED_COUNT gems at fixed positions;
        #   * when one is eaten it respawns at a random empty spot, so the fixed count stays at
        #     two except during the brief wait for the respawn;
        #   * separately a MOVING gem appears at a random time and place, drifting at about the
        #     spike ball's speed, rare - order five across a whole match;
        #   * beans, the speed item and the slowdown item never move.
        #
        # That is what makes the persistence test legitimate, and also what limits it: it works
        # for the fixed pair BECAUSE they do not move, and a moving gem would defeat it, since a
        # gem drifting through a cell never accumulates the frames the test needs.  So:
        #
        #   fixed   a candidate that has held this cell for DIA_FIXED_MIN_RUN frames is accepted
        #           even on a weak edge - measured, a real fixed gem's ring fraction wanders
        #           0.029-0.139 while it sits still, so the edge cannot be the criterion for it.
        #   moving  cannot use persistence, so it needs a genuinely strong edge, and it is
        #           reported with a distinct tag so nothing downstream treats it as fixed.
        out = []
        for cell, (a, frac) in found.items():
            hist = self.seen.setdefault(cell, [])
            hist.append(self.frame)
            del hist[:-8]
            recent = sum(1 for fr in hist if self.frame - fr <= DIA_FIXED_MEMORY)
            pos=self.centers[('dia',cell[0],cell[1])]
            observations=self.positions.setdefault(cell,[])
            if not observations or stamp!=observations[-1][0]:
                observations.append((stamp,*pos))
            del observations[:-8]
            stable=(len(observations)>=2 and stamp-observations[0][0]>=.20-1e-6
                    and max(p[1] for p in observations)-min(p[1] for p in observations)<=6
                    and max(p[2] for p in observations)-min(p[2] for p in observations)<=6)
            strong_recent=stamp-self.strong_seen.get(cell,float('-inf'))<=DIA_FIXED_MEMORY/15.0
            if frac>=WIN_DIA_DARK_MIN:
                self.strong_seen[cell]=stamp
            if recent >= DIA_FIXED_MIN_RUN and a>=120 and (stable or strong_recent):
                # Coarse-cell repetition alone is not stationarity: the old
                # rule promoted 74..103px floor chips beside a moving head.
                # Require elapsed time AND actual centroid stability, or earlier
                # strong shape evidence. Duplicate control frames add no time.
                out.append((cell[0], cell[1], a, round(frac, 3), "fixed"))
            elif frac >= DIA_MOVING_MIN_DARK:
                # cannot be confirmed by stationarity BY CONSTRUCTION, so it needs a real edge
                out.append((cell[0], cell[1], a, round(frac, 3), "moving"))
            elif frac >= WIN_DIA_DARK_MIN:
                out.append((cell[0], cell[1], a, round(frac, 3), "candidate"))
        # forget cells not seen for a long time
        for cell in list(self.seen):
            self.seen[cell] = [fr for fr in self.seen[cell]
                               if self.frame - fr <= DIA_FIXED_MEMORY * 2]
            if not self.seen[cell]:
                del self.seen[cell]
                self.positions.pop(cell,None)
                self.strong_seen.pop(cell,None)
        return out

    # ------------------------------------------------------------------ obstacles
    def obstacles(self, img_full, box, family_masks, own_cells, balls, slow):
        """Window obstacles: the three deadly things, measured where they matter.

          ball   a spike ball overlapping the window - contact is fatal, so it outranks gems
          body   any non-player snake body inside the window, expanded by the head margin
          wall   the window crossing the arena border

        `own_cells` is the player's own body, excluded: its own body is not an obstacle.
        Returns [(cell_x, cell_y, kind, area_or_zero)].
        """
        x0, y0, x1, y1 = box
        out = []
        for cx, cy, a in balls:
            px, py = cx * CELL + CELL // 2, cy * CELL + CELL // 2
            if x0 <= px < x1 and y0 <= py < y1:
                out.append((int(cx), int(cy), "ball", int(a)))
        for cx, cy, a in slow:
            px, py = cx * CELL + CELL // 2, cy * CELL + CELL // 2
            if x0 <= px < x1 and y0 <= py < y1:
                out.append((int(cx), int(cy), "slow", int(a)))
        # enemy bodies: the family masks already exclude the player, so any big blob in the
        # window that is not ours is an obstacle
        for fam, mask in family_masks.items():
            if mask is None:
                continue
            sub = mask[max(0, y0):y1, max(0, x0):x1]
            if sub.size == 0 or not sub.any():
                continue
            n, lab, st, ce = cv2.connectedComponentsWithStats(sub.view(np.uint8), 8)
            for i in range(1, n):
                a = int(st[i, cv2.CC_STAT_AREA])
                if a < 200:                      # skin the texture, keep bodies
                    continue
                gx = (int(ce[i][0]) + max(0, x0)) // CELL
                gy = (int(ce[i][1]) + max(0, y0)) // CELL
                if (gx, gy) in own_cells:
                    continue
                out.append((gx, gy, "body", a))
        # arena border: if the window leaves the board, the cells outside are walls
        c0x, c0y = x0 // CELL, y0 // CELL
        c1x, c1y = (x1 - 1) // CELL, (y1 - 1) // CELL
        if c0x < 0 or c0y < 0 or c1x >= COLS or c1y >= ROWS:
            for cx in range(max(0, c0x), min(COLS - 1, c1x) + 1):
                for cy in range(max(0, c0y), min(ROWS - 1, c1y) + 1):
                    if cx == 0 or cy == 0 or cx == COLS - 1 or cy == ROWS - 1:
                        out.append((cx, cy, "wall", 0))
        return out


# ============================================================ the periphery
class PeripheralVision:
    """Samples the whole frame occasionally, and carries the result forward in between.

    The reading is decayed rather than deleted: an item that was measured PERIPHERAL_MAX_AGE
    frames ago is still interesting, but it is marked with its age and the state analysis
    discounts it.  Deleting it instead would make the policy forget a gem it is walking toward
    the moment the periphery looks away, which is worse than a slightly stale position.
    """

    def __init__(self):
        self.diamonds = []      # (cx, cy, area, ring) as last measured
        self.beans = []
        self.balls = []
        self.slow = []
        self.speed = []
        self.age = 999
        self.runs = 0

    def due(self, frame):
        return self.age >= PERIPHERAL_EVERY

    def observe(self, items):
        # keep the identity tag if it is there, so a moving gem stays labelled
        self.diamonds = [tuple(e[:5]) for e in items.get("dia", [])]
        self.beans = [tuple(e[:3]) for e in items.get("bean", [])]
        self.balls = [tuple(e[:3]) for e in items.get("ball", [])]
        self.slow = [tuple(e[:3]) for e in items.get("slow", [])]
        self.speed = [tuple(e[:3]) for e in items.get("grn", [])]
        self.age = 0
        self.runs += 1

    def tick(self):
        self.age += 1

    def fresh_enough(self):
        return self.age <= PERIPHERAL_MAX_AGE

    def as_lists(self):
        """-> (dia, bean, ball, slow) each with the age appended, or empty when too stale."""
        if not self.fresh_enough():
            return ([], [], [], [])
        a = self.age
        return ([d + (a,) for d in self.diamonds],
                [b + (a,) for b in self.beans],
                [b + (a,) for b in self.balls],
                [s + (a,) for s in self.slow])


# ============================================================ the fixed diamond pair
class FixedGemPair:
    """Exactly DIA_FIXED_COUNT fixed gems, tracked across frames by their stationarity.

    The gems wobble and their brightness breathes, so a single frame is a poor witness - measured on
    luping3 the fixed pair is missed for runs of 3-5 frames and then reappears at the same cells.  A
    per-frame test therefore cannot be right; what is right is a persistent model of the two
    positions, updated by whatever the frame reports and corrected by the rule that there are two.
    """

    def __init__(self):
        from config import DIA_FIXED_COUNT, DIA_FIXED_MEMORY
        self.slots = []          # [dict(cell=..., seen=..., last=...)] at most DIA_FIXED_COUNT
        self.count = DIA_FIXED_COUNT
        self.window = DIA_FIXED_MEMORY
        self.frame = 0
        self.notes = []

    def observe(self, gems, frame):
        """`gems` = [(cx, cy, area, ...)] sightings this frame. -> (fixed, extra)."""
        self.frame = frame
        self.notes = []
        cells = [(int(g[0]), int(g[1])) for g in gems]
        # ---- 1. refresh a slot for every sighting at, or one cell from, its position
        taken = set()
        for i, c in enumerate(cells):
            best, bestd = None, 1e9
            for slot in self.slots:
                d = max(abs(c[0] - slot["cell"][0]), abs(c[1] - slot["cell"][1]))
                if d <= 1 and d < bestd:
                    best, bestd = slot, d
            if best is not None:
                # the slot keeps the cell it has been seen at most: the gem wobbles in place, and
                # following the wobble makes the position jitter
                best["hits"][c] = best["hits"].get(c, 0) + 1
                best["cell"] = max(best["hits"].items(), key=lambda kv: kv[1])[0]
                best["last"] = frame
                taken.add(i)
        # ---- 2. unfilled slots take the newest unexplained candidate
        for i, c in enumerate(cells):
            if i in taken:
                continue
            if len(self.slots) < self.count:
                self.slots.append(dict(cell=c, hits={c: 1}, last=frame))
                taken.add(i)
        # ---- 3. a slot unseen for a long time was eaten; its replacement is the newest candidate
        for slot in self.slots:
            if frame - slot["last"] > self.window:
                cand = [(i, c) for i, c in enumerate(cells) if i not in taken
                        and len(gems[i])>4 and gems[i][4]=='fixed']
                if cand:
                    i, c = cand[-1]
                    taken.add(i)
                    slot.update(cell=c, hits={c: 1}, last=frame)
                    self.notes.append(f"respawn -> {c}")
        # ---- 4. everything else is NOT a fixed gem: it is the moving gem or a false positive
        extra = [c for i, c in enumerate(cells) if i not in taken]
        # A stationary gem remains there until collection or a verified
        # replacement. A timeout alone is missing vision, not game movement.
        fixed = [tuple(s["cell"]) for s in self.slots]
        return fixed, extra

    def respawning(self):
        """Slots that have gone unseen - the window in which only one fixed gem is on the board."""
        return [tuple(s["cell"]) for s in self.slots
                if self.frame - s["last"] > 2]
