"""Perception: screen -> structured game state.  OpenCV, full pipeline 10-18 ms.

The implementation is the validated one from `_an/bot_cv.py`; it is copied here
so this folder is self-contained.  Each non-obvious rule carries the measurement
that justifies it.

Measured constants
------------------
  board            29x25 cells, 32 px each (928/29 = 800/25 = 32)
  snake body       32 px + ~1 px dark outline per side
  eye slits        two white bars 12 px apart
        normal/key     ~3x8 px, elongation 2.4-3.9, symmetric about the axis
        item accel     V-tilted, the pair asymmetric along the axis
        DEATH BUFFER   two compact 'x', elongation < 1.8
  body flash       luminance span 0.6-2.0 at rest, 20-100 while accelerating,
                   sweeping head->tail about once per second
  blue snake       mask pixels have saturation p25 >= 0.95
        (the yellow snake's tail tip during a bean drop is 0.54-0.71 - that is
         the false positive the saturation test removes)
  whiteout         the respawn wash DESATURATES the arena (S 0.78 -> 0.30-0.45);
                   it does not brighten it, so the test is saturation-based
  layers, bottom->top: grid < snakes < speed token < bean = diamond < ball
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import (GAME_HZ, BALL_SPAWN_T,
                    _BALL_SMIN, _BALL_VMIN,
                    COLS, ROWS, CELL, FRAME_W, FRAME_H, WORK_SCALE,
                    OPEN_R, CLOSE_R, MIN_SNAKE_PX, JOIN_TOL, EYE_NEAR_F, EYE_SPACING,
                    EYE_SLIT_MIN_PX, EYE_SLIT_MAX_PX, EYE_WARM_MIN, EYE_BRIDGE_R,
                    EYE_HOLE_R, EYE_HOLE_R_SEQ, EYE_GATE_R, EYE_RING_D, EYE_RING_MIN,
                    EYE_PAIR_MIN_PX, EYE_PAIR_MAX_PX, EYE_PAIR_MIN_ELONG,
                    EYE_PAIR_QUALITY_PX, EYE_QUALITY_BONUS,
                    BODY_MIN_FRAC, SHORT_SNAKE_MIN_PX,
                    BLUE_SAT_P25, WHITEOUT_SAT, WHITEOUT_VAL,
                    BUFFER_ELONG, ITEM_ASYM, ITEM_MAX_PX, DIA_MAX_COUNT, MAX_OPP_PX_WHITEOUT,
                    ITEM_MAX_PX_OF, ITEM_MAX_ELONG_OF, DIA_DARK_VMAX, DIA_DARK_MIN, DIA_RMAX, DIA_ERODE,
                    BALL_MIN_PX, BALL_MAX_PX, BALL_BOX_MIN, BALL_BOX_MAX,
                    BALL_FILL_MIN, BODY_GROW_FRAC, RED_TAIL_TOL,
                    SLOW_MIN_PX, SLOW_MAX_PX, SLOW_BOX_MIN, SLOW_BOX_MAX,
                    SLOW_FILL_MIN, BALL_DARK_MIN, BALL_DARK_VMAX,
                    BALL_BAND_PAD, EYE_WINDOW, EYE_WINDOW_CELLS, EYE_WINDOW_MATCH_PX)

WORK_W, WORK_H = FRAME_W // WORK_SCALE, FRAME_H // WORK_SCALE
CELL_W = CELL // WORK_SCALE

# OpenCV HSV ranges (H 0-179, S/V 0-255), calibrated against the recordings.
#
# The orange upper hue bound is 19, and that is load-bearing.  Measured over 7
# matches x 7 frames of pure body pixels, the orange snake occupies H 8-19
# (median 12) while real bean cores sit at H 20-25.  The old bound of 29 reached
# into the beans, which is why a blanket "drop anything with B >= 150" guard was
# bolted on to keep them out - and that guard also deleted the snake's own body
# wherever the acceleration flash had washed it pale, so the blob fragmented:
# measured on luping3 f236, the whole snake (47 cells) was reported as a 4-cell
# fragment.  With hue doing the work the guard is gone and those pixels stay.
#
# KNOWN LIMIT, measured, NOT yet fixed: when the flash is over the orange snake
# its hue falls to H 9-11 at S 0.24-0.42, which is inside the RED range, and the
# two are then indistinguishable - the red snake's own body is a light coral with
# G-B median 52, and the pale orange is G-B median 53, so neither hue nor
# green-minus-blue separates them (a G-B split was tried and broke red detection
# outright: red alive fell from 953/975 frames to 407/975 on luping3).  So a
# flashed orange snake leaves pixels in the red mask, and the fix has to be at
# blob level - pick the red group by continuity with the previous frame rather
# than by size - not per pixel.
_RED = (0, 11, 45, 204)
_ORG = (11, 29, 45, 204)
_ORG_BMAX = None            # optional blue-channel ceiling on the orange mask
_PUR = (124, 167, 55, 90)
# The blue saturation floor was 127, which is what made the blue snake disappear
# outright: the acceleration flash desaturates it exactly like the others, but
# here it takes the WHOLE snake below the floor instead of splitting it.  Measured
# over 1110 frames, blue was missing in 23% of them, and in 89% of those a
# blue-hued blob of 300+ px was sitting right there with a median saturation of 86.
# The floor is now 60, and the bean-drop tail tip is excluded by the rule section
# 9.1 already prescribes - persistence - rather than by colour, because colour
# cannot tell them apart (the tip measures 0.54-0.71, the flashed snake lower).
_BLU = (97, 124, 60, 64)
_GRN = (50, 82, 75, 140)
_BALL_SMIN, _BALL_VMAX = 60, 255
_BALL_VMIN = 130
# A yellow bean is H 20-35 with a saturated core; the pale halo around it is the
# same hue at S ~ 0.25, and a flashed orange body pixel is H ~ 11, so hue plus a
# saturation floor separates all three.  The old rule was pure RGB
# (R>200 & G>140 & B<215); measured, it also matched the flashed orange body
# (255,143,74) and put phantom beans on an opponent: 27 "beans" in one frame.
_BEAN = (18, 100, 150)
# Diamond, at FULL resolution only (see Perception.diamonds_full).  Hue 80-95 is the
# gap between the gem (H 88) and the floor (H 92-95); the blue snake is H 102.
_DIA = (80, 100, 190, 95)                  # hlo, smin, vmin, hhi  (full resolution)
DIA_MIN_PX, DIA_MAX_PX = 60, 2500           # gems measure 126-611 px; the floor blob is ~700k

UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)
DIRS = (UP, DOWN, LEFT, RIGHT)

_MOVES4 = DIRS
_MOVES8 = DIRS + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _hue_deg(r, g, b):
    """Hue in degrees, 0 = red, matching OpenCV's H * 2 on the 0-179 scale."""
    mx, mn = max(r, g, b), min(r, g, b)
    if mx == mn:
        return 0.0
    d = mx - mn
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6.0)
    elif mx == g:
        h = 60.0 * (((b - r) / d) + 2.0)
    else:
        h = 60.0 * (((r - g) / d) + 4.0)
    return h


def _is_warm_px(rgb):
    """Is this pixel player-snake body?  Red or orange, and not greyed out.

    The hue windows are the two family masks expressed in degrees (red H 0-11 and
    175-179 on OpenCV's scale is 0-23 and 350-360 degrees; orange H 11-29 is 22-58
    degrees), widened to 23-60 so that the flash-pale coral at the head still counts.
    Saturation floor 0.30 and value floor 130 reject white, the cyan floor and the
    pale yellow halo around a bean's green glow.
    """
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    mx, mn = max(r, g, b), min(r, g, b)
    if mx < 130:
        return False
    if mx == 0 or (mx - mn) / mx < 0.30:
        return False
    hd = _hue_deg(r, g, b)
    return hd < 23.0 or 23.0 <= hd <= 60.0


def _is_pale_red_px(rgb):
    """High-value R-dominant coral, not neutral white/yellow/cyan sparkle.

    The visible heads in luping3 f156/f411 have only 13-20% saturation,
    below the ordinary warm embedding floor. JPEG can put B just above G,
    so both sides of red's hue wrap are accepted.
    """
    r, g, b = map(int, rgb)
    if r < 210 or r - g < 20 or r - b < 25:
        return False
    hd = _hue_deg(r, g, b)
    return hd < 30.0 or hd > 345.0


def _pale_red_mask(img):
    r, g, b = (img[..., i].astype(np.int16) for i in range(3))
    hue = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 0]
    return ((r >= 210) & (r - g >= 20) & (r - b >= 25)
            & ((hue < 15) | (hue > 172)))


class _EyePair(tuple):
    """Preserve the five-field diagnostic tuple with internal recovery metadata."""
    def __new__(cls, score, mid, state, asym, ring, pale_supported=False,travel_axis=None):
        pair = super().__new__(cls, (score, mid, state, asym, ring))
        pair.pale_supported = bool(pale_supported)
        pair.travel_axis=travel_axis
        return pair


@dataclass
class SnakeObs:
    family: str
    n_px: int = 0
    cells: set = field(default_factory=set)
    length: int = 0                      # cells occupied == score (within ~2)
    bbox: tuple = (0, 0, 0, 0)
    head_px: tuple | None = None         # eye-pair midpoint, full-res pixels
    tail_px: tuple | None = None
    eye_state: str | None = None         # 'normal' | 'item' | 'buffer'
    eye_asym: float = 0.0
    ramp_span: float = 0.0
    path: list = field(default_factory=list)     # ordered cells, head -> tail
    head_cell: tuple | None = None
    tail_cell: tuple | None = None
    head_age: int = 0                    # frames since the head was last seen
    facing: tuple | None = None          # (dx,dy) of travel, from the body layout
    head_from_eyes: bool = False         # the head end was fixed by the eye pair
    head_from_pixels: bool = False       # short-gap snout recovery with native red support
    orientation: str = "unknown"         # 'eyes' | 'motion' | 'unknown'
    path_trust: str = "absent"           # revalidated after head/orientation changes
    eye_extrapolated: bool = False       # True = state carried over a gap,
                                        # discount it (see temporal.py)
    eye_travel_axis: tuple | None = None # unsigned axis perpendicular to measured eye separation

    @property
    def alive(self):
        return self.n_px >= MIN_SNAKE_PX


class Perception:
    """Frame -> structured state.

    Not quite stateless: `prev_cells` remembers each family's occupied cells from
    the previous frame, and it is used for one thing only - choosing which blob is
    that family's snake when the mask splits into several.  Processing frames out
    of order, or after reset(), still works; it just falls back to "largest blob".
    """

    def __init__(self):
        # eye-window state, created here as well as in reset(): __init__ does not call reset(), so without
        # this a Perception used before its first reset raises AttributeError in run().
        self.eye_window_used = 0
        self.eye_window_fallbacks = 0
        self.eye_window_mismatch = 0
        self._eye_win_head = None
        self._k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * OPEN_R + 1,) * 2)
        self._k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CLOSE_R + 1,) * 2)
        self._k_open_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._k_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * EYE_NEAR_F + 1,) * 2)
        # Bridges the holes the white eyes punch through the warm mask and a neck
        # washed out by the acceleration flash: measured gaps are 12 px (eye spacing)
        # and 6-8 px.  EYE_BRIDGE_R is in working-resolution pixels.
        self._k_bridge = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * EYE_BRIDGE_R + 1,) * 2)
        # ring used for the diamond outline test
        self._k_dia_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        # larger ring for the ball outline test: the spikes make it a wide border
        self._k_ball_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        # gate used to decide whether an accepted eye pair belongs to the accepted body
        self._k_eye_gate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * EYE_GATE_R + 1,) * 2)
        self._warm_cache = None
        self.last_ms = 0.0
        self.last_evidence = None
        self.last_eye_entry = None
        self.prev_red_px = 0
        self.prev_head_px = {}
        self.prev_head_frame = {}
        self.prev_head_vel = {}
        self.prev_eye_frame = {}
        self.item_centers = {}            # (kind,cx,cy) -> full-resolution centroid
        self._ball_recent = []            # measured centers only; never publish predictions
        self.ball_dropped = []
        self.frame_no = 0
        self.match_t = 0.0
        self.prev_cells = {}

    # ------------------------------------------------------------------ eye window
    def _eye_slits_windowed(self, img_full, warm_mask):
        """Eye slits, searched in a window around the last known head.

        Returns `(slits, pairs, used_window)`.  Falls back to the full frame whenever the window yields no
        PAIR, so the result is never worse than a full search; `used_window` says which path was taken.
        """
        last = getattr(self, "_eye_win_head", None)
        if last is None:
            return None
        half = max(1, int(EYE_WINDOW_CELLS)) * CELL
        x0 = max(0, int(last[0]) - half)
        y0 = max(0, int(last[1]) - half)
        x1 = min(img_full.shape[1], int(last[0]) + half)
        y1 = min(img_full.shape[0], int(last[1]) + half)
        if x1 - x0 < 2 * CELL or y1 - y0 < 2 * CELL:
            return None
        crop = img_full[y0:y1, x0:x1]
        # `_warm_ring` indexes `warm` at WORKING resolution (`warm[y // WORK_SCALE, ...]`) while indexing
        # `img_full` at FULL resolution, so the two arguments live in the SAME coordinate frame.  Slicing the
        # warm mask at full-resolution offsets would put it in a different frame and make the ring test sample
        # the wrong body pixels, so it is sliced in the crop's own frame instead, via the work-resolution
        # offsets, and `_warm_ring` is asked about the crop.
        wx0, wy0 = x0 // WORK_SCALE, y0 // WORK_SCALE
        wx1, wy1 = max(wx0 + 1, x1 // WORK_SCALE), max(wy0 + 1, y1 // WORK_SCALE)
        crop_warm = warm_mask[wy0:wy1, wx0:wx1]
        slits = self.white_slits(crop)
        if len(slits) < 2:
            return None
        # The crop is full resolution while the mask is at WORKING resolution, so the sampler maps a
        # crop pixel back to the mask's own coordinates.  Without it the two live in different frames.
        _wx, _wy = x0 // WORK_SCALE, y0 // WORK_SCALE

        def _warm_of(cx, cy, _wx=_wx, _wy=_wy, _wm=crop_warm):
            mx = _wx + cx // WORK_SCALE
            my = _wy + cy // WORK_SCALE
            if 0 <= my < _wm.shape[0] and 0 <= mx < _wm.shape[1]:
                return bool(_wm[my, mx])
            return False

        pairs = self.eye_pair_candidates(slits, crop, crop_warm, warm_of=_warm_of)
        if not pairs:
            return None
        # translate to full-frame coordinates: a slit's `c` is a (x, y) midpoint, and the pair's midpoint
        # has to move by the same offset or the head would be reported at the crop's origin
        off = (x0, y0)
        moved = []
        for sl in slits:
            d = dict(sl)
            d["c"] = (sl["c"][0] + off[0], sl["c"][1] + off[1])
            moved.append(d)
        moved_pairs = []
        for pr in pairs:
            mid = pr[1]
            moved_pairs.append(_EyePair(pr[0], (mid[0] + off[0], mid[1] + off[1]),
                                        pr[2], pr[3], pr[4],getattr(pr,'pale_supported',False),
                                        getattr(pr,'travel_axis',None)))
        return moved, moved_pairs, True

    def _note_eye_head(self, head_px):
        """Remember the head that centres the next frame's eye window."""
        if head_px is not None:
            self._eye_win_head = (float(head_px[0]), float(head_px[1]))

    def reset(self):
        """Forget identity and motion evidence at a new match/life boundary."""
        self.prev_cells = {}
        self.prev_head_px = {}
        self.prev_head_frame = {}
        self.prev_head_vel = {}
        self.prev_eye_frame = {}
        self.prev_red_px = 0
        self._warm_cache = None
        self.last_evidence = None
        self.last_eye_entry = None
        self.item_centers = {}
        self._ball_recent = []
        self.ball_dropped = []
        self.frame_no = 0
        self.match_t = 0.0
        # eye-window state: the centre for the next frame's window, plus the counters that show whether the
        # window is used or is silently falling back to a full search every frame
        self.eye_window_used = 0
        self.eye_window_fallbacks = 0
        self.eye_window_mismatch = 0
        self._eye_win_head = None

    # ------------------------------------------------------------ rescale
    @staticmethod
    def work(img):
        if WORK_SCALE == 1:
            return img
        return cv2.resize(img, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)

    # -------------------------------------------------------- colour masks
    def families(self, img):
        """Six cv2.inRange calls plus three RGB conditions.

        inRange is a single C++ pass; building the same masks with numpy
        broadcasting cost 12.5 ms and ~20 temporaries per frame.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        R, G, B = img[..., 0], img[..., 1], img[..., 2]
        Ri = R.astype(np.int16)
        red = cv2.inRange(hsv, (0, _RED[2], _RED[3]), (_RED[1], 255, 255))
        red = cv2.bitwise_or(red, cv2.inRange(hsv, (175, _RED[2], _RED[3]), (179, 255, 255)))
        org = cv2.inRange(hsv, (_ORG[0], _ORG[2], _ORG[3]), (_ORG[1], 255, 255))
        if _ORG_BMAX is not None:
            org[img[..., 2] >= _ORG_BMAX] = 0
        pur = cv2.inRange(hsv, (_PUR[0], _PUR[2], _PUR[3]), (_PUR[1], 255, 255))
        blu = cv2.inRange(hsv, (_BLU[0], _BLU[2], _BLU[3]), (_BLU[1], 255, 255))
        # Spike ball: KEEP THIS TIGHT.  The mask is subtracted from red and orange below
        # (the ball is drawn on top of snakes), so widening it deletes the player's own
        # snake: measured on luping1 f540 a loosened version claimed 15370 px and `red`
        # went from 79 cells to zero, which cost 322 of 900 frames.  This window is the
        # ball's bright saturated core, measured at 259-419 px on f634/f660/f700.
        # The lower value bound is essential: dark spikes/rims can bridge a
        # ball to a neighbouring snake. With V=0 luping3 f655's 415px core
        # became a 1745px merged blob and was rejected as a whole.
        ball = cv2.inRange(hsv, (0, _BALL_SMIN, _BALL_VMIN), (12, 255, 255))
        ball = cv2.bitwise_or(ball, cv2.inRange(hsv, (170, _BALL_SMIN, _BALL_VMIN),
                                                (179, 255, 255)))
        grn = cv2.inRange(hsv, (_GRN[0], _GRN[2], _GRN[3]), (_GRN[1], 255, 255))
        # The ball is drawn on top of snakes, so its pixels must not be read as snake -
        # but only where a ball was actually VALIDATED.  A blanket `red[ball > 0] = 0`
        # deletes whatever else falls in the ball's hue window, and the red snake's own
        # coral body does: measured on luping1 f1, red went 5 cells -> 0.
        ball_ok = self._valid_balls(ball)
        if ball_ok is not None:
            red[ball_ok] = 0
            org[ball_ok] = 0
        bean = cv2.inRange(hsv, (_BEAN[0], _BEAN[1], _BEAN[2]), (35, 255, 255))
        # NOTE: diamonds are deliberately NOT built here.  This RGB rule matches the
        # arena floor itself (measured: 93-97% of the frame, one blob of ~178,000 px),
        # so it is computed at full resolution in diamonds_full() instead.
        return (dict(red=(red > 0), org=(org > 0), pur=(pur > 0), blu=(blu > 0),
                     ball=(ball > 0), grn=(grn > 0),
                     bean=(bean > 0)),
                hsv, hsv[..., 1], hsv[..., 2])

    # ------------------------------------------------------------- diamonds
    def diamonds_full(self, img_full, origin=(0,0)):
        """Diamonds, found at FULL resolution.  Returns [(cell_x, cell_y, area)].

        The diamond is a bright cyan hexagon whose dark outline is what separates it
        from the arena floor.  At working resolution the 2-3 px outline is blurred
        away by the 2x area resize, so the gem merges into one blob with the floor.

        That was not a small error.  The old rule was pure RGB and the FLOOR passes
        it: measured on a live frame it claimed 96.6% of the frame as a single blob
        of 178,495 px, centred on cell (14,12).  So the item list always contained a
        gigantic phantom "diamond" at the middle of the board, and the policy drove
        at the middle of the board all game - which is exactly what the live logs
        show its head doing, and what "it does not know where the diamonds are"
        looks like from outside.

        At full resolution the outline holds and the gems are separate small blobs
        (measured 126-611 px) while the floor is one huge blob, so keeping only
        60-2500 px removes the floor and keeps the gems.  The hue bound removes the
        other bright cyan thing on the board, the blue snake (H 102 vs the gem's 88).

        Written with cv2 primitives rather than numpy masks: the numpy version built
        three full-resolution int arrays and cost 23 ms of the 66 ms frame budget,
        this one is a handful of C++ passes.

        KNOWN LIMIT, guarded rather than fixed: during the respawn whiteout the arena is
        desaturated, the floor stops being one big blob and shatters into small cyan
        fragments that all satisfy the size test.  Measured on luping6 f895: 272
        "diamonds" in one frame.  A whiteout frame does not need gems - the bot is not
        steering - and there are never more than about five on the board, so a count
        over DIA_MAX_COUNT is discarded outright.  Measured over luping6 this removed
        every frame above 100 and left the median at 4.
        """
        hsv = cv2.cvtColor(img_full, cv2.COLOR_RGB2HSV)
        R, G, B = img_full[..., 0], img_full[..., 1], img_full[..., 2]
        m = cv2.inRange(hsv, (_DIA[0], _DIA[1], _DIA[2]), (_DIA[3], 255, 255))
        m = cv2.bitwise_and(m, cv2.inRange(B, 161, 255))
        m = cv2.bitwise_and(m, cv2.inRange(G, 141, 255))
        # R<=194 was too tight: a gem that is spawning or being collected carries a WHITE
        # sparkle, and the white lifts R to 200-235 over most of the gem's face.  Measured
        # on luping1 f124 cell (17,1): the gem is plainly on screen but the R test left a
        # broken thin ring that fell under DIA_MIN_PX, so the frame reported one gem where
        # two were visible.  The relaxed test still requires the pixel to be cyan-dominant
        # (B-R>=40 and G-R>=20 below), which the floor and the blue snake do not satisfy.
        m = cv2.bitwise_and(m, cv2.inRange(R, 0, DIA_RMAX))
        m = cv2.bitwise_and(m, cv2.threshold(cv2.subtract(B, R), 40, 255,
                                             cv2.THRESH_BINARY)[1])
        m = cv2.bitwise_and(m, cv2.threshold(cv2.subtract(G, R), 20, 255,
                                             cv2.THRESH_BINARY)[1])
        # SEPARATE THE GEM FROM THE FLOOR.  The gem's interior shares the floor's cyan
        # family and they are joined by 1-2 px bridges (the gem's highlight touching a grid
        # line, the anti-aliased rim).  Measured on luping1 f122 cell (17,1): the gem is
        # plainly on screen, 1466 mask px sit inside its 72x72 box, and yet there is no
        # gem-sized component anywhere - everything is ONE 666835 px blob with the floor,
        # which the 60..2500 px window then discards.  That is why f107/f124/f125 reported
        # one gem where two were visible, and it is not a threshold problem: no colour
        # bound fixes a connectivity problem.
        # One erosion removes those bridges (measured: the gem becomes a separate 165-231 px
        # component at f107/f122/f124 on all four conditions) while the floor stays one huge
        # blob, because the floor's own joins are wider than 1 px.
        if DIA_ERODE > 1:
            # An OPENING (erode then dilate), not a bare erosion: the erosion is what
            # breaks the bridges, the dilation gives the gem its true extent back so the
            # area and the dark-ring fraction are measured on the gem as it actually is.
            # A bare erosion raises the ring fraction enough to need a stricter floor,
            # which then rejects the weak-ring gems - measured, that cost 4 of 15 frames.
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DIA_ERODE, DIA_ERODE))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        n, lab, st, ce = cv2.connectedComponentsWithStats(m, 8)
        # Dark-outline ring: the gem has a dark hexagonal border, a bright patch of the
        # background does not.  Computed once for the whole frame, at full resolution.
        dark = (img_full.max(axis=2) < DIA_DARK_VMAX)
        out = []
        for i in range(1, n):
            a = int(st[i, cv2.CC_STAT_AREA])
            if not (DIA_MIN_PX <= a <= DIA_MAX_PX):
                continue
            x, y = int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP])
            w, h = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
            x0, y0 = max(0, x - 5), max(0, y - 5)
            x1, y1 = min(FRAME_W, x + w + 5), min(FRAME_H, y + h + 5)
            sub = (lab[y0:y1, x0:x1] == i)
            grown = cv2.dilate(sub.view(np.uint8), self._k_dia_ring) > 0
            ring = grown & ~sub
            ring_n = int(ring.sum())
            if ring_n:
                frac = float((dark[y0:y1, x0:x1] & ring).sum()) / ring_n
            else:
                frac = 0.0
            if frac >= DIA_DARK_MIN:
                # the 4th field is the ring strength, which the temporal layer uses to
                # trust a gem outright instead of waiting for a confirming frame
                center=(float(ce[i][0]+origin[0]),float(ce[i][1]+origin[1]))
                cx, cy = int(center[0] // CELL), int(center[1] // CELL)
                out.append((cx, cy, a, round(frac, 3)))
                key = ("dia", cx, cy)
                if a >= getattr(self, "_item_center_areas", {}).get(key, -1):
                    self.item_centers[key] = center
                    if not hasattr(self, "_item_center_areas"):
                        self._item_center_areas = {}
                    self._item_center_areas[key] = a
        return out

    # ---------------------------------------------------------- morphology
    def clean(self, mask):
        """Opening deletes the ~3 px floor texture, which shares the cyan family
        with the blue snake; closing refills the outline gap.  The closing is
        skipped for tiny masks because it is the costlier of the two."""
        u8 = mask.view(np.uint8) if mask.dtype == np.bool_ else mask
        o = cv2.morphologyEx(u8, cv2.MORPH_OPEN, self._k_open)
        if cv2.countNonZero(o) < 600:
            return o != 0
        return cv2.morphologyEx(o, cv2.MORPH_CLOSE, self._k_close) != 0

    def clean_small(self, mask):
        u8 = mask.view(np.uint8) if mask.dtype == np.bool_ else mask
        return cv2.morphologyEx(u8, cv2.MORPH_OPEN, self._k_open_s) != 0

    # -------------------------------------------------------------- grid
    @staticmethod
    def to_grid(mask):
        """Boolean pixel mask -> occupancy grid, one cell per 32x32 block.

        The occupancy bar is 30% of a cell and it is right for a long snake: measured
        on luping1 the reported length matches the user-confirmed final score exactly
        (peak 106-109 against a score of 109).  It is wrong for a SHORT one, and the
        two cases it gets wrong are both real:

          * one cell of snake can straddle four cells at 25% each and vanish - which
            is what a respawn looks like - handled by the looser second pass below;
          * a partially occluded head is a fragment of a cell.  Measured on luping1
            f507: the eyes decoded exactly, and the warm body the anchor could reach
            through the white the eyes punched out was 64 px = 0.4 of a cell, so the
            snake was reported with length 0 and no path at all while 11071 px of its
            body was plainly on screen.

        A fragment is still evidence of a snake being THERE, so for a snake short
        enough that this matters it is credited as a single cell at its centroid.  A
        long snake's cells are all above the bar anyway, so the rule cannot touch the
        validated length readings.
        """
        m = mask.astype(np.float32)
        h, w = m.shape
        hh, ww = h // ROWS, w // COLS
        g = m[:hh * ROWS, :ww * COLS].reshape(ROWS, hh, COLS, ww).sum(axis=(1, 3))
        strict = g > (0.30 * hh * ww)
        if not strict.any() and g.max() > 0:
            strict = g > (0.15 * hh * ww)
        cells = strict
        if not cells.any() and int(m.sum()) >= SHORT_SNAKE_MIN_PX:
            # Nothing reached even the loose bar, so all that is left is a fragment.  A
            # fragment is still evidence that a snake is THERE, so credit it as one cell
            # where it actually is.  A one-cell snake at a half-cell offset is four
            # quarter-cells = 240 px, which passes the loose bar anyway and must keep its
            # OWN four cells - this pass is only for what is left after that, so it ADDS
            # to the grid rather than replacing it.
            ys, xs = np.nonzero(m)
            if ys.size:
                cy = int(ys.mean() // hh)
                cx = int(xs.mean() // ww)
                cells = cells.copy()
                cells[min(ROWS - 1, cy), min(COLS - 1, cx)] = True
        # 2026-10-18 撤回"孤立岛过滤"：它是错的。
        # 用户指正："加速的白光是周期性从蛇头扫到蛇尾，常常会在蛇身中间让你把蛇识别成两截。
        # 死亡的闪光是五颜六色的粒子效果，这个才确实符合你的丢弃判断。"
        # 被白光切断的两截**都是红色身体**，丢掉后半截等于把蛇读短（连带影响路径、尾巴与安全判定）；
        # 只有死亡闪光的碎片才该丢，而那种帧（泛白/缓冲）本来就不下发控制。所以恢复原样。
        return cells

    @staticmethod
    def _drop_islands(cells):
        """只保留与最大连通域相连的格子（形状级识别的第一层）。

        用户 10-18："按原计划做形状级识别（覆盖率 + 连通形状）"。
        覆盖率（30% 门槛）已经在上面生效，但它挡不住**成团的假格**：
        加速白光扫过蛇身、以及死亡闪光，都会在一小片区域里一次点亮好几个互不相连的格子。
        真正的蛇是一条**连续的身体**，所以任何与主体不连通的格子都不可能是蛇 —— 直接丢掉。
        实测意义（第 18 局 / round_003 的两次误判）：这类假格会让"自己的身体"被高估，
        可用空间被当成障碍，随后转向撞上真身体。

        只在格子数 >= 3 时做过滤：一两格的短蛇本身就是"一块"，过滤会把它误删。
        """
        if cells is None:
            return cells
        idx = np.argwhere(cells)
        if len(idx) < 3:
            return cells
        seen = np.zeros_like(cells, dtype=bool)
        best = None
        best_size = 0
        h, w = cells.shape
        for y0, x0 in idx:
            if seen[y0, x0]:
                continue
            stack = [(int(y0), int(x0))]
            seen[y0, x0] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and cells[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            if len(comp) > best_size:
                best_size = len(comp)
                best = comp
        if best is None or best_size == len(idx):
            return cells
        out = np.zeros_like(cells, dtype=bool)
        for y, x in best:
            out[y, x] = True
        return out

    @staticmethod
    def grid_path(g):
        """Ordered cell path through a snake-shaped occupancy grid.

        Two move sets are tried and whichever traversal covers MORE of the body
        wins.  This matters because a snake's cell set is often two cells thick
        (the outline and the closing morphology spill it over the 30% cell
        threshold), and 8-connectivity happily cuts diagonally down such a ribbon,
        skipping every other cell: measured over 1614 red-snake frames the
        8-connected path covered only 76.4% of the body - a median of 4 cells
        missing, and only 11.3% of frames complete.  Everything downstream that
        walks the body inherits that error: the tail-vacating rule in policy.py
        reads cell positions off this path, and `facing` is path[0]-path[1], which
        a diagonal step turns into a non-axis direction.

        4-connectivity must walk the ribbon instead: 90.5% coverage, complete in
        59.0% of frames.  Taking the better of the two reaches 90.8%, so the
        8-connected traversal is kept purely as a fallback for a region that is not
        4-connected at all, such as a staircase turn.
        """
        pts = {(int(x), int(y)) for y, x in zip(*np.nonzero(g))}
        if not pts:
            return []
        best = []
        for moves in (_MOVES4, _MOVES8):
            path = Perception._bfs_path(pts, moves)
            if len(path) > len(best):
                best = path
        return best

    @staticmethod
    def _bfs_path(pts, moves):
        """Double BFS: from any end to the farthest cell, then back, giving the
        longest shortest-path between two endpoints of the region."""

        def nb(p):
            x, y = p
            return [(x + dx, y + dy) for dx, dy in moves if (x + dx, y + dy) in pts]

        def bfs(start):
            seen = {start: None}
            q, last = [start], start
            while q:
                nxt = []
                for p in q:
                    last = p
                    for t in nb(p):
                        if t not in seen:
                            seen[t] = p
                            nxt.append(t)
                q = nxt
            return last, seen

        ends = [p for p in pts if len(nb(p)) <= 1]
        far, _ = bfs(ends[0] if ends else next(iter(pts)))
        other, prev = bfs(far)
        path, cur = [], other
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        return path

    # ------------------------------------------------------------- blobs
    @staticmethod
    def blobs(mask, min_px):
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.view(np.uint8), 8)
        out = []
        for i in range(1, n):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a >= min_px:
                out.append((a, i, stats[i], cent[i]))
        out.sort(key=lambda t: -t[0])
        return out, lab

    @staticmethod
    def _valid_balls(ball):
        """Mask of ball blobs that look like a spike ball; None when there is none.

        A spike ball's bright core measures 259-419 px in a 21-25 px box with fill
        0.56-0.70 (luping1 f634/f660/f700); the red snake's body shares the hue window
        and is 4000-14000 px of fill 0.17-0.30.  Returning only the validated blobs
        bounds the red/orange subtraction to the ball itself.
        """
        n, lab, st, _ce = cv2.connectedComponentsWithStats(ball, 8)
        keep = None
        for i in range(1, n):
            a = int(st[i, cv2.CC_STAT_AREA])
            if not (BALL_MIN_PX <= a <= BALL_MAX_PX):
                continue
            w, h = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
            if not (BALL_BOX_MIN <= w <= BALL_BOX_MAX and
                    BALL_BOX_MIN <= h <= BALL_BOX_MAX):
                continue
            if a / max(1.0, float(w * h)) < BALL_FILL_MIN:
                continue
            if keep is None:
                keep = np.zeros_like(ball)
            keep[lab == i] = 255
        return keep

    def _choose_balls(self, candidates):
        """Rank measured blobs by one-to-one recent support when there are >3.

        A coral snout also passes round/outline tests occasionally (f711). True
        slow balls have recent measured centers, while that snout is a new claim.
        Missing measurements are NOT filled from this history.
        """
        recent = [r for r in self._ball_recent if self.frame_no - r['frame'] <= 20]
        edges = []
        for i, c in enumerate(candidates):
            for j, r in enumerate(recent):
                age = self.frame_no - r['frame']
                dist = math.hypot(c['px'][0] - r['px'][0], c['px'][1] - r['px'][1])
                # Ball centers move ~1.5px/frame plus segmentation jitter.
                if dist <= CELL * 0.75 + 2 * age:
                    edges.append((dist, i, j))
        assigned, used = {}, set()
        for dist, i, j in sorted(edges):
            if i not in assigned and j not in used:
                assigned[i] = (j, dist)
                used.add(j)
        for i, c in enumerate(candidates):
            j, dist = assigned.get(i, (-1, float('inf')))
            c['hits'] = recent[j]['hits'] + 1 if j >= 0 else 1
            c['matched'] = j >= 0
            c['distance'] = dist if j >= 0 else None
        ranked = sorted(candidates, key=lambda c: (
            not c['matched'], -c['hits'], -c['ring'], abs(c['entry'][2] - 400)))
        chosen = ranked[:3]
        self.ball_dropped = [dict(c, reason='more-than-three: weaker measured continuity')
                             for c in ranked[3:]]
        keep_history = [r for j, r in enumerate(recent) if j not in used]
        keep_history.extend(dict(px=c['px'], frame=self.frame_no, hits=c['hits'])
                            for c in chosen)
        self._ball_recent = keep_history[-12:]
        return chosen

    @staticmethod
    def _grow_component(warm, comp, r):
        """Expand a component through the warm mask across a gap of up to r pixels.

        Single-linkage within the given radius: the warm mask is dilated by r (which
        joins anything within 2r) and everything in the component's new blob is taken.
        Used only to recover from a lost eye pair, and capped by EYE_HOLE_R_SEQ, whose
        largest value is 17 working px - two full cells of clear floor is 32, so a
        neighbouring snake that is merely passing by is not swallowed.
        """
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        grown = cv2.dilate(comp.view(np.uint8), k) > 0
        return warm & grown

    @staticmethod
    def _elong(ys, xs):
        """Elongation of a blob from its second moments (major/minor axis ratio).

        The discriminator between a bean and a length of snake tail: measured bean
        1.3-2.0, tail fragments 2.4-4.8.  Cheap - it is four means over the blob's own
        pixels, which the caller already extracted for other reasons.
        """
        yy = ys.astype(np.float64) - ys.mean()
        xx = xs.astype(np.float64) - xs.mean()
        mu20 = float((xx * xx).mean())
        mu02 = float((yy * yy).mean())
        mu11 = float((xx * yy).mean())
        common = math.sqrt(max(1e-9, 4 * mu11 ** 2 + (mu20 - mu02) ** 2))
        l1 = (mu20 + mu02 + common) / 2
        l2 = (mu20 + mu02 - common) / 2
        return math.sqrt(max(l1, 1e-6) / max(l2, 1e-6))

    # -------------------------------------------------------------- eyes
    def eyes_full(self, img_full, snake_mask_full, roi):
        """White slit blobs at FULL resolution (a slit is ~3x8 px; at half res it
        collapses to a 2-6 px blob whose moments are noise)."""
        x0, y0, x1, y1 = roi
        R = img_full[y0:y1, x0:x1, 0]
        G = img_full[y0:y1, x0:x1, 1]
        B = img_full[y0:y1, x0:x1, 2]
        white = ((R > 213) & (G > 213) & (B > 208)).view(np.uint8)
        cnt = int(cv2.countNonZero(white))
        if not (4 <= cnt <= 2000):
            return []
        near = cv2.dilate(snake_mask_full[y0:y1, x0:x1].view(np.uint8), self._k_near)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(white & near, 8)
        if n <= 2:
            return []
        out = []
        for i in range(1, n):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if not (8 <= a <= 400):
                continue
            ys, xs = np.nonzero(lab == i)
            if ys.size < 5:
                continue
            y = ys.astype(np.float64) - ys.mean()
            x = xs.astype(np.float64) - xs.mean()
            mu20 = float((x * x).mean()); mu02 = float((y * y).mean()); mu11 = float((x * y).mean())
            common = math.sqrt(max(1e-9, 4 * mu11 ** 2 + (mu20 - mu02) ** 2))
            l1 = (mu20 + mu02 + common) / 2; l2 = (mu20 + mu02 - common) / 2
            out.append(dict(n=a, c=(float(cent[i][0]) + x0, float(cent[i][1]) + y0),
                            elong=math.sqrt(max(l1, 1e-6) / max(l2, 1e-6)),
                            ang=0.5 * math.atan2(2 * mu11, mu20 - mu02)))
        return out

    @staticmethod
    def eye_state(eyes):
        """-> (midpoint | None, 'normal'|'item'|'buffer'|None, asym).

        Rotation-invariant: the camera is fixed but the snake turns, so the state
        is read from the pair's geometry relative to its own axis.
          buffer : both slits compact crosses (elongation < 1.8)
          normal : the pair is SYMMETRIC about the travel axis (both slits equally
                   far forward) -> the separation vector is perpendicular to it
          item   : the pair is V-tilted, one slit further forward -> the
                   separation vector has a component ALONG the axis
        """
        cands = [e for e in eyes if e["n"] >= 8]
        best = None
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                d = math.hypot(a["c"][0] - b["c"][0], a["c"][1] - b["c"][1])
                if 6.0 <= d <= 28.0 and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            return None, None, 0.0
        _, a, b = best
        mid = ((a["c"][0] + b["c"][0]) / 2.0, (a["c"][1] + b["c"][1]) / 2.0)
        if a["elong"] < BUFFER_ELONG and b["elong"] < BUFFER_ELONG:
            return mid, "buffer", 0.0
        ax = np.array([math.cos(a["ang"]), math.sin(a["ang"])])
        bx = np.array([math.cos(b["ang"]), math.sin(b["ang"])])
        if float(np.dot(ax, bx)) < 0.0:          # align before averaging
            bx = -bx
        axis = ax + bx
        n = float(np.linalg.norm(axis))
        if n < 1e-6:
            return mid, "normal", 0.0
        axis = axis / n
        sep = np.array([b["c"][0] - a["c"][0], b["c"][1] - a["c"][1]])
        perp = np.array([-axis[1], axis[0]])
        asym = abs(float(np.dot(sep, axis))) / max(abs(float(np.dot(sep, perp))), 1e-6)
        return mid, ("item" if asym > ITEM_ASYM else "normal"), asym

    @staticmethod
    def eye_state_from(cands):
        """-> (state, asym, head_px) from already-computed (state, asym, midpoint).

        The eye state has to come from a pair that is on OUR snake, and there is no
        restriction left to make: white_slits() only returns slits that have body on
        both sides, and only the player's snake carries the marker (section 3).  So
        the closest-spaced pair wins, both halves agreeing, which is the same
        preference order eye_pair_candidates() applies.

        The head pixel IS the midpoint between the eyes.  It used to be recomputed by
        re-detecting the slits inside the snake's bounding box, which failed whenever
        the pair straddled the box edge - measured on luping1 f444, where the snake's
        head was at the frame edge and the pair was dropped by the bbox test.
        """
        if not cands:
            return None, 0.0, None
        best = min(cands, key=lambda t: t[1])
        state, asym, mid = best
        if state is None:
            return None, 0.0, None
        return state, float(asym), (float(mid[0]), float(mid[1]))

    def white_slits(self, img_full, restrict=None):
        """Eye-slit candidates at full resolution: (midpoint, state, asym, stats).

        Used to find the player's snake WITHOUT depending on its red mask.  The red
        family only covers the head half of the body, because the body's hue slides
        from red at the head to orange at the tail (section 9.6); when the snake is
        short - most importantly right after a respawn - the red part can be under
        MIN_SNAKE_PX and the snake is reported as absent.  Measured against the
        running game: no red snake at all in 45% of in-match frames, and the eyes
        readable in 27%.  The eye pair is the one certain identity marker (section 3:
        only the player's snake has two white slits), and it is findable on its own.

        Two changes over the earlier version, both measured on luping1 frames 1-900:

        * the blob area ceiling is 400 -> EYE_SLIT_MAX_PX.  When the acceleration
          flash is over the head, the two slits, their bloom and the washed-out snout
          merge into one white blob: measured 1326 px on luping1 f444, where the eye
          pair is plainly visible to a human but every blob was over the ceiling, so
          the frame decoded as "no eyes" - one of 60 such frames.
        * an EMBEDDING test is applied per blob.  The eyes sit inside the head, so a
          few pixels either side of the slit, perpendicular to it, are always snake
          body.  White things that are NOT eyes - the diamond sparkle, the green
          token's glow, the potion star, the grid-line intersections, the arena's
          static corner squares - have floor or empty space there.  Measured: of the
          23 white blobs on luping1 f723, only the 2 real slits pass.  Without the
          test the pair search sees more than 20 candidates and every frame with
          extra sparkle reports "no valid pair" (60 frames on luping1 alone).

        Returns a list of dicts: c (full-res midpoint), elong, ang, n, and `warm`
        (how many of the 8 perpendicular probes were warm).
        """
        R, G, B = img_full[..., 0], img_full[..., 1], img_full[..., 2]
        # A slit has to be bright and near-neutral.  A second, looser threshold (accept
        # anything bright with a lifted blue channel) was tried to recover eyes the
        # acceleration flash had washed out, and REJECTED: measured on luping1 f862 it
        # turned two clean 26 px and 21 px eye marks into one 46 px and one 37 px blob,
        # and on f444/f445 it merged the eyes with the bloom into a single 80x52 px
        # blob that no longer fits the slit geometry at all - worse, not better.  The
        # strict threshold loses those frames, but it keeps the geometry usable, and the
        # frames it loses are the ones where an eye is genuinely not separable.
        pale_body = _pale_red_mask(img_full)
        # A pale coral head can satisfy all three brightness cutoffs itself.
        # Remove R-dominant skin ONLY in a locally dense low-saturation head:
        # isolated tinted antialias pixels on a saturated head are real eye
        # edges (luping7 f201 Xs:19/20px, reduced to10/8 by global subtraction).
        low_skin = pale_body & ((R.astype(np.int16) - np.minimum(G, B)) <= 0.30 * R)
        width = 2 * (CELL // 4) + 1
        skin_support = cv2.boxFilter(low_skin.view(np.uint8), cv2.CV_32F,
            (width, width), normalize=False, borderType=cv2.BORDER_CONSTANT)
        dense_skin = pale_body & (skin_support >= SHORT_SNAKE_MIN_PX * WORK_SCALE ** 2)
        white = (R > 213) & (G > 213) & (B > 208) & ~dense_skin
        if restrict is not None:
            white = white & restrict
        cnt = int(np.count_nonzero(white))
        if not (4 <= cnt <= 60000):
            return []
        n, lab = cv2.connectedComponents(white.view(np.uint8), 8)[:2]
        ys, xs = np.nonzero(white)
        ids = lab[ys, xs]
        order = np.argsort(ids, kind="stable")
        ys, xs, ids = ys[order], xs[order], ids[order]
        edges = np.searchsorted(ids, np.arange(1, n + 1))
        out = []
        start = 0
        for i in range(1, n):
            # edges[i] ends label i; edges[i-1] starts it. The old indexing
            # silently omitted the final white component (often the second eye).
            end = int(edges[i])
            count = end - start
            if EYE_SLIT_MIN_PX <= count <= EYE_SLIT_MAX_PX:
                yy = ys[start:end].astype(np.float64)
                xx = xs[start:end].astype(np.float64)
                cy, cx = yy.mean(), xx.mean()
                dy, dx = yy - cy, xx - cx
                mu20 = float((dx * dx).mean())
                mu02 = float((dy * dy).mean())
                mu11 = float((dx * dy).mean())
                common = math.sqrt(max(1e-9, 4 * mu11 ** 2 + (mu20 - mu02) ** 2))
                l1 = (mu20 + mu02 + common) / 2
                l2 = (mu20 + mu02 - common) / 2
                elong = math.sqrt(max(l1, 1e-6) / max(l2, 1e-6))
                ang = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
                warm = self._warm_around_warm(img_full, cx, cy, ang)
                pale_hits = 0
                pale_supported = False
                if warm < EYE_WARM_MIN:
                    px, py = math.cos(ang + math.pi / 2), math.sin(ang + math.pi / 2)
                    for d in (4, 8):
                        for sign in (-1, 1):
                            xx = int(round(cx + sign * d * px))
                            yy = int(round(cy + sign * d * py))
                            if (0 <= xx < img_full.shape[1] and 0 <= yy < img_full.shape[0]
                                    and pale_body[yy, xx]):
                                pale_hits += 1
                    # This is a LOCAL recovery, not a general lower white/warm bar:
                    # both sides of a slit must be coral and a head-sized patch
                    # must contain actual R-dominant body pixels, not yellow glow.
                    half = CELL // 2
                    x0, x1 = max(0, int(cx) - half), min(img_full.shape[1], int(cx) + half + 1)
                    y0, y1 = max(0, int(cy) - half), min(img_full.shape[0], int(cy) + half + 1)
                    body_px = int(pale_body[y0:y1, x0:x1].sum())
                    pale_supported = (pale_hits >= 3 and
                        body_px >= SHORT_SNAKE_MIN_PX * WORK_SCALE * WORK_SCALE)
                    prev = self.prev_head_px.get("red")
                    age = self.frame_no - self.prev_head_frame.get("red", -999)
                    if (pale_supported and prev is not None and 1 <= age <= 3
                            and math.hypot(cx - prev[0], cy - prev[1]) > 4 * CELL * age):
                        pale_supported = False
                if warm >= EYE_WARM_MIN or pale_supported:
                    out.append(dict(n=count, c=(float(cx), float(cy)), elong=elong,
                                    ang=ang, warm=warm, pale=pale_supported,
                                    pale_hits=pale_hits))
            start = end
        return out

    @staticmethod
    def _warm_around_warm(img_full, cx, cy, ang):
        """Per-blob warm probe count, used to drop obvious decoys early.

        The angle comes from the blob's own second moments and that is NOT reliable
        enough to place probes precisely - the slit is cut by its own specular
        highlight, so the moments describe the surviving half.  Measured on luping1
        f444 a bean's white shine scores elong 1.85, ang -1.10 rad, a diagonal it does
        not have.  So this is a coarse gate only; the decision is made by _warm_ring,
        which probes around the PAIR midpoint along the pair's own axis.
        """
        h, w = img_full.shape[:2]
        px, py = math.cos(ang + math.pi / 2), math.sin(ang + math.pi / 2)
        hits = 0
        for d in (4, 8):
            for sgn in (1, -1):
                x, y = int(round(cx + sgn * d * px)), int(round(cy + sgn * d * py))
                if not (0 <= x < w and 0 <= y < h):
                    continue
                if _is_warm_px(img_full[y, x]):
                    hits += 1
        return hits

    @classmethod
    def _warm_ring(cls, img_full, mid, sep_ang, warm, allow_pale=False, warm_of=None):
        """Sample a ring around an eye midpoint: -> (warm hits, probed points).

        Only the player's snake carries two white slits (section 3), so a candidate
        pair sitting on the snake's head must be SURROUNDED by red/orange body.  This
        is the test that separates a real eye pair from the white specular shine on a
        yellow bean, which is the same size, the same roundness and, measured on
        luping1 f444, sits with beans under half of its neighbourhood.

        The probes are placed from the MIDPOINT and directed by the pair's own
        separation axis, not by either blob's moments: the two slits are mirrored
        about the head axis, so the line between them is perpendicular to it and,
        unlike a single blob's second moments, is not disturbed by the highlight that
        cuts each slit in two.

        Distances start at 7 px because the mask used here is dilated by 6 px when the
        body is re-joined across the eye holes, and a probe inside that band would
        report body for anything the eyes are merely NEAR.
        """
        h, w = img_full.shape[:2]
        # perpendicular to the separation direction == the head's travel axis
        px, py = math.cos(sep_ang + math.pi / 2), math.sin(sep_ang + math.pi / 2)
        sx, sy = math.cos(sep_ang), math.sin(sep_ang)
        hits = probes = 0
        for d in EYE_RING_D:
            for t in (-1.0, -0.5, 0.0, 0.5, 1.0):
                for sgn in (1, -1):
                    x = int(round(mid[0] + sgn * d * px + t * d * 0.8 * sx))
                    y = int(round(mid[1] + sgn * d * py + t * d * 0.8 * sy))
                    probes += 1
                    if not (0 <= x < w and 0 <= y < h):
                        continue
                    # `warm_of` lets the caller place the mask in the same frame as `img_full`; the
                    # default is the original working-resolution lookup, so the full-frame path is
                    # byte-identical to before.
                    _wm = (warm_of(x, y) if warm_of is not None
                           else warm[y // WORK_SCALE, x // WORK_SCALE])
                    if (_wm or _is_warm_px(img_full[y, x])
                            or (allow_pale and _is_pale_red_px(img_full[y, x]))):
                        hits += 1
        return hits, probes

    @staticmethod
    def _pair_axis(a, b):
        """Direction of the line joining two slits, in radians."""
        return math.atan2(b["c"][1] - a["c"][1], b["c"][0] - a["c"][0])

    @classmethod
    def eye_pair_candidates(cls, slits, img_full=None, warm=None, warm_of=None):
        """Plausible eye pairs, best first: (score, midpoint, state, asym, ring).

        Preference order matters now that the search covers the whole frame instead
        of one snake's bounding box: the measured slit spacing is 12 px (section 3),
        and a pair of actual slits is elongated rather than round.

        When `img_full` and `warm` are given, every candidate pair is also put through
        the ring test and pairs that are not surrounded by body are DROPPED.  That is
        what removes the white specular shine on a bean, which otherwise pairs up with
        its neighbour and puts a phantom head - and a phantom `normal` accel state -
        on an item.
        """
        cands = [e for e in slits
                 if EYE_PAIR_MIN_PX <= e["n"] <= EYE_PAIR_MAX_PX
                 and e["elong"] >= EYE_PAIR_MIN_ELONG]
        pairs = []
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                d = math.hypot(a["c"][0] - b["c"][0], a["c"][1] - b["c"][1])
                if not (6.0 <= d <= 28.0):
                    continue
                mid, state, asym = cls.eye_state([a, b])
                if mid is None:
                    continue
                ring = None
                if img_full is not None and warm is not None:
                    pale_pair = bool(a.get("pale") or b.get("pale"))
                    hits, probes = cls._warm_ring(img_full, mid, cls._pair_axis(a, b),
                                                  warm, allow_pale=pale_pair, warm_of=warm_of)
                    ring = hits
                    if hits < EYE_RING_MIN:
                        continue
                # Distance is the primary term, but a pair that is two halves of one
                # merged blob is not a pair: real slits are near-parallel and mirrored
                # about the head axis.  Two independent slits on unrelated white
                # objects are not, and admitting them put a phantom head on a sparkle.
                # The geometric match decides which pair is the eyes, so it is weighted
                # heavily; the rest are tie-breakers only.  Measured on luping1 f507,
                # where a pair of stray marks 3 cells behind the head also passed the
                # ring test at 22 of 40 against the real eyes' 20: with the small weights
                # the stray pair won and put our head 3 cells off.  The real pair is the
                # one whose separation is 12 px, because that separation IS the anatomy
                # (section 3).  Scale: a 1 px error costs 100 points, and the whole
                # tie-break budget is 30.
                score = 100.0 * abs(d - EYE_SPACING)
                if min(a["elong"], b["elong"]) < BUFFER_ELONG:
                    score += 12.0         # both eye forms are at least somewhat slit-like
                ax = np.array([math.cos(a["ang"]), math.sin(a["ang"])])
                bx = np.array([math.cos(b["ang"]), math.sin(b["ang"])])
                if float(np.dot(ax, bx)) < 0.0:
                    bx = -bx
                score += 18.0 * (1.0 - float(np.dot(ax, bx)))
                if ring is not None:
                    score -= 0.2 * ring   # a well-surrounded pair is the likelier one
                sep_x=abs(a['c'][0]-b['c'][0]);sep_y=abs(a['c'][1]-b['c'][1])
                travel_axis=((1,0) if sep_y>sep_x else (0,1)) if max(sep_x,sep_y)>2*min(sep_x,sep_y) else None
                pairs.append(_EyePair(score, mid, state, asym, ring,
                                      a.get("pale") or b.get("pale"),travel_axis))
        pairs.sort(key=lambda t: t[0])
        return pairs

    # ------------------------------------------------- warm component helpers
    def _warm_labels(self, warm):
        """Cached component labels of a warm mask, keyed on the mask's CONTENT.

        The key must not be the array's identity.  This method is called on temporary
        masks (the warm mask with the eye hole filled in), and a temporary is freed
        as soon as the call returns - so CPython reuses its address for the next
        temporary, and an id()-based key then matches a completely different mask.
        Measured: that collision handed the scorer a component of 167643 px on a
        464x400 frame, which is more pixels than the frame has.

        The fingerprint is a checksum of a small strided sample plus the shape and
        the exact pixel count: two masks that agree on all three are the same mask
        for this purpose, and 2000 bytes is cheaper to hash than a labelling pass.

        Returns (lab, areas, bboxes, centroids), all indexed by component id.
        """
        key = (warm.shape, int(warm.sum()),
               hash(warm[::16, ::16].tobytes()) if warm.size else 0)
        hit = self._warm_cache
        if hit is not None and hit[0] == key:
            return hit[1]
        n, lab, st, ce = cv2.connectedComponentsWithStats(warm.view(np.uint8), 8)
        areas = [int(st[i, cv2.CC_STAT_AREA]) for i in range(n)]
        # Label 0 is the BACKGROUND, and OpenCV reports its "area" as every pixel not
        # in a component - i.e. 167643 px of a 464x400 frame, which is larger than any
        # snake can be.  It must never be mistaken for a candidate: measured on
        # luping1 f507 the scorer ranked it top and then took `warm & (lab == 0)`,
        # which is empty, so the frame decoded as "no snake" while the eyes and the
        # whole body were both plainly present.
        areas[0] = 0
        bboxes = [(int(st[i, cv2.CC_STAT_LEFT]), int(st[i, cv2.CC_STAT_TOP]),
                   int(st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] - 1),
                   int(st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] - 1))
                  for i in range(n)]
        centroids = [(float(ce[i][0]), float(ce[i][1])) for i in range(n)]
        self._warm_cache = (key, (lab, areas, bboxes, centroids))
        return self._warm_cache[1]

    def _nearest_components(self, lab, areas, centroids, wx, wy, count):
        """The `count` nearest components to a working-resolution point."""
        ordered = []
        for i in range(1, len(areas)):
            if areas[i] < 16:
                continue
            cx, cy = centroids[i]
            ordered.append((math.hypot(cx - wx, cy - wy), i))
        ordered.sort(key=lambda t: t[0])
        return [i for _d, i in ordered[:count]]

    # --------------------------------------------------- player identification
    def _clusters(self, warm):
        """Warm mask bridged into clusters: (lab, areas, bboxes, centroids, clab).

        The player's snake is ONE object, but when the eye slits punch a hole in it
        or the acceleration flash washes a neck out, the mask breaks it into two or
        more components.  Bringing back components within EYE_BRIDGE_R of each other
        reconstructs the object.  Measured gaps: 12 px (the eye spacing) and 6-8 px
        (a washed neck), against 16 working px for one clear cell between two snakes.
        """
        lab, areas, bboxes, centroids = self._warm_labels(warm)
        clab = cv2.connectedComponents(
            (cv2.dilate(warm.view(np.uint8), self._k_bridge) > 0).astype(np.uint8), 8)[1]
        return lab, areas, bboxes, centroids, clab

    def _nearest_cluster(self, warm, clab, areas, bboxes, centroids, wx, wy,
                         max_d=64.0):
        """Cluster id of the warm body nearest a working-resolution point.

        The eye midpoint sits BETWEEN the slits, i.e. inside the white hole, so the
        point itself is usually on no component at all.  Nearest is measured from the
        component's bounding box, which is what makes an L-shaped body work: its
        centroid can be cells away from the part next to the eyes.
        """
        best_cid, best_d = None, 1e9
        for i in range(1, len(areas)):
            if areas[i] < 16:
                continue
            x0, y0, x1, y1 = bboxes[i]
            dx = max(x0 - wx, 0, wx - x1)
            dy = max(y0 - wy, 0, wy - y1)
            d = math.hypot(dx, dy)
            if d < best_d:
                best_cid, best_d = int(clab[y0, x0]), d
        if best_cid is None or best_d > max_d:
            return None
        return best_cid

    def _prior_overlap(self, clab, cid, prev):
        """Fraction of last frame's player cells that this cluster still covers.

        This is the strongest identity signal there is: a snake is a physical object
        and cannot teleport, so the body occupying where our body was a frame ago is
        ours.  It is what keeps the snake identified through the frames where the eyes
        are hidden by the acceleration flash or by an opponent's head - the cases
        where the colour-first method used to grab whichever blob was biggest.
        """
        if prev is None or not prev.any():
            return -1.0
        up = np.kron(prev, np.ones((CELL_W, CELL_W), bool))
        up = up[:clab.shape[0], :clab.shape[1]]
        hit = int((up & (clab == cid)).sum())
        return hit / max(1.0, float(up.sum()))

    def player_region(self, img_full, cleaned, slits, pairs):
        """Player's snake -> (mask, evidence, eye_entry) or (None, why, None).

        Every available route to the identity is scored in the same currency instead of
        being tried in a chain, because any one of them can fail on its own:
          * the eyes            - definitive when found, but absent in 36% of luping4;
          * overlap with the previous frame - definitive when the body is visible at
            all, but useless on a respawn;
          * plain area          - last resort, and only accepted with real overlap.

        Two measured traps this has to survive, both of which the old eye-only anchor
        fell into:

        * the eyes are TRUE but the body they reach is a fragment.  The two slits are
          white, so they punch a hole through the warm mask and cut the head off the
          neck.  Measured on luping1 f507: eyes detected exactly at (583,585) and
          (595,586) 12 px apart, the body is 11071 px running the length of the board,
          and the component the midpoint landed on was the 64 px snout stub below it -
          the frame decoded as a ZERO-cell snake.  The fill disks in EYE_HOLE_R_SEQ
          step outwards until a component of at least one cell is reached.
        * the eyes are FALSE. On luping1 f104, the bean sparkle on a purple
          snake passed the warm/ring test and claimed (16,6), while the actual
          player head was (8,19). Position continuity plus supported old body
          rejects this claim; area consistency alone was not sufficient.
          Beware of treating every short eye-anchored component as false eyes:
          f862 has real white V eyes at (1,1), but the potion overlay severs
          their small snout component from the large true body. That is a
          reconnection failure, not evidence that the genuine eyes are sparkle.
        """
        warm = cleaned["red"] | cleaned["org"]
        if not warm.any():
            return None, "no-warm-mask", None
        min_area = max(32, MIN_SNAKE_PX // 4)
        _lab, areas, bboxes, centroids, clab = self._clusters(warm)
        prev = self.prev_cells.get("red")
        prev_npx = self.prev_red_px
        ncl = int(clab.max()) + 1
        up = None
        supported_large = np.zeros_like(warm)
        if prev is not None and prev_npx:
            up = np.kron(prev, np.ones((CELL_W, CELL_W), bool))
            up = up[:clab.shape[0], :clab.shape[1]]
            # Reuse the existing body-growth target, not a new size heuristic:
            # a large component must itself have this much prior native support.
            target = max(MIN_SNAKE_PX, int(BODY_GROW_FRAC * prev_npx))
            for c in np.unique(clab[up]):
                if c:
                    body = warm & (clab == c)
                    if int((body & up).sum()) >= target:
                        supported_large |= body

        hands = []          # (mask, evidence, eye_entry, eye_quality)
        for entry in pairs[:4]:
            mid = entry[1]
            pair_sc = entry[0] if len(entry) > 4 else 0.0
            wx, wy = int(mid[0]) // WORK_SCALE, int(mid[1]) // WORK_SCALE
            if not (0 <= wx < WORK_W and 0 <= wy < WORK_H):
                continue
            comp = gate = None
            entry_warm = warm
            if getattr(entry, "pale_supported", False):
                # Reconnect only the verified pale eye's local coral head/neck.
                # Do not add pale pixels globally or around ordinary star pairs.
                rad = CELL // WORK_SCALE
                x0, x1 = max(0, wx - rad), min(WORK_W, wx + rad + 1)
                y0, y1 = max(0, wy - rad), min(WORK_H, wy + rad + 1)
                local = self.work(img_full)[y0:y1, x0:x1]
                entry_warm = warm.copy()
                entry_warm[y0:y1, x0:x1] |= _pale_red_mask(local)
            previous_head = self.prev_head_px.get("red")
            head_age = self.frame_no - self.prev_head_frame.get("red", -999)
            need_large = (previous_head is not None and 1 <= head_age <= 3
                          and math.hypot(mid[0] - previous_head[0],
                                         mid[1] - previous_head[1]) <= 2 * CELL
                          and supported_large.any())
            for r in EYE_HOLE_R_SEQ:
                fill = np.zeros(warm.shape, np.uint8)
                cv2.circle(fill, (int(round(mid[0] / WORK_SCALE)),
                                  int(round(mid[1] / WORK_SCALE))), r, 1, -1)
                disk = fill > 0
                filled = self._clusters(entry_warm | disk)[4]
                cid = int(filled[wy, wx])
                if cid == 0:
                    cid = self._nearest_cluster(entry_warm, filled, areas, bboxes, centroids,
                                                wx, wy)
                if not cid:
                    continue
                comp = entry_warm & (filled == cid)
                gate = comp | disk
                # Usually one cell is enough. A verified near-previous eye can
                # sit on a sufficiently large but disconnected snout (f862:260px),
                # so do not stop until the SAME prior-supported large body is reached.
                # Existing radii suffice (r17:23981 native px). Disk pixels never
                # enter comp; they only establish connectivity through the overlay.
                if (int(comp.sum()) >= MIN_SNAKE_PX
                        and (not need_large or (comp & supported_large).any())):
                    break
            if comp is None or (need_large and not (comp & supported_large).any()):
                continue
            hands.append((comp, "eyes", (entry, gate), pair_sc))
            if len(hands) >= 3:
                break
        if prev is not None:
            if up is None:
                up = np.kron(prev, np.ones((CELL_W, CELL_W), bool))
                up = up[:clab.shape[0], :clab.shape[1]]
            for c in np.unique(clab[up]):
                if c:
                    hands.append((warm & (clab == c), "prev", None, None))
        for c in sorted(range(1, ncl), key=lambda c: -int((clab == c).sum()))[:2]:
            hands.append((warm & (clab == c), "area", None, None))
        if not hands:
            return None, "no-candidate", None
        # ---- fallback growth: the eyes were lost this frame, so grow the body outward
        # instead of publishing a fragment as the whole snake.  The user's description of
        # these frames ("红蛇过长且多处自身相邻紧贴") matches the measurement: on luping1
        # f540 the head end is covered by another snake, so the largest component is a
        # few cells of a 79-cell snake.  The stop condition is the length the previous
        # frame saw, so a genuinely short snake is never inflated.
        if prev_npx:
            target = max(MIN_SNAKE_PX, int(BODY_GROW_FRAC * prev_npx))
            start = max((m for m, ev, _e, _q in hands if ev != "eyes"),
                        key=lambda m: int(m.sum()), default=None)
            if start is not None and int(start.sum()) < target:
                grown = start
                for r in EYE_HOLE_R_SEQ:
                    if int(grown.sum()) >= target:
                        break
                    grown = self._grow_component(warm, grown, r)
                if int(grown.sum()) > int(start.sum()):
                    hands.append((grown, "grown", None, None))

        # Two hands can reach the SAME body and then the only thing left to choose on is
        # how good the eye pair was.  That is not a corner case: measured on luping1
        # f507, the real eyes at cell (18,18) and a pair of stray marks at (16,16) both
        # grow into the same 14000 px body and score 24.29 against 24.31 - a 0.02 gap
        # that flipped our head three cells.  So a pair's own geometric quality (how
        # close its separation is to the measured 12 px, EYE_PAIR_QUALITY_PX) is the
        # dominant term for an eye hand, worth up to EYE_QUALITY_BONUS, which is larger
        # than the entire body-fit term.  Body fit then decides between non-eye hands,
        # and continuity-vs-eye is decided by the quality bonus.
        best = None
        # A sparkle can have excellent pair geometry, but cannot become our head
        # thirteen cells away while a substantial continuation of our body remains.
        # Do not impose this gate across a genuine respawn: without a supported
        # continuation the new eye reading must be allowed to re-acquire the snake.
        prev_head = self.prev_head_px.get("red")
        has_continuation = any(
            self._prior_overlap_cluster(m, prev) >= 0.15
            and int(m.sum()) >= max(min_area, int(BODY_MIN_FRAC * prev_npx))
            for m, _ev, _en, _qs in hands)
        for comp, ev, entry, pair_sc in hands:
            # Warm colours alone do not assign ownership. A newly connected
            # orange body may not extend our tail by twenty cells in one frame.
            # Keep previous occupied tiles plus the newly measured head sweep.
            if (prev_head is not None and prev is not None and up is not None
                    and self.frame_no-self.prev_head_frame.get('red',-999)<=3):
                mid=entry[0][1] if entry is not None else prev_head
                if math.hypot(mid[0]-prev_head[0],mid[1]-prev_head[1])<=2.5*CELL:
                    allowed=cv2.dilate(up.astype(np.uint8),np.ones((5,5),np.uint8))
                    cv2.circle(allowed,(int(mid[0]/WORK_SCALE),int(mid[1]/WORK_SCALE)),
                               int(1.25*CELL_W),1,-1)
                    # Reacquire current coral only where the measured head
                    # trail still witnesses our body. Washed orange can also
                    # enter red's hue band, so colour alone is not ownership.
                    hints=np.zeros_like(allowed)
                    for cx,cy in getattr(self,'player_body_hint',()):
                        if 0<=cx<COLS and 0<=cy<ROWS:
                            hints[cy*CELL_W:(cy+1)*CELL_W,cx*CELL_W:(cx+1)*CELL_W]=1
                    if getattr(self,'player_body_hint_complete',False) and hints.any():
                        # A previous ownership mistake must not become permanent
                        # through its own prior mask (r051: orange column7).
                        # The independently witnessed live body trail supersedes
                        # that prior; only the new measured head sweep extends it.
                        allowed=hints.copy()
                        cv2.circle(allowed,(int(mid[0]/WORK_SCALE),int(mid[1]/WORK_SCALE)),
                                   int(1.25*CELL_W),1,-1)
                    else:allowed |= hints
                    comp=comp & (allowed>0)
            npx = int(comp.sum())
            if npx < min_area:
                continue
            eyes_here = "eyes" in ev
            if eyes_here and prev_head is not None and has_continuation:
                mid = entry[0][1]
                age = max(1, self.frame_no - self.prev_head_frame.get("red", self.frame_no - 1))
                if math.hypot(mid[0] - prev_head[0], mid[1] - prev_head[1]) > 4 * CELL * age:
                    continue
            ov = self._prior_overlap_cluster(comp, prev)
            if not eyes_here and ov < 0.15:
                continue          # no eye evidence and no continuity: not our snake
            # A snake does not shrink by 99x in one frame.  This is what rejects the
            # potion star posing as our head, and it can only fire when we HAVE a
            # previous size, so a respawn (one cell, no history) is unaffected.
            if prev_npx and npx < BODY_MIN_FRAC * prev_npx:
                continue
            sc = 90.0 * (min(1.0, ov / 0.15) if ov >= 0.0 else 0.0)
            sc += 6.0 * min(1.0, npx / 20000.0)
            if eyes_here:
                q = 1.0 - min(1.0, (pair_sc or 0.0) / EYE_PAIR_QUALITY_PX)
                sc += EYE_QUALITY_BONUS * q
                ev = ev + ("+prev" if ov >= 0.15 else "")
            else:
                ev = ev + "+prev"
            if best is None or sc > best[0]:
                best = (sc, comp, ev, npx, ov, entry)
        if best is None:
            return None, "no-supported-candidate", None
        _sc, comp, ev, npx, _ov, entry = best
        self.prev_red_px = npx
        # The accepted body and the accepted eye pair have to belong together: when the
        # eyes were rejected as a potion star, publishing them anyway would put our head
        # back on the star.  The test uses the SAME disk that reconnected the body to
        # the eyes, because that disk is exactly the white the slits punch out of the
        # head: measured on luping1 f124 the eye midpoint is 6.3 working px from the
        # nearest body pixel, so a fixed 4 px gate threw away a pair whose ring score
        # was 26 of 40 and whose position was visibly correct.
        if entry is not None:
            pair_entry, gate = entry
            mxx, myy = int(pair_entry[1][0]) // WORK_SCALE, int(pair_entry[1][1]) // WORK_SCALE
            if not (0 <= mxx < WORK_W and 0 <= myy < WORK_H and gate[myy, mxx]):
                pair_entry = None
            entry = pair_entry
        return comp, ev, entry

    @staticmethod
    def _prior_overlap_cluster(comp, prev):
        """Fraction of last frame's player cells that this mask covers."""
        if prev is None or not prev.any():
            return -1.0
        up = np.kron(prev, np.ones((CELL_W, CELL_W), bool))
        h, w = comp.shape
        up = up[:h, :w]
        return float((up & comp).sum()) / max(1.0, float(up.sum()))

    # -------------------------------------------------------------- ramp
    @staticmethod
    def ramp_span(gray, path_cells):
        """Acceleration marker valid for every snake: the white streak shows as a
        large luminance span along the body (0.6-2.0 at rest, 20-100 accelerating)."""
        if len(path_cells) < 6:
            return 0.0
        vals = np.array([gray[min(WORK_H - 1, y * CELL_W + CELL_W // 2),
                              min(WORK_W - 1, x * CELL_W + CELL_W // 2)]
                         for x, y in path_cells], dtype=np.float32)
        k = min(9, max(3, (len(vals) // 12) * 2 + 1))
        # ``same`` zero-pads the ends: a constant 200-gray six-cell snake
        # acquired a fictitious 66.67 brightness ramp and looked accelerated.
        # Repeat real edge samples instead, preserving length and crop semantics.
        sm = np.convolve(np.pad(vals, (k // 2, k // 2), mode="edge"),
                         np.ones(k) / k, mode="valid")
        core = sm[k:len(sm) - k] if len(sm) > 3 * k else sm
        return float(core.max() - core.min())

    # ------------------------------------------------------------------ path trust
    @staticmethod
    def path_trust(path, head_cell, cells=None):
        """Is `path` a real head-to-tail ordering?  -> "ordered" | "uncertain" | "absent".

        Two conditions, both necessary:

          * CONTIGUOUS - a snake's body is 4-connected, so consecutive cells must be neighbours.  A
            path with a gap is a collection of fragments, and an ordering of fragments is not an
            ordering of the snake;
          * ENDS BRACKET THE HEAD - `path[0]` must BE the resolved head cell.  `grid_path` finds the
            two ends but not which is which, so a path whose ends do not include the head cell tells
            us nothing about which end is the head.

        This matters because the previous code sorted the cell set and called its first and last
        elements "the two ends": on a bent snake those are the topmost-left and bottommost-right
        cells, which need not be either end.  Measured on luping3 that is the f277 / f155-162 class
        of error.
        """
        if not path or len(path) < 2:
            return "absent"
        # Double-BFS finds a diameter, not a full snake ordering. A continuous
        # diameter can omit branches/loops and cannot schedule their tail vacancy.
        pts = [(int(c[0]), int(c[1])) for c in path]
        if len(set(pts)) != len(pts):
            return "uncertain"
        if cells is not None and set(pts) != {(int(c[0]), int(c[1])) for c in cells}:
            return "uncertain"
        for a, b in zip(path, path[1:]):
            if abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) != 1:
                return "uncertain"
        if head_cell is None:
            return "uncertain"
        return ("ordered" if (int(path[0][0]), int(path[0][1]))
                == (int(head_cell[0]), int(head_cell[1])) else "uncertain")

    # --------------------------------------------------------------- run
    def run(self, img_full, t=None, allow_balls=None, restrict_box=None):
        """One frame -> (snakes, items, flags).  `img_full` is the 928x800 capture.

        `t` is the elapsed match time in seconds when the caller knows it (bot.py does).  It
        `restrict_box` is (x0, y0, x1, y1) in FULL-resolution pixels.  When given, the DIAMOND SCAN -
        the single largest per-pixel cost, 21.19 ms of a 59.9 ms frame - runs on that crop instead of
        the whole frame, and its cells are translated back to the board.  Everything else stays global:
        the colour families, the per-family labelling, the eye search and the path extraction (so the
        head, bodies, opponents and eye state are unchanged), and also the spike balls and the dark
        mask that serves their ring test.

        Why balls are deliberately NOT restricted: they kill on contact with no death buffer, they
        drift, and the bot decodes with `allow_balls=True` on a cold start precisely so its first
        decision sees the balls that already exist.  A window around the head cannot see a ball across
        the board, and an earlier version that restricted them made the first decision report zero
        balls (caught by test_runtime's cold-f650 case).  Gems, by contrast, are targets the snake
        travels TO, and the policy only acts on one within its lookahead.

        `None` is the previous behaviour exactly.

        ``allow_balls=True`` lets an explicitly resumed mid-match run look for
        existing balls without fabricating an elapsed match time. Geometry and
        outline checks still apply. ``t`` otherwise is used for the
        "spike balls do not exist yet" gate: without it the gate falls
        back to counting frames seen by this Perception object, which is identical for a live
        run that starts when the match does (the case the bot is in) but DIVERGES for any tool
        that starts mid-match - measured, a harness starting at frame 600 produced frame_no=31
        there, so the gate never opened and the balls were invisible to it.
        """
        t0 = time.perf_counter()
        self.item_centers = {}
        self._item_center_areas = {}
        self.frame_no += 1
        if t is not None:
            self.match_t = float(t)
        else:
            self.match_t = self.frame_no * (1.0 / GAME_HZ)
        img = self.work(img_full)
        fam, hsv, S, V = self.families(img)

        # whiteout is needed here (not just at the end) because during it the arena
        # itself desaturates into the blue family's range, and the pale-blue rescue
        # below must not run then: measured in section 5.3, the blue mask swallows
        # the whole floor (1096 px -> 45490 px) once it is desaturated.
        mean_val = float(V.mean() / 255.0)
        mean_sat = float(S.mean() / 255.0)
        whiteout = (mean_sat < WHITEOUT_SAT and mean_val > WHITEOUT_VAL)
        # ---- the end-game countdown.  A large pale numeral is drawn over the arena for the final
        # seconds, and the user reports it disturbs recognition and that the snake then dies in the last
        # few seconds.
        #
        # It is detected by its PALE FILL.  The first version looked for a large DARK component in the
        # interior (the digits carry a dark stroke), and measurement killed it: on real end-of-match
        # frames the biggest dark components are the SNAKE's own outline - up to 1460 px and 314 px tall,
        # far larger than a numeral's stroke - so that test selected the snake and could never separate
        # the two.  The countdown is bright and near-neutral, while the arena is saturated cyan and every
        # snake is a saturated warm or cool colour, so a large bright low-saturation blob in the interior
        # is the distinguishing signature.  Gems are excluded by size: they are a fraction of a cell and
        # this is a numeral.
        countdown_like = False
        try:
            _R, _G, _B = img[..., 0], img[..., 1], img[..., 2]
            _pale = (_R > 213) & (_G > 213) & (_B > 208)
            _h, _w = _pale.shape[:2]
            _inner = np.zeros((_h, _w), dtype=bool)
            _m = max(2, int(2 * CELL // WORK_SCALE))
            _inner[_m:_h - _m, _m:_w - _m] = True
            _pale = _pale & _inner
            if _pale.any():
                _n, _lab, _st, _ce = cv2.connectedComponentsWithStats(_pale.astype(np.uint8), 8)
                for _i in range(1, _n):
                    _a = int(_st[_i, cv2.CC_STAT_AREA])
                    if _a < COUNTDOWN_MIN_PX:
                        continue
                    _bw = int(_st[_i, cv2.CC_STAT_WIDTH])
                    _bh = int(_st[_i, cv2.CC_STAT_HEIGHT])
                    if _bh >= COUNTDOWN_MIN_H and _bw >= COUNTDOWN_MIN_W:
                        countdown_like = True
                        break
        except Exception:
            countdown_like = False

        # -------- items: one shared labelling pass (families are hue-disjoint)
        # (restored: an edit that replaced the countdown detector deleted this block by accident, which
        # showed up immediately as `NameError: item_ids is not defined`)
        item_ids = np.zeros((WORK_H, WORK_W), np.uint8)
        item_id_of = {"ball": 1, "grn": 2, "bean": 3}
        for key, i in item_id_of.items():
            item_ids[self.clean_small(fam[key])] = i
        dark_full = ((img_full[:, :, 0] < BALL_DARK_VMAX) &
                     (img_full[:, :, 1] < BALL_DARK_VMAX) &
                     (img_full[:, :, 2] < BALL_DARK_VMAX))
        items = {k: [] for k in item_id_of}
        # the slowdown item rides in the ball mask but is a different thing entirely
        items["slow"] = []
        # The diamond scan is the single largest per-pixel cost (21.19 ms of a 59.9 ms frame) and it
        # scales with area, so `restrict_box` can run it on a crop and translate the cells back to the
        # board.  MEASURED, and the reason the bot no longer passes a box: a window sees only 9% of the
        # gems on the board (2.00 per frame globally against 0.18 windowed, for an 18 ms saving), and
        # diamonds are +5 cells each and the entire win condition - so the saving cost the objective.
        # The mechanism is kept because it is right for anything genuinely local.
        if restrict_box is None:
            items["dia"] = self.diamonds_full(img_full)
        else:
            _rx0, _ry0, _rx1, _ry1 = restrict_box
            # Quantise AFTER adding the pixel origin. floor(local/32)+
            # floor(origin/32) is wrong by one tile at non-aligned crops.
            # The measured item centre must be in the same global coordinates.
            items["dia"] = self.diamonds_full(img_full[_ry0:_ry1, _rx0:_rx1],origin=(_rx0,_ry0))
        # Balls do not exist before BALL_SPAWN_T (~f630).  A ball can only be recognised
        # from its bright core, and a few snake-body highlights match that core early in
        # a match: measured on luping1 f2/f12/f13/f15/f26/f27/f30.  The clock is a free
        # and exact filter here - the game itself cannot show one yet.
        balls_live = (allow_balls is True) or self.match_t >= BALL_SPAWN_T
        ball_candidates = []
        self.ball_dropped = []
        nit, ilab, istats, icent = cv2.connectedComponentsWithStats(item_ids, 8)
        for i in range(1, nit):
            a = int(istats[i, cv2.CC_STAT_AREA])
            # An item is a small thing.  A cap as well as a floor stops any mask
            # that accidentally matches the floor from ever becoming an item again.
            if a < 40 or a > ITEM_MAX_PX:
                continue
            w = int(istats[i, cv2.CC_STAT_WIDTH])
            h = int(istats[i, cv2.CC_STAT_HEIGHT])
            bx = int(istats[i, cv2.CC_STAT_LEFT])
            by = int(istats[i, cv2.CC_STAT_TOP])
            # Work on the blob's own BOUNDING BOX, not the whole frame.  `ilab == i`
            # built a full-frame boolean per item and then scanned it: measured at 13 ms
            # of a 34 ms frame with ~20 blobs, because every iteration touched 185600 px.
            sub = ilab[by:by + h, bx:bx + w]
            hit = (sub == i)
            vals, counts = np.unique(item_ids[by:by + h, bx:bx + w][hit], return_counts=True)
            if not len(vals):
                continue
            fid = int(vals[np.argmax(counts)])
            key = next((k2 for k2, k3 in item_id_of.items() if k3 == fid), None)
            if key is None:
                continue
            # ---- per-kind shape/size gate (see ITEM_* in config for the measurements)
            if key in ITEM_MAX_PX_OF:
                if a > ITEM_MAX_PX_OF[key]:
                    continue
                ys, xs = np.nonzero(hit)
                if self._elong(ys, xs) > ITEM_MAX_ELONG_OF[key]:
                    continue
            if key == "ball":
                # A single-cell round red blob is the slowdown item, not a spike ball.
                if (SLOW_MIN_PX <= a <= SLOW_MAX_PX and
                        SLOW_BOX_MIN <= w <= SLOW_BOX_MAX and
                        SLOW_BOX_MIN <= h <= SLOW_BOX_MAX and
                        a / max(1.0, float(w * h)) >= SLOW_FILL_MIN):
                    cx, cy = int(icent[i][0] // CELL_W), int(icent[i][1] // CELL_W)
                    items["slow"].append((cx, cy, a))
                    self.item_centers[("slow", cx, cy)] = (
                        float(icent[i][0] * WORK_SCALE + (WORK_SCALE - 1) / 2.0), float(icent[i][1] * WORK_SCALE + (WORK_SCALE - 1) / 2.0))
                    self._item_center_areas[("slow", cx, cy)] = a
                    continue
                if not balls_live:
                    continue
                # What is detected is the ball's bright core, not the ball: measured
                # 259-419 px in a 21-25 px box on f634, where the ball itself is ~2 cells
                # across including its dark spikes and outline.  So the gates are on the
                # core's size and roundness (the red snake's body shares this hue family
                # and is 4000-14000 px of fill 0.17-0.30), and the reported cell is the
                # core's cell - the policy treats a ball as a 2x2 obstacle, which is
                # centred on that same cell.
                # Size breathes and an occluder can leave a partial core (f733:
                # 259/347px; f717:346px). Keep it above the one-cell slowdown
                # ceiling and rely on bbox/fill/dark outline, not a 350px floor.
                # Birth has a real scale overshoot (f635:619-635px,33-34px boxes).
                # This is local to item detection: _valid_balls retains its old
                # subtraction gate so small player masks are never deleted.
                birth = 0.0 <= self.match_t - BALL_SPAWN_T < 1.0
                unknown_clock = allow_balls is True and t is None
                min_ball = SLOW_MAX_PX + 1
                max_ball = BALL_BOX_MAX ** 2 if birth or unknown_clock else BALL_MAX_PX
                if not (min_ball <= a <= max_ball):
                    continue
                if not (BALL_BOX_MIN <= w <= BALL_BOX_MAX and
                        BALL_BOX_MIN <= h <= BALL_BOX_MAX):
                    continue
                if a / max(1.0, float(w * h)) < BALL_FILL_MIN:
                    continue
                # Dark-outline test: a spike ball has a thick dark spiky border, our own body
                # against the cyan floor does not.  The statistic is the dark fraction of the
                # ring just outside the blob.
                #
                # Implementation note: the ring is sampled at FULL resolution directly, one
                # pixel per 2x2 block, rather than gathering every ring pixel.  The quantity is
                # the mean dark fraction over the ring, so sampling it on the grid the dark
                # mask is actually used at (each full-res 2x2 block corresponds to one
                # working-resolution pixel, evaluated at that pixel's top-left) is the same
                # number with a quarter of the gather.
                #
                # NOTE: two cheaper-looking rewrites were tried and REJECTED because they
                # measured a DIFFERENT quantity and silently rejected every real ball (0 balls
                # reported in 322 of 346 frames): the expanded bounding BOX includes the
                # corners, and four edge strips at full-res coordinates are not the same
                # neighbourhood as the elliptical ring.  A wrong ball count is worse than the
                # few ms this costs, so the ring stays.
                pad = BALL_BAND_PAD
                ex0, ey0 = max(0, bx - pad), max(0, by - pad)
                ex1, ey1 = min(WORK_W, bx + w + pad), min(WORK_H, by + h + pad)
                sub = (ilab[ey0:ey1, ex0:ex1] == i)
                grown = cv2.dilate(sub.view(np.uint8), self._k_ball_ring) > 0
                ring = grown & ~sub
                ring_n = int(ring.sum())
                if ring_n:
                    rys, rxs = np.nonzero(ring)
                    fy = (rys + ey0) * WORK_SCALE
                    fx = (rxs + ex0) * WORK_SCALE
                    ring_frac = float(dark_full[fy, fx].mean())
                    if ring_frac < BALL_DARK_MIN:
                        continue
                else:
                    continue
                cx, cy = int(icent[i][0] // CELL_W), int(icent[i][1] // CELL_W)
                ball_candidates.append(dict(entry=(cx, cy, a), ring=ring_frac,
                    px=(float(icent[i][0] * WORK_SCALE + (WORK_SCALE - 1) / 2.0),
                        float(icent[i][1] * WORK_SCALE + (WORK_SCALE - 1) / 2.0))))
                continue
            cx, cy = int(icent[i][0] // CELL_W), int(icent[i][1] // CELL_W)
            items[key].append((cx, cy, a))
            ckey = (key, cx, cy)
            if a >= self._item_center_areas.get(ckey, -1):
                self.item_centers[ckey] = (float(icent[i][0] * WORK_SCALE + (WORK_SCALE - 1) / 2.0),
                                          float(icent[i][1] * WORK_SCALE + (WORK_SCALE - 1) / 2.0))
                self._item_center_areas[ckey] = a

        for ball in self._choose_balls(ball_candidates):
            cx, cy, a = ball['entry']
            items['ball'].append((cx, cy, a))
            key = ('ball', cx, cy)
            if a >= self._item_center_areas.get(key, -1):
                self.item_centers[key] = ball['px']
                self._item_center_areas[key] = a

        def _grow(m, r):
            """Dilate a boolean mask by r pixels (a small kernel, built per call)."""
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            return cv2.dilate(m.view(np.uint8), k) > 0

        # -------- snakes: label PER FAMILY.  A single shared labelling merged
        # neighbouring snakes into one component (they touch, and 8-connectivity
        # bridges them) and silently reassigned the player's snake.
        snakes = {}
        gray = None
        red_region = None
        # cleaned once and shared: the red branch needs red|org before the per-family
        # loop runs, and the loop needs the same masks again
        cleaned = {k: self.clean(fam[k]) for k in ("red", "org", "pur", "blu")}
        # Eye slits and their pairing are computed ONCE, before the family loop, and
        # reused for the player's identity and for the eye-state decode below.  They
        # used to be found twice - once in the anchor, once inside the red branch -
        # which doubled the cost of the single most expensive step in perception.
        # The warm mask goes in as well, because the pair test needs to know whether a
        # candidate pair is surrounded by our body or is the shine on a bean.
        warm_mask = cleaned["red"] | cleaned["org"]
        # ---- the eye search.  The window is a CANDIDATE GENERATOR and its answer is verified against the
        # full frame before use, because measured it disagreed with the full search on 10 of 41 frames (one
        # case put the head at (458.9, 692.3) against the true (184.1, 479.2)): a 6-cell crop can contain a
        # white blob that satisfies the pair test locally while the real eyes lie outside it.  Verifying keeps
        # the OUTPUT identical to a full search by construction; the counters say what it bought.
        # The window is OFF by default (see EYE_WINDOW in config: measured, it cannot pay for the
        # verification that makes it safe).  When enabled, its result is verified against the full
        # frame before use, so the output stays identical either way.
        windowed = self._eye_slits_windowed(img_full, warm_mask) if EYE_WINDOW else None
        full_slits = self.white_slits(img_full)
        full_pairs = (self.eye_pair_candidates(full_slits, img_full, warm_mask)
                      if len(full_slits) >= 2 else [])
        if windowed is not None and full_pairs:
            w_slits, w_pairs, _w = windowed
            wm, fm = w_pairs[0][1], full_pairs[0][1]
            if (abs(wm[0] - fm[0]) <= EYE_WINDOW_MATCH_PX
                    and abs(wm[1] - fm[1]) <= EYE_WINDOW_MATCH_PX):
                slits, pairs = w_slits, w_pairs
                self.eye_window_used += 1
            else:
                slits, pairs = full_slits, full_pairs
                self.eye_window_mismatch += 1
        else:
            slits, pairs = full_slits, full_pairs
            if getattr(self, '_eye_win_head', None) is not None:
                self.eye_window_fallbacks += 1
        for kind in ("red", "org", "pur", "blu"):
            mask = cleaned[kind]
            if kind == "org" and red_region is not None:
                # See the note on the red branch below: the player's body runs
                # through the orange family, so those pixels are ours, not an
                # opponent's.  Without this the bot sees a phantom enemy sitting
                # against its own back.
                mask = mask & ~red_region

            gm = g_cells = None
            if kind == "red":
                # The player's snake is identified FIRST and by all available signals
                # at once - the eye pair, continuity with the previous frame, area -
                # rather than by the colour blob test, which only ever sees the red
                # half of the body.  See Perception.player_region for the measured
                # failure rates that motivated the change.
                anchored, why, eye_entry = self.player_region(img_full, cleaned,
                                                              slits, pairs)
                self.last_evidence = why
                self.last_eye_entry = eye_entry
                if anchored is not None:
                    gm = anchored
                    g_cells = self.to_grid(gm)

            if gm is not None:
                parts = lab = groups = None
            else:
                parts, lab = self.blobs(mask, MIN_SNAKE_PX)
                if not parts:
                    continue
                if kind == "blu":
                    keep = []
                    prev_blu = self.prev_cells.get("blu")
                    for a, i, st, ce in parts:
                        # During the respawn whiteout the desaturated floor shatters into
                        # blue fragments that 8-connect into one giant component: measured
                        # on luping6 f717, a single "blue snake" of 322 cells (the real one
                        # is 14).  The pale-blue rescue below is already skipped during
                        # whiteout for the same reason; this is the other half of it.
                        if whiteout and a > MAX_OPP_PX_WHITEOUT:
                            continue
                        ys, xs = np.nonzero(lab == i)
                        p25 = float(np.percentile(S[ys, xs], 25)) if ys.size else 0.0
                        if p25 >= BLUE_SAT_P25:
                            keep.append((a, i, st, ce))
                        elif (not whiteout and prev_blu is not None
                              and np.count_nonzero(self.to_grid(lab == i) & prev_blu)):
                            # Pale blue, but it is standing exactly where the blue
                            # snake stood a frame ago, so it is that snake with the
                            # flash over it and not the yellow snake's tail tip, which
                            # lasts 1-2 frames and does not appear there.
                            keep.append((a, i, st, ce))
                    parts = keep
                    if not parts:
                        continue
                groups = []
                for a, i, st, ce in parts:
                    bx0, by0 = int(st[cv2.CC_STAT_LEFT]), int(st[cv2.CC_STAT_TOP])
                    bx1 = bx0 + int(st[cv2.CC_STAT_WIDTH])
                    by1 = by0 + int(st[cv2.CC_STAT_HEIGHT])
                    for g in groups:
                        gx0 = min(int(s[cv2.CC_STAT_LEFT]) for _, _, s, _ in g)
                        gy0 = min(int(s[cv2.CC_STAT_TOP]) for _, _, s, _ in g)
                        gx1 = max(int(s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH])
                                  for _, _, s, _ in g)
                        gy1 = max(int(s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT])
                                  for _, _, s, _ in g)
                        if not (gx1 + JOIN_TOL < bx0 or bx1 + JOIN_TOL < gx0 or
                                gy1 + JOIN_TOL < by0 or by1 + JOIN_TOL < gy0):
                            g.append((a, i, st, ce)); break
                    else:
                        groups.append([(a, i, st, ce)])
                # Which group is this family's snake?  Taking the biggest one silently
                # reassigns the snake every time the mask splits, and it splits often:
                # measured over 1106 frames, the orange snake came out as a SINGLE
                # component in only 21-25% of frames, because the acceleration flash
                # desaturates a section and because snakes occlude each other.  The
                # result was a head that jumped 4-10 cells between consecutive frames
                # (measured on luping7 f33/f39/f47/f53/f57/f62), i.e. the snake was
                # being swapped for its own stray fragment.  So pick by continuity with
                # the previous frame first - a snake cannot teleport - and fall back to
                # area only when nothing overlaps, which is what a respawn looks like.
                prev = self.prev_cells.get(kind)
                if prev is not None:
                    # Score each candidate group on how much of it sits where this snake's
                    # body was a frame ago, AND on how close it is to where the HEAD is
                    # predicted to be.  The head term is what fixes the fragmentation case
                    # the user reported as the orange snake's "death effect" (f629-f712):
                    # measured on luping1, the orange area oscillates 12231 <-> 1952 px and
                    # its cell count 51 <-> 9 between consecutive frames, because the picker
                    # took whichever FRAGMENT was biggest rather than the fragment that
                    # continues the snake.  Head prediction is the strongest constraint
                    # available - a snake cannot teleport, and its head moves about one cell
                    # per frame, so the true fragment is always the near one.
                    prev_head = self.prev_head_px.get(kind)
                    scored = []
                    for g in groups:
                        cand = np.isin(lab, [t[1] for t in g])
                        cells = self.to_grid(cand)
                        ov = int(np.count_nonzero(cells & prev))
                        # distance from the predicted head to the nearest cell of the group
                        hd = 1e9
                        if prev_head is not None:
                            hx = prev_head[0] / WORK_SCALE / CELL_W
                            hy = prev_head[1] / WORK_SCALE / CELL_W
                            ys2, xs2 = np.nonzero(cells)
                            if ys2.size:
                                hd = float(np.min(np.maximum(np.abs(xs2 - hx),
                                                             np.abs(ys2 - hy))))
                        scored.append((ov, -hd, sum(t[0] for t in g), cand, cells))
                    # prefer overlap; break ties by whichever fragment is nearest the
                    # predicted head; only then fall back to size
                    scored.sort(key=lambda t: (-t[0], -t[1], -t[2]))
                    ov, neg_hd, _area, gm, g_cells = scored[0]
                    if ov == 0:
                        gm = g_cells = None
                if gm is None:
                    best = max(groups, key=lambda g: sum(t[0] for t in g))
                    gm = np.isin(lab, [t[1] for t in best])
                    g_cells = self.to_grid(gm)

            if kind == "red":
                if anchored is not None:
                    # Ownership must agree with the accepted identity mask. The
                    # old red-neighbourhood subtraction left an orange distal
                    # tail published a second time as an opponent. player_region
                    # already gates ownership by eyes and recent native support.
                    # Remove only accepted current pixels, never all red-family
                    # pixels or a dilation into a neighbouring opponent.
                    red_region = anchored
                else:
                    # We could not identify our snake this frame at all.  Do NOT
                    # invent one out of the largest red blob: that is how the bot
                    # ended up steering an opponent's body.  Publish nothing, which
                    # the tracker already handles (`player_alive` false), and let the
                    # continuity route pick it up again next frame.
                    red_region = None
                    # Otherwise the generic colour-blob fallback was still
                    # published and seeded prev_cells['red'] after respawn.
                    continue

            path = self.grid_path(g_cells)
            partial_head = None
            if kind == "red" and self.last_eye_entry is None:
                # Opening can erase the flash-covered snout (luping1 f104: 44
                # native working pixels -> 0 cleaned pixels). Recover only a
                # short-gap prediction supported by real red pixels at a body end.
                old = self.prev_head_px.get("red")
                vel = self.prev_head_vel.get("red")
                age = self.frame_no - self.prev_head_frame.get("red", -999)
                eye_gap = self.frame_no - self.prev_eye_frame.get("red", -999)
                if (old is not None and vel is not None and 1 <= age <= 3
                        and eye_gap <= 3 and path):
                    pred = (old[0] + vel[0] * age, old[1] + vel[1] * age)
                    hc = (int(pred[0] // CELL), int(pred[1] // CELL))
                    if (0 <= hc[0] < COLS and 0 <= hc[1] < ROWS
                            and min(_cheb(hc, path[0]), _cheb(hc, path[-1])) <= 1):
                        yy = slice(hc[1] * CELL_W, (hc[1] + 1) * CELL_W)
                        xx = slice(hc[0] * CELL_W, (hc[0] + 1) * CELL_W)
                        native_red = fam["red"][yy, xx] > 0
                        if int(native_red.sum()) >= SHORT_SNAKE_MIN_PX:
                            gm = gm.copy()
                            gm[yy, xx] |= native_red
                            g_cells = g_cells.copy()
                            g_cells[hc[1], hc[0]] = True
                            path = self.grid_path(g_cells)
                            partial_head = pred
            obs = SnakeObs(family=kind, n_px=int(gm.sum()),
                           cells={(x, y) for x, y in zip(*np.nonzero(g_cells.T))},
                           length=int(g_cells.sum()), path=path,
                           head_from_pixels=partial_head is not None)
            ys, xs = np.nonzero(gm)
            obs.bbox = (int(xs.min() * WORK_SCALE), int(ys.min() * WORK_SCALE),
                        int(xs.max() * WORK_SCALE), int(ys.max() * WORK_SCALE))
            if path:
                obs.head_cell, obs.tail_cell = path[0], path[-1]
            if len(path) >= 6:
                if gray is None:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                obs.ramp_span = self.ramp_span(gray, path)
            # eyes: reuse the slits and pairs already found for the identity step,
            # keeping the ones whose midpoint actually falls on this snake's body.
            # Re-detecting them here cost a second full-resolution white scan and a
            # second pairing pass - measured at 3-4 ms of a 20 ms frame.
            if kind == "red":
                # Only the pair the identity step ACCEPTED is published.  Another
                # pair in the same frame can sit on a different object: measured on
                # luping1 f862 the runner-up pair is the shine on a potion star, and
                # publishing it reported eye=item, a speed state the snake was not in.
                entry = self.last_eye_entry
                mine = [] if entry is None else [(entry[2], entry[3], entry[1])]
                obs.eye_state, obs.eye_asym, obs.head_px = self.eye_state_from(mine)
                obs.eye_travel_axis=getattr(entry,'travel_axis',None)
                # A one-cell respawn still has measured eyes. Requiring two
                # path cells withheld native identity until it somehow moved.
                obs.head_from_eyes=bool(entry is not None and obs.head_px is not None and partial_head is None)
                if obs.head_from_eyes:obs.orientation='eyes'
                if obs.head_px is None and partial_head is not None:
                    obs.head_px = partial_head
            else:
                # The other three have no eye pair; a white slit is a marker only the
                # player's snake carries, and looking for one on an opponent is what
                # used to put a head on a diamond sparkle.
                obs.head_px, obs.eye_state, obs.eye_asym = None, None, 0.0

            # Orient the path head->tail.  grid_path() finds the two ends with a
            # double BFS and cannot know which is which, so path[0] used to be an
            # arbitrary end: tail_cell, and any heading derived from the body
            # layout, were coin flips.  The eye pair settles it for the player's
            # snake; the tracker settles it for the others by motion.
            if obs.head_px is not None and len(path) >= 2:
                hx, hy = obs.head_px

                def _d2(c):
                    return ((c[0] * CELL + CELL / 2.0 - hx) ** 2 +
                            (c[1] * CELL + CELL / 2.0 - hy) ** 2)

                if _d2(path[-1]) < _d2(path[0]):
                    path.reverse()
                obs.head_from_eyes = partial_head is None
                obs.orientation = "eyes" if obs.head_from_eyes else "motion"
            if obs.head_px is not None:
                obs.head_cell = (int(obs.head_px[0] // CELL), int(obs.head_px[1] // CELL))
            if path:
                # Orient by the RESOLVED head cell as well as by head_px: the two can disagree, and
                # that disagreement is what left path[0] pointing at a cell that was not the head.
                if obs.head_cell is not None and len(path) >= 2:
                    hc = (int(obs.head_cell[0]), int(obs.head_cell[1]))
                    if (int(path[-1][0]), int(path[-1][1])) == hc:
                        path.reverse()
                obs.path = path
                # The eye midpoint is authoritative even when the occupancy mask
                # missed its snout. Never overwrite it with an arbitrary diameter end.
                if obs.head_px is None:
                    obs.head_cell = path[0]
                obs.tail_cell = path[-1]
                if len(path) >= 2:
                    obs.facing = (path[0][0] - path[1][0], path[0][1] - path[1][1])
                obs.path_trust = Perception.path_trust(path, obs.head_cell, obs.cells)
            if kind != "red" and parts:
                # Body occupancy includes every valid native fragment of this
                # family. A flash can split one physical snake into many groups;
                # choosing one group is useful for orientation, not for erasing
                # its other still-visible obstacles. No old pixels are added.
                visible=np.isin(lab,[part[1] for part in parts])
                all_cells=self.to_grid(visible)
                obs.cells={(x,y) for x,y in zip(*np.nonzero(all_cells.T))}
                obs.length=len(obs.cells)
                obs.n_px=int(visible.sum())
                obs.path_trust=Perception.path_trust(obs.path,obs.head_cell,obs.cells)
                g_cells=all_cells
            snakes[kind] = obs
            self.prev_cells[kind] = g_cells
            if obs.head_px is not None:
                old = self.prev_head_px.get(kind)
                age = max(1, self.frame_no - self.prev_head_frame.get(kind, self.frame_no - 1))
                if old is not None:
                    vel = ((obs.head_px[0] - old[0]) / age,
                           (obs.head_px[1] - old[1]) / age)
                    if math.hypot(*vel) <= 3 * CELL:
                        self.prev_head_vel[kind] = vel
                self.prev_head_px[kind] = obs.head_px
                self.prev_head_frame[kind] = self.frame_no
                if obs.head_from_eyes:
                    self.prev_eye_frame[kind] = self.frame_no
                if kind == "red":
                    # centre the next frame's eye window on THIS frame's resolved red head.  Without this
                    # the window has no centre, `_eye_slits_windowed` returns None every frame, and the
                    # optimisation silently degrades to the full-frame search it was meant to replace -
                    # which is why the fallback counter is reported rather than assumed.
                    self._note_eye_head(obs.head_px)

        mean_val = float(V.mean() / 255.0)
        mean_sat = float(S.mean() / 255.0)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return snakes, items, dict(whiteout=whiteout, countdown_like=countdown_like,
                                   mean_val=mean_val, mean_sat=mean_sat, ms=self.last_ms,
                                   item_centers=dict(self.item_centers),
                                   ball_dropped=list(self.ball_dropped))


# ================================================================== tracker
class SnakeTracker:
    """Temporal layer.  Three rules that each fix a specific misreading:

    * HEAD EXTRAPOLATION - during a head-to-head grind the opponent covers the
      player's head and the eyes vanish; the head is predicted from the recent
      heading and marked with `head_age` so the policy can discount it.
    * HEAD-TO-HEAD BY GRIND - one frame of proximity is NOT an event.  The event
      requires the opponent's cell count to fall by >= HH_MIN_DROP over 5..25
      frames while the heads stay within one cell.  In the recordings a real
      head-to-head grinds for 15-25 frames.
    * CONTACT IS ALSO THE BUFFER - the same whole-snake pause appears in both,
      so a sustained grind is reported as `grind` and treated as decision time.
    """

    HH_MIN_DROP = 3
    HH_MIN_SPAN, HH_MAX_SPAN = 5, 25

    def __init__(self):
        self.head_hist = {k: deque(maxlen=6) for k in ("red", "org", "pur", "blu")}
        self.frame = 0
        self.t = 0.0
        self.events = []
        self._grind = None
        self.cells_hist = {}                 # family -> occupied cells last frame
        self.head_end = {}                   # family -> last resolved head cell
        self.item_accel_until = 0.0
        self._grn_seen = set()
        self._grn_pending = {}

    # ------------------------------------------------------- head orientation
    def _orient(self, k, s, prev_cells):
        """Decide which end of the cell path is the head.

        The eye pair is definitive and perception has already used it.  For the
        other three snakes the body advances every step, so the NEW head cell is
        one that was not occupied a frame ago while the tail cell just left is
        gone: the end absent from the previous cell set is the head.  A snake that
        is growing (just ate) keeps its tail end old and still reads correctly.
        When both ends were already occupied - a stationary buffer/grind frame -
        only continuity from the last resolved head is left.
        """
        if len(s.path) < 2:
            return
        if s.head_from_eyes:
            s.orientation = "eyes"
            # The head end is right, but the BODY LAYOUT may not be: perception reverses
            # the path after setting `facing`, so `facing` can still be None or describe
            # the pre-reversal direction.  It is the only heading available on the first
            # frames of a life (SnakeTracker.heading falls back to it before any motion
            # has been observed), and `facing is None` also crashed the self-test.
            s.head_cell, s.tail_cell = s.path[0], s.path[-1]
            s.facing = (s.path[0][0] - s.path[1][0], s.path[0][1] - s.path[1][1])
            return
        a, b = s.path[0], s.path[-1]
        ref = self.head_end.get(k)
        head = None
        if prev_cells:
            a_new, b_new = (a not in prev_cells), (b not in prev_cells)
            if a_new != b_new:
                head = a if a_new else b
                # A head cannot teleport.  When the acceleration flash desaturates
                # the head end of a snake, the mask keeps only a fragment, both its
                # ends are "old" or the wrong one looks new, and the growth rule
                # happily names the far end the head - measured as a 10-cell jump in
                # one frame.  If the picked end is far from where the head was while
                # the other end is right there, continuity wins.  A genuine respawn
                # leaves BOTH ends far away, so it still falls through to the rule.
                if ref is not None:
                    other = b if head is a else a
                    if _cheb(head, ref) > 2 and _cheb(other, ref) <= 2:
                        head = other
        if head is None and ref is not None:
            head = a if _cheb(a, ref) <= _cheb(b, ref) else b
        if head is None:
            s.orientation = "unknown"
            return
        if head is b:
            s.path.reverse()
        s.head_cell, s.tail_cell = s.path[0], s.path[-1]
        s.facing = (s.path[0][0] - s.path[1][0], s.path[0][1] - s.path[1][1])
        s.orientation = "motion"

    def update(self, snakes, items, flags, dt):
        self.frame += 1
        self.t += dt
        self.events = []
        st = {"frame": self.frame, "t": self.t, "snakes": {}, "items": items,
              "whiteout": flags["whiteout"], "perception_ms": flags.get("ms", 0.0),
              # initialised here: these are appended to further down, and a
              # missing key raised KeyError('events') on the first contact frame
              "events": [], "grind": None, "item_accel_until": self.item_accel_until}

        for k, s in snakes.items():
            self._orient(k, s, self.cells_hist.get(k))
            if s.orientation == "unknown":
                # Both ends were already occupied last frame (a stationary snake)
                # or there is no history at all (the spawn frame), so the cell path
                # cannot say which end is the head.  Measured: publishing a coin
                # flip here put an opponent's head at its TAIL for the first ~19
                # frames of luping7, then jumped 5 cells when it started moving.
                # Publishing nothing for that frame is strictly better.
                s.head_cell = None
                s.facing = None
            head = s.head_px
            if head is None and s.orientation == "motion" and s.head_cell is not None:
                # an opponent has no eye pair, so the resolved head cell IS the
                # head.  Feeding it through head_px keeps head_hist alive, which is
                # what the grind extrapolation below needs for them too.
                head = (s.head_cell[0] * CELL + CELL // 2, s.head_cell[1] * CELL + CELL // 2)
            if head is None:
                hh = self.head_hist[k]
                if len(hh) >= 2:
                    (x0, y0, _), (x1, y1, _) = hh[-2], hh[-1]
                    head = (x1 + (x1 - x0), y1 + (y1 - y0))
                elif hh:
                    head = (hh[-1][0], hh[-1][1])
                else:
                    head = None
                s.head_age = (self.frame - hh[-1][2]) if hh else 99
            else:
                self.head_hist[k].append((head[0], head[1], self.frame))
                s.head_age = 0
            s.head_px = head
            if head is not None:
                s.head_cell = (int(head[0] // CELL), int(head[1] // CELL))
            # Remember the last RESOLVED head separately from the published one.
            # Writing s.head_cell (which the unknown case clears) straight in here
            # erased the continuity reference, so a snake that stood still for a
            # while lost its head entirely and flipped to the other end when it
            # moved again - measured as 200+ apparent head "flips" per match.
            if s.head_cell is not None:
                self.head_end[k] = s.head_cell
            self.cells_hist[k] = set(s.cells)
            s.path_trust = Perception.path_trust(s.path, s.head_cell, s.cells)
            st["snakes"][k] = s

        red = st["snakes"].get("red")
        # Deliberately NOT `length >= 3`.  Items are drawn ON TOP of snakes
        # (section 6), so a diamond or a green token sitting on our body hides it
        # and the measured length drops to 1-2 for a frame at a time - measured
        # live, red detection flickered on and off about 8 times a second while the
        # snake was perfectly healthy, and the bot went inert for 63% of the match
        # because of it.  "We can see our snake's head and the arena is not washing
        # out" is the honest test; the respawn wash is handled by `whiteout`.
        st["player_alive"] = (red is not None and red.head_cell is not None
                              and not flags["whiteout"])
        st["player_respawn_pending"] = not st["player_alive"]

        # ---- head-to-head: require a grind, not a single frame
        if red is not None and red.head_cell:
            for k in ("org", "pur", "blu"):
                s = st["snakes"].get(k)
                if s is None or not s.head_cell:
                    continue
                d = max(abs(red.head_cell[0] - s.head_cell[0]),
                        abs(red.head_cell[1] - s.head_cell[1]))
                if d <= 1:
                    if self._grind is None or self._grind["opponent"] != k:
                        self._grind = {"opponent": k, "start": self.frame,
                                       "cells0": s.length, "best": s.length}
                    else:
                        self._grind["best"] = min(self._grind["best"], s.length)
                    st["events"].append(("contact", k, d))
            if self._grind is not None:
                g = self._grind
                span = self.frame - g["start"]
                drop = g["cells0"] - g["best"]
                st["grind"] = dict(opponent=g["opponent"], span=span, drop=drop)
                if drop >= self.HH_MIN_DROP and span >= self.HH_MIN_SPAN:
                    st["events"].append(("head_to_head_kill", g["opponent"], drop))
                    self.events.append(("kill", g["opponent"], self.t, drop))
                    self._grind = None
                elif span > self.HH_MAX_SPAN:
                    self._grind = None
            else:
                st["grind"] = None
        else:
            self._grind = None
            st["grind"] = None

        # ---- persistent item acceleration timer (the green token, ~10 s).
        # Adjacency alone is not a collect. Require a nearby previously seen
        # token to vanish, corroborated by item-eyes or two further missing frames.
        greens = {(int(e[0]), int(e[1])) for e in items.get("grn", [])}
        if red is not None and red.head_cell is not None:
            for cell in self._grn_seen - greens:
                if _cheb(cell, red.head_cell) <= 1:
                    self._grn_pending.setdefault(cell, self.frame)
        for cell, gone_frame in list(self._grn_pending.items()):
            if cell in greens:
                del self._grn_pending[cell]  # breathing/flicker, not a collect
            elif ((red is not None and red.eye_state == "item")
                  or self.frame - gone_frame >= 2):
                self.item_accel_until = self.t + 10.0
                st["events"].append(("item_accel", cell, 0))
                del self._grn_pending[cell]
        self._grn_seen = greens
        st["item_accel_until"] = self.item_accel_until
        return st

    @staticmethod
    def key_held(bean_history):
        """Beans on the floor climbing steadily == the accelerate key is held
        (holding A leaks a bean from the tail roughly every 5.7 s)."""
        return len(bean_history) >= 15 and (bean_history[-1] - bean_history[0]) >= 2

    @staticmethod
    def heading(head_hist, snake=None):
        """Direction of travel, or None when it cannot be established.

        Two independent sources, in order of trust:
          * observed motion - the head moved between the last two frames, which is
            what the snake is really doing;
          * body layout (`snake.facing`) - valid on the very first frame, because
            the head is fixed by the eye pair and the neck cell is behind it.  This
            is what removes the need for a probe step at spawn.
        """
        if len(head_hist) >= 2:
            (x0, y0, _), (x1, y1, _) = head_hist[-2], head_hist[-1]
            if (x0, y0) != (x1, y1):
                if abs(x1 - x0) >= abs(y1 - y0):
                    return (1 if x1 >= x0 else -1, 0)
                return (0, 1 if y1 >= y0 else -1)
        if snake is not None and snake.facing is not None:
            return tuple(snake.facing)
        return None
