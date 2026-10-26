"""Game-on-screen detection -- the same idea as luping.py, but with a reference
template built from the game itself instead of a desktop grab.

luping.py waited for the detect region (950,650)-(1650,1050) on the 2560x1440
screen to match start.png at 95%, then started recording.  `start.png` turned
out to be a desktop capture, and -- more importantly -- that reference was only
used to find the START overlay, so the recording began at the moment START
appeared.

This module does the same job for the bot, with two additions that matter:

  * the template is taken from the game's own START screen (assets/
    start_template.png), so it matches that overlay reliably;
  * a score threshold alone is not sufficient - a plain desktop can score 0.45 -
    so the region must ALSO look like the arena: bright and cyan-dominated.
    Both conditions together separate every case tested:

        START overlay    score 0.989-1.000
        in-play frames   score -0.05..0.10
        desktop capture  score 0.16   (after the same normalisation)
        black screen     score 0.00

Usage
-----
    det = GameDetector()
    ok, info = det.is_game_on(frame)     # frame = 928x800 RGB uint8
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from config import FRAME_W, FRAME_H, CAPTURE_BBOX, CELL

# detect region on the 2560x1440 screen, exactly as start.py / luping.py used
BOX_SCREEN = (950, 650, 1650, 1050)

# the same region in saved-frame coordinates
#   x_c = (x - 643) * 928/1276 ,  y_c = (y - 269) * 800/1100
_SX = FRAME_W / float(CAPTURE_BBOX[2]-CAPTURE_BBOX[0])
_SY = FRAME_H / float(CAPTURE_BBOX[3]-CAPTURE_BBOX[1])
BOX_FRAME = (int(round((BOX_SCREEN[0] - CAPTURE_BBOX[0]) * _SX)),
             int(round((BOX_SCREEN[1] - CAPTURE_BBOX[1]) * _SY)),
             int(round((BOX_SCREEN[2] - CAPTURE_BBOX[0]) * _SX)),
             int(round((BOX_SCREEN[3] - CAPTURE_BBOX[1]) * _SY)))

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "assets", "start_template.png")


class LobbyDetector:
    """Recognise the observed X/匹配 prompt on the 2560x1440 game screen."""

    BOX = {"left": 2080, "top": 1290, "width": 420, "height": 125}

    def __init__(self):
        import mss
        self.sct = mss.mss()
        path = os.path.join(os.path.dirname(TEMPLATE_PATH), 'lobby_footer.png')
        ref = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if ref is None or ref.shape != (125, 420):
            raise RuntimeError('The lobby X template is missing or has the wrong size')
        # Static X icon and 匹配 lettering; the arrows around them animate.
        self.ref = ref[20:110, 40:230]

    def score(self):
        frame = np.asarray(self.sct.grab(self.BOX))[:, :, :3]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.matchTemplate(gray[20:110, 40:230], self.ref,
                                       cv2.TM_CCOEFF_NORMED)[0, 0])

    def visible(self):
        return self.score() >= 0.90

# tuned on the recordings; see the table in the module docstring
START_THRESHOLD = 0.80      # luping.py used 0.95 against a desktop reference
MIN_SATURATION = 60         # the arena is a saturated cyan
MIN_VALUE = 100             # reject a black / blank screen
MIN_CYAN_FRACTION = 0.40    # the arena floor is cyan-dominated
_CYAN_H = (80, 100)         # OpenCV hue for cyan


class GameDetector:
    def __init__(self, template_path=TEMPLATE_PATH, threshold=START_THRESHOLD):
        self.threshold = threshold
        self.tpl = None
        if os.path.exists(template_path):
            tpl = cv2.imread(template_path, cv2.IMREAD_COLOR)
            if tpl is not None:
                # cv2 loads BGR; work in RGB to match the rest of the codebase
                self.tpl = cv2.resize(cv2.cvtColor(tpl, cv2.COLOR_BGR2RGB),
                    (BOX_FRAME[2]-BOX_FRAME[0],BOX_FRAME[3]-BOX_FRAME[1]),interpolation=cv2.INTER_AREA)
        self.result_tpl = None
        result_path = os.path.join(os.path.dirname(TEMPLATE_PATH), 'result_screen.png')
        if os.path.exists(result_path):
            result = cv2.imread(result_path, cv2.IMREAD_GRAYSCALE)
            if result is not None:
                # Names, scores and podium heights all change between matches.
                # Only the A/继续 footer stays fixed in the result screen.
                result = cv2.resize(result, (FRAME_W, FRAME_H))
                self.result_tpl = result[748:798, 392:546].copy()
        finish_path = os.path.join(os.path.dirname(TEMPLATE_PATH), 'finish_word.png')
        word = cv2.imread(finish_path, cv2.IMREAD_COLOR)
        self.finish_tpl = None if word is None else cv2.resize(
            self._pale_word(cv2.cvtColor(word, cv2.COLOR_BGR2RGB)),
            (236, 54), interpolation=cv2.INTER_AREA)
        self.last_finish_score = 0.0
        self.last = {}

    @staticmethod
    def _pale_word(rgb):
        """Actual light-blue glyph interiors, not the cyan gridded arena."""
        a = rgb.astype(np.int16)
        ink = ((a[..., 0] > 125) & (a[..., 1] > 165)
               & (a[..., 2] > 180) & (a[..., 2] - a[..., 0] > 16))
        return (ink.astype(np.uint8) * 255)

    @staticmethod
    def live_player(red, require_eyes=False):
        """Reject coloured avatar squares; attachment needs real slit eyes."""
        if red is None or not red.alive:
            return False
        if require_eyes and (getattr(red, 'head_px', None) is None
                             or getattr(red, 'eye_state', None) not in ('normal', 'item')):
            return False
        path = getattr(red, 'path', ())
        cells = getattr(red, 'cells', ())
        if require_eyes:
            # Long curled bodies can have an uncertain skeleton despite exact
            # slit eyes.  Do not wait seconds for path ordering we do not need.
            if len(cells) < 4:
                return False
            hx, hy = red.head_px[0] / CELL, red.head_px[1] / CELL
            if not any(max(abs(x - hx), abs(y - hy)) <= 1.5 for x, y in cells):
                return False
            xs, ys = zip(*cells)
        else:
            if getattr(red, 'path_trust', None) != 'ordered' or len(path) < 4:
                return False
            if len(path) < len(cells) * 0.6:
                return False
            xs, ys = zip(*path)
        return max(max(xs) - min(xs), max(ys) - min(ys)) >= 3

    def menu_kind(self, frame):
        """Result footer + podium, excluding the live board's repeating grid.

        A cropped A/继续 match authorises the known result-menu action.  A bare
        scoreboard without that footer only stops play ('scoreboard'); it must
        never authorise an A press on an unrecognised screen.
        """
        if self.result_tpl is None or frame.shape != (FRAME_H, FRAME_W, 3):
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # FINISH is displayed over the live grid *before* the scoreboard; there
        # is no podium or A prompt yet.  Its broad pale-blue lettering occupies
        # most of the central band.  START has similar letters, so the authentic
        # START template explicitly excludes it.  'finish' stops steering but
        # does not authorise the menu controller to press A.
        band = frame[308:465, 170:755].astype(np.int16)
        letters = ((band[..., 0] > 125) & (band[..., 1] > 165)
                   & (band[..., 2] > 180)
                   & (band[..., 2] - band[..., 0] > 16))
        if (int(letters.sum()) > 14000
                and int((letters.sum(axis=1) >= 180).sum()) >= 65
                and self.start_score(frame) < START_THRESHOLD):
            return 'finish'
        # A dark continuous podium baseline is present at the same y on both
        # recorded and real scoreboards; the live grid has no such long bar.
        baseline = float((gray[687:695, 80:846] < 90).mean())
        if baseline < 0.65:
            return None
        # The arena grid covers this empty upper-left board region; result
        # screens have a smooth backdrop.  Lobby artwork has abundant edges.
        board_edges = cv2.Canny(gray[28:196, 32:352], 12, 27)
        if float((board_edges > 0).mean()) > 0.035:
            return None
        footer = gray[748:798, 392:546]
        score = float(cv2.matchTemplate(footer, self.result_tpl,
                                       cv2.TM_CCOEFF_NORMED)[0, 0])
        return 'result' if score >= 0.86 else 'scoreboard'

    # ------------------------------------------------------------------ box
    @staticmethod
    def region(frame):
        x0, y0, x1, y1 = BOX_FRAME
        return frame[y0:y1, x0:x1]

    # ------------------------------------------------------------- detection
    def is_game_on(self, frame, require_start=True):
        """-> (bool, info dict).

        `require_start=True` (the default, and what the bot uses) only accepts a
        frame that shows the game's START overlay.  That is the strictest and
        safest gate: the loose "looks like a cyan arena" test alone is passed by
        a colourful desktop (measured: S=105, cyan fraction 0.83), whereas the
        START template scores 0.99-1.00 on the overlay and 0.16 there.

        `require_start=False` additionally accepts a frame that merely looks like
        the arena, for when the bot is attached to a match already in progress.
        """
        if not isinstance(frame, np.ndarray) or frame.shape != (FRAME_H, FRAME_W, 3):
            info = dict(score=0.0, S=0.0, V=0.0, cyan=0.0, arena_like=False,
                        start_overlay=False, menu_kind=None, reason='invalid_frame')
            self.last = info
            return False, info
        r = self.region(frame)
        hsv = cv2.cvtColor(r, cv2.COLOR_RGB2HSV)
        S = float(hsv[..., 1].mean())
        V = float(hsv[..., 2].mean())
        cyan = float(((hsv[..., 0] >= _CYAN_H[0]) & (hsv[..., 0] <= _CYAN_H[1])).mean())

        score = 0.0
        if self.tpl is not None and r.shape == self.tpl.shape:
            res = cv2.matchTemplate(r, self.tpl, cv2.TM_CCOEFF_NORMED)
            score = float(cv2.minMaxLoc(res)[1])

        arena_like = (S >= MIN_SATURATION and V >= MIN_VALUE
                      and cyan >= MIN_CYAN_FRACTION)
        menu_kind = self.menu_kind(frame)
        start_overlay = arena_like and score >= self.threshold and menu_kind is None
        info = dict(score=round(score, 4), S=round(S, 1), V=round(V, 1),
                    cyan=round(cyan, 3), arena_like=arena_like,
                    start_overlay=start_overlay, menu_kind=menu_kind)
        self.last = info
        if require_start:
            return start_overlay, info
        return (start_overlay or arena_like), info

    def start_score(self, frame):
        r = self.region(frame)
        if self.tpl is None or r.shape != self.tpl.shape:
            return 0.0
        return float(cv2.minMaxLoc(cv2.matchTemplate(r, self.tpl, cv2.TM_CCOEFF_NORMED))[1])


if __name__ == "__main__":
    """Validate against the recordings and against a desktop grab."""
    from PIL import Image
    ROOT = r"C:\Users\Administrator\Desktop\shex"
    det = GameDetector()
    print("box_frame", BOX_FRAME, "template",
          None if det.tpl is None else det.tpl.shape)

    def load(tag, fr):
        return np.asarray(Image.open(rf"{ROOT}\{tag}\{fr}.jpg").convert("RGB")).astype(np.uint8)

    cases = [("START", "luping1", 1), ("START", "luping5", 5), ("START", "luping6", 1),
             ("START", "luping7", 5),
             ("PLAY", "luping4", 200), ("PLAY", "luping4", 600),
             ("PLAY", "luping6", 212), ("PLAY", "luping7", 201),
             ("PLAY", "luping7", 676)]
    print(f"\n{'expect':6s} {'case':18s} {'on?':5s} {'start?':7s} score    S     V    cyan")
    bad = 0
    for label, tag, fr in cases:
        on, info = det.is_game_on(load(tag, fr))
        want_on = (label == "START")          # strict gate: only START counts
        ok = on == want_on
        bad += (not ok)
        print(f"{label:6s} {tag+' f'+str(fr):18s} {str(on):5s} "
              f"{str(info['start_overlay']):7s} {info['score']:6.3f} {info['S']:6.1f} "
              f"{info['V']:6.1f} {info['cyan']:.3f}{'' if ok else '   <-- WRONG'}")

    # negative controls
    for name, img in (("desktop start.png",
                       np.asarray(Image.open(rf"{ROOT}\start.png").convert("RGB"))),
                      ("black", np.zeros((FRAME_H, FRAME_W, 3), np.uint8)),
                      ("white", np.full((FRAME_H, FRAME_W, 3), 255, np.uint8))):
        img = cv2.resize(img, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA) \
            if img.shape[:2] != (FRAME_H, FRAME_W) else img
        on, info = det.is_game_on(img.astype(np.uint8))
        ok = (on is False)
        bad += (not ok)
        print(f"{'NOGAME':6s} {name:18s} {str(on):5s} {str(info['start_overlay']):7s} "
              f"{info['score']:6.3f} {info['S']:6.1f} {info['V']:6.1f} {info['cyan']:.3f}"
              f"{'' if ok else '   <-- WRONG'}")
    print(f"\n{'PASS' if bad == 0 else str(bad) + ' FAILURES'}")
