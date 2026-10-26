"""Live bot: from the START screen all the way to the result screen.

    D:\\ProgramData\\miniconda3\\envs\\she\\python.exe bot.py
    ... --dry       perceive and decide but send nothing to the pad
    ... --log       write logs/live_<timestamp>.jsonl
    ... --in-match  safely recover a missed START (default: shape + eyes + motion)
    ... --start-only  wait strictly for START; disable mid-match recovery

Key behaviours
--------------
* The bot will NOT act until the game is actually on screen.  The gate is a
  template match of the detect region against assets/start_template.png, the same
  approach luping.py used to decide when to start recording.  Without this the
  bot happily "finds" a red snake in an unrelated window and starts steering
  (observed: a red blob of length 7 on a desktop produced continuous input).
* The snake can already move while START is on screen, so steering begins the
  moment START is up - the match clock is already running.
* The two starting diamonds sit at mid-height, one left of centre and one right.
* The accelerate key always costs beans, so it is used for travel on open
  straights and for mashing when the tail needs shortening (see policy.py).
"""
from __future__ import annotations

import argparse
from endgame_strategy import EndgameSchedule
import json
import math
import os
import sys
import time

# Windows Python 3.10 monotonic uses 15.625ms GetTickCount64.
# Use QPC consistently for vision age, motion and controller deadlines.
control_time = time.perf_counter
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (LOG_DIR, LOOP_HZ, GAME_HZ, START_DIAMONDS,
                    MATCH_MAX_SECONDS, RESULT_ABSENT,
                    MENU_RETRY_SECONDS, MENU_A_TO_X_SECONDS, CONTROL_DELAY_MS, CONTROL_DELAY_MS_NORMAL, CONTROL_DELAY_MS_BOOST, CONTROL_DELAY_MS_BOOST_GREEN, COLS, ROWS)
import config

FRAME_DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames_dump")
TIMER_DUMP = LOG_DIR   # 直接写进已有的 logs/：本会话沙箱不允许在新建目录写文件
from capture import Vision
from perception import Perception, SnakeTracker
from temporal import TemporalSense
from attention_core import Attention
from state_analysis import StateAnalysis
from policy import Policy
from game_start import GameDetector, LobbyDetector
from window_perception import window_box


class Phase:
    """WAITING -> PLAYING -> RESULT.

    START is the fastest gate.  A missed overlay can also be recovered through
    ordered snake-body + slit-eye + sustained pixel-motion evidence, never through
    arena colour alone.  WAITING itself does not authorise controller input.
    """

    def __init__(self, in_match=True):
        self.phase = "WAITING"
        self.in_match = in_match
        self.entered = False
        self._white_since = None
        self._absent_since = None
        self._board_seen = 0
        self._playing_since = None
        self.detector = GameDetector()
        self.last_info = {}
        self._motion = deque(maxlen=8)
        self.entry_reason = None
        self.match_time_known = False
        self.input_allowed = False
        self._player_evidence_until = 0.0
        # Once START is proved, a short eyeless player can still be followed by
        # consecutive native-mask endpoint moves.  This is NOT an entry signal.
        self._native_cells = None
        self._native_head = None
        self._native_at = None
        self._native_moves = deque(maxlen=2)

    def _short_player_motion(self, red, now):
        """Two recent mask-supported head steps; never extrapolate a missing snake."""
        if red is None or not red.alive:
            self._native_cells = self._native_head = self._native_at = None
            self._native_moves.clear()
            return False
        cells = set(getattr(red, 'cells', ()) or ())
        path = list(getattr(red, 'path', ()) or ())
        pixels = getattr(red, 'n_px', 0)
        # Scope the fallback to the 2-5-cell mask that defeated the normal gate.
        # A result graphic or merged opponent cannot authenticate a player.
        if (not 2 <= len(cells) <= 5 or not 2 <= len(path) <= 5
                or getattr(red, 'path_trust', None) != 'ordered'
                or not 200 <= pixels <= 1800 or not set(path).issubset(cells)):
            self._native_cells = self._native_head = self._native_at = None
            self._native_moves.clear()
            return False
        previous, old_head, old_at = self._native_cells, self._native_head, self._native_at
        self._native_cells, self._native_at = cells, now
        if previous is None or old_at is None or not 0 < now - old_at <= .25:
            self._native_head = None
            self._native_moves.clear()
            return False
        new = cells - previous
        endpoints = {tuple(path[0]), tuple(path[-1])}
        tip = next(iter(new & endpoints)) if len(new & endpoints) == 1 else None
        # One newly occupied endpoint and shared old body are actual movement,
        # unlike a static coloured icon or an extrapolated/head-history point.
        if (tip is None or len(new) != 1 or not (cells & previous)
                or (old_head is not None and
                    max(abs(tip[0] - old_head[0]), abs(tip[1] - old_head[1])) > 1)):
            if new:
                self._native_moves.clear()
                self._native_head = None
            elif self._native_moves and now - self._native_moves[-1] > .40:
                self._native_moves.clear()
            return False
        self._native_head = tip
        self._native_moves.append(now)
        return (len(self._native_moves) >= 2
                and now - self._native_moves[0] <= .50)

    def _moving_player(self, red, info, flags, now):
        if (not self.in_match or not info.get('arena_like')
                or info.get('menu_kind') or flags.get('whiteout')
                or not self.detector.live_player(red, require_eyes=True)):
            # Brief eye occlusion does not combine evidence across different scenes.
            if self._motion and now - self._motion[-1][0] > 0.25:
                self._motion.clear()
            if not info.get('arena_like') or info.get('menu_kind'):
                self._motion.clear()
            return False
        point = tuple(red.head_px)
        if self._motion:
            prev_t, prev = self._motion[-1]
            dist = float(np.linalg.norm(np.subtract(point, prev)))
            if now - prev_t > 0.25 or dist > 128 or now <= prev_t:
                self._motion.clear()
        self._motion.append((now, point))
        if len(self._motion) < 3:
            return False
        points = [p for t, p in self._motion if now - t <= 0.5]
        moves = [float(np.linalg.norm(np.subtract(b, a)))
                 for a, b in zip(points, points[1:])]
        return sum(m >= 3.0 for m in moves) >= 2 and sum(moves) >= 12.0

    # how the bot decides the game is really up
    NEED_FRAMES = 2          # consecutive frames of evidence

    def update(self, frame, snakes, flags, now):
        on, info = self.detector.is_game_on(frame, require_start=not self.in_match)
        self.last_info = info
        red = snakes.get("red")
        moving_player = self._moving_player(red, info, flags, now)
        info['moving_player'] = moving_player
        snake_here = red is not None and red.alive

        # ---- when is the match over?
        #
        # NOT "whiteout and our snake is gone".  That was the old rule and it is
        # wrong: it is the DEATH signature, not the end of the match.  Section 5.2-5.3
        # is death -> dissolve -> whiteout wash -> respawn, and a live run showed
        # exactly that at t=85.5 (snake gone) followed at t=86.7 by a fresh 5-cell
        # snake respawning at (19,19).  Every live attempt so far was cut short by
        # this rule firing on our own death, a few seconds into the match.
        #
        # A respawn takes 1-2 s, so an absence that lasts many seconds means the
        # board is not coming back - that is the match ending.  The match clock is
        # the hard backstop.
        red = snakes.get("red")
        snake_here = red is not None and red.alive

        if self.phase == "PLAYING" and not self.match_time_known:
            if red is None and not flags["whiteout"]:
                if self._absent_since is None:
                    self._absent_since = now
                elif now - self._absent_since > RESULT_ABSENT:
                    self.phase = "RESULT"
                    self._board_seen = 0
            else:
                self._absent_since = None

        # ...or the match simply ran its length.  A result screen is not reliably
        # separable from a live board by looking at it (see config.MATCH_MAX_SECONDS),
        # so the clock is the backstop that stops the bot steering a score screen and
        # pressing A on its menu, where A means "continue".
        if (self.phase == "PLAYING" and self._playing_since is not None
                and now - self._playing_since > MATCH_MAX_SECONDS):
            self.phase = "RESULT"
            self._board_seen = 0

        if self.phase == "WAITING":
            # Fast path: two START observations.  Recovery path: real slit eyes,
            # extended body and three temporally plausible moving-head observations.
            # Cyan + a coloured blob alone never authorises entry.
            if info.get('start_overlay') and not flags.get('whiteout'):
                self._board_seen += 1
            else:
                self._board_seen = 0
            if self._board_seen >= self.NEED_FRAMES or moving_player:
                self.phase = 'PLAYING'
                self.entry_reason = 'start_overlay' if self._board_seen >= self.NEED_FRAMES else 'moving_player'
                self.match_time_known = self.entry_reason == 'start_overlay'
                self._board_seen = 0
                self._playing_since = now
        elif self.phase == "RESULT":
            # leave RESULT for WAITING so the next match's START is awaited
            if not flags["whiteout"] and red is None:
                self.phase = "WAITING"
                self._board_seen = 0
                self._playing_since = None
        native_body = (self.entry_reason == 'start_overlay' and snake_here
                       and getattr(red, 'head_cell', None) is not None
                       and getattr(red, 'path_trust', None) == 'ordered'
                       and len(getattr(red, 'cells', ()) or ()) >= 4)
        strong_player = (self.detector.live_player(red) or native_body
                         or (snake_here and getattr(red, 'head_px', None) is not None
                             and getattr(red, 'eye_state', None) in ('normal', 'item')))
        native_scene = (self.phase == 'PLAYING' and self.entry_reason == 'start_overlay'
                        and info.get('arena_like', False) and not info.get('menu_kind')
                        and not flags.get('whiteout'))
        short_motion = self._short_player_motion(red if native_scene else None, now)
        if info.get('start_overlay') or strong_player:
            self._player_evidence_until = now + (0.75 if native_body else 0.25)
        elif native_scene and short_motion:
            self._player_evidence_until = now + 0.40
        early_finish = (info.get('menu_kind') == 'finish' and
                        (flags.get('whiteout') or
                         (self.match_time_known and self._playing_since is not None
                          and now - self._playing_since < 55.0)))
        info['early_finish_ignored']=bool(early_finish)
        self.input_allowed = (self.phase == 'PLAYING' and info.get('arena_like', False)
                              and (not info.get('menu_kind') or early_finish) and snake_here
                              and not flags.get('whiteout')
                              and now <= self._player_evidence_until)
        # The START overlay can momentarily resemble the large FINISH letters.
        # A normal round lasts about 65 seconds, so an early "finish" is not an
        # end-of-round signal.  The result footer remains a separate positive
        # screen and can still stop the controller immediately.
        if self.phase == 'PLAYING' and info.get('menu_kind') and not early_finish:
            self.phase = 'RESULT'
            self.input_allowed = False
        info['input_allowed'] = self.input_allowed
        info['short_player_motion'] = bool(short_motion)
        info['player_evidence_remaining_ms'] = round(
            max(0.0, self._player_evidence_until - now) * 1000, 1)
        info['entry_reason'] = self.entry_reason
        return self.phase


class MenuGate:
    """No timer-only presses.  Unknown menus need a human, not guessed X input."""

    def __init__(self):
        self.kind = None
        self.seen = 0
        self.next = 0.0
        self.lobby_pressed = False

    def action(self, info, now):
        kind = info.get('menu_kind')
        if info.get('start_overlay') or kind not in ('result', 'lobby'):
            self.kind, self.seen = None, 0
            self.lobby_pressed = False
            return None
        self.seen = self.seen + 1 if kind == self.kind else 1
        self.kind = kind
        if self.seen < 2 or now < self.next:
            return None
        # A matchmaking press can fail to register.  After a full queue-length
        # delay, a still-visible X/匹配 lobby is evidence that we never left it.
        self.next = now + (MENU_A_TO_X_SECONDS if kind == 'result' else MENU_RETRY_SECONDS)
        if kind == 'lobby':
            self.lobby_pressed = True
        return 'a' if kind == 'result' else 'x'


class HeadGuard:
    """Reject impossible head jumps without forgetting the last real position."""

    def __init__(self):
        self.last = None
        self.at = None

    def accept(self, red, now):
        if red is None or red.head_cell is None or red.head_px is None:
            return False
        head = tuple(red.head_cell)
        if self.last is None or self.at is None:
            if red.eye_state not in ('normal', 'item'):
                return False
        else:
            dt = now - self.at
            if dt <= 0:
                return False
            # A colour/body endpoint with carried eye state is not a fresh
            # eye position. r027 f411 jumped three tiles to such an endpoint
            # for one frame and replaced an otherwise correct food approach.
            native=(getattr(red,'head_from_eyes',False) or
                    getattr(red,'head_from_pixels',False))
            if not native and max(abs(head[i]-self.last[i]) for i in (0,1))>1:
                return False
            if dt > 0.35:
                fresh_buffer=(red.eye_state=='buffer' and getattr(red,'head_from_eyes',False)
                              and getattr(red,'head_age',99)==0 and dt<=.7
                              and max(abs(head[i]-self.last[i]) for i in (0,1))<=2)
                if red.eye_state not in ('normal', 'item') and not fresh_buffer:
                    return False
            else:
                limit = max(2, math.ceil(2 * GAME_HZ * dt))
                if max(abs(head[0] - self.last[0]),
                       abs(head[1] - self.last[1])) > limit:
                    return False
        self.last, self.at = head, now
        return True


def fixed_accel_decision(decision, mode):
    """The user's chosen fixed mode overrides adaptive policy releases."""
    if mode in ('hold','off'):
        return decision[0], mode == 'hold', False, decision[3]
    return decision


def main(dry=None, log_on=None, minutes=None, after_start=None, auto_start=None,
         in_match=None, accel_mode=None):
    """Run the bot.  Any argument left as None comes from the command line, so
    both `python bot.py --dry` and `bot.main(dry=True)` work."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="do not touch the pad")
    ap.add_argument("--log", action="store_true", help="write a jsonl log")
    ap.add_argument("--minutes", type=float, default=10.0,
                    help="hard stop this many minutes after launch (a safety net; the "
                         "normal stop is --after-start)")
    ap.add_argument("--after-start", type=float, default=70.0, metavar="SECONDS",
                    help="stop this many seconds after the match starts, i.e. after the "
                         "START overlay is recognised (default 60)")
    ap.add_argument('--in-match', action='store_true', default=True,
                    help='safely recover a missed START using moving snake evidence (default)')
    ap.add_argument('--start-only', dest='in_match', action='store_false',
                    help='require the START overlay; disable safe mid-match recovery')
    ap.add_argument("--save-frames", type=int, default=0, metavar="N",
                    help="during PLAYING, save every Nth frame as a PNG next to a "
                         "red-mask overlay, for diagnosing what the mask really saw")
    ap.add_argument('--capture-threads',type=int,choices=(1,2),default=1,
                    help='2: capture worker plus the sole control thread, latest frame only')
    ap.add_argument("--attention", dest="attention", action="store_true", default=True,
                    help="enable the 7x7 head window, peripheral vision and the "
                         "state-analysis layer (default: on)")
    ap.add_argument("--no-attention", dest="attention", action="store_false",
                    help="disable the layers above and use the raw decode path "
                         "(kept for A/B and as a fallback)")
    ap.add_argument('--auto-start', action='store_true',
                    help='drive only positively recognised menus; unknown lobby needs a human')
    ap.add_argument('--accel-mode', choices=('hold', 'off', 'adaptive'), default='hold',
                    help='keep A held for the match, keep it released, or use the older adaptive mode')
    ap.add_argument("--save-blind", type=int, default=0, metavar="N",
                    help="save up to N in-match frames in which our snake was NOT "
                         "found at all, to see what the mask is missing")
    args = ap.parse_args()
    if dry is not None:
        args.dry = dry
    if log_on is not None:
        args.log = log_on
    if minutes is not None:
        args.minutes = minutes
    if after_start is not None:
        args.after_start = after_start
    if auto_start is not None:
        args.auto_start = auto_start
    if in_match is not None:
        args.in_match = in_match
    if accel_mode is not None:
        if accel_mode not in ('hold', 'off', 'adaptive'):
            raise ValueError('accel_mode must be hold, off, or adaptive')
        args.accel_mode = accel_mode

    if args.capture_threads==2:
        from latest_capture import LatestCapture
        vision=LatestCapture(Vision)
    else:
        vision=Vision()
    # 顶部计时器不参与控制（用户 10-18："不要惦记游戏顶部的计时器，知道大致偏差调整实际计时就行"）。
    # 比赛时钟改用固定偏差修正：见 config.MATCH_CLOCK_OFFSET 与 policy 里的对应处理。
    timer_clock=None
    timer_err=False
    diagnostics=None
    if args.save_frames:
        from diagnostic_writer import DiagnosticWriter
        diagnostics=DiagnosticWriter(FRAME_DUMP,Perception)
    per = Perception()
    track = SnakeTracker()
    sense = TemporalSense()
    # opt-in: the 7x7 head window + peripheral vision + state analysis
    attention = Attention(per, sense) if args.attention else None
    analyser = StateAnalysis()
    policy = Policy()
    phase = Phase(in_match=args.in_match)
    lobby = LobbyDetector()
    head_guard = HeadGuard()

    pad = None
    pad_creation_s = 0.0
    menu = MenuGate()

    def drive_menus(now):
        """Explicit auto mode still requires CURRENT recognised menu evidence."""
        action = menu.action(phase.last_info, now)
        if action is None:
            return
        creating = pad is None
        ensure_pad()
        if pad is not None:
            if ((action == 'x' and not lobby.visible())
                    or (action != 'x' and
                        ((creating and pad_creation_s > 1.0 / GAME_HZ)
                         or control_time() - captured_at > 2.0 / GAME_HZ))):
                pad.reset()
                menu.next = control_time()
                menu.lobby_pressed = False
                return  # Next frame must confirm the menu again, without sleeping.
            print(f'[auto] pressing {action.upper()} on recognised menu')
            pad.tap(action)

    def ensure_pad():
        """Create the virtual pad, but only once the match is actually on.

        Creating it up front means a virtual controller exists during WAITING, and
        a real XInput game may then take its input from the virtual pad instead of
        the physical one - so the human could not press A to leave the result
        screen or X to start matchmaking.  Measured: the pad becomes visible to
        XInput within ~0.6 s of creation, and the snake can already move during
        START, so waiting costs well under a second of a 65 s match.
        """
        nonlocal pad, pad_creation_s
        if args.dry or pad is not None:
            return pad
        try:
            from input_pad import Pad
            creation_started = control_time()
            pad = Pad()
            pad_creation_s = control_time() - creation_started
            pad.reset()
            print(f'[bot] virtual pad created after positive game/menu evidence '
                  f'({pad_creation_s * 1000:.1f} ms)')
        except Exception as e:
            print(f"[bot] could NOT create the virtual pad ({type(e).__name__}: {e}) "
                  f"- continuing in observe-only mode")
            args.dry = True
        return pad

    def return_to_lobby():
        """Press A only on a recognised result prompt, then verify the lobby."""
        deadline = control_time() + 18.0
        result_seen = 0
        next_press = 0.0
        presses = 0
        result_visible_since=None
        while control_time() < deadline:
            tick = control_time()
            if pad is not None:
                pad.service(tick)
            if lobby.visible():
                print('[auto] matching lobby confirmed')
                return True
            screen = vision.grab()
            kind = phase.detector.menu_kind(screen)
            result_seen = result_seen + 1 if kind == 'result' else 0
            if kind!='result':result_visible_since=None
            elif result_visible_since is None:result_visible_since=tick
            if (result_seen >= 2 and tick >= next_press and result_visible_since is not None
                    and tick-result_visible_since>=1.0):
                if presses == 0:
                    import cv2
                    os.makedirs(LOG_DIR, exist_ok=True)
                    result_path = os.path.join(
                        LOG_DIR, f"result_{time.strftime('%Y%m%d_%H%M%S')}.png")
                    cv2.imwrite(result_path, cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))
                    print(f'[auto] saved result screen: {result_path}')
                ensure_pad()
                if pad is not None:
                    pad.reset()
                    if pad.tap('a', hold_ms=250):
                        presses += 1
                        next_press = tick + 1.5
                        print(f'[auto] result prompt confirmed; pressing A '
                              f'(attempt {presses})')
            time.sleep(0.08)
        print('[auto] lobby could not be confirmed after the match')
        return False

    log = None
    if args.log:
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, f"live_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
        log = open(path, "w", encoding="utf-8")
        print("logging to", path)

    print('waiting for START or safe moving-player evidence (no guessed inputs) ...')
    if args.auto_start:
        print('[auto] only known menu templates permit presses; unknown lobby/X needs a human')
    t_start = control_time()
    last = t_start
    frames = 0
    ms_sum = 0.0
    worst = 0.0
    bean_hist = []
    blind_saved = [0]
    recent = deque(maxlen=4)          # rolling window of the last in-match frames
    _last_keep = 0.0
    announced_start = False
    _last_wait_print = 0.0
    match_started = None              # when the START overlay was recognised
    bootstrap_pending = False         # discard visual history from the START title
    was_whiteout = False
    match_accel_mode = args.accel_mode
    endgame_schedule = EndgameSchedule(args.accel_mode)

    def fresh_tracking():
        nonlocal track, sense, attention, analyser, policy, bean_hist, head_guard
        per.reset()
        track = SnakeTracker()
        sense = TemporalSense()
        attention = Attention(per, sense) if args.attention else None
        analyser = StateAnalysis()
        policy = Policy()
        head_guard = HeadGuard()
        bean_hist = []
        phase._native_cells = phase._native_head = phase._native_at = None
        phase._native_moves.clear()
        phase._player_evidence_until = 0.0

    # head from the previous frame, for the hybrid window (see HYBRID_WINDOW)
    hybrid_prev_head = None
    head_jump_streak = 0          # 连续 head_jump 帧数（见下面的保持航向分支）
    HEAD_JUMP_HOLD_FRAMES = 20    # 最多保持这么久，再长就承认读数真的坏了
    try:
        while True:
            now = control_time()
            match_accel_mode = endgame_schedule.mode(now, match_started)
            endgame_schedule.enforce(pad)
            policy.key_boost = match_accel_mode == 'hold'
            if pad is not None:
                pad.service(now)
            if now - t_start > args.minutes * 60:
                print("time limit reached")
                break
            if match_started is not None and now - match_started >= args.after_start:
                print(f"{args.after_start:.0f} s of match elapsed - stopping "
                      f"(started at t={match_started - t_start:.2f}s)")
                break
            captured_at = control_time()
            # per-phase wall-clock marks, to attribute the ~94 ms live cycle.  `cycle_at` is the same
            # instant as the previous frame's `loop_end`, so the deltas add up to the cycle exactly and
            # nothing can hide between them.
            cycle_at = time.perf_counter()
            # default for frames that never reach the policy (stale frame, whiteout, no head)
            timer_px = 0
            _delay_ms = CONTROL_DELAY_MS_NORMAL
            _speed_item = False
            # phase marks, defaulted to the frame start so every early-continue path still logs a
            # consistent set and the deltas always sum to the cycle.  `loop_end` MUST be in this list: it is
            # read by the per-frame log, which runs BEFORE the pacing block assigns it, and omitting it raised
            # `UnboundLocalError: local variable 'loop_end' referenced before assignment` on every frame - the
            # same failure mode as `_delay_ms` earlier, which cost two matches.  Seven were lost to this one.
            grabbed_at = perceived_at = decided_now = applied_at = loop_end = cycle_at
            pad_was_ready = pad is not None
            if args.capture_threads==2:
                img,captured_at=vision.grab_sample()
                now=captured_at
            else:
                img = vision.grab()
            # 每 4 帧读一次计时器（约 3 ms）；样本图用于离线标定。
            if timer_clock is not None and frames % 4 == 0:
                try:
                    import cv2 as _cv2
                    _tpx,_tbgr=timer_clock.read()
                    timer_px=_tpx
                    if frames % 40 == 0:
                        _ok=_cv2.imwrite(os.path.join(TIMER_DUMP,f"timer_{frames:05d}.png"),_tbgr)
                        if not _ok:
                            print("[bot] timer dump write failed",flush=True)
                        open(os.path.join(TIMER_DUMP,"px.log"),"a").write(f"{frames} {_tpx}\n")
                except Exception as _te:
                    if not timer_err:
                        print(f"[bot] timer read error {type(_te).__name__}: {_te}",flush=True)
                        timer_err=True
                    timer_px=0
            grabbed_at = time.perf_counter()
            # pass the match clock when we have it so the "balls do not exist yet" gate is
            # driven by real match time rather than by how many frames this process has seen
            # ---- global head + window detail.  The box is centred on the head from the PREVIOUS
            # frame, because this frame's head is not known until the decode below has run - the order
            # the live loop actually executes in, which is also the order ab_hybrid.py measured.
            restrict = None
            if config.HYBRID_WINDOW and hybrid_prev_head is not None:
                # 用户 10-18 要求提高反应力：窗口扫描把感知从 ~52 ms 压到 ~37 ms，但窗口外看不见钻石
                # （实测 gems/frame 1.8 -> 0.6，目标选择立刻变瞎）。折中：每 HYBRID_GLOBAL_EVERY 帧做一次
                # 全局重扫，其余帧只扫头部窗口 —— 速度收益保留大半，远处钻石的刷新不会被切断。
                if (frames % max(1, config.HYBRID_GLOBAL_EVERY)) != 0:
                    restrict = window_box(hybrid_prev_head)
            per.high_speed_mode=bool(getattr(policy.follower,'fast_active',False) and
                                     now < getattr(policy.follower,'fast_deadline_wall',0.) and match_accel_mode=='hold')
            body_history=getattr(policy,'body_history',None)
            per.player_body_hint=(body_history.occupied() if body_history is not None else frozenset())
            per.player_body_hint_complete=bool(per.player_body_hint)
            snakes, items, flags = per.run(
                img, t=(now - match_started) if match_started is not None and phase.match_time_known else None,
                allow_balls=True if phase.phase == 'PLAYING' and not phase.match_time_known else None,
                restrict_box=restrict)
            _hr = snakes.get('red')
            if _hr is not None and _hr.head_px is not None:
                hybrid_prev_head = _hr.head_px
            perceived_at = time.perf_counter()
            dt = now - last
            last = now
            ms_sum += flags["ms"]
            worst = max(worst, flags["ms"])

            red = snakes.get('red')
            ph = phase.update(img, snakes, flags, now)
            if ph == 'WAITING' and not phase.last_info.get('arena_like'):
                if lobby.visible():
                    phase.last_info['menu_kind'] = 'lobby'
                    phase.last_info['lobby_score'] = round(lobby.score(), 3)
            if ph == 'PLAYING' and match_started is None and not phase.match_time_known:
                # Entry was proved by moving-player evidence, not a known START.
                # Re-decode once so the FIRST decision also sees existing balls;
                # t=None alone would still use this process's tiny frame counter.
                snakes, items, flags = per.run(img, allow_balls=True,
                                                restrict_box=restrict)
                red = snakes.get('red')
            # The clock for "play for N seconds" starts the moment the match is
            # recognised, and only ever once per run: keyed off the phase itself
            # rather than off Phase.entered, which is a UI flag set further down.
            if ph == "PLAYING" and match_started is None:
                match_started = now
                authentic_red = (red is not None and red.eye_state in ('normal', 'item')
                                 and red.head_px is not None and len(red.cells) >= 4)
                bootstrap_pending = (phase.entry_reason == 'start_overlay'
                                     and not authentic_red)
                print(f"match recognised at t={now - t_start:.2f}s "
                      f"(start_score={phase.last_info.get('score', 0):.3f}) - "
                      f"playing for {args.after_start:.0f}s")
            # A confidently identified snake may act DURING the START overlay: the user asks for it twice
            # ("start出现即可行动，不要等start消失了才行动"), and waiting costs seconds of a 65 s match.  The
            # authentication test below is unchanged, so the START artwork - which decodes as a static
            # three-cell snake with no eyes - still cannot steer.
            if (ph == 'PLAYING' and bootstrap_pending and authentic_red):
                fresh_tracking()
                bootstrap_pending = False
                phase.last_info['start_overlay'] = None
                print('[bot] START still on screen but the snake is identified - steering now')
            if ph == 'PLAYING' and bootstrap_pending:
                ensure_pad()
                if pad is not None and match_accel_mode == 'hold':pad.set_accel(True)
                if pad is not None:
                    pad.apply_decision(None, evidence=False,
                                       preserve_accel=match_accel_mode == 'hold')
                if not phase.last_info.get('start_overlay'):
                    # The START artwork itself sometimes decodes as a static
                    # three-cell red snake with no eyes.  Carrying that identity
                    # into play hid the real slit-eyed snake for several seconds.
                    fresh_tracking()
                    bootstrap_pending = False
                    print('[bot] START cleared; fresh snake tracking begins')
                frames += 1
                continue
            if ph == 'PLAYING':
                if match_accel_mode == 'hold':
                    ensure_pad()
                    if pad is not None:pad.set_accel(True)
                if flags['whiteout']:
                    if not was_whiteout and match_accel_mode == 'hold':
                        if pad is not None:
                            pad.rearm_accel(wait_for_motion=True)
                        print('[bot] death whiteout; release A and wait for active respawn motion')
                    was_whiteout = True
                elif was_whiteout:
                    was_whiteout = False
                    if pad is not None:
                        pad.apply_decision(None, evidence=False,
                                           preserve_accel=match_accel_mode == 'hold')
                    fresh_tracking()
                    if match_accel_mode == 'hold' and pad is not None:
                        pad.rearm_accel(wait_for_motion=True)
                        print('[bot] whiteout cleared; wait for active snake before the 10-frame release')
                    print('[bot] whiteout cleared; tracking the new red snake')
                    frames += 1
                    continue
            if ph == "WAITING":
                # Unknown menus need a human: do not keep a virtual slot that
                # would prevent their physical controller from starting the game.
                if pad is not None and not phase.last_info.get('menu_kind') and not args.auto_start:
                    pad.reset()
                    pad = None
                    print('[bot] unknown screen: virtual controller released to the human')
                if args.auto_start:
                    drive_menus(now)
                if now - _last_wait_print > 2.0:
                    _last_wait_print = now
                    info = phase.last_info
                    print(f"  waiting: start_score={info.get('score', 0):.3f} "
                          f"arena_like={info.get('arena_like')} "
                          f"cyan={info.get('cyan')} S={info.get('S')} "
                          f"({now - t_start:.1f}s)")
                frames += 1
                time.sleep(0.05)
                continue
            if ph == "RESULT":
                import cv2 as _cv2
                _stop_path=os.path.join(LOG_DIR,"finish_detect_"+time.strftime("%Y%m%d_%H%M%S")+".png")
                _cv2.imencode(".png",_cv2.cvtColor(img,_cv2.COLOR_RGB2BGR))[1].tofile(_stop_path)
                # Neutralise immediately, before saving diagnostic images.
                if pad:
                    pad.apply_decision(None, evidence=False)
                if log:
                    log.write(json.dumps(dict(
                        f=frames + 1, t=round(now - t_start, 3), phase=ph,
                        arena_like=phase.last_info.get('arena_like'),
                        menu_kind=phase.last_info.get('menu_kind'),
                        short_player_motion=phase.last_info.get('short_player_motion'),
                        player_evidence_remaining_ms=phase.last_info.get('player_evidence_remaining_ms'),
                        input_allowed=False, dir=None, accel=False,
                        execution=pad.last_execution if pad else None)) + "\n")
                # The settlement page is the authority on length (the user reads the
                # real number off it), but our end-of-match trigger is the 80 s clock
                # and the game can have moved on by then - a live run saved a profile
                # screen instead.  So keep a rolling window of the last few in-match
                # frames and write those: whenever the trigger fires, the settlement
                # page is inside the window.
                import cv2
                os.makedirs(LOG_DIR, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                for k, (ft, fimg) in enumerate(recent):
                    shot = os.path.join(LOG_DIR, f"tail_{stamp}_{k}.png")
                    cv2.imwrite(shot, cv2.cvtColor(fimg, cv2.COLOR_RGB2BGR))
                print(f"result screen detected - stopping. last {len(recent)} in-match "
                      f"frames saved as {os.path.join(LOG_DIR, 'tail_' + stamp + '_*.png')}")
                if red is not None:
                    print(f"  our last measured red length was {red.length} cells "
                          f"(px={red.n_px}, eye={red.eye_state})")
                else:
                    print("  our snake was not visible on the last frame")
                if args.auto_start and not args.dry:
                    return_to_lobby()
                break

            flags['sample_t'] = now
            # ---- temporal layer.  Filters item flicker, keeps a running head window on
            # the player's snake, and rejects impossible length changes.  It never
            # creates a body, so a frame it cannot improve passes through unchanged.
            items, dropped_items = sense.filter_items(items)
            st = track.update(snakes, items, flags, dt)
            st['match_elapsed'] = (now - match_started if match_started is not None
                                   and phase.match_time_known else None)
            st["snakes"], st["sense_notes"] = sense.update(st["snakes"], items, flags,
                                                           flags["whiteout"])
            st["items_dropped"] = dropped_items
            # 用户（10-18）："死亡判定用白屏/泛白即可，因为别的蛇死亡也会有特效，区别在于不会全屏泛白"。
            # 把全屏泛白标志交给策略：泛白期间画面无信息，红蛇往往会以"死亡闪光"的假位置出现，
            # 必须整帧拒绝，而不是拿这个假头去决策。
            st["whiteout"] = bool(flags["whiteout"])

            # ---- attention + state analysis.  The 7x7 window on the head is re-measured every
            # frame (diamonds, obstacles, beans) and the periphery only every
            # PERIPHERAL_EVERY frames; state_analysis then judges the readings against the
            # game's mechanics before the policy sees them. Enabled by default;
            # --no-attention keeps the legacy observation path for controlled A/B.
            if attention is not None:
                _mt = (now - match_started) if match_started is not None and phase.match_time_known else None
                af = attention.scan(img, st["snakes"], items, flags,
                                    red_cells=None, match_t=_mt)
                belief = analyser.analyse(af)
                st["attention"] = af
                st["belief"] = belief
                st["head_cell"] = belief.head_cell or st.get("head_cell")
                if belief.anomalies:
                    st.setdefault("anomalies", []).extend(belief.anomalies)

            bean_hist.append(len(items.get("bean", [])))
            st["key_held"] = SnakeTracker.key_held(bean_hist)

            if args.save_frames and frames % args.save_frames == 0:
                diagnostics.submit(frames,img)

            red = st["snakes"].get("red")
            head_guard_ok = head_guard.accept(red, now)
            if pad is not None and match_accel_mode == 'hold':
                if pad.confirm_respawn_motion(red,phase.input_allowed and head_guard_ok,now):
                    print('[bot] active respawn motion confirmed; A released for >=10 game frames before holding',flush=True)
            if head_guard_ok:
                head_jump_streak = 0
            # Phase sees the raw frame before the temporal tracker repairs a
            # partially hidden eye/head.  In the live run it revoked input for
            # 1.5 s while the tracked one-cell snake still had a fresh normal
            # eye and walked straight into the lower wall.  Accept that fresh
            # tracked evidence only after START has authenticated the match.
            tracked_player = (red is not None and red.head_px is not None
                              and red.head_cell is not None
                              and red.eye_state in ('normal', 'item', 'buffer')
                              and red.head_age <= 2)
            if (not phase.input_allowed and phase.match_time_known
                    and phase.phase == 'PLAYING' and tracked_player
                    and phase.last_info.get('arena_like')
                    and (not phase.last_info.get('menu_kind') or phase.last_info.get('early_finish_ignored'))
                    and not flags['whiteout']):
                phase.input_allowed = True
                phase.last_info['input_allowed'] = True
            if now - _last_keep >= 3.0:
                _last_keep = now
                recent.append((now, img.copy()))
            if (args.save_blind and (red is None or not red.alive)
                    and blind_saved[0] < args.save_blind):
                import cv2
                os.makedirs(FRAME_DUMP, exist_ok=True)
                shot = os.path.join(FRAME_DUMP, f"blind{blind_saved[0]:03d}.png")
                cv2.imwrite(shot, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"[diag] our snake was not found here - saved {shot}")
                blind_saved[0] += 1
            if not phase.entered:
                phase.entered = True
                info = phase.last_info
                print(f"game detected at t={now-t_start:.2f}s via "
                      f"start_score={info.get('score', 0):.3f} "
                      f"arena_like={info.get('arena_like')} - now steering")
                ensure_pad()
            if red is not None and red.alive and not announced_start:
                announced_start = True
                print(f"red snake found at t={now-t_start:.2f}s "
                      f"len={red.length} eye={red.eye_state}")

            frame_age_s = control_time() - captured_at
            slow_pad_creation = (pad is not None and not pad_was_ready
                                 and pad_creation_s > 1.0 / GAME_HZ)
            if slow_pad_creation or frame_age_s > 2.0 / GAME_HZ:
                if pad:
                    pad.apply_decision(None, evidence=False,
                                       preserve_accel=match_accel_mode == 'hold')
                dec = (None, False, False,
                       'pad_creation_refresh' if slow_pad_creation else 'stale_frame')
            elif not phase.input_allowed:
                if pad:
                    pad.apply_decision(None, evidence=False,
                                       preserve_accel=match_accel_mode == 'hold')
                # Keep A held in `hold` mode: the DIRECTION is withheld because the reading is
                # unreliable, but releasing the boost for a brief vision gap is the mistake the
                # user's rule forbids (and it was 39% of all boost-off frames, measured).
                dec = (None, match_accel_mode == 'hold', False, 'no_game_evidence')
            elif red is None or red.head_cell is None or red.head_px is None:
                if pad:
                    pad.apply_decision(None, evidence=False,
                                       preserve_accel=match_accel_mode == 'hold')
                # Keep A held in `hold` mode: the DIRECTION is withheld because the reading is
                # unreliable, but releasing the boost for a brief vision gap is the mistake the
                # user's rule forbids (and it was 39% of all boost-off frames, measured).
                dec = (None, match_accel_mode == 'hold', False, 'no_player_snake')
            elif not head_guard_ok:
                # 用户（10-18 第 11 局实录）："540 帧已经复活可以开始行动了却一直呆住直到 595 帧，
                # 浪费了大量时间"。查证：f=538—595 每一帧都是 `head_jump`，而同一段头部轨迹
                # (9,1)->(10,3) 其实是在正常移动 —— 复活后红蛇识别抖动，守卫把正常移动也判成跳变，
                # 而原来的分支在 evidence=False 时会**松开方向**，于是整段 3.7 秒完全失控。
                # 机制文档："活着的时候蛇头一直在动，不会停滞超过 7 帧"。所以读数不可信时，
                # 保持上一帧方向比松开方向安全；只有在长时间（HEAD_JUMP_HOLD_FRAMES）都不可信时才放弃。
                _hold_dir = getattr(policy, '_sent_dir', None)
                if _hold_dir is not None and head_jump_streak < HEAD_JUMP_HOLD_FRAMES:
                    head_jump_streak += 1
                    if pad:
                        pad.apply_decision(tuple(_hold_dir), evidence=True,
                                           fixed_accel=(True if match_accel_mode == 'hold' else None),
                                           preserve_accel=match_accel_mode == 'hold')
                    dec = (tuple(_hold_dir), match_accel_mode == 'hold', False, 'head_jump_hold')
                else:
                    if pad:
                        pad.apply_decision(None, evidence=False,
                                           preserve_accel=match_accel_mode == 'hold')
                    # Keep A held in `hold` mode: the DIRECTION is withheld because the reading is
                    # unreliable, but releasing the boost for a brief vision gap is the mistake the
                    # user's rule forbids (and it was 39% of all boost-off frames, measured).
                    dec = (None, match_accel_mode == 'hold', False, 'head_jump')
            elif flags['whiteout'] or not st['player_alive']:
                if pad:
                    pad.apply_decision(None, evidence=False,
                                       preserve_accel=match_accel_mode == 'hold')
                dec = (None, False, False, 'dead_or_whiteout')
            else:
                dec = policy.decide(st, belief=st.get('belief'))
                dec = fixed_accel_decision(dec, match_accel_mode)
                decided_now = time.perf_counter()
                # The control delay, chosen by the snake's speed state, computed HERE so that every later
                # use - the per-frame log and the pacing block - sees it.  It was previously computed only in
                # the pacing block, and the log referenced it first, which raised UnboundLocalError on every
                # frame that logged without pacing (two whole matches ran with no steering).
                # The speed-item flag comes from what the POLICY publishes: the earlier version read
                # `st['item_accel_until']`, which the policy consumes and never writes, so that tier was dead
                # code (the user noticed: "吃到绿豆后反应不足").
                _speed_item = bool(getattr(policy, 'last', {}).get('speed_item_active'))
                if _speed_item and dec[1] is True:
                    _delay_ms = CONTROL_DELAY_MS_BOOST_GREEN
                elif dec[1] is True:
                    _delay_ms = CONTROL_DELAY_MS_BOOST
                else:
                    _delay_ms = CONTROL_DELAY_MS_NORMAL
                # The policy's `accel` is used as-is.  There used to be an override here:
                #     if match_accel_mode != 'adaptive':
                #         dec = (dec[0], match_accel_mode == 'hold', False, dec[3])
                # which replaced the decision with the CONSTANT True in `hold` mode - the mode run.py
                # uses - so every deliberate release the policy made was discarded before it was logged
                # or applied.  That is why the log could print `why=travel+starved+release accel=True`:
                # `+starved` and `+release` are only appended when the policy set accel=False.
                #
                # `hold` is now expressed as the policy's DEFAULT (ACCEL_DEFAULT_HOLD=True) rather than
                # as an override, which preserves the user's requirement - A is held while alive and is
                # not toggled for incidental reasons, with `preserve_accel` covering a brief loss of
                # visual evidence - while letting a substantive release (starvation, boxed in, a speed
                # item expiring) actually take effect.
                frame_age_s = control_time() - captured_at
                if frame_age_s > 2.0 / GAME_HZ:
                    # as above: withhold the direction, keep the boost
                    dec = (None, match_accel_mode == 'hold', False, 'stale_frame')
                    if pad:
                        pad.apply_decision(None, evidence=False,
                                           preserve_accel=match_accel_mode == 'hold')
                elif pad:
                    observed_dir = policy.last.get('heading')
                    if observed_dir is None:
                        observed_dir = st.get('belief').facing if st.get('belief') is not None else red.facing
                    if observed_dir not in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                        observed_dir = None
                    # `hold` used to pass the CONSTANT True here, which `apply_decision` writes
                    # straight to the pad (`if fixed_accel is not None: set_accel(fixed_accel)`) and
                    # therefore silently discarded every deliberate release the policy made - the
                    # starvation brake in particular, whose whole purpose was to stop paying for the
                    # boost early.  The policy's own decision is now what holds or releases A;
                    # `preserve_accel` still protects A from incidental releases (brief loss of visual
                    # evidence), which is the behaviour the user asked for.
                    pad.apply_decision(dec[0], dec[1], dec[2],
                                       observed_dir=observed_dir, now=control_time(),
                                       fixed_accel=(False if dec[1] is False
                                                    else (None if match_accel_mode == 'adaptive'
                                                          else match_accel_mode == 'hold')),
                                       preserve_accel=match_accel_mode == 'hold')
                    applied_at = time.perf_counter()

            frames += 1
            if log:
                log.write(json.dumps(dict(
                    f=frames, t=round(now - t_start, 3), phase=ph,
                    whiteout=flags["whiteout"], ms=round(flags["ms"], 2),
                    red_len=red.length if red else 0,
                    # Distinguishes "the mask is tiny" from "the mask is fine but
                    # the cell conversion is failing".  A live match read length 2
                    # for 62% of frames and that difference decides which half of
                    # the pipeline to fix.
                    red_px=red.n_px if red else 0,
                    red_cells=len(red.cells) if red else 0,
                    path_len=len(red.path) if red else 0,
                    orient=(getattr(red, "orientation", None) if red else None),
                    red_head=list(red.head_cell) if red and red.head_cell else None,
                    # Sub-cell head position.  `red_head` above is CELL-QUANTIZED, so at roughly one
                    # cell of travel per control frame it reads "1 cell" almost every frame and cannot
                    # distinguish 0.6 from 1.4 - which is exactly what a turn-lead / control-delay
                    # analysis needs.  This is the same position without quantisation.
                    head_px_f=([round(float(px), 2) for px in red.head_px]
                               if red is not None and red.head_px is not None else None),
                    # what the follower believes, for prediction-vs-observation comparison
                    pred_px=([round(float(v), 2) for v in policy.follower.previous_px]
                             if getattr(policy.follower, 'previous_px', None) is not None else None),
                    speed_px=round(float(policy.follower.speed_px), 2),
                    speed_px_s=round(float(policy.follower.speed_s), 2),
                    steering_speed_px_s=round(float(getattr(policy.follower,'last_steer_speed',0.)),2),
                    turn_lead_s=round(float(policy.follower.lead_s), 4),
                    turn_error_px=policy.follower.last_turn_error,
                    precision_turn_event=getattr(policy.follower,'turn_event',None),
                    fast_reaction_mode=bool(getattr(per,'high_speed_mode',False)),
                    parallel_diamonds=bool(flags.get('parallel_diamonds',False)),
                    fast_remaining_s=round(max(0.,getattr(policy.follower,'fast_deadline_wall',0.)-control_time()),3),
                    timing_mode=getattr(policy.follower,'mode',None),
                    timing_mode_reason=getattr(policy.follower,'mode_reason',None),
                    timing_length=int(getattr(policy.follower,'length',0)),
                    timing_length_slope=float(getattr(policy.follower,'length_slope',0.)),
                    growth_debt=policy._growth_debt,
                    future_turn_cell=policy._future_permission,
                    captured_at=round(captured_at - t_start, 4),
                    early_finish_ignored=phase.last_info.get('early_finish_ignored',False),
                    decided_at=round(control_time() - t_start, 4),
                    head_guard_ok=head_guard_ok,
                    red_eye=red.eye_state if red else None,
                    head_age=red.head_age if red else None,
                    ramp=round(red.ramp_span, 1) if red else 0,
                    grind=st.get("grind"),
                    key_held=st["key_held"],
                    dir=list(dec[0]) if dec[0] else None,
                    accel=dec[1], mash=dec[2], why=dec[3],
                    accel_mode=match_accel_mode,
                    requested_accel_mode=args.accel_mode,
                    endgame_release=bool(endgame_schedule.latched),
                    long_strategy_active=bool(policy.last.get("long_strategy_active")),
                    long_survival_priority=bool(policy.last.get("long_survival_priority")),
                    long_escape_hints=policy.last.get("long_escape_hints",0),
                    long_head_conflicts=policy.last.get("long_head_conflicts",0),
                    endgame_release_elapsed=endgame_schedule.released_elapsed,
                    entry_reason=phase.entry_reason,
                    match_time_known=phase.match_time_known,
                    timer_px=timer_px,
                    match_elapsed=(round(float(st['match_elapsed']), 2)
                                   if isinstance(st.get('match_elapsed'), (int, float)) else None),
                    input_allowed=phase.input_allowed,
                    arena_like=phase.last_info.get('arena_like'),
                    menu_kind=phase.last_info.get('menu_kind'),
                    short_player_motion=phase.last_info.get('short_player_motion'),
                    player_evidence_remaining_ms=phase.last_info.get('player_evidence_remaining_ms'),
                    execution=pad.last_execution if pad else None,
                    pad_accel_held=bool(pad is not None and pad._accel),
                    boost_rearm_active=bool(pad is not None and (pad._rearm_at is not None or pad._rearm_waiting)),
                    boost_rearm_events=pad.rearm_events if pad is not None else [],
                    frame_age_ms=round(frame_age_s * 1000, 2),
                    capture_threads=args.capture_threads,
                    capture_work_ms=(round(vision.capture_ms,2) if args.capture_threads==2
                                     else round((grabbed_at-cycle_at)*1000,2)),
                    capture_skipped=(vision.skipped if args.capture_threads==2 else 0),
                    diagnostic_dropped=(diagnostics.dropped if diagnostics else 0),
                    pad_creation_ms=round(pad_creation_s * 1000, 2),
                    runtime_input_allowed=bool(phase.input_allowed and not slow_pad_creation
                                               and frame_age_s <= 2.0 / GAME_HZ),
                    policy_heading=policy.last.get('heading'),
                    red_identity_evidence=per.last_evidence,
                    planning_hard_cells=policy.last.get('planning_hard_cells'),
                    legal_directions=policy.last.get('legal_directions'),
                    requested_target=policy.last.get('requested_target'),
                    opponent_memory_cells=policy.last.get('opponent_memory_cells'),
                    secured_length=policy.last.get('secured_length'),
                    policy_body_path=policy.last.get('policy_body_path'),
                    policy_body_source=policy.last.get('body_source'),
                    policy_path_trust=policy.last.get('path_trust'),
                    body_length_bound=getattr(policy,'_witnessed_body_bound',None),
                    body_trace_size=len(policy.body_history.trace),
                    planning_growth=getattr(policy,'_geometry_growth',None),
                    opponent_heads=policy.last.get('opponent_heads'),
                    measured_speed_px=round(policy.follower.speed_px,1),
                    waypoint=policy.last.get('waypoint'),
                    target=policy.last.get('target'),
                    path_trust=policy.last.get('path_trust'),
                    own_path=[[int(v) for v in c] for c in red.path] if red else [],
                    snake_cells={k:[[int(v) for v in c] for c in s.cells] for k,s in st['snakes'].items()},
                    planned_route=([[int(v) for v in c] for c in policy.last['route']]
                                   if policy.last.get('route') is not None else None),
                    runway=policy.last.get('runway'),
                    release_why=policy.last.get('release_why'),
                    delay_ms=_delay_ms,
                    speed_item_active=_speed_item,
                    # per-phase attribution of the cycle, in ms.  These are the only numbers that can
                    # explain the difference between the stage profile (56 ms perception) and the measured
                    # 94 ms cycle, so they are recorded per frame rather than sampled.
                    ph_cycle=round((loop_end - cycle_at) * 1000, 2),
                    ph_grab=round((grabbed_at - cycle_at) * 1000, 2),
                    ph_perceive=round((perceived_at - grabbed_at) * 1000, 2),
                    ph_decide=round((decided_now - perceived_at) * 1000, 2),
                    ph_apply=round((applied_at - decided_now) * 1000, 2),
                    ph_log=round((loop_end - applied_at) * 1000, 2),
                    pickup_guard=policy.last.get('pickup_guard'),
                    decision_risk=policy.last.get('risk'),
                    # the whiteout test's own inputs, so its margin is visible: it fires on
                    # (mean_sat < WHITEOUT_SAT 0.58) and (mean_val > WHITEOUT_VAL 0.70), and
                    # neither number used to be recorded anywhere.
                    red_flash_recovered_px=int(flags.get('red_flash_recovered_px',0)),
                    mean_sat=round(float(flags.get('mean_sat', 0.0)), 4),
                    mean_val=round(float(flags.get('mean_val', 0.0)), 4),
                    countdown_like=bool(flags.get('countdown_like')),
                    beans=len(items["bean"]), dias=len(items["dia"]),
                    # WHERE the gems were, so "a gem by the wall was never eaten" can be checked
                    # against the board record instead of argued about.
                    dia_cells=[[int(e[0]), int(e[1])] for e in items["dia"]],
                    gem_best=[[int(e[0]), int(e[1])]
                              for e in (st['attention'].gem_best
                                        if st.get('attention') is not None else ())],
                    gem_sources=[[int(e[0]),int(e[1]),str(e[4])]
                                 for e in (st['attention'].gem_best if st.get('attention') is not None else ())],
                    candidate_review=policy._candidate_review,
                    pickup_committed=bool(policy._committed_route),
                    # how far each gem is from the nearest wall, so 'are border gems being
                    # collected' can be answered from the record instead of argued about
                    gem_wall_dist=[min(int(e[0]), int(e[1]), COLS - 1 - int(e[0]),
                                       ROWS - 1 - int(e[1]))
                                   for e in (st['attention'].gem_best
                                             if st.get('attention') is not None else ())],
                    balls=len(items["ball"]),
                    ball_cells=[list(x[:2]) for x in items['ball']],
                    ball_tracks=(st['belief'].ball_tracks if st.get('belief') else []),
                    danger_cells=(st['belief'].danger_cells if st.get('belief') else []),
                )) + "\n")

            if frames % 60 == 0:
                print(f"  t={now-t_start:5.1f}s len={red.length if red else 0:3d} "
                      f"eye={red.eye_state if red else None} "
                      f"why={dec[3]} accel={dec[1]} mashing={dec[2]} "
                      f"[{flags['ms']:.1f} ms]")

            slack = 1.0 / LOOP_HZ + _delay_ms / 1000.0 - (control_time() - now)
            if slack > 0.002:
                time.sleep(slack)
            loop_end = time.perf_counter()
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        if pad:
            try:
                pad.reset()
            except Exception:
                pass
        if hasattr(vision,'close'):
            vision.close()
        if hasattr(per,'close'):
            per.close()
        if diagnostics is not None:
            diagnostics.close()
        if log:
            log.close()
        if frames:
            print(f"frames={frames} mean={ms_sum/frames:.1f} ms worst={worst:.1f} ms")
        print("done")


if __name__ == "__main__":
    main()
