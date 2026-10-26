"""Input: virtual Xbox pad (vgamepad/ViGEmBus) and physical pad reader (XInput).

Both the left stick and the D-pad steer the snake, and button A is the
accelerate key.  The bot drives the stick because that is what a human uses, but
`USE_DPAD` switches to the D-pad if the game turns out to prefer buttons.

XInput is read straight from xinput1_4.dll through ctypes: no extra dependency,
and it gives the raw state as the game itself sees it, which is what the
recording script needs to log a human's play.
"""
from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

from config import GAME_HZ, MASH_MIN_INTERVAL, DIR_PULSE_MS, DIR_HOLD

USE_DPAD = False                 # False = left stick, True = D-pad

# direction vectors in screen space (y grows downward)
UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)


# ============================================================ virtual pad
class Pad:
    """Writes to a virtual X360 pad."""

    def __init__(self, backend=None, vg=None, clock=None):
        # Dependency injection keeps offline tests completely away from ViGEm.
        if vg is None:
            import vgamepad as vg
        self._vg = vg
        self.pad = backend if backend is not None else vg.VX360Gamepad()
        self._clock = clock or time.perf_counter
        self._dir = None
        self._last_command = None
        self._accel = False
        self._rearm_at = None
        self._rearm_waiting = False
        self._respawn_motion = []
        self.rearm_events = []
        self._a_released_at = float('-inf')
        self._mash_last = float('-inf')
        # A 30 ms pulse can fall entirely between two 15 Hz samples.
        self._pulse_s = max(DIR_PULSE_MS / 1000.0, 1.25 / GAME_HZ)
        self._release_at = None
        self._button_release_at = None
        self._tap_button = None
        self._turn_pending = None
        self._turn_boost_after = 0.0
        self.last_execution = {}

    def reset(self):
        self.pad.reset()
        self.pad.update()
        self._dir = None
        self._last_command = None
        self._accel = False
        self._rearm_at = None
        self._rearm_waiting = False
        self._respawn_motion = []
        self._a_released_at = self._clock()
        self._release_at = None
        self._button_release_at = None
        self._tap_button = None
        self._turn_pending = None
        self._turn_boost_after = 0.0

    def _push(self, d):
        if USE_DPAD:
            B = self._vg.XUSB_BUTTON
            table = {(1, 0): B.XUSB_GAMEPAD_DPAD_RIGHT, (-1, 0): B.XUSB_GAMEPAD_DPAD_LEFT,
                     (0, 1): B.XUSB_GAMEPAD_DPAD_DOWN, (0, -1): B.XUSB_GAMEPAD_DPAD_UP}
            for btn in table.values():
                self.pad.release_button(button=btn)
            self.pad.press_button(button=table[d])
        else:
            self.pad.left_joystick_float(x_value_float=float(d[0]), y_value_float=float(-d[1]))
        self.pad.update()

    def _release(self):
        if USE_DPAD:
            B = self._vg.XUSB_BUTTON
            for btn in (B.XUSB_GAMEPAD_DPAD_RIGHT, B.XUSB_GAMEPAD_DPAD_LEFT,
                        B.XUSB_GAMEPAD_DPAD_DOWN, B.XUSB_GAMEPAD_DPAD_UP):
                self.pad.release_button(button=btn)
        else:
            self.pad.left_joystick_float(x_value_float=0.0, y_value_float=0.0)
        self.pad.update()

    def hold_dir(self, d):
        """Push the stick (or D-pad) and LEAVE it pushed.  Idempotent."""
        if d is None:
            return
        d = (int(d[0]), int(d[1]))
        if d == self._dir:
            return
        self._push(d)
        self._dir = d

    def release_dir(self):
        """Recentre.  The snake keeps travelling the way it was going."""
        if self._dir is None:
            return
        self._release()
        self._dir = None
        self._release_at = None

    def service(self, now=None):
        """Finish scheduled pulses; call once per control frame (no sleeps)."""
        now = self._clock() if now is None else now
        if self._rearm_at is not None and now >= self._rearm_at:
            self._rearm_at = None
            self.set_accel(True)
            self.rearm_events.append(dict(kind='held',at=now))
        if self._button_release_at is not None and now >= self._button_release_at:
            self.pad.release_button(button=self._tap_button)
            self.pad.update()
            if self._tap_button == self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A:
                self._accel = False
                self._a_released_at = now
            self._tap_button = None
            self._button_release_at = None
        if self._release_at is not None and now >= self._release_at:
            self.release_dir()

    @staticmethod
    def _direction(d):
        if d is None:
            return None
        d = tuple(d)
        if d not in (UP, DOWN, LEFT, RIGHT):
            raise ValueError('direction must be one cardinal unit vector')
        return d

    def set_dir(self, d):
        """Non-blocking sampled pulse, or persistent hold if DIR_HOLD is set."""
        now = self._clock()
        self.service(now)
        d = self._direction(d)
        if d is None:
            self.release_dir()
            return
        if d != self._dir:
            self.hold_dir(d)
            self._release_at = None if DIR_HOLD else now + self._pulse_s
        self._last_command = d

    def apply_decision(self, d, accel=False, mash=False, *, observed_dir=None,
                       turn_ready=True, evidence=True, now=None,
                       fixed_accel=None, preserve_accel=False):
        """Execute a decision without blocking the perception loop.

        Adaptive acceleration releases A for turns.  fixed_accel=True/False
        instead keeps A in one state while the match is active.  A brief loss
        of visual evidence may release the stick without changing that state.
        turn_ready=False defers a planned turn until the pixel head reaches its
        waypoint; callers must provide that spatial condition.
        """
        now = self._clock() if now is None else now
        self.service(now)
        if not evidence:
            if preserve_accel:
                self.release_dir()
                self._last_command = None
                self._turn_pending = None
            else:
                self.reset()
            self.last_execution = dict(dir=None, accel=self._accel, pending=None,
                                       reason='no_game_evidence')
            return self.last_execution
        d = self._direction(d)
        observed_dir = self._direction(observed_dir)
        if d is None:
            if preserve_accel:
                self.release_dir()
                self._last_command = None
                self._turn_pending = None
            else:
                self.reset()
            self.last_execution = dict(dir=None, accel=self._accel, pending=None,
                                       reason='no_direction')
            return self.last_execution
        if fixed_accel is not None:
            mash = False
            self.set_accel(fixed_accel)
        if self._turn_pending is not None and observed_dir == self._turn_pending:
            self._turn_pending = None
        locked = (self._release_at is not None and now < self._release_at)
        if locked and self._dir is not None and d != self._dir:
            d = self._dir
        turning = d != (observed_dir or self._last_command)
        if turning and not turn_ready:
            if fixed_accel is None:
                self.set_accel(False)
            self.last_execution = dict(dir=self._last_command, accel=self._accel,
                                       pending=d, reason='await_waypoint')
            return self.last_execution
        if turning:
            if self._turn_pending != d:
                self._turn_boost_after = now + self._pulse_s
            self._turn_pending = d
        boost = (bool(fixed_accel) if fixed_accel is not None else
                 bool(accel and not turning and self._turn_pending is None
                      and now >= self._turn_boost_after and not mash))
        # Update A first, even when it was held on the previous frame.
        if not (mash and self._tap_button == self._vg.XUSB_BUTTON.XUSB_GAMEPAD_A):
            self.set_accel(boost)
        if d != self._last_command or (self._turn_pending is not None and self._dir is None):
            self.hold_dir(d)
            self._release_at = None if DIR_HOLD else now + self._pulse_s
        self._last_command = d
        if mash:
            self.mash_accel(now=now)
        self.last_execution = dict(dir=d, accel=self._accel,
                                   pending=self._turn_pending,
                                   reason='turn_ack' if self._turn_pending else 'applied')
        return self.last_execution

    def set_accel(self, on):
        """Button A.  Holding A gives speed AND leaks a bean from the tail about
        every 5.7 s (measured: +0.375 beans/s with A held vs +0.200 without)."""
        on = bool(on)
        if on and self._rearm_waiting:
            return
        if on and self._rearm_at is not None and self._clock() < self._rearm_at:
            return
        if on == self._accel:
            return
        B = self._vg.XUSB_BUTTON
        if on:
            self.pad.press_button(button=B.XUSB_GAMEPAD_A)
        else:
            self.pad.release_button(button=B.XUSB_GAMEPAD_A)
        self.pad.update()
        self._accel = on
        if not on:
            self._a_released_at = self._clock()

    def stop_accel_schedule(self):
        self._rearm_at = None
        self._rearm_waiting = False
        self._respawn_motion = []
        self.set_accel(False)

    def rearm_accel(self,wait_for_motion=False):
        """Wait for an active new snake, then release ten frames before holding."""
        self.set_accel(False)
        self._rearm_waiting=wait_for_motion
        self._respawn_motion=[]
        self._rearm_at=None if wait_for_motion else self._clock()+max(.70,10./GAME_HZ)
        self.rearm_events.append(dict(kind='wait_motion' if wait_for_motion else 'release',at=self._clock()))

    def confirm_respawn_motion(self,red,allowed,now):
        """Fresh slit eyes and consecutive observed movement end the spawn wait."""
        if not self._rearm_waiting:return False
        if (not allowed or red is None or not red.alive or red.head_px is None
                or red.eye_state not in ('normal','item') or red.head_age!=0):
            self._respawn_motion=[]
            return False
        point=tuple(red.head_px)
        if self._respawn_motion:
            stamp,previous=self._respawn_motion[-1]
            distance=sum(abs(point[i]-previous[i]) for i in (0,1))
            if now-stamp>.35 or distance>64:self._respawn_motion=[]
            elif distance<3:return False
        self._respawn_motion.append((now,point))
        self._respawn_motion=self._respawn_motion[-3:]
        if len(self._respawn_motion)>=3 and now-self._respawn_motion[0][0]>=.12:
            self._rearm_waiting=False
            self._rearm_at=self._clock()+max(.70,10./GAME_HZ)
            self.rearm_events.append(dict(kind='motion_confirmed_release10',at=self._clock(),
                                          press_due=self._rearm_at))
            return True
        return False

    def tap(self, name, hold_ms=110):
        """Press and release one named button.

        Needed to drive the game's own menus (A leaves the result screen, X starts
        matchmaking) so a run does not depend on a human being there.  Named rather
        than raw so the call sites read as what they do.
        """
        B = self._vg.XUSB_BUTTON
        table = {"a": B.XUSB_GAMEPAD_A, "b": B.XUSB_GAMEPAD_B, "x": B.XUSB_GAMEPAD_X,
                 "y": B.XUSB_GAMEPAD_Y, "start": B.XUSB_GAMEPAD_START,
                 "back": B.XUSB_GAMEPAD_BACK,
                 "lb": B.XUSB_GAMEPAD_LEFT_SHOULDER,
                 "rb": B.XUSB_GAMEPAD_RIGHT_SHOULDER}
        btn = table.get(name.lower())
        if btn is None:
            return False
        now = self._clock()
        self.service(now)
        if self._button_release_at is not None:
            return False
        self.pad.press_button(button=btn)
        self.pad.update()
        self._tap_button = btn
        self._button_release_at = now + max(hold_ms / 1000.0, self._pulse_s)
        if name.lower() == 'a':
            self._accel = True
        return True

    def mash_accel(self, force=False, now=None):
        """Schedule one sampled A press and a full sampled release interval."""
        now = self._clock() if now is None else now
        self.service(now)
        interval = max(MASH_MIN_INTERVAL, self._pulse_s + 1.0 / GAME_HZ)
        if self._button_release_at is not None:
            return False
        if not force and now - self._mash_last < interval:
            return False
        if now - self._a_released_at < 1.0 / GAME_HZ:
            return False
        # A genuinely new edge is required; holding A is not a mash.
        if self._accel:
            self.set_accel(False)
            self._mash_last = now
            return False
        B = self._vg.XUSB_BUTTON
        self.pad.press_button(button=B.XUSB_GAMEPAD_A)
        self.pad.update()
        self._accel = True
        self._tap_button = B.XUSB_GAMEPAD_A
        self._button_release_at = now + self._pulse_s
        self._mash_last = now
        return True


# ========================================================== physical pad
class XInputState(ctypes.Structure):
    _fields_ = [("dwPacketNumber", wintypes.DWORD),
                ("wButtons", wintypes.WORD),
                ("bLeftTrigger", ctypes.c_ubyte),
                ("bRightTrigger", ctypes.c_ubyte),
                ("sThumbLX", ctypes.c_short),
                ("sThumbLY", ctypes.c_short),
                ("sThumbRX", ctypes.c_short),
                ("sThumbRY", ctypes.c_short)]


_BTN = {"dpad_up": 0x0001, "dpad_down": 0x0002, "dpad_left": 0x0004, "dpad_right": 0x0008,
        "start": 0x0010, "back": 0x0020, "left_thumb": 0x0040, "right_thumb": 0x0080,
        "left_shoulder": 0x0100, "right_shoulder": 0x0200,
        "a": 0x1000, "b": 0x2000, "x": 0x4000, "y": 0x8000}


class PhysicalPad:
    """Reads the human's real controller through XInput."""

    def __init__(self, index=0):
        self.index = index
        self.dll = None
        for name in ("xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"):
            try:
                self.dll = ctypes.windll.LoadLibrary(name)
                break
            except OSError:
                continue
        if self.dll is None:
            raise RuntimeError("no XInput DLL available")

    def read(self):
        st = XInputState()
        rc = self.dll.XInputGetState(ctypes.c_uint(self.index), ctypes.byref(st))
        if rc != 0:
            return None
        btns = {k: bool(st.wButtons & v) for k, v in _BTN.items()}
        lx = st.sThumbLX / 32767.0
        ly = st.sThumbLY / 32767.0
        dead = 0.25
        if abs(lx) < dead:
            lx = 0.0
        if abs(ly) < dead:
            ly = 0.0
        d = None
        if abs(lx) > abs(ly):
            d = (1, 0) if lx > 0 else (-1, 0)
        elif ly != 0:
            # XInput Y is positive upward; screen y grows downward
            d = (0, -1) if ly > 0 else (0, 1)
        return dict(buttons=btns, stick=(round(lx, 3), round(ly, 3)),
                    dir=d, accel=btns["a"],
                    triggers=(st.bLeftTrigger, st.bRightTrigger),
                    right_stick=(round(st.sThumbRX / 32767.0, 3),
                                 round(st.sThumbRY / 32767.0, 3)))


if __name__ == "__main__":
    p = PhysicalPad()
    print("reading physical pad for 5 s (Ctrl+C to stop) ...")
    t0 = time.time()
    while time.time() - t0 < 5:
        s = p.read()
        if s is None:
            print("  no controller on index 0"); time.sleep(0.5); continue
        held = [k for k, v in s["buttons"].items() if v]
        print(f"  dir={s['dir']} stick={s['stick']} accel={s['accel']} held={held}")
        time.sleep(0.2)
