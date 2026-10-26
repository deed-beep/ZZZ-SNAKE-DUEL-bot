"""Mode-local timing calibration; no target, collision or route selection changes."""
from collections import deque
from statistics import median
import time
from navigation_candidate_sweep import RouteFollower as BaselineFollower
from config import CELL


class PrecisionFollower(BaselineFollower):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lead_s = .170
        self.fast_active = False
        self.fast_until = 0.
        self.fast_deadline_wall = 0.
        self.mode = 'held'
        self.mode_changed_at = 0.
        self.mode_reason = 'initial'
        self.length = 10
        self.length_slope = 0.
        self._item_eyes = False
        self._item_sightings = deque(maxlen=3)
        self._banks = {}
        self._rates_by_length = {'held': deque(maxlen=240), 'stacked': deque(maxlen=240)}
        self.turn_event = None
        self.turn_sequence = 0
        self.native_timing = True
        self._straight_rates = deque(maxlen=5)
        self.last_steer_speed = 0.

    def set_context(self, now, length, item_until, item_eyes, native=True):
        now = float(now)
        # The tracker supplies actual pickup events. Eyes can rescue a missed
        # first pickup, but continuous V-eyes NEVER refresh a ten-second timer.
        self._item_sightings.append((now,bool(item_eyes and native)))
        corroborated=sum(valid and 0<=now-at<=.25 for at,valid in self._item_sightings)>=2
        if item_eyes and corroborated and self.fast_until <= now and not self._item_eyes:
            self.fast_until = now + 10.
            self.mode_reason = 'two_native_item_eyes'
            self._item_eyes = True
        if not item_eyes and not corroborated:
            self._item_eyes = False
        if float(item_until or 0.) > self.fast_until:
            self.fast_until = float(item_until)
            self.mode_reason = 'confirmed_pickup'
        active = self.fast_until > now
        mode = 'stacked' if active else 'held'
        if mode != self.mode:
            self._banks[self.mode] = (float(self.lead_s), self._turn_delays)
            self.lead_s, self._turn_delays = self._banks.get(mode, (.170, deque(maxlen=7)))
            self.mode = mode
            self.mode_changed_at = now
            self._native_rates.clear()
            self._straight_rates.clear()
            # A command already sent still needs its visual acknowledgement.
            # It is retained, but not used to teach the other speed mode.
        self.fast_active = bool(active)
        self.fast_deadline_wall = time.perf_counter()+max(0., self.fast_until-now)
        self.native_timing = bool(native)
        new_length = max(1, int(length))
        samples = self._rates_by_length[mode]
        # Fit ONLY within one speed mode, with enough actual length contrast.
        # No arbitrary multiplier for length 80: unsupported lengths remain
        # governed by current observed pixel speed.
        self.length_slope = 0.
        if len(samples) >= 24:
            ordered = sorted(samples)
            lo, hi = ordered[:len(ordered)//3], ordered[-len(ordered)//3:]
            short, long = median(x[0] for x in lo), median(x[0] for x in hi)
            if long-short >= 20:
                a, b = median(x[1] for x in lo), median(x[1] for x in hi)
                self.length_slope = max(0., min(.0015, (b/max(1., a)-1.)/(long-short)))
        if new_length != self.length and self.speed_s > 40:
            self.speed_s *= max(.96, min(1.04, 1.+self.length_slope*(new_length-self.length)))
        self.length = new_length

    def observe(self, px, t):
        if self.previous_px is not None and t > (self.previous_t or 0.):
            dx, dy = px[0]-self.previous_px[0], px[1]-self.previous_px[1]
            distance = abs(dx)+abs(dy)
            dt = t-self.previous_t
            # At 500 px/s, 100 ms legitimately exceeds the old 48 px cap.
            # The time-scaled ceiling still rejects endpoint swaps/teleports.
            credible = distance <= min(4*CELL, max(1.5*CELL, 700.*dt))
            if credible:
                w = .82 if distance < self.speed_px else .45
                self.speed_px = (1-w)*self.speed_px+w*distance
                if .025 <= dt <= .3 and distance >= 2:
                    rate = distance/dt
                    self._native_rates.append(rate)
                    w = .8 if rate < self.speed_s else .4
                    self.speed_s = (1-w)*self.speed_s+w*rate
                    if (self.native_timing and distance >= 6 and max(abs(dx),abs(dy)) >= .9*distance
                            and 100 <= rate <= 650 and self.turn_pending is None):
                        self._rates_by_length[self.mode].append((self.length, float(rate)))
                        self._straight_rates.append((float(t),float(rate),self.length))
                pending = self.turn_pending
                if pending is not None:
                    d = pending['dir']; axis = 0 if d[0] else 1; old_axis = 1-axis
                    moved = (dx, dy)
                    snapped = moved[old_axis]*pending['from'][old_axis] < -2
                    if moved[axis]*d[axis] > 4 and (abs(moved[axis]) > abs(moved[old_axis]) or snapped):
                        error = float((px[old_axis]-pending['corner'][old_axis])*pending['from'][old_axis])
                        self.last_turn_error = round(error, 2)
                        self.turn_sequence += 1
                        self.turn_event = dict(sequence=self.turn_sequence, at=float(t),
                            issued_at=float(pending['at']), mode=pending.get('mode', self.mode),
                            error_px=round(error, 2), speed=float(pending.get('issued_speed', self.speed_s)),
                            lead_s=float(pending.get('issued_lead', self.lead_s)),
                            remaining=float(pending.get('remaining', 0.)),
                            learn=bool(pending.get('learn', True)), native=bool(self.native_timing),
                            length=int(self.length))
                        if (pending.get('learn', True) and self.native_timing and
                                pending.get('mode', self.mode) == self.mode and
                                t-pending['at'] < .6 and abs(error) < 1.5*CELL and self.speed_s > 40):
                            speed = pending.get('issued_speed', self.speed_s)
                            remaining = pending.get('remaining', self.lead_s*speed)
                            delay = self.lead_s if abs(error) < .5*CELL else (remaining+error)/max(40., speed)
                            if .02 <= delay <= .40:
                                self._turn_delays.append(delay)
                            if len(self._turn_delays) >= 3:
                                self.lead_s = max(.07, min(.32, .5*self.lead_s+.5*median(self._turn_delays)))
                        self.turn_pending = None
        self.previous_px = tuple(px)
        self.previous_t = float(t)

    def command(self, *args, **kwargs):
        previous = self.turn_pending
        measured = self.speed_s
        fresh=[rate*max(.96,min(1.04,1.+self.length_slope*(self.length-length)))
               for at,rate,length in self._straight_rates if 0 <= (self.previous_t or 0.)-at <= .35]
        # A corner's sampled pause is not a real reduction of item boost.
        # Use only very recent SAME-mode straight measurements, with at most
        # 60 px/s compensation; buffer/expiry immediately bypass this floor.
        if self.fast_active and self.native_timing and len(fresh)>=2:
            self.speed_s=max(measured,min(median(fresh),measured+60.))
        self.last_steer_speed=float(self.speed_s)
        try:
            result = super().command(*args, **kwargs)
        finally:
            self.speed_s=measured
        if self.turn_pending is not None and self.turn_pending is not previous:
            self.turn_pending.update(mode=self.mode, issued_lead=float(self.lead_s))
        return result
