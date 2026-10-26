"""Match-wide acceleration schedule, independent of life/reset and turn timing."""


class EndgameSchedule:
    def __init__(self, requested='hold', release_at=45.):
        self.requested = requested
        self.release_at = float(release_at)
        self.latched = False
        self.released_elapsed = None

    def mode(self, now, match_started):
        if match_started is not None and now-match_started >= self.release_at:
            if not self.latched:
                self.released_elapsed = float(now-match_started)
            self.latched = True
        return 'off' if self.latched else self.requested

    def enforce(self, pad):
        if not self.latched or pad is None:
            return
        # Cancel a previously scheduled respawn press BEFORE service() runs.
        # Steering/pending bends are deliberately untouched.
        pad.stop_accel_schedule()

