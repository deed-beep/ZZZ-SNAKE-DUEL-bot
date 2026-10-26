"""Pixel-level execution of a grid route (no game or pad dependencies).

Entering an item's tile is not collection. Keep the last waypoint until its
pixel centre is reached; brake for corners/the final approach, and never ask
for an instantaneous 180-degree turn. A route is revalidated by Policy every
frame; this controller may keep a command only if it remains legal.
"""
from __future__ import annotations
from config import CELL
from config import (TURN_PHASE_GATE, TURN_PHASE_BEST, TURN_PHASE_TOL,
                    TURN_PHASE_MAX_WAIT, HUG_ON_ESCAPE, GEM_MIN_DWELL)
from config import PICKUP_COMMIT_CELLS, PICKUP_COMMIT_CELLS_NO_BOOST, NO_BOOST
from config import (TURN_LEAD_SPEED_MULT_FAST, TURN_LEAD_SPEED_MULT_SLOW,
                    TURN_LEAD_MAX_CELLS_FAST, TURN_LEAD_MAX_CELLS_SLOW,
                    TURN_LEAD_FAST_SPEED_PX)

OPPOSITE = {(1,0):(-1,0),(-1,0):(1,0),(0,1):(0,-1),(0,-1):(0,1)}

class RouteFollower:
    ALIGN_TOL = 4.0  # px, deliberately less than half an item's radius
    PICKUP_LANE = .35 * CELL  # head must actually cross the item's hitbox

    def __init__(self, no_boost=None):
        # the commit radius depends on the speed mode; see NO_BOOST in config
        self.no_boost = bool(NO_BOOST if no_boost is None else no_boost)
        self.waypoint = None
        self.direction = None
        self.target = None
        self.previous_px = None
        self.speed_px = 0.0
        self.previous_t = None
        self.pending_dir = None    # a wanted turn, waiting for the best cell phase
        self.frames = 0
        self._turn_issued_at = -99
        self.speed_s = 0.0
        self.lead_s = .20
        self.turn_pending = None
        self.last_turn_error = None

    def observe(self, px, t):
        if self.previous_px is not None and t > (self.previous_t or 0):
            dx, dy = px[0]-self.previous_px[0], px[1]-self.previous_px[1]
            distance = abs(dx)+abs(dy)
            # A one-off end swap must not become an absurd speed estimate.
            if distance <= 1.5 * CELL:
                # Slowdowns must reduce the turn lead immediately.  A symmetric
                # average kept the previous boosted speed for several samples
                # and made the bot steer too early after a speed change.
                weight = .82 if distance < self.speed_px else .45
                self.speed_px = (1-weight)*self.speed_px + weight*distance
                dt = t-self.previous_t
                if .025 <= dt <= .3 and distance >= 2:
                    rate=distance/dt
                    w=.8 if rate < self.speed_s else .4
                    self.speed_s=(1-w)*self.speed_s+w*rate
                pending=self.turn_pending
                if pending is not None:
                    d=pending['dir']
                    turn_axis=0 if d[0] else 1
                    moved=(dx,dy)
                    if moved[turn_axis]*d[turn_axis]>4 and abs(moved[turn_axis])>abs(moved[1-turn_axis]):
                        old_axis=1-turn_axis
                        error=(px[old_axis]-pending['corner'][old_axis])*pending['from'][old_axis]
                        if t-pending['at']<.6 and abs(error)<2*CELL and self.speed_s>40:
                            self.lead_s=max(.07,min(.32,self.lead_s+.25*error/self.speed_s))
                            self.last_turn_error=round(float(error),2)
                        self.turn_pending=None
        self.previous_px = tuple(px)
        self.previous_t = t


    # ---------------------------------------------------------------- turn phase gate
    def _phase_ok(self, px, turn):
        """Is the head, ALONG THE AXIS THE TURN TAKES, at a phase where a turn takes effect?

        The user's mechanic: "按键转向实际上仅在蛇头处于未占满当前格的时候有效，0.5格的时候控制转向最佳".  A turn is
        therefore issued while the head's fractional position within its cell is inside
        TURN_PHASE_BEST +/- TURN_PHASE_TOL, and ALWAYS once it is past TURN_PHASE_MAX_WAIT so the gate can
        never make the snake miss its corner.

        Only the axis the turn moves ALONG matters.  The first version tested both axes with `any()`, so a
        head at x-phase 0.00 and y-phase 0.50 passed the gate on the wrong axis and turned early - and the
        probe fixture sat at y=400 (y-phase 0.50), which hid it.
        """
        if not TURN_PHASE_GATE or turn is None:
            return True
        axis = 0 if turn[0] else 1
        phase = ((px[0] / CELL) % 1.0) if axis == 0 else ((px[1] / CELL) % 1.0)
        lo, hi = TURN_PHASE_BEST - TURN_PHASE_TOL, TURN_PHASE_BEST + TURN_PHASE_TOL
        return lo <= phase <= hi or phase >= TURN_PHASE_MAX_WAIT

    def _gate(self, px, heading, d, legal):
        """Return the direction to actually send, deferring a turn that is too early in the cell.

        A turn is QUEUED (`pending_dir`) and the current heading is held, so the snake keeps moving and the
        turn is issued on a later control frame - the loop runs at about 19 Hz against a 15 Hz game, so a cell
        takes several frames to cross and there are several chances to hit the window.
        """
        if d is None or d == heading or not TURN_PHASE_GATE:
            return d
        if self._phase_ok(px, d):
            self.pending_dir = None
            return d
        self.pending_dir = d
        return heading if (heading in legal) else d

    def command(self, px, heading, route, target_key, target_px, legal, pickup=None):
        self.frames += 1
        """Return (direction, precision_brake, waypoint). Directions are prechecked.

        `legal` contains next-cell safe commands. RouteFollower never turns that
        geometric permission into a permission to reverse through its own neck.
        """
        legal = set(legal)
        if heading in OPPOSITE:
            legal.discard(OPPOSITE[heading])
        if not legal:
            return None, True, None
        if self.no_boost:
            return self._command_no_boost(px,heading,route,target_key,target_px,legal,pickup)
        tol = self.ALIGN_TOL
        # Under a held boost the game applies a new stick direction after the
        # captured head has travelled another 1-2 tiles.  Send an upcoming
        # corner early, while the currently measured next cell permits it.
        if heading in OPPOSITE and self.speed_px >= 20 and len(route) >= 3:
            first = (route[1][0]-route[0][0],route[1][1]-route[0][1])
            if first == heading:
                for j in range(2,min(len(route),6)):
                    upcoming = (route[j][0]-route[j-1][0],
                                route[j][1]-route[j-1][1])
                    if upcoming != heading:
                        corner = ((route[j-1][0]+.5)*CELL,
                                  (route[j-1][1]+.5)*CELL)
                        axis = 0 if heading[0] else 1
                        distance = (corner[axis]-px[axis])*heading[axis]
                        lateral = abs(corner[1-axis]-px[1-axis])
                        # How early to commit to the corner, as a multiple of the measured speed.
                        # Named in config so the next adjustment is an evidence-backed edit rather
                        # than a formula change; see the TURN_LEAD_* notes for why it is not zero.
                        lead = (min(TURN_LEAD_MAX_CELLS_FAST*CELL,
                                    TURN_LEAD_SPEED_MULT_FAST*self.speed_px)
                                if self.speed_px >= TURN_LEAD_FAST_SPEED_PX else
                                min(TURN_LEAD_MAX_CELLS_SLOW*CELL,
                                    TURN_LEAD_SPEED_MULT_SLOW*self.speed_px))
                        if (upcoming in legal and -.25*CELL <= distance <= lead
                                and lateral <= .7*CELL):
                            self.target = target_key
                            self.direction = upcoming
                            self.waypoint = corner
                            return upcoming, True, corner
                        break
        # Finish an in-progress edge even after floor(px/CELL) changes. Replanning
        # immediately on tile entry was the source of early corners and zigzags.
        if self.target == target_key and self.waypoint is not None and self.direction in legal:
            axis = 0 if self.direction[0] else 1
            remaining = (self.waypoint[axis]-px[axis])*self.direction[axis]
            lateral = abs(self.waypoint[1-axis]-px[1-axis])
            next_direction = (route[1][0]-route[0][0],route[1][1]-route[0][1]) if len(route)>1 else None
            corner = next_direction is not None and next_direction != self.direction
            # Vision and the next game sample each add latency.  Release an
            # intermediate waypoint slightly before its centre when a corner
            # follows.  The lead adapts to measured head movement per frame.
            lead = min(.55*CELL, max(tol, .55*self.speed_px)) if corner else tol
            if lead < remaining <= 1.25*CELL and lateral <= 0.7*CELL:
                close = abs(target_px[0]-px[0])+abs(target_px[1]-px[1]) <= .85*CELL
                aligned_pickup = (abs(target_px[1-axis]-px[1-axis]) <= tol
                                  and (target_px[axis]-px[axis])*self.direction[axis] > 0)
                close = close and not aligned_pickup
                return self.direction, close or (corner and remaining < .7*CELL), self.waypoint
        self.target = target_key
        # Last mile: aim at measured item centroid, not merely its tile. Stay on
        # the current axis until aligned, then turn, even when route distance=0.
        dx, dy = target_px[0]-px[0], target_px[1]-px[1]
        # Commit to the item's centre whenever it is close, NOT only when it is the last waypoint or the
        # one after.  The old gate was `len(route) <= 2 and max(abs(dx),abs(dy)) <= 1.6*CELL`, and the
        # route to a gem is normally long, so this branch - the one that drives THROUGH the centre - was
        # skipped exactly when it was needed, and the follower steered for the next cell instead.  That is
        # the "turns away one moment before eating" the user reported.
        # The commit radius follows the speed mode: a base-speed snake crosses a cell in about 2.7 frames
        # against 1.4 when boosted, so the window can be wider without overshooting (see NO_BOOST in config).
        _commit = (PICKUP_COMMIT_CELLS_NO_BOOST if self.no_boost else PICKUP_COMMIT_CELLS)
        if max(abs(dx),abs(dy)) <= _commit*CELL:
            desired = None
            if heading in legal:
                axis = 0 if heading[0] else 1
                delta = (dx,dy)[axis]
                lateral = abs((dx,dy)[1-axis])
                if delta*heading[axis] > tol and lateral <= self.PICKUP_LANE:
                    desired = heading
            if desired is None:
                # Prefer the axis the head is ALREADY on when it has a gap, because that needs no turn;
                # only then fall back to the larger gap.  Sorting purely by gap size made the follower close
                # the row gap first and turn down while still short on x, so the head crossed the gem's
                # column one row below its centre and missed it (probed: dx=12, dy=-20 -> (0,-1)).
                # ORIGINAL ORDER RESTORED.  Closing the smaller gap first was tried and REVERTED: it
                # inverted the documented intent of this branch, which is to align the LATERAL lane before
                # driving forward (`test_last_mile_aligns_lateral_lane_before_driving_forward`), and it
                # broke four tests.  The remaining explanation for a missed pickup is recorded in
                # PICKUP_COMMIT_CELLS and in LIVE_RUNS.md rather than fixed by reordering here.
                axes = sorted((0,1), key=lambda a: -abs((dx,dy)[a]))
                if heading in OPPOSITE:
                    travel_axis = 0 if heading[0] else 1
                    lateral_axis = 1-travel_axis
                    if abs((dx,dy)[lateral_axis]) > self.PICKUP_LANE:
                        axes = [lateral_axis, travel_axis]
                for axis in axes:
                    delta = (dx,dy)[axis]
                    if abs(delta) <= tol:
                        continue
                    d = ((1 if delta > 0 else -1),0) if axis == 0 else (0,(1 if delta > 0 else -1))
                    if d in legal:
                        desired = d
                        break
            if desired is None and max(abs(dx),abs(dy)) <= tol:
                # Drive THROUGH the object. Do not pivot away the instant its tile
                # is entered; disappearance/collection is verified upstream.
                desired = heading if heading in legal else next(iter(sorted(legal)))
            if desired is not None:
                if desired != heading:
                    self.pending_dir = desired
                self.direction = desired
                self.waypoint = tuple(target_px)
                turning = heading is not None and desired != heading
                close = max(abs(dx),abs(dy)) <= .85*CELL
                axis = 0 if desired[0] else 1
                aligned_pickup = abs((dx,dy)[1-axis]) <= tol and (dx,dy)[axis]*desired[axis] > 0
                # A straight crossing does not require re-pressing A. Slow only
                # for an actual lateral correction/turn, not every pickup.
                close = close and not aligned_pickup
                _d = self._gate(px, heading, desired, legal)
                return _d, (turning and _d == desired) or close, self.waypoint
        if len(route) < 2:
            return None, True, None
        cell = route[1]
        d = (cell[0]-route[0][0],cell[1]-route[0][1])
        if d not in legal:
            return None, True, None
        waypoint = (cell[0]*CELL+CELL/2, cell[1]*CELL+CELL/2)
        if len(route) == 2:
            waypoint = tuple(target_px)
        # every direction change passes the phase gate, including this route branch: it returns directly,
        # so before this it bypassed the gate entirely (probed: a turn was issued at phase 0.81)
        d = self._gate(px, heading, d, legal)
        self.direction, self.waypoint = d, waypoint
        # Corners need time to be sampled. Open straights still hold acceleration.
        turning = heading is not None and d != heading
        close = abs(dx)+abs(dy) <= .85*CELL
        return d, turning or close, waypoint

    def _command_no_boost(self, px, heading, route, target_key, target_px, legal,pickup=None):
        """Follow checked route corners with time-based actuation prediction.

        Nearby items cannot replace a route with an unchecked diagonal shortcut.
        A queued perpendicular command stays stable until motion acknowledges it.
        """
        pending=self.turn_pending
        now=self.previous_t or 0.
        if pending is not None:
            if (self.target==target_key and pending['dir'] in legal and
                    heading!=pending['dir'] and now-pending['at']<=.45):
                self.direction=pending['dir']
                self.waypoint=pending['corner']
                return self.direction,True,self.waypoint
            self.turn_pending=None
        self.target=target_key
        if not route:
            return None,True,None
        if len(route)==1:
            if heading in legal:
                axis=0 if heading[0] else 1
                forward=(target_px[axis]-px[axis])*heading[axis]
                if forward>=-self.ALIGN_TOL and abs(target_px[1-axis]-px[1-axis])<=self.PICKUP_LANE:
                    self.direction,self.waypoint=heading,tuple(target_px)
                    return heading,False,self.waypoint
            return None,True,None
        directions=[(b[0]-a[0],b[1]-a[1]) for a,b in zip(route,route[1:])]
        first=directions[0]
        if first not in legal:
            return None,True,None
        # If the route turns immediately, the corner is the current tile's
        # centre. Otherwise preview the first bend while still on its straight.
        turn_index=0 if first!=heading else next((i for i,d in enumerate(directions) if d!=heading),None)
        if turn_index is None:
            self.direction,self.waypoint=first,tuple(target_px)
            return first,False,self.waypoint
        desired=directions[turn_index]
        cell=route[turn_index]
        corner=[(cell[0]+.5)*CELL,(cell[1]+.5)*CELL]
        # On the final straight, aim along the measured pickup lane.
        if all(d==desired for d in directions[turn_index:]):
            turn_axis=0 if desired[0] else 1
            corner[1-turn_axis]=target_px[1-turn_axis]
        # The pickup can be an INTERMEDIATE corner now: queue the exit turn
        # before reaching it, while still crossing its measured centre.
        if pickup is not None:
            pickup_cell,pickup_px=pickup
            if tuple(cell)==tuple(pickup_cell):
                corner=list(pickup_px)
            elif tuple(route[-1])==tuple(pickup_cell) and all(d==desired for d in directions[turn_index:]):
                turn_axis=0 if desired[0] else 1
                corner[1-turn_axis]=pickup_px[1-turn_axis]
        corner=tuple(corner)
        if heading in OPPOSITE:
            axis=0 if heading[0] else 1
            remaining=(corner[axis]-px[axis])*heading[axis]
            lead=min(3*CELL,max(self.ALIGN_TOL,self.speed_s*self.lead_s))
            if remaining>lead and heading in legal:
                self.direction,self.waypoint=heading,corner
                return heading,False,corner
        if desired not in legal:
            return (heading,False,corner) if heading in legal else (None,True,None)
        self.direction,self.waypoint=desired,corner
        if heading in OPPOSITE and desired!=heading:
            self.turn_pending=dict(dir=desired,corner=corner,at=now,from_=heading)
            self.turn_pending['from']=heading
        return desired,True,corner
