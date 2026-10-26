"""Length-seeking control with checked body physics and pixel-level collection.

Survival constraints precede item utility: a death destroys the length objective.
The grid planner chooses a corridor; RouteFollower executes it without treating
'entered the diamond tile' as 'collected the diamond'. All measurements are from
vision, not internal game memory. Open-loop recordings cannot prove a win rate.
"""
from __future__ import annotations
from collections import deque
from copy import copy
from heapq import heappop, heappush
import math
from causal_planner import RoutePlans, tail_witness, advance, packing_escape
from body_history import OwnBodyHistory
from config import (COLS, ROWS, CELL, BALL_CELLS, ITEM_VALUE, ITEM_FIXED_COST,
                    MIN_ESCAPE_AREA, ESCAPE_AREA_CAP, LOW_VALUE_MAX_DETOUR,
                    TARGET_STICKY, CARRIED_MAX_AGE, GEM_BONUS_WINDOW, GEM_BONUS_PAIR,
                    ONLY_WINDOW_LOW_VALUE, WALL_MARGIN, ACCEL_DEFAULT_HOLD,
                    SHED_TRAPPED_FRAMES, STRATEGY, STARVED_LEN, STARVATION_BRAKE, GEM_WALL_APPROACH_CELLS, ALIGN_BRAKE_CELLS, RELEASE_ON_LOW_TRUST, ACCEL_MIN_RUNWAY, ACCEL_MAX_RISK, HOLD_BOOST_EXCEPT_SAFETY, ACCEL_WALL_RESERVE, ENDGAME_WINDOW, ENDGAME_MIN_CLEARANCE, ENDGAME_BALL_WEIGHT, MATCH_SECONDS, ENDGAME_ARM_MIN, MATCH_SCORE_SECONDS, EYE_BUFFER_CONFIRM, BUFFER_MAX_FRAMES, PICKUP_LANE_GUARD_CELLS, NO_BOOST, NECK_EXEMPT_CELLS, WALL_GEM_GREED, WALL_GEM_GREED_CELLS, WALL_GEM_GREED_RADIUS, TAIL_CHASE, TAIL_CHASE_MIN_LEN, TAIL_CHASE_AREA, OPP_TAIL_CHASE, OPP_HEAD_BLOCK, OPP_TAIL_MOVING_FRAMES, OPP_BLOCK_AHEAD, OPP_BLOCK_MAX_DIST, HUG_WALL)
from navigation import RouteFollower

OPPOSITE = {(1,0):(-1,0),(-1,0):(1,0),(0,1):(0,-1),(0,-1):(0,1)}
NEIGH = ((1,0),(-1,0),(0,1),(0,-1))
INF = 10**6
EDGE_TURN_RESERVE = 3            # cells needed before a 15 Hz turn is visible

def in_board(c):
    return 0 <= c[0] < COLS and 0 <= c[1] < ROWS

class Policy:
    def __init__(self, no_boost=None):
        # NO_BOOST is a MODE, not a global: the boosted path still exists and is tested, so the
        # caller chooses.  Defaults to the configured value.
        self.no_boost = bool(NO_BOOST if no_boost is None else no_boost)
        self.mode = 'length'
        self.head_hist = deque(maxlen=6)
        self._boxed_frames = 0
        self._target = None
        self.last = {}
        self.follower = RouteFollower(no_boost=self.no_boost)
        self.body_history = OwnBodyHistory()
        self._heading = None
        self._last_observation = None
        self._risk = {}
        self._first_allowed = None
        self._escape_until_frame = 0
        self._pickup_clear_until_frame = 0
        self._skip_target_until = {}
        self._opp_tail_hist = {}
        self._recovery_dir = None
        self._recovery_dir_since = 0
        self._ball_motion = []
        self._opponent_motion = {}
        self.plans = RoutePlans()
        self._tail_goal = None
        self._roam_goal = None
        self._growth_debt = 0
        self._growth_point = None
        self._growth_distance = 0.
        self._future_permission = None
        self._committed_route = None
        self._candidate_review = []

    def decide(self, st, belief=None):
        red = st['snakes'].get('red')
        if red is None or red.head_cell is None or red.head_px is None:
            self.last = {'why':'no_head','target':None}
            return None, False, False, 'no_head'
        if belief is not None:
            st['belief'] = belief
        ds = self._decision_state(st, belief or st.get('belief'))
        head, length = ds['head'], ds['length']
        # Local coherent geometry only; never rewrite another consumer's vision.
        red = copy(red)
        red.head_cell,red.head_px,red.facing = head,ds['head_px'],ds['facing']
        red.path,red.path_trust = list(ds['path']),ds['path_trust']
        self._pickup_guard = None
        self._future_permission = None
        red.length = length
        frame = int(st.get('frame', 0))
        # ---- crash buffer, with confirmation.  A single `buffer` reading used to arm an 18-frame escape and
        # release the boost on every one of those frames; measured, most buffer episodes are ONE frame while
        # the real buffer is a stopped snake, so those were misreads costing the boost for up to 17 frames at
        # a time (traced: 101 boost-off frames after one respawn).  Now the state must repeat, and the escape
        # lasts the buffer's own measured lifetime.
        if red.eye_state == 'buffer':
            self._buffer_streak = getattr(self, '_buffer_streak', 0) + 1
            if self._buffer_streak >= (1 if self.no_boost else EYE_BUFFER_CONFIRM):
                self._escape_until_frame = max(self._escape_until_frame, frame + BUFFER_MAX_FRAMES)
        else:
            self._buffer_streak = 0
        recovering = frame < self._escape_until_frame
        endgame = (st.get('match_elapsed') is not None
                   and st['match_elapsed'] >= 50.0)
        # The FINAL window is a stricter mode than `endgame`: see ENDGAME_WINDOW in config.  Measured, the
        # snake was grinding in and out of the crash buffer for ~2.9 s before dying, with `endgame` active
        # the whole time, so stopping item chasing is not enough on its own.
        _el = st.get('match_elapsed')
        # Armed on the EARLIER of the computed window and a floor, because `match_elapsed` is derived from
        # when the START overlay was recognised and that lagged by seconds, leaving the endgame essentially
        # unarmed (see ENDGAME_ARM_MIN).  Being cautious early costs a few beans; being late costs the score.
        # Timed against the SCORE moment (60 s), not the end of the match: see MATCH_SCORE_SECONDS.  The
        # strict window covers the stretch after the score is taken, when a death is no longer paid for and
        # nothing can be gained.
        final_seconds = _el is not None and (_el >= (MATCH_SCORE_SECONDS - ENDGAME_WINDOW)
                                             or _el >= ENDGAME_ARM_MIN)
        if self.no_boost:
            # Collect through the scoring moment.  Old boosted endgame rules
            # stopped useful collection ten seconds early.
            endgame = final_seconds = _el is not None and _el >= MATCH_SCORE_SECONDS
        pending_growth=max(0,min(5,int(getattr(st.get('belief'),'predicted_growth',0))))
        if self.no_boost:
            if self._growth_point is not None:
                distance=sum(abs(ds['head_px'][i]-self._growth_point[i]) for i in (0,1))
                if distance<=2*CELL:
                    self._growth_distance+=distance
                    consumed=int(self._growth_distance//CELL)
                    self._growth_debt=max(0,self._growth_debt-consumed)
                    self._growth_distance-=consumed*CELL
            self._growth_point=ds['head_px']
            if pending_growth:
                self._growth_debt=max(self._growth_debt,pending_growth)
                self._growth_distance=0.
            pending_growth=self._growth_debt
        own = self.own_index(red, ds['path'], ds['path_trust'], growth=pending_growth)
        hard = self.hard_blocked(st)
        # the balls, for the final-seconds ranking below
        items_ball = st.get('items', {}).get('ball') if isinstance(st.get('items'), dict) else None
        heading = self._observed_heading(ds, red, st)
        self._heading = heading
        self._risk = self._risk_map(st, hard)
        self.follower.observe(ds['head_px'], st.get('t',0.0))
        need = self.escape_need(length)
        steps = []
        for d in NEIGH:
            if heading and d == OPPOSITE.get(heading):
                continue  # never reverse, even if a fragmented mask forgot the neck
            nxt = (head[0]+d[0],head[1]+d[1])
            if not self._passable(nxt, hard, own, 1):
                continue
            area, _ = self.flood(nxt, 1, hard, own)
            runway = self._runway(head, d, hard, own)
            steps.append(dict(d=d, cell=nxt, area=area, runway=runway,
                              clearance=self._clearance(nxt,hard), gap=self._gap(nxt,hard),
                              body_gap=self._body_gap(nxt,own,red.path),
                              risk=self._risk.get(nxt,0.0), straight=int(d==heading),
                              wall=int(self._near_wall(nxt))))
        if self.no_boost and heading and self.follower.speed_s>0:
            # Predict travel during capture/actuation latency. This only
            # rejects motion INTO an occupied cell; adjacent lanes stay legal.
            horizon=self.follower.speed_s*self.follower.lead_s
            moving_safe=[]
            for step in steps:
                if step['d']==heading:
                    collision=False
                    for k in range(1,int(horizon/4)+1):
                        p=(ds['head_px'][0]+heading[0]*k*4,ds['head_px'][1]+heading[1]*k*4)
                        c=(int(p[0]//CELL),int(p[1]//CELL))
                        if c!=head and not self._passable(c,hard,own,max(1,k*4//CELL)):
                            collision=True
                            break
                    if collision:
                        continue
                moving_safe.append(step)
            if moving_safe:
                steps=moving_safe
        # In the live run the first loss came after repeatedly steering from
        # row 22 into rows 23/24 for edge diamonds.  A turn pulse may take one
        # game tick to land, so leave two cells to brake and turn.  While already
        # in that margin, move inward as soon as a legal direction exists.
        att=st.get('attention')
        # Most gems spawn near the border.  Keep the usual turn reserve for
        # ordinary travel, but allow a measured nearby gem one extra lane so
        # the planner can collect it instead of cruising past it forever.
        # Unlock the wall approach for any non-extra gem in `gem_best` within GEM_WALL_APPROACH_CELLS.
        # The old test additionally required wall distance 1-2 and distance <= 10, which deadlocked: the
        # step filter below refuses to move toward the wall until this is true, and it could not become
        # true until the snake was already closer.  `gem_best` is the policy's own authoritative diamond
        # list (provenance-filtered), and it is available before target selection.
        #
        # Note: an earlier attempt read a `hot_cells` field off the attention frame.  No such field
        # exists, and the `getattr(..., ())` default silently made the condition False - the fix would
        # have shipped doing nothing.  Checked against AttentionFrame's fields.
        # Relax the wall reserve to 1 whenever a reachable gem exists.  The previous test also required
        # `manhattan(head, gem) <= GEM_WALL_APPROACH_CELLS`, and since `_wall_distance` is the distance to the
        # NEAREST wall, a gem at wall distance 1 could only satisfy an 18-cell radius if the head were within
        # 13.2 px of it - under half a cell.  Until then the 3-cell reserve stayed and EVERY step toward such a
        # gem was filtered out, so gems beside a wall were never approached at all.  The user reports exactly
        # that ("几乎完全避开了刷新在墙旁的钻石和道具"), and gems "tend to spawn near the border" per the note
        # above, so the filter was hitting the most common placement.
        #
        # The reserve stays at 1 rather than 0, which still keeps the snake one cell off the wall.
        edge_gem = bool(att and any(
            (e[4] if len(e) > 4 else '') != 'extra'
            and 0 <= int(e[0]) < COLS and 0 <= int(e[1]) < ROWS
            for e in att.gem_best))
        # The user asked for a deliberate risk on border gems ("墙边的钻石增加冒险去吃的策略").  When a gem is
        # both CLOSE to the head and NEAR the wall, the approach is allowed to give up the one-cell margin
        # entirely (reserve 0).  `_passable` still forbids wall cells, opponents and balls, so this is a
        # tighter line, not a blind one.
        _wall_greed = False
        if WALL_GEM_GREED and att is not None and head is not None:
            try:
                for e in (att.gem_best or ()):
                    if (e[4] if len(e) > 4 else '') == 'extra':
                        continue
                    gc = (int(e[0]), int(e[1]))
                    if self._wall_distance(gc) <= WALL_GEM_GREED_RADIUS and \
                            max(abs(gc[0] - head[0]), abs(gc[1] - head[1])) <= WALL_GEM_GREED_CELLS:
                        _wall_greed = True
                        break
            except Exception:
                _wall_greed = False
        # HUG_WALL (the user's instruction, three times): in the no-boost collection mode the margin is a
        # preference for route planning, not a safety rule, so the reserve goes to 0 and the head may run along
        # the outermost playable line.  `_passable` still forbids entering a wall cell, a body or a ball.
        if HUG_WALL and self.no_boost:
            _reserve = 0
        else:
            _reserve = 0 if _wall_greed else (1 if edge_gem else EDGE_TURN_RESERVE)
        if not self.no_boost:
            steps = self._prefer_inner_steps(steps, head, reserve=_reserve)
        # A nearby snake or moving ball must be cleared before pursuing an
        # item.  The live loss at ~8 s followed a diamond through the purple
        # snake's one-cell halo (risk 1-2) while a lower-risk exit existed.
        low_risk = [s for s in steps if s['risk'] < 0.8]
        if low_risk:
            steps = low_risk
        self._first_allowed = {s['cell'] for s in steps}
        # A self-trap has to actually be caused by own body, not merely the map
        # border or a padded opponent. Count its persistence independently.
        self_blocks = sum((head[0]+d[0],head[1]+d[1]) in own
                          and not self._passable((head[0]+d[0],head[1]+d[1]),set(),own,1)
                          for d in NEIGH)
        self_boxed=(self_blocks>=3 and (not steps or
                    (max(s['area'] for s in steps)<need and
                     max(s['runway'] for s in steps)<=1)))
        self._boxed_frames = self._boxed_frames+1 if self_boxed else 0
        mash = (not self.no_boost and self._boxed_frames >= SHED_TRAPPED_FRAMES and length>3
                and red.n_px >= 750)  # grid overlap can make a single segment look like four
        if not steps:
            self.last = dict(why='trapped', area=0,best_area=0,need=need,n_steps=0,
                             target=None,length=length,head=head,path_trust=ds['path_trust'])
            return None,False,mash,'trapped'
        safe = [s for s in steps if s['area']>=need]
        viable = safe or steps
        self._first_allowed = {s['cell'] for s in viable}
        previous_target = self.last.get('target')
        target = self._continue_pickup(st,red,head,hard,own) if self.no_boost and not recovering else None
        if target is None and not recovering and not endgame:
            target=self.choose_target(st,red,head,hard,own,need)
        # A route to the actual tail is the first survival fallback. During
        # the scoring half, refuse long detours and redundant speed pickups.
        if self.no_boost and not recovering:
            late=st.get('match_elapsed',0.) or 0.
            protect=(late>=35 and length>=51) or (late>=48 and length>=20 and
                       (target is None or target['dist']>8 or target['kind']=='grn'))
            if target is None or (protect and not target.get('committed')):
                tail=self._safe_tail_target(st,red,head,hard,own)
                if tail is not None:
                    target=tail
        # ---- tail chasing, as a FALLBACK.  User: "蛇太长的时候可以追着自己蛇尾跑，这样是非常安全的".
        # Only when the policy has no other target, so collecting always wins; the tail goes through the
        # normal target path so every safety rule still applies.
        # ---- opponent tactics, as fallbacks (an item target always wins)
        opp_why = None
        if target is None and not self.no_boost and not recovering and not endgame and head is not None:
            _ot = self._opp_tail_target(st)
            _ob = self._opp_block_target(st, red, head)
            for _kind, _cell in (('opp_tail', _ot), ('opp_block', _ob)):
                if _cell is None:
                    continue
                try:
                    _dist, _parent = self.route(head, hard, own)
                    _rt = self._route_cells(_parent, _cell) if _cell in _dist else None
                except Exception:
                    _rt = None
                if _rt:
                    # `dist` is REQUIRED: decide() reads target['dist'] for the log, and omitting it raised
                    # KeyError out of decide() - the same fatal shape as the earlier _delay_ms/loop_end bugs.
                    target = dict(kind=_kind, cell=_cell, route=_rt, dist=int(_dist.get(_cell, len(_rt))),
                                  pixel=(_cell[0]*CELL+CELL/2, _cell[1]*CELL+CELL/2),
                                  extra=True)
                    opp_why = _kind
                    self.last[opp_why] = True
                    break
        for _k in ('opp_tail', 'opp_block'):
            if _k != opp_why:
                self.last.pop(_k, None)
        tail_why = None
        if target is None and not self.no_boost and not recovering and not endgame:
            _area = max((s['area'] for s in viable), default=0)
            _tc = self._tail_chase_target(st, red, own, head, _area)
            if _tc is not None and _tc != head:
                try:
                    # use the SAME routing the normal targets use, so every safety property (reachability,
                    # the parent tree) is identical; only the destination differs
                    _dist, _parent = self.route(head, hard, own)
                    rt = self._route_cells(_parent, _tc) if _tc in _dist else None
                except Exception:
                    rt = None
                if rt:
                    target = dict(kind='tail', cell=_tc, route=rt,
                                  dist=int(_dist.get(_tc, len(rt))),
                                  pixel=(_tc[0]*CELL+CELL/2, _tc[1]*CELL+CELL/2),
                                  extra=True)
                    tail_why = 'tail_chase'
                    self.last['tail_chase'] = True
        if tail_why is None:
            self.last.pop('tail_chase', None)
        if self.no_boost and target is None and not recovering:
            target=self._safe_block_target(st,red,head,hard,own)
            if target is None:
                target=self._packing_target(red,head,hard,own)
            if target is None:
                target=self._safe_roam_target(st,red,head,hard,own)
        if self.no_boost and target is not None and target.get('pickup'):
            self._remember_pickup(st,target)
        if (not self.no_boost and previous_target and previous_target[0] == 'dia'
                and previous_target[2] <= 2
                and (target is None or target['cell'] != tuple(previous_target[1]))):
            # A newly collected gem can add five body cells.  Keep travelling
            # out of its cramped pickup lane before turning toward the next gem.
            self._pickup_clear_until_frame = max(self._pickup_clear_until_frame, frame + 6)
        chosen = None
        precision = False
        waypoint = None
        why = 'travel'
        if target is not None and not mash:
            # An anticipated turn executes at a FUTURE corner. Its permission
            # must be checked there, not against the unrelated lateral tile
            # beside the captured head. The checked approach itself remains
            # physically passable throughout the reaction interval.
            future=self._future_turn_step(target,red,head,hard,own,viable)
            if future is not None:
                viable=viable+[future]
                self._future_permission=future['cell']
            direction, precision, waypoint = self.follower.command(
                ds['head_px'],heading,target['route'],(target['kind'],target['cell']),
                target.get('drive_pixel',target['pixel']),{s['d'] for s in viable},
                pickup=target.get('pickup'))
            chosen = next((s for s in viable if s['d']==direction),None)
            if chosen is not None:
                why = 'eat_'+target['kind']
            if target['kind'] == 'dia':
                self.last['gem_dwell'] = self.last.get('gem_dwell', 0) + 1
        if chosen is None:
            if recovering:
                pinned = next((s for s in viable if s['d']==self._recovery_dir
                               and s['risk']<0.8 and s['runway']>=2),None)
                if pinned is not None and frame-self._recovery_dir_since<4:
                    chosen=pinned
                else:
                    # RISK FIRST.  This branch is the escape-from-collision path, so a roomy cell must
                    # not outrank a cell a spike ball is about to sweep: a ball kills instantly with no
                    # death buffer, whereas a crowded cell only limits the next move.  The old key put
                    # `s['area']>=need` first, which let the snake re-enter a cell it had itself flagged
                    # as ball-danger (measured, 4th live match, final frames).
                    chosen = max(viable,key=lambda s:(-s['risk'],s['area']>=need,
                                                       s['runway'],s['area'],s['straight'])
                                 if self.no_boost else
                                 (-s['risk'],s['area']>=need,s['runway'],s['body_gap'],
                                  s['gap'],s['area'],s['straight'],-s['wall']))
                    self._recovery_dir=chosen['d']
                    self._recovery_dir_since=frame
            elif self.no_boost:
                # No empty-cell clearance requirement or distance preference.
                # Feasibility is physical occupancy plus predicted crossings.
                chosen = max(viable,key=lambda s:(-s['risk'],s['area']>=need,
                                                   s['straight'],s['runway'],s['area']))
            elif final_seconds:
                # Keep the snake in the open, away from balls and the wall, and prefer to continue
                # straight: a turn in a crowd is what starts a crash buffer, and the buffer's ~7-frame
                # limit makes a second contact fatal.  Distance from the nearest ball is the first term
                # because a ball is the only thing that kills without a buffer.
                def _ball_gap(cell):
                    best = 99
                    for e in (items_ball or ()):
                        try:
                            best = min(best, max(abs(int(e[0]) - cell[0]), abs(int(e[1]) - cell[1])))
                        except Exception:
                            continue
                    return best if best < 99 else 99
                chosen = max(viable, key=lambda s: (
                    _ball_gap(s['cell']) >= 5,
                    ENDGAME_BALL_WEIGHT * min(6, _ball_gap(s['cell'])),
                    s['clearance'] >= ENDGAME_MIN_CLEARANCE,
                    s['clearance'],
                    s['straight'],
                    s['area'] >= need,
                    -s['risk'],
                    -s['wall'],
                    s['area']))
            elif endgame:
                # RISK FIRST, for the same reason as the buffer branch above: this phase exists only to
                # survive, so it must never trade a ball risk for open space.
                chosen = max(viable,key=lambda s:(-s['risk'],s['area']>=need,
                                                   s['runway']>=3,s['body_gap'],
                                                   s['runway'],s['gap'],
                                                   s['straight'],s['area'],-s['wall']))
            else:
                chosen = max(viable,key=lambda s:(s['area']>=need,-s['risk'],s['gap'],
                                                   s['straight'],s['runway'],s['area'],-s['wall']))
            target = None
            self.follower.waypoint = None
            why = ('buffer_recover' if recovering else 'endgame_final' if final_seconds
                   else 'endgame_survive' if endgame
                   else 'shed_escape' if mash else 'travel')
        if frame < self._pickup_clear_until_frame and not recovering:
            straight = next((s for s in viable if s['d'] == heading
                             and s['runway'] >= 2 and s['risk'] < 0.8), None)
            if straight is not None:
                chosen = straight
                target = None
                self.follower.waypoint = None
                why = 'pickup_clear'
        if red.eye_state == 'buffer':
            # A stopped snake needs a perpendicular exit, not 'keep going' just
            # because the large flood area behind a lethal contact looks good.
            turning = [s for s in viable if s['d'] != heading]
            if turning:
                pinned=next((s for s in turning if s['d']==self._recovery_dir
                             and frame-self._recovery_dir_since<5 and s['risk']<.8),None)
                chosen = pinned or max(turning,key=lambda s:(-s['risk'],s['area'],s['runway'])
                                       if self.no_boost else (-s['risk'],s['area'],s['gap']))
                target = None
                self._committed_route=None
                if pinned is None:
                    self._recovery_dir=chosen['d']
                    self._recovery_dir_since=frame
            why = 'buffer_turn'
            precision = True
        # ---- do not turn off a gem's line while its centre is still ahead.  The follower aligns the
        # lateral lane before driving forward by design, so a diagonal approach turns BEFORE crossing the
        # gem; measured live, match 6 "错过了钻石才转弯".  This guard is in the policy so the follower and its
        # tests are untouched, and it only removes steps that are legal, gem-directed and increasing the
        # perpendicular offset - it can never force an unsafe move.
        if not self.no_boost and target is not None and target.get('kind') == 'dia' and chosen is not None:
            try:
                tpx = target.get('pixel')
                if tpx is not None:
                    _dx = float(tpx[0]) - float(ds['head_px'][0])
                    _dy = float(tpx[1]) - float(ds['head_px'][1])
                    _r = PICKUP_LANE_GUARD_CELLS * CELL
                    if max(abs(_dx), abs(_dy)) <= _r:
                        # the offset to the gem line now, and after the planned step
                        _now = min(abs(_dx), abs(_dy))
                        _nx = _dx - chosen['d'][0] * CELL
                        _ny = _dy - chosen['d'][1] * CELL
                        _after = min(abs(_nx), abs(_ny))
                        if _after > _now:
                            _better = [s2 for s2 in viable
                                       if min(abs(_dx - s2['d'][0] * CELL),
                                              abs(_dy - s2['d'][1] * CELL)) <= _now]
                            if _better:
                                _old_d = chosen['d']
                                chosen = max(_better, key=lambda s2: (s2['area'], -s2['risk'],
                                                                      s2['straight']))
                                waypoint = None
                                # record it: a steering change that never fires, or fires without changing
                                # the direction, is not the fix it was supposed to be
                                self._pickup_guard = (tuple(_old_d), tuple(chosen['d']),
                                                      round(_now, 1), round(_after, 1))
            except Exception:
                pass
        # NO_BOOST: the whole match is spent collecting, so the key is never requested.  Written as an
        # explicit term so the intent is testable rather than implied by ACCEL_DEFAULT_HOLD.
        accel = bool(ACCEL_DEFAULT_HOLD) and not self.no_boost
        # High speed is maintained on useful straights, not into a wall/corner,
        # an uncertain head, or the final alignment of an item.
        # `low_trust` no longer releases the key by itself (see RELEASE_ON_LOW_TRUST): the user's rule is
        # that a brief loss of visual evidence must not toggle acceleration, and measured, head_jump and
        # no_game_evidence released it on 146 of 748 frames of a match.  It is still computed and still
        # reported, and path safety still governs the release.
        low_trust = bool(RELEASE_ON_LOW_TRUST and (getattr(red,'head_age',0)>2
                                                   or ds['head_px_uncertain']))
        # ---- the starvation brake.  Holding the boost costs about one cell per 5.7 s, so it needs
        # roughly +0.18 cells/s of intake just to break even.  Measured live in the third match: early
        # (t20-45 s) the net was +0.16 cells/s and the snake went 5 -> 2 -> 1 -> 0 while the key was
        # held, i.e. it was LOSING length; late (t45-79 s) the net was +0.29 and it grew 8 -> 18 with the
        # boost paying for itself.  Below STARVED_LEN the key is released to keep the cells, and the
        # brake resumes by itself once the snake is long enough.  A speed item already running is exempt
        # because that cost is already sunk, and the brake NEVER presses the key - it only stops holding.
        # Whether a speed item (绿豆) is currently running.  The bot needs this to pick its control
        # delay tier, and it must be PUBLISHED: the previous attempt read `st['item_accel_until']`,
        # which the policy consumes but never writes, so that tier was dead code.
        self.speed_item_active = bool(st.get('item_accel_until', 0.0) > st.get('t', 0.0))
        starved = (STARVATION_BRAKE and length < STARVED_LEN
                   and not (st.get('item_accel_until', 0.0) > st.get('t', 0.0)))
        # `precision` means the planner is on its final approach; brake only when the item is within
        # ALIGN_BRAKE_CELLS, so the approach itself is made at speed and the pickups are still landed on
        # the centre.
        near_item = True
        if precision and target is not None:
            try:
                near_item = (abs(int(target['cell'][0]) - head[0]) <= ALIGN_BRAKE_CELLS
                             and abs(int(target['cell'][1]) - head[1]) <= ALIGN_BRAKE_CELLS)
            except Exception:
                near_item = True
        # ---- which releases survive.  With HOLD_BOOST_EXCEPT_SAFETY only the lethal ones do: a step
        # that is not passable, no room for the planned turn, and the crash-buffer turn.  The rest
        # (alignment, runway, risk, low_trust, starved) are caution, and caution costs the speed that
        # reaches the next diamond - measured, the key was being toggled on every approach to food.
        # The terms are evaluated individually so the log can say WHICH one released the boost.  Reasoning
        # about this from `runway`/`risk` alone was wrong twice: those two were above their thresholds on
        # frames that still released, because the cause was `not safe` or the wall reserve.
        release_why = None
        if HOLD_BOOST_EXCEPT_SAFETY:
            if not safe:
                release_why = 'release:not_safe'
            elif self._wall_distance(chosen['cell']) < ACCEL_WALL_RESERVE:
                # 2 cells, not EDGE_TURN_RESERVE+1: instrumented with `release_why`, the old 4-cell demand
                # caused 291 of 297 releases in a run, because it spans columns 0-3/26-28 and rows 0-3/22-24
                # of a 29x25 board.  See ACCEL_WALL_RESERVE for why 2 still protects the turn.
                release_why = 'release:wall'
            elif why == 'buffer_turn':
                release_why = 'release:buffer_turn'
            elif mash:
                release_why = 'release:mash'
            release = release_why is not None
        else:
            release = ((precision and near_item) or why=='buffer_turn' or mash or low_trust or starved
                       or chosen['runway']<ACCEL_MIN_RUNWAY or chosen['risk']>=ACCEL_MAX_RISK
                       or not safe
                       or self._wall_distance(chosen['cell']) < EDGE_TURN_RESERVE + 1)
        if release:
            accel = False
            if starved:
                why += '+starved'
        if precision and why.startswith('eat_'):
            why += '+align'
        elif not accel and why != 'buffer_turn':
            why += '+release'
        self.head_hist.append((*ds['head_px'],st.get('frame',0)))
        self.last = dict(why=why,area=chosen['area'],best_area=max(s['area'] for s in steps),
                         need=need,n_steps=len(steps),length=length,heading=heading,head=head,
                         waypoint=waypoint,head_px=ds['head_px'],runway=chosen['runway'],
                         clearance=chosen['clearance'],gap=chosen['gap'],risk=chosen['risk'],
                         release_why=release_why,
                         speed_item_active=getattr(self, 'speed_item_active', False),
                         pickup_guard=getattr(self, '_pickup_guard', None),
                         path_trust=ds['path_trust'],body_source=ds['body_source'],
                         body_coverage=ds['body_coverage'],orientation=red.orientation,
                         target=None if target is None else (target['kind'],target['cell'],target['dist']),
                         route=None if target is None else target['route'])
        return chosen['d'],accel,mash,why

    def _observed_heading(self,ds,red,st):
        facing = ds['facing']
        if facing not in NEIGH:
            facing = None
        px = ds['head_px']
        prev = self._last_observation
        t = st.get('t',0.0)
        heading = facing
        if prev is not None and 0<t-prev[1]<=0.3:
            dx,dy = px[0]-prev[0][0],px[1]-prev[0][1]
            if 2<=abs(dx)+abs(dy)<=CELL*1.5:
                heading = (1 if dx>0 else -1,0) if abs(dx)>abs(dy) else (0,1 if dy>0 else -1)
        self._last_observation = (px,t)
        return heading or self._heading

    def _decision_state(self,st,belief):
        red = st['snakes'].get('red')
        head = tuple(int(v) for v in red.head_cell) if red and red.head_cell else None
        px = tuple(float(v) for v in red.head_px) if red and red.head_px else None
        length = red.length if red else 0
        path = [tuple(int(v) for v in c) for c in (getattr(red,'path',None) or [])]
        trust = getattr(red,'path_trust','absent')
        facing = getattr(red,'facing',None)
        uncertain_px = False
        if belief is not None:
            if belief.head_cell is not None:
                bh = tuple(int(v) for v in belief.head_cell)
                if head != bh:
                    # A corrected tile does not magically supply a measured pixel
                    # centre. Brake, and do not aim a last-mile turn with wrong pixels.
                    px = (bh[0]*CELL+CELL/2,bh[1]*CELL+CELL/2)
                    uncertain_px = True
                head = bh
            if belief.length:
                length = belief.length
            if getattr(belief,'facing',None) in NEIGH:
                facing = belief.facing
        if trust=='ordered' and (not path or path[0]!=head):
            trust = 'uncertain'
        source='visual'
        estimate=self.body_history.update(red,st,head_cell=head,head_px=px,
                                          verified=False if uncertain_px else None)
        # A partial matched prefix is not proof of the physical tail. Adopt only
        # complete, independently witnessed history; unobserved cells remain INF.
        if estimate.ordered and estimate.tail_confirmed and trust!='ordered':
            path=list(estimate.path)
            trust='ordered'
            source='head_history'
        if estimate.reason=='body_collapse_or_occlusion':
            trust='uncertain'
        return dict(head=head,head_px=px,length=int(length),path=path,path_trust=trust,
                    facing=facing,head_px_uncertain=uncertain_px,body_source=source,
                    body_coverage=estimate.coverage)

    @staticmethod
    def collision_cells(st):
        blocked=set()
        for kind,s in st['snakes'].items():
            if kind!='red':
                blocked.update((int(x),int(y)) for x,y in s.cells)
        centers=st.get('item_centers',{})
        att=st.get('attention')
        if att is not None:
            centers=getattr(att,'item_centers',centers)
        for bx,by,_ in st['items'].get('ball',[]):
            center=centers.get(('ball',int(bx),int(by)),((float(bx)+.5)*CELL,(float(by)+.5)*CELL))
            blocked.update(Policy.ball_footprint(center))
        return blocked

    @staticmethod
    def ball_footprint(center):
        # Item coordinates are CENTROIDS, not top-left corners. Pixel evidence
        # in luping1 f634 shows (103,636) occupies x2,3, not old x3,4.
        radius=.75*CELL
        x0,x1=int((center[0]-radius)//CELL),int((center[0]+radius)//CELL)
        y0,y1=int((center[1]-radius)//CELL),int((center[1]+radius)//CELL)
        return {(x,y) for x in range(x0,x1+1) for y in range(y0,y1+1) if in_board((x,y))}

    @staticmethod
    def hard_blocked(st):
        # Safety halos are COSTS, not walls. This avoids being 'trapped' by a
        # one-cell margin even when the recorded player has a physical exit.
        blocked=Policy.collision_cells(st)
        belief=st.get('belief')
        if belief is not None:
            # Full sweep is a soft risk below; only immediate fitted positions
            # belong in the hard set, rather than arbitrary top six sweep cells.
            for mode,threat,cell,ahead in getattr(belief,'ball_tracks',[]):
                if ahead and threat>0:
                    blocked.update(Policy.ball_footprint(((ahead[0]+.5)*CELL,(ahead[1]+.5)*CELL)))
        return blocked

    def _risk_map(self,st,hard):
        risk={}
        if self.no_boost:
            # Exact predicted occupancy, not an obstacle halo.  Adjacent cells
            # remain usable, including the lane directly beside another body.
            current=[]
            now=st.get('t',0.)
            att=st.get('attention')
            centers=getattr(att,'item_centers',st.get('item_centers',{}))
            unused=list(self._ball_motion)
            for bx,by,*_ in st.get('items',{}).get('ball',[]):
                p=tuple(centers.get(('ball',int(bx),int(by)),((bx+.5)*CELL,(by+.5)*CELL)))
                old=min(unused,key=lambda o:abs(o['p'][0]-p[0])+abs(o['p'][1]-p[1]),default=None)
                velocity=(0.,0.)
                if old is not None and 0<now-old['t']<=.4 and max(abs(p[i]-old['p'][i]) for i in (0,1))<2*CELL:
                    unused.remove(old)
                    dt=now-old['t']
                    velocity=tuple(.6*old['v'][i]+.4*(p[i]-old['p'][i])/dt for i in (0,1))
                current.append(dict(p=p,t=now,v=velocity))
                # Only the actual swept footprint, no additional safety ring.
                for dt in (.1,.2,.3):
                    predicted=tuple(p[i]+velocity[i]*dt for i in (0,1))
                    for c in self.ball_footprint(predicted):
                        if c not in hard:
                            risk[c]=2.0
            self._ball_motion=current
            for kind,s in st['snakes'].items():
                if kind=='red' or not s.head_cell:
                    continue
                facing=getattr(s,'facing',None)
                point=getattr(s,'head_px',None) or tuple((v+.5)*CELL for v in s.head_cell)
                old=self._opponent_motion.get(kind)
                velocity=(0.,0.)
                if old is not None and .025<now-old['t']<=.35:
                    dt=now-old['t']
                    delta=tuple(point[i]-old['p'][i] for i in (0,1))
                    if sum(abs(v) for v in delta)<=3*CELL:
                        velocity=tuple(.5*old['v'][i]+.5*delta[i]/dt for i in (0,1))
                self._opponent_motion[kind]=dict(p=point,t=now,v=velocity)
                for dt in (.12,.24,.36):
                    predicted=tuple(int((point[i]+velocity[i]*dt)//CELL) for i in (0,1))
                    if in_board(predicted) and predicted not in hard:
                        risk[predicted]=2.0
                if facing in NEIGH:
                    n=(s.head_cell[0]+facing[0],s.head_cell[1]+facing[1])
                    if in_board(n) and n not in hard:
                        risk[n]=2.0
            return risk
        for c in hard:
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    n=(c[0]+dx,c[1]+dy)
                    if in_board(n) and n not in hard:
                        risk[n]=max(risk.get(n,0),0.8)
        # The balls move less than a cell on many frames, so grid-quantised
        # tracks often read "still" even while a ball is closing on the snake.
        # A boosted head travels about one cell per control frame.  Reserve its
        # approach lane several cells ahead regardless of fitted ball velocity.
        ball_cells={(int(x),int(y)) for x,y,*_ in st['items'].get('ball',[])}
        b=st.get('belief')
        for _mode,_threat,cell,_ahead in getattr(b,'ball_tracks',[]) if b else []:
            if cell is not None:
                ball_cells.add((int(cell[0]),int(cell[1])))
        for bx,by in ball_cells:
            for dx in range(-4,5):
                for dy in range(-4,5):
                    n=(bx+dx,by+dy)
                    if not in_board(n):
                        continue
                    radius=max(abs(dx),abs(dy))
                    if radius<=2:
                        risk[n]=max(risk.get(n,0),1.5)
                    elif radius<=4 and min(abs(dx),abs(dy))<=1:
                        risk[n]=max(risk.get(n,0),0.9)
        for kind,s in st['snakes'].items():
            if kind=='red' or not s.head_cell:
                continue
            hx,hy=s.head_cell
            boost=1.0 if getattr(s,'ramp_span',0)>=12 else 0.5
            # Potential head crossing is dangerous even without perfect direction.
            for dx in range(-2,3):
                for dy in range(-2,3):
                    n=(int(hx)+dx,int(hy)+dy)
                    if in_board(n) and n not in hard:
                        risk[n]=max(risk.get(n,0),boost*(3-max(abs(dx),abs(dy))))
        for c,threat,_ in getattr(b,'danger_cells',[]) if b else []:
            c=tuple(int(v) for v in c)
            # A spike ball occupies roughly 2x2 cells and moves between vision
            # samples.  Its forecast centre alone is too narrow a collision
            # zone for a boosted snake approaching from the adjacent lane.
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n=(c[0]+dx,c[1]+dy)
                    if in_board(n):
                        scale=2.0 if dx == dy == 0 else 1.0
                        risk[n]=max(risk.get(n,0),scale*threat)
        return risk



    def tight_area(self, head, own, path, limit=64):
        """Reachable cells from the head with the body treated as SOLID except the tail.

        `flood` answers a different question: it lets the snake move into cells its own body will have vacated,
        so on a long coil it still reports most of the board (measured: 705 of 725 on a 21-cell coil).  That
        made the tail-chase gate impossible to satisfy.  This is the "how boxed in am I" measure instead, and
        it stops as soon as `limit` cells are found, so it is cheap.
        """
        body = {(int(c[0]), int(c[1])) for c in (path or ())}
        tail = (int(path[-1][0]), int(path[-1][1])) if path else None
        if tail is not None:
            body.discard(tail)
        seen = {head}
        frontier = [head]
        while frontier and len(seen) < limit:
            nxt = []
            for c in frontier:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (c[0] + dx, c[1] + dy)
                    if n in seen or n in body:
                        continue
                    if not in_board(n):
                        continue
                    seen.add(n)
                    nxt.append(n)
                    if len(seen) >= limit:
                        break
                if len(seen) >= limit:
                    break
            frontier = nxt
        return len(seen)

    def _tail_chase_target(self, st, red, own, head, area):
        """The own tail cell, when chasing it is the safe thing to do.  -> cell or None.

        User: "蛇太长的时候可以追着自己蛇尾跑，这样是非常安全的".  A long snake's body follows the head, so a head
        that stays behind its own tail has nothing to collide with.  This is used only as a FALLBACK when the
        free area around the head is small and no item is being pursued, so it never overrides collecting.
        """
        if not TAIL_CHASE:
            return None
        length = int(getattr(red, 'length', 0) or 0)
        # the gate uses the BODY-AWARE area: `area` from flood() counts cells the body will vacate, so it
        # is large even when the snake is coiled up (measured 705 of 725), and the gate never opened
        tight = self.tight_area(head, own, getattr(red, 'path', None), limit=TAIL_CHASE_AREA + 1)
        if length < TAIL_CHASE_MIN_LEN or tight > TAIL_CHASE_AREA:
            return None
        path = getattr(red, 'path', None) or []
        if len(path) < 2:
            return None
        tail = path[-1]
        return (int(tail[0]), int(tail[1]))

    def _remember_pickup(self,st,target):
        if target.get('committed'):
            return
        reach=max(3,math.ceil(self.follower.speed_s*self.follower.lead_s/CELL)+1)
        if target['dist']>reach:
            return
        route=target['route']
        index=route.index(target['cell'])
        segment=max(1,min(3,math.ceil(max(0.,self.follower.speed_s*self.follower.lead_s-8)/CELL)))
        stop=min(len(route)-1,index+max(2,segment))
        if stop<=index:
            return
        saved=dict(target)
        saved['route']=list(route[:stop+1])
        saved['drive_pixel']=tuple((v+.5)*CELL for v in route[stop])
        saved['committed']=True
        saved['commit_at']=st.get('t',0.)
        self._committed_route=saved

    def _continue_pickup(self,st,red,head,hard,own):
        saved=self._committed_route
        if saved is None:
            return None
        self._committed_route=None
        route=saved['route']
        if st.get('t',0.)-saved['commit_at']>1.4 or head not in route:
            return None
        cut=route.index(head)
        route=route[cut:]
        if len(route)<2:
            return None
        first=(route[1][0]-head[0],route[1][1]-head[1])
        if first==OPPOSITE.get(self._heading) or route[1] not in self._first_allowed:
            return None
        if any(c in hard or self._risk.get(c,0.)>=.8 for c in route[1:]):
            return None
        if red.path_trust=='ordered':
            if advance(red.path,route,self._growth_debt) is None:
                return None
        elif any(own.get(c,-1)>=i for i,c in enumerate(route[1:],1)):
            return None
        saved=dict(saved,route=route,dist=route.index(saved['cell']) if saved['cell'] in route else 0)
        self._committed_route=saved
        return saved

    def _packing_target(self,red,head,hard,own):
        segment=max(1,min(3,math.ceil(max(0.,self.follower.speed_s*self.follower.lead_s-8)/CELL)))
        ordered=red.path_trust=='ordered' and red.path and tuple(red.path[0])==head
        if ordered:
            body=red.path;blocked=hard;growth=self._growth_debt
        else:
            body=[head];blocked=set(hard)|set(red.cells);blocked.discard(head)
            # The unobserved old body remains solid, while the NEW trail
            # cannot disappear before a full body length has passed.
            growth=max(0,red.length-1)+self._growth_debt
        initial=0
        if self._heading in NEIGH:
            axis=0 if self._heading[0] else 1
            to_center=((head[axis]+.5)*CELL-red.head_px[axis])*self._heading[axis]
            initial=max(0,math.ceil((self.follower.speed_s*self.follower.lead_s-8-to_center)/CELL))
        route=packing_escape(body,self._heading,blocked,growth,segment,initial,
                             partial=not ordered)
        if not route or len(route)<3 or route[1] not in self._first_allowed:
            return None
        # Execute a short sweep before updating moving opponents and tail.
        route=route[:9];goal=route[-1]
        return dict(kind='packing',cell=goal,route=route,dist=len(route)-1,
                    pixel=tuple((v+.5)*CELL for v in goal),extra=True)

    def _safe_tail_target(self, st, red, head, hard, own):
        if red.path_trust!='ordered' or len(red.path)<8:
            return None
        previous=self.last.get('target')
        goal=self._tail_goal if previous and previous[0]=='tail' else None
        if goal is None or goal==head or goal in hard or max(abs(goal[i]-head[i]) for i in (0,1))<=1:
            goal=tuple(red.path[-1])
        self._tail_goal=goal
        path=self.plans.approach(('tail',goal),head,red.head_px,self._heading,hard,own,
            self.follower.speed_s,self.follower.lead_s,red.length+self._growth_debt,self._risk)
        if not path or len(path)<2 or path[1] not in self._first_allowed:
            return None
        moved=advance(red.path,path,self._growth_debt)
        if moved is None:
            return None
        return dict(kind='tail',cell=goal,route=path,dist=len(path)-1,
                    pixel=((goal[0]+.5)*CELL,(goal[1]+.5)*CELL),extra=True)

    def _future_turn_step(self,target,red,head,hard,own,viable):
        if not self.no_boost or self._heading not in NEIGH or self.follower.speed_s<80:
            return None
        route=target.get('route',[])
        h=self._heading
        dirs=[(b[0]-a[0],b[1]-a[1]) for a,b in zip(route,route[1:])]
        if not dirs or dirs[0]!=h:
            return None
        j=next((i for i,d in enumerate(dirs) if d!=h),None)
        if j is None or j==0 or any(s['d']==dirs[j] for s in viable):
            return None
        corner=route[j]
        axis=0 if h[0] else 1
        remaining=((corner[axis]+.5)*CELL-red.head_px[axis])*h[axis]
        lead=min(3*CELL,self.follower.speed_s*self.follower.lead_s)
        entry=route[j+1]
        if not(-.25*CELL<=remaining<=lead) or self._risk.get(entry,0)>=.8:
            return None
        if any(not self._passable(c,hard,own,i) for i,c in enumerate(route[1:j+2],1)):
            return None
        template=max(viable,key=lambda s:s['area']).copy()
        template.update(d=dirs[j],cell=entry,risk=self._risk.get(entry,0),straight=0)
        return template

    def _safe_roam_target(self, st, red, head, hard, own):
        # Keep an executable multi-cell exit plan while vision cannot prove a
        # tail, instead of choosing an isolated neighbour by flood area.
        dist,_=self.route(head,hard,own)
        goals=[c for c,d in dist.items() if 7<=d<=16]
        h=self._heading or (0,0)
        goals.sort(key=lambda c:(abs(dist[c]-11),-((c[0]-head[0])*h[0]+(c[1]-head[1])*h[1])))
        if self._roam_goal in dist and 3<=dist[self._roam_goal]<=25:
            goals=[self._roam_goal]+[g for g in goals if g!=self._roam_goal]
        for goal in goals[:5]:
            path=self.plans.approach(('roam',goal),head,red.head_px,self._heading,hard,own,
                self.follower.speed_s,self.follower.lead_s,red.length+self._growth_debt,self._risk)
            if (path and len(path)>1 and path[1] in self._first_allowed and
                    self._pickup_escape(path,red,hard,0,require_tail=red.length>=12,debt=self._growth_debt)):
                self._roam_goal=goal
                return dict(kind='roam',cell=goal,route=path,dist=len(path)-1,
                            pixel=((goal[0]+.5)*CELL,(goal[1]+.5)*CELL),extra=True)
        return None

    def _safe_block_target(self, st, red, head, hard, own):
        # Only occupy a crossing we can reach and clear BEFORE the opponent.
        # Late scoring preservation and teammate interference take precedence.
        if (st.get('match_elapsed',0.) or 0.)>=30 or red.length<10 or red.path_trust!='ordered':
            return None
        for kind in ('pur','blu'):
            opponent=st['snakes'].get(kind)
            if opponent is None or not opponent.head_cell or opponent.facing not in NEIGH:
                continue
            oh=opponent.head_cell;d=opponent.facing
            goal=(oh[0]+4*d[0],oh[1]+4*d[1])
            if not in_board(goal) or goal in hard:
                continue
            path=self.plans.approach(('block',goal),head,red.head_px,self._heading,hard,own,
                self.follower.speed_s,self.follower.lead_s,red.length,self._risk)
            if not path or len(path)<2 or path[1] not in self._first_allowed:
                continue
            our_eta=(len(path)-1+.6)*CELL/max(120.,self.follower.speed_s)
            # Assume a fast opponent (ten cells/second); never race its head.
            if our_eta+.12>=.4:
                continue
            if self._pickup_escape(path,red,hard,0,require_tail=True):
                return dict(kind='block',cell=goal,route=path,dist=len(path)-1,
                            pixel=((goal[0]+.5)*CELL,(goal[1]+.5)*CELL),extra=True)
        return None

    @staticmethod
    def own_index(red,path=None,trust=None,growth=0):
        """cell -> release delay minus one; tail=0, neck=L-2, head=L-1.

        The old head->tail index released the HEAD first: a dangerous inverted
        model. Unordered/missing body cells are never scheduled as empty.
        """
        cells=getattr(red,'cells',None) or ()
        path=path if path is not None else (getattr(red,'path',None) or [])
        trust=trust if trust is not None else getattr(red,'path_trust','absent')
        idx={}
        if trust=='ordered' and len(path)>=2:
            for i,c in enumerate(path):
                c=tuple(int(v) for v in c)
                idx[c]=max(idx.get(c,0),len(path)-1-i+growth)
        for c in cells:
            idx.setdefault(tuple(int(v) for v in c),INF)
        return idx

    @staticmethod
    def _free(c,hard):
        return in_board(c) and c not in hard

    @staticmethod
    def _passable(c,hard,own,d):
        return in_board(c) and c not in hard and own.get(c,-1)<d

    def flood(self,start,d0,hard,own,limit=None):
        dist={start:d0}
        q=deque([start])
        while q:
            c=q.popleft()
            d=dist[c]
            if limit is not None and d>=limit:
                continue
            for dx,dy in NEIGH:
                n=(c[0]+dx,c[1]+dy)
                if n not in dist and self._passable(n,hard,own,d+1):
                    dist[n]=d+1
                    q.append(n)
        return len(dist),dist

    def escape_need(self,length):
        return max(MIN_ESCAPE_AREA,min(length+2,ESCAPE_AREA_CAP))

    def route(self,start,hard,own,limit=None):
        """Dijkstra: penalise proximity throughout a route, not after BFS.

        Distances are number of moves, while costs include wall/risk preference.
        Returned parent pointers retain the legacy public replay interface.
        """
        dist={start:0}
        cost={start:0.0}
        parent={start:None}
        queue=[(0.0,start)]
        while queue:
            value,c=heappop(queue)
            if value>cost.get(c,float('inf')):
                continue
            d=dist[c]
            if limit is not None and d>=limit:
                continue
            for dx,dy in NEIGH:
                n=(c[0]+dx,c[1]+dy)
                if c==start:
                    if self._heading and (dx,dy)==OPPOSITE.get(self._heading):
                        continue
                    if self._first_allowed is not None and n not in self._first_allowed:
                        continue
                if not self._passable(n,hard,own,d+1):
                    continue
                # A nonnegative per-edge wall cost can never outweigh all useful
                # item utility or make target-stickiness compare negative scores.
                added=1.0+3.0*self._risk.get(n,0)+(1.5 if not self.no_boost and self._along_wall(n) else 0)
                nv=value+added
                if nv<cost.get(n,float('inf')):
                    cost[n]=nv
                    dist[n]=d+1
                    parent[n]=c
                    heappush(queue,(nv,n))
        self._route_cost=cost
        return dist,parent

    @staticmethod
    def _route_cells(parent,cell):
        out=[]
        seen=set()
        while cell is not None and cell not in seen:
            seen.add(cell)
            out.append(cell)
            cell=parent.get(cell)
        return out[::-1]


    # ---------------------------------------------------------------- opponent tactics
    def _opp_tail_target(self, st):
        """The opponent tail cell, but ONLY while that tail is actually moving.

        User: "别的蛇的蛇尾也是安全的，除非它蛇尾不动".  A tail that is moving vacates its cell as the body
        advances, so following it is safe; a tail that has stopped is a static body cell and driving at it is a
        collision.  The test is therefore on MOVEMENT, tracked from the observations themselves.
        """
        if not OPP_TAIL_CHASE:
            return None
        frame = int(st.get('frame', 0))
        best = None
        for kind, s in (st.get('snakes') or {}).items():
            if kind == 'red':
                continue
            tail = getattr(s, 'tail_cell', None)
            path = getattr(s, 'path', None) or []
            if tail is None and path:
                tail = path[-1]
            if tail is None:
                continue
            tail = (int(tail[0]), int(tail[1]))
            seen = self._opp_tail_hist.setdefault(kind, [])
            if not seen or seen[-1][0] != tail:
                seen.append((tail, frame))
                del seen[:-8]
            moved_at = seen[-1][1] if seen else frame
            # still (or never observed moving) -> not a tail worth following
            if frame - moved_at > OPP_TAIL_MOVING_FRAMES:
                continue
            length = int(getattr(s, 'length', 0) or 0)
            if best is None or length > best[1]:
                best = (tail, length)
        return None if best is None else best[0]

    def _opp_block_target(self, st, red, head):
        """A cell ahead of the nearest opponent's head, to restrict where it can go.

        User: "堵别的蛇头" and "建议尝试将其它蛇堵死而非完全避开".  The cell is `OPP_BLOCK_AHEAD` in front of the
        opponent's heading (or toward us if its heading is unknown), and it is only RETURNED - the caller still
        requires it to be reachable through the routing tree, so the snake cannot be sent into a body.
        """
        if not OPP_HEAD_BLOCK:
            return None
        cand = []
        for kind, s in (st.get('snakes') or {}).items():
            if kind == 'red' or not getattr(s, 'head_cell', None):
                continue
            oh = (int(s.head_cell[0]), int(s.head_cell[1]))
            d = max(abs(oh[0] - head[0]), abs(oh[1] - head[1]))
            if d > OPP_BLOCK_MAX_DIST:
                continue
            facing = getattr(s, 'facing', None)
            if not facing or (facing[0] == 0 and facing[1] == 0):
                dx, dy = head[0] - oh[0], head[1] - oh[1]
                if dx == 0 and dy == 0:
                    continue
                facing = (1 if dx > 0 else -1, 0) if abs(dx) >= abs(dy) else (0, 1 if dy > 0 else -1)
            cell = (oh[0] + facing[0] * OPP_BLOCK_AHEAD, oh[1] + facing[1] * OPP_BLOCK_AHEAD)
            if in_board(cell):
                cand.append((d, cell, kind))
        if not cand:
            return None
        cand.sort()
        return cand[0][1]

    def _candidates(self,st,hard):
        b=st.get('belief')
        att=st.get('attention')
        chosen={}
        for kind,val in ITEM_VALUE.items():
            if b is not None and kind=='dia':
                continue
            entries=b.items.get(kind,[]) if b is not None else st['items'].get(kind,[])
            for e in entries:
                c=(int(e[0]),int(e[1]))
                if not in_board(c) or c in hard:
                    continue
                if not self.no_boost and self._wall_distance(c) < (1 if kind=='dia' else EDGE_TURN_RESERVE):
                    continue
                if b is not None:
                    status=str(e[3]) if len(e)>3 else 'new'
                    if status in ('eaten','forgotten','phantom'):
                        continue
                    if status.startswith('carried('):
                        try:
                            if int(status[8:-1])>CARRIED_MAX_AGE:
                                continue
                        except ValueError:
                            continue
                if kind!='dia' and not (self.no_boost and kind=='grn') and ONLY_WINDOW_LOW_VALUE and att is not None:
                    box=att.window_box
                    if box:
                        px,py=(c[0]+.5)*CELL,(c[1]+.5)*CELL
                        if not(box[0]<=px<box[2] and box[1]<=py<box[3]):
                            continue
                # The free ten-second speed item improves collection rate.
                # Do not repeatedly chase another one while it is active.
                if self.no_boost and kind == 'grn':
                    late=(st.get('match_elapsed',0.) or 0.)>=35
                    val = 1.5 if late or st.get('item_accel_until',0) > st.get('t',0) else 4.5
                chosen[(kind,c)]=(val,0.0)
        if b is not None and att is not None:
            for e in att.gem_best:
                c=(int(e[0]),int(e[1]))
                src=e[4] if len(e)>4 else 'peripheral'
                if src=='extra' or not in_board(c) or c in hard:
                    continue
                # A gem ON the border used to be discarded here, and measured it is 26.4% of all gems
                # sighted (321 of 1217) with 0% ever chosen as a target.  That is not a safety rule - the
                # BODY's clearance is enforced by `_prefer_inner_steps` above - so it was simply throwing away
                # the largest single category of the only thing that grows the snake.  Gems "tend to spawn
                # near the border" (see the note above), and the user reports them being avoided and then a
                # border gem causing a spiral ("内耗").  Kept as a validity check only.
                if not (0 <= c[0] < COLS and 0 <= c[1] < ROWS):
                    continue
                statuses=[str(x[3]) for x in b.items.get('dia',[]) if (int(x[0]),int(x[1]))==c]
                if 'eaten' in statuses:
                    continue
                bonus=GEM_BONUS_WINDOW if bool(e[5]) else GEM_BONUS_PAIR if src=='pair' else 0
                chosen[('dia',c)]=(ITEM_VALUE['dia'],bonus)
        return [(k,v,c,bonus) for (k,c),(v,bonus) in chosen.items()]

    def _item_pixel(self,st,kind,cell):
        centers=st.get('item_centers',{})
        att=st.get('attention')
        if att is not None:
            centers=getattr(att,'item_centers',centers)
        return tuple(centers.get((kind,*cell),((cell[0]+.5)*CELL,(cell[1]+.5)*CELL)))

    def choose_target(self,st,red,head,hard,own,need):
        cands=self._candidates(st,hard)
        if not cands:
            self._target=None
            return None
        dist,parent=self.route(head,hard,own)
        frame=int(st.get('frame',0))
        scored=[]
        for kind,val,cell,bonus in cands:
            key=(kind,cell)
            if self._skip_target_until.get(key,0)>frame:
                continue
            d=dist.get(cell)
            if d is None or (kind=='bean' and d>LOW_VALUE_MAX_DETOUR):
                continue
            # With a long body, a target on the other side of our own coil can
            # require a huge loop.  Refreshing that route each frame makes the
            # snake orbit the gem and wastes the match; choose another item.
            direct=abs(cell[0]-head[0])+abs(cell[1]-head[1])
            if kind=='dia' and d>max(20,direct+14):
                continue
            if not self.no_boost and kind in ('dia','bean') and self._heading in NEIGH and direct<=3:
                forward=(cell[0]-head[0])*self._heading[0]+(cell[1]-head[1])*self._heading[1]
                lateral=abs((cell[0]-head[0])*self._heading[1]
                            -(cell[1]-head[1])*self._heading[0])
                if (forward<0 or (lateral and forward<=1)):
                    # At full boost a fresh stick turn lands about two cells
                    # later.  A target beside/behind the head now requires a
                    # tight loop; retire it briefly so we can clear our body.
                    self._skip_target_until[key]=frame+25
                    continue
            pixel=self._item_pixel(st,kind,cell)
            if d == 0 and self._heading in NEIGH:
                axis=0 if self._heading[0] else 1
                if (pixel[axis]-red.head_px[axis])*self._heading[axis] < -6:
                    # The head already passed this item's centre.  Asking for
                    # it every frame causes a tight orbit because 180-degree
                    # turns are illegal.  Leave the tile and replan an approach.
                    continue
            path=self._route_cells(parent,cell)
            # Do NOT skip d==0. Need to reach the actual object's centre.
            area,_=self.flood(cell,max(1,d),hard,own)
            growth = 5 if kind=='dia' else 1 if kind=='bean' else 0
            if not self.no_boost and area<self.escape_need(max(need,red.length+growth)):
                continue
            if not self.no_boost and not self._prefix_safe(path,red,hard,growth=growth):
                continue
            if self._risk.get(cell, 0.0) >= 0.8:
                continue
            score=val/(self._route_cost[cell]+ITEM_FIXED_COST[kind])+bonus
            scored.append(dict(kind=kind,cell=cell,dist=d,first=path[1] if d else head,
                               area=area,score=score,route=path,pixel=pixel,value=val,bonus=bonus))
        if not scored:
            self._target=None
            return None
        # Check the body AFTER the complete approach and growth, not merely
        # the first eight cells or one empty cell beside the pickup.
        if self.no_boost:
            self._candidate_review=[]
            scored.sort(key=lambda s:s['score'], reverse=True)
            accepted=[]
            segment=max(1,min(3,math.ceil(max(0.,self.follower.speed_s*self.follower.lead_s-8)/CELL)))
            for candidate in scored[:5]:
                path=self.plans.approach((candidate['kind'],candidate['cell']),head,red.head_px,
                    self._heading,hard,own,self.follower.speed_s,self.follower.lead_s,
                    length=red.length+self._growth_debt,risk=self._risk)
                if not path or (len(path)>1 and path[1] not in self._first_allowed):
                    self._candidate_review.append((candidate['kind'],candidate['cell'],'approach'))
                    continue
                growth=5 if candidate['kind']=='dia' else 1 if candidate['kind']=='bean' else 0
                escape=self._pickup_escape(path, red, hard, growth,segment=segment,
                    require_tail=red.length>=12,debt=self._growth_debt,return_route=True)
                if escape:
                    candidate['route']=path+escape[1:9]
                    candidate['pickup']=(candidate['cell'],candidate['pixel'])
                    candidate['drive_pixel']=tuple((v+.5)*CELL for v in candidate['route'][-1])
                    candidate['dist']=len(path)-1
                    candidate['first']=path[1] if len(path)>1 else head
                    # Only reward beans actually crossed by this checked path;
                    # do not interrupt an approach for a sideways bean detour.
                    beans={tuple(map(int,e[:2])) for e in st['items'].get('bean',[])}
                    incidental=len(beans.intersection(path[1:]))
                    candidate['score']=(candidate['value']+.45*incidental)/(len(path)-1+ITEM_FIXED_COST[candidate['kind']])+candidate['bonus']
                    self._candidate_review.append((candidate['kind'],candidate['cell'],'accepted'))
                    accepted.append(candidate)
                    if len(accepted)>=2:
                        break
                else:
                    self._candidate_review.append((candidate['kind'],candidate['cell'],'exit'))
            scored=accepted
            if not scored:
                self._target=None
                return None
        best=max(scored,key=lambda s:s['score'])
        current=next((c for c in scored if (c['kind'],c['cell'])==self._target),None)
        if current and current['score']>=TARGET_STICKY*best['score']:
            best=current
        self._target=(best['kind'],best['cell'])
        return best

    @staticmethod
    def _pickup_escape(path, red, hard, growth, segment=1, require_tail=False, debt=0,return_route=False):
        """Check the complete approach, then return an executable exit witness.

        Reaching a boundary pickup can require a turn AT its centre, queued
        while approaching. The last approach leg determines the next turn's
        spacing; it does not automatically require another straight segment.
        """
        def result(route):
            return route if return_route else bool(route)
        if not path:
            return result(None)
        facing=(path[-1][0]-path[-2][0],path[-1][1]-path[-2][1]) if len(path)>1 else red.facing
        run=0
        for a,b in reversed(list(zip(path,path[1:]))):
            if (b[0]-a[0],b[1]-a[1])!=facing:
                break
            run+=1
        initial=max(0,segment-run) if len(path)>1 else 0
        ordered=getattr(red,'path_trust','')=='ordered' and red.path and tuple(red.path[0])==path[0]
        if ordered:
            moved=advance(red.path,path,debt)
            if moved is None:
                return result(None)
            body,remaining=moved
            growth+=remaining
            if require_tail:
                route=tail_witness([body[0]],body,facing,hard,growth=growth,
                                   segment=segment,initial=initial)
                if route:
                    return result(route)
                # A distant old-tail cell is not the only possible exit. A
                # simulated full body turnover can also prove a packed sweep.
                if len(path)>11:
                    return result(None)
        else:
            # Unordered cells have NO scheduled tail. Keep all of them solid
            # and simulate the new head trail through the approach and escape.
            hard=set(hard)|set(map(tuple,red.cells))
            hard.discard(path[0])
            if any(c in hard for c in path[1:]):
                return result(None)
            body=tuple(reversed(path))[:max(1,red.length)]
            growth+=max(0,red.length-len(body))+debt
        route=packing_escape(body,facing,hard,growth,segment,initial,
                             depth=None if require_tail and ordered else 12)
        return result(route)

    @staticmethod
    def _prefix_safe(path,red,hard,growth=0):
        """Simulate the near body, including cells newly occupied by our route.

        A release-time lookup alone forgets the bot's own intervening moves. It
        cannot be a safety proof. Reject inconsistent geometry instead of
        guessing the neck is a tail; replan each visual frame.
        """
        if getattr(red,'path_trust','')!='ordered' or not red.path or not path:
            return True  # own_index has already treated uncertain body as solid
        if tuple(red.path[0])!=tuple(path[0]):
            return True
        body=[tuple(int(v) for v in c) for c in red.path]
        for c in path[1:9]:
            if c in hard or c in body[:-1]:
                return False
            body=[c]+body[:-1]
        # On a near pickup the tail stalls for growth; entering an item then
        # relying on the tail immediately leaving is not an escape route.
        if growth and len(path)<=9:
            head=body[0]
            facing=(path[-1][0]-path[-2][0],path[-1][1]-path[-2][1]) if len(path)>=2 else red.facing
            neck_reverse=OPPOSITE.get(facing)
            exits=[(head[0]+dx,head[1]+dy) for dx,dy in NEIGH
                   if (dx,dy)!=neck_reverse and in_board((head[0]+dx,head[1]+dy))
                   and (head[0]+dx,head[1]+dy) not in hard
                   and (head[0]+dx,head[1]+dy) not in body]
            if not exits:
                return False
        return True

    def _runway(self,head,d,hard,own):
        n=0
        for k in range(1,5):
            c=(head[0]+d[0]*k,head[1]+d[1]*k)
            if not self._passable(c,hard,own,k):
                break
            n+=1
        return n

    def _clearance(self,cell,hard):
        # Legacy diagnostic only. A count of free neighbours is NOT a one-cell
        # minimum-distance guarantee; _gap computes that separate quantity.
        return sum(self._free((cell[0]+dx,cell[1]+dy),hard)
                   for dx in (-1,0,1) for dy in (-1,0,1) if dx or dy)

    @staticmethod
    def _gap(cell,hard):
        border=min(cell[0],cell[1],COLS-1-cell[0],ROWS-1-cell[1])
        obstacle=min((max(abs(cell[0]-c[0]),abs(cell[1]-c[1]))-1 for c in hard),default=3)
        return max(0,min(border,obstacle,3))

    @staticmethod
    def _body_gap(cell,own,path):
        # Distance to our own body, EXCLUDING the neck.  The head and the first few neck cells follow the head
        # by construction and cannot be collided with, so penalising them makes the snake refuse tight turns
        # around itself - the user's complaint that it "cannot turn right up against its own body" and their
        # instruction that the neck "does not have to be kept a cell away from the head".
        neck=set(path[:NECK_EXEMPT_CELLS]) if path else set()
        closest=min((max(abs(cell[0]-c[0]),abs(cell[1]-c[1]))
                     for c,delay in own.items()
                     if delay>=NECK_EXEMPT_CELLS-1 and c not in neck),
                    default=4)
        return max(0,min(3,closest-1))

    @staticmethod
    def _wall_distance(cell):
        return min(cell[0], cell[1], COLS-1-cell[0], ROWS-1-cell[1])

    @classmethod
    def _prefer_inner_steps(cls, steps, head, reserve=EDGE_TURN_RESERVE):
        if not steps:
            return steps
        margin = cls._wall_distance(head)
        if margin >= reserve:
            kept = [s for s in steps if cls._wall_distance(s['cell']) >= reserve]
        else:
            kept = [s for s in steps if cls._wall_distance(s['cell']) > margin]
            if not kept:
                kept = [s for s in steps if cls._wall_distance(s['cell']) == margin]
        return kept or steps

    @staticmethod
    def _near_wall(cell):
        return min(cell[0],cell[1],COLS-1-cell[0],ROWS-1-cell[1])==0

    @staticmethod
    def _along_wall(cell):
        return min(cell[0],cell[1],COLS-1-cell[0],ROWS-1-cell[1])<=WALL_MARGIN

    @staticmethod
    def _winning_grind(st,red):
        g=st.get('grind')
        other=st['snakes'].get(g['opponent']) if g else None
        return other is not None and red.length>other.length

    def mode_for(self,st):
        return 'lost' if st.get('player_respawn_pending') or not st['snakes'].get('red') else STRATEGY
