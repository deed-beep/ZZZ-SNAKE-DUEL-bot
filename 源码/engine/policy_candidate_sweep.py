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
from causal_planner import RoutePlans, tail_witness, advance, packing_escape, search
from sweep_planner import pocket,plan_sweep,walk
from body_history import OwnBodyHistory
from opponent_heads import NativeOpponentHeads,possible_crossings
from config import (COLS, ROWS, CELL, BALL_CELLS, ITEM_VALUE, ITEM_FIXED_COST,
                    MIN_ESCAPE_AREA, ESCAPE_AREA_CAP, LOW_VALUE_MAX_DETOUR,
                    TARGET_STICKY, CARRIED_MAX_AGE, GEM_BONUS_WINDOW, GEM_BONUS_PAIR,
                    ONLY_WINDOW_LOW_VALUE, WALL_MARGIN, ACCEL_DEFAULT_HOLD,
                    SHED_TRAPPED_FRAMES, STRATEGY, STARVED_LEN, STARVATION_BRAKE, GEM_WALL_APPROACH_CELLS, ALIGN_BRAKE_CELLS, RELEASE_ON_LOW_TRUST, ACCEL_MIN_RUNWAY, ACCEL_MAX_RISK, HOLD_BOOST_EXCEPT_SAFETY, ACCEL_WALL_RESERVE, ENDGAME_WINDOW, ENDGAME_MIN_CLEARANCE, ENDGAME_BALL_WEIGHT, MATCH_SECONDS, ENDGAME_ARM_MIN, MATCH_SCORE_SECONDS, EYE_BUFFER_CONFIRM, BUFFER_MAX_FRAMES, PICKUP_LANE_GUARD_CELLS, NO_BOOST, NECK_EXEMPT_CELLS, WALL_GEM_GREED, WALL_GEM_GREED_CELLS, WALL_GEM_GREED_RADIUS, TAIL_CHASE, TAIL_CHASE_MIN_LEN, TAIL_CHASE_AREA, OPP_TAIL_CHASE, OPP_HEAD_BLOCK, OPP_TAIL_MOVING_FRAMES, OPP_BLOCK_AHEAD, OPP_BLOCK_MAX_DIST, HUG_WALL,
                    LATE_GUARD_SECONDS, LATE_GUARD_STRICT_SECONDS, LATE_HOLD_LENGTH, LATE_BEAN_DIST,
                    LATE_STRICT_DIAMOND_DIRECT, GREEN_MAX_DIRECT, GREEN_LATE_VALUE, GREEN_NORMAL_VALUE, PROTECT_LENGTH,
                    BALL_ADJ_RISK, OPP_BODY_ADJ_RISK,
                    BALL_DIAG_ADJ_RISK, BALL_DIAG_RADIUS, BALL_DIAG_HARD_RADIUS, BALL_AXIS_ADJ_RISK, BUFFER_DIR_HOLD,
                    DIAMOND_MAX_ROUTE, DIAMOND_NEAR_DIRECT, BLOCK_GEM_NEAR, BLOCK_GEM_BONUS,
                    RESPAWN_GUARD_LEN, RESPAWN_GUARD_FRAMES, TAIL_FOLLOW_MIN_LENGTH, TIGHT_GUARD_LENGTH, TIGHT_GUARD_AREA, TIGHT_GUARD_AREA_SHORT, GEM_CLUSTER_CELLS, GEM_CLUSTER_BONUS, MATCH_CLOCK_OFFSET, TAIL_CHASE_ONLY_WHEN_TIGHT, STACK_ESCAPE_MIN_LEN, STACK_ESCAPE_MARGIN, NORMAL_DIR_HOLD, HOLD_AREA_MARGIN, NARROW_LANE_PENALTY, TARGET_KEEP_ROUTE, ESCAPE_ROUTE_CELLS, DIA_MEMORY_FRAMES, DIA_MEMORY_HITS, CIRCLE_SPAN, CIRCLE_WINDOW_FRAMES, CIRCLE_MIN_SAMPLES, CIRCLE_BREAK_FRAMES, SHORT_SNAKE_LEN, FINAL_GUARD_SECONDS, FINAL_GUARD_LEN, LONG_SNAKE_LEN, GEM_EXIT_ROOM_MARGIN, TURN_WINDOW_SECONDS, TURN_REPEAT_LIMIT, TURN_REPEAT_LIMIT_RECOVERING, TURN_LIMIT_MIN_AREA)
from navigation_candidate_sweep import RouteFollower

OPPOSITE = {(1,0):(-1,0),(-1,0):(1,0),(0,1):(0,-1),(0,-1):(0,1)}
NEIGH = ((1,0),(-1,0),(0,1),(0,-1))
INF = 10**6
EDGE_TURN_RESERVE = 3            # cells needed before a 15 Hz turn is visible

def in_board(c):
    return 0 <= c[0] < COLS and 0 <= c[1] < ROWS

class Policy:
    variant="buffer_absolute_near_chain_20261017"
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
        self._witnessed_body_bound = 0
        self._heading = None
        self._opposite_motion_count = 0
        self._last_observation = None
        self._risk = {}
        self._hug = set()
        self._route_risk = {}
        self._first_allowed = None
        self._escape_until_frame = 0
        self._pickup_clear_until_frame = 0
        self._skip_target_until = {}
        self._opp_tail_hist = {}
        self._recovery_dir = None
        self._recovery_dir_since = 0
        self._ball_motion = []
        self._opponent_motion = {}
        self._opponent_native=NativeOpponentHeads()
        self._opponent_heads={}
        self.plans = RoutePlans()
        self._tail_goal = None
        self._roam_goal = None
        self._growth_debt = 0
        self._geometry_growth = 0
        self._packing_saved = None
        self._item_accel_until = 0.
        self._growth_point = None
        self._growth_distance = 0.
        self._future_permission = None
        self._committed_route = None
        self._candidate_review = []
        self._sweep = None
        self._food = {}
        # final-stretch flags (set every frame by decide(); defaulted here so the helpers are usable
        # on their own, e.g. from a targeted test that calls choose_target directly)
        self._late = False
        self._late_strict = False
        self._late_hold = False
        # 复活保护（decide() 每帧维护）
        self._prev_len = None
        self._respawn_guard_until = 0
        # 方向连续性：上一帧真正下发的方向与它的起始帧
        self._sent_dir = None
        self._sent_since = -10 ** 6
        # 转头计数：(t, sign) 序列，只保留"连续同向"的一段（见 TURN_REPEAT_LIMIT）
        self._turn_seg = []
        # 小圈检测（见 config 的 CIRCLE_*）：头部轨迹 + 触发后的外拐截止帧与圈心
        self._head_trail = deque(maxlen=80)
        self._circling_until = 0
        self._circle_center = None
        self._broke_circle = False
        self._loop_break = False
        self._final_guard = False
        self._had_target = False
        self._opp_memory = {}
        self._opp_seen_at = {}
        self._dia_memory = {}

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
        red.cells=set(ds['body_cells'])
        self._pickup_guard = None
        self._candidate_review=[]
        self._future_permission = None
        red.length = length
        now=float(st.get('t',0.))
        if red.eye_state=='item':self._item_accel_until=max(self._item_accel_until,now+.35)
        elif (red.eye_state=='normal' and getattr(red,'head_from_eyes',False)
              and getattr(red,'head_age',99)==0):self._item_accel_until=now
        st['item_accel_until']=max(st.get('item_accel_until',0.),self._item_accel_until)
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
        # ---- 复活保护.  The user reported it twice ("复活后不能及时行动（刚复活为1格）", third batch #10
        # "结束时追尾巴但没有正确").  A respawn is visible as a large DROP in measured length (a body of
        # 20-40 cells becomes 1-5), which is the only respawn evidence available inside the policy: the
        # whiteout flag lives in bot.py's per-frame log, not in `st`.  For a few frames afterwards the head
        # is re-acquired badly (the batch logs show 40-110 `head_jump` frames after each respawn), so do not
        # chase anything and do not inherit the pre-death route: look for open space instead.
        _prev_len = getattr(self, '_prev_len', None)
        if (self.no_boost and _prev_len is not None and length <= RESPAWN_GUARD_LEN
                and _prev_len - length >= 4):
            self._respawn_guard_until = frame + RESPAWN_GUARD_FRAMES
        self._prev_len = length
        # ---- 死亡判定改用"全屏泛白"（用户 10-18："死亡判定用白屏/泛白即可，因为别的蛇死亡也会有特效，
        # 区别在于不会全屏泛白"）。原先想靠"长度骤降"判定，实测死亡前后长度是 8→8，根本抓不到。
        # 泛白期间画面没有可用信息，红蛇常以死亡闪光的假位置出现 —— 这一帧直接整帧拒绝、不发方向；
        # 泛白一结束（红蛇重新可见）就立刻正常行动，不再维护额外的"等待复活"状态机。
        if st.get('whiteout') is True:
            self.last = dict(why='whiteout_wait', target=None, length=length, head=head,
                             waypoint=None, n_steps=0)
            return None, False, False, 'whiteout_wait'
        guarding = bool(self.no_boost and frame < self._respawn_guard_until)
        endgame = (st.get('match_elapsed') is not None
                   and st['match_elapsed'] >= 50.0)
        # The FINAL window is a stricter mode than `endgame`: see ENDGAME_WINDOW in config.  Measured, the
        # snake was grinding in and out of the crash buffer for ~2.9 s before dying, with `endgame` active
        # the whole time, so stopping item chasing is not enough on its own.
        _el = st.get('match_elapsed')
        if _el is not None and not self.no_boost:
            # 固定偏差补偿（见 config.MATCH_CLOCK_OFFSET）：START overlay 与识别延迟让 bot 的时钟偏小。
            _el = _el + MATCH_CLOCK_OFFSET
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
        # ---- FINAL-STRETCH DISCIPLINE (see LATE_GUARD_SECONDS in config).  `endgame` above is a hard
        # stop on item chasing and stays tied to the score moment, which is why in no-boost mode it never
        # fired before the deaths that actually cost this batch (45-58 s).  These three flags are graded
        # instead, and only ever REMOVE a target when the trade is clearly losing.
        self._late = bool(self.no_boost and _el is not None
                          and _el >= MATCH_SCORE_SECONDS - LATE_GUARD_SECONDS)
        self._late_strict = bool(self.no_boost and _el is not None
                                 and _el >= MATCH_SCORE_SECONDS - LATE_GUARD_STRICT_SECONDS)
        # Tile occupancy oscillates with subcell phase. In r024 it briefly
        # read 43, yet the real settlement was 39. Do not protect that apparent
        # surplus or stop reachable pickups below the actual acceptance goal.
        secured_length=min(length,int(red.n_px/((CELL/2)**2)))
        self._late_hold = bool(self._late and secured_length >= 83)
        # 末段纯逃避（见 config.FINAL_GUARD_*）：用户 10-18"最后几秒错误操作…死得很愚蠢导致得分很低"。
        # 计分那一刻在死亡动画里就是 0 分，所以最后几秒只保命，不再接任何目标。
        self._final_guard = bool(self.no_boost and _el is not None
                                 and _el >= MATCH_SCORE_SECONDS - FINAL_GUARD_SECONDS
                                 and secured_length >= 83)
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
        # A visible tail tile is not empty at the instant the next head step
        # starts: the physical rectangles still overlap before that cell is
        # fully vacated (r028 f316). Simulate one step of phase uncertainty;
        # this delays release, without padding any obstacle's spatial extent.
        self._geometry_growth=pending_growth+(1 if self.no_boost else 0)
        own = self.own_index(red, ds['path'], ds['path_trust'], growth=self._geometry_growth)
        # Briefly bridge missing native observations, but never keep a departed
        # body cell forever. The old size-ratio union accumulated phantom walls
        # after a shrink, death or tail reclassification for the rest of the life.
        now=float(st.get('t',0.))
        for kind,snake in st.get('snakes',{}).items():
            if kind=='red':continue
            for c in snake.cells:self._opp_seen_at[(kind,tuple(c))]=now
        self._opp_seen_at={k:stamp for k,stamp in self._opp_seen_at.items()
                           if 0<=now-stamp<=.32}
        self._opp_memory={}
        for (kind,c),stamp in self._opp_seen_at.items():
            self._opp_memory.setdefault(kind,set()).add(c)
        hard=self.hard_blocked(st)|{c for cells in self._opp_memory.values() for c in cells}
        # ---- 钻石跨帧记忆（机制文档 §8.2 的"呼吸灯"；见 config.DIA_MEMORY_*）
        # 实录：同一批钻石在 f=577/583/586/589 能检出全部 3 颗，而 f=580 等帧检出 0 颗 —— 画面里钻石一直在。
        # 固定钻石不会动，所以"最近几帧稳定出现过的那几格"可以直接补回本帧的钻石列表。
        _items = st.get('items')
        if self.no_boost and isinstance(_items, dict):
            for _e in list(_items.get('dia') or []):
                try:
                    _c = (int(_e[0]), int(_e[1]))
                except Exception:
                    continue
                _last, _hits = self._dia_memory.get(_c, (-10 ** 6, 0))
                _hits = _hits + 1 if frame - _last <= DIA_MEMORY_FRAMES else 1
                self._dia_memory[_c] = (frame, _hits)
            _seen = set()
            for _e in (_items.get('dia') or []):
                try:
                    _seen.add((int(_e[0]), int(_e[1])))
                except Exception:
                    pass
            for _c, (_last, _hits) in list(self._dia_memory.items()):
                if frame - _last > DIA_MEMORY_FRAMES:
                    self._dia_memory.pop(_c, None)
                    continue
                if _hits >= DIA_MEMORY_HITS and _c not in _seen:
                    _items.setdefault('dia', []).append((_c[0], _c[1], 300, 0.5))
                    _seen.add(_c)
        # the balls, for the final-seconds ranking below
        items_ball = st.get('items', {}).get('ball') if isinstance(st.get('items'), dict) else None
        heading = self._observed_heading(ds, red, st)
        self._heading = heading
        self._risk = self._risk_map(st, hard)
        # Route cost = real hazard + the lanes beside an opponent's body.  `_risk` itself must NOT gain
        # those lanes (it also gates whether a gem can be approached at all), so the two are merged into
        # a separate map that only the search pays for: an approach route goes around a body instead of
        # hugging it, while a gem that can only be reached through a hugging lane stays reachable.
        self._route_risk = dict(self._risk)
        for c in self._hug:
            self._route_risk[c] = max(self._route_risk.get(c, 0.), OPP_BODY_ADJ_RISK)
        self._food=self._route_food(st)
        fresh_native=(not ds['head_px_uncertain'] and getattr(red,'head_age',99)==0
                      and (getattr(red,'head_from_eyes',False) or getattr(red,'head_from_pixels',False)))
        if fresh_native and red.eye_state!='buffer':
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
                              risk=max(self._risk.get(nxt,0.0),3.0 if self._ball_swept_hit(ds['head_px'],d,
                                  max(CELL,self.follower.speed_s*self.follower.lead_s)) else 0.), straight=int(d==heading),
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
        for step in steps:
            if step['risk']>=3.:self._risk[step['cell']]=max(self._risk.get(step['cell'],0.),step['risk'])
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
        # 长蛇的"留在大空间"护栏：见 config 的 TIGHT_GUARD_*。第一批第 8 局峰值 60、末尾被 `trapped`
        # 封死、结算 0 —— 长蛇最致命的不是吃不到钻石，而是把自己塞进窄缝后出不来。
        # 只在还有别的选择时剔除，全部方向都变窄时不过滤，因此不会把蛇钉死。
        if self.no_boost and length >= TIGHT_GUARD_LENGTH:
            _need_area = TIGHT_GUARD_AREA if length >= 35 else TIGHT_GUARD_AREA_SHORT
            _roomy = []
            for _s in viable:
                if self.tight_area(_s['cell'], own, getattr(red, 'path', None),
                                   limit=_need_area + 1) >= _need_area:
                    _roomy.append(_s)
            if _roomy:
                viable = _roomy
        # ---- 小圈检测与外拐（用户 10-18："意识到蛇行成一个小圈时一定要往外拐"）。
        # 今天 28 次死亡里 17 次是 `trapped`，很多发生在体长 9—21 的短蛇上：那不是长蛇缠住自己，
        # 而是它在几个格里反复绕圈，把出口绕没了。这里在"最近 CIRCLE_WINDOW_FRAMES 帧都没走出
        # CIRCLE_SPAN 格"时判定绕圈，随后一段时间只允许"不朝圈心"的方向。
        if head is not None:
            self._head_trail.append((frame, (int(head[0]), int(head[1]))))
        # 两种"绕圈"都要抓（见 2026-10-18 07 批第 9 局的死亡前 4 帧）：
        #   * 小圈：最近 CIRCLE_WINDOW_FRAMES 帧都没走出 CIRCLE_SPAN 格；
        #   * **大圈（真正的杀手）**：画面里红蛇在右侧画了一个跨 10+ 格的环，把头部关在里面 ——
        #     包围盒检测完全看不到它。判据改成"头部回到自己 20—70 帧前走过的位置附近"。
        if self.no_boost and head is not None and len(self._head_trail) >= 25:
            _came_back = False
            _old = [(f, c) for f, c in self._head_trail if 20 <= frame - f <= 70]
            if _old:
                _near = min(abs(head[0] - c[0]) + abs(head[1] - c[1]) for _f, c in _old)
                _came_back = _near <= 2
            if _came_back:
                self._circling_until = frame + CIRCLE_BREAK_FRAMES
                self._circle_center = None      # 用"远离历史轨迹"代替"远离圈心"
                self._loop_break = True
        if self.no_boost and frame % 8 == 0 and len(self._head_trail) >= CIRCLE_MIN_SAMPLES:
            _recent = [c for f, c in self._head_trail if frame - f <= CIRCLE_WINDOW_FRAMES]
            if len(_recent) >= CIRCLE_MIN_SAMPLES:
                _xs = [c[0] for c in _recent]
                _ys = [c[1] for c in _recent]
                if (max(_xs) - min(_xs)) <= CIRCLE_SPAN and (max(_ys) - min(_ys)) <= CIRCLE_SPAN:
                    self._circling_until = frame + CIRCLE_BREAK_FRAMES
                    self._circle_center = (sum(_xs) / float(len(_xs)),
                                          sum(_ys) / float(len(_ys)))
        # Revisiting a coordinate does not prove a self enclosure. An item
        # approach can legitimately pass an old head position after its tail
        # has left. Keep the signal as diagnostics, without reducing viable
        # to one outward step before checking a food route (r021 f178).
        self._broke_circle=False
        # Width alone cannot invalidate a physically checked packed escape.
        # r027 f430 discarded the saved right/up fold only because its next
        # tile was a one-cell lane, then sent a fatal down turn instead.
        self._first_allowed = {s['cell'] for s in viable}
        # Opponent-body adjacency is a MOVE cost, applied only now that `_first_allowed` is fixed, so it
        # re-ranks the four candidate steps without taking any item target off the board (see `_hug`).
        if self.no_boost and self._hug:
            for s in steps:
                if s['cell'] in self._hug:
                    s['risk'] = max(s['risk'], OPP_BODY_ADJ_RISK)
        previous_target = self.last.get('target')
        # 复活保护期间：不吃钻、不继承死前路线，直接去找开阔方向（见上面的 RESPAWN_GUARD_*）。
        target = (self._enclosed_route(st,red,head,hard,own)
                  if self.no_boost and not recovering and not guarding else None)
        if target is None and self.no_boost and not recovering and not guarding:
            target=self._continue_pickup(st,red,head,hard,own)
        if target is None and not recovering and not endgame and not guarding:
            target=self.choose_target(st,red,head,hard,own,need)
        # A route to the actual tail is the first survival fallback. During
        # the scoring half, refuse long detours and redundant speed pickups.
        if self.no_boost and not recovering:
            late=st.get('match_elapsed',0.) or 0.
            # Stable >=40 settlement is the goal. A small visual surplus
            # absorbs cell quantisation; below 40 we must still collect -- which is why this gate stays at
            # the goal length and the final-stretch work is done by the trade filter in choose_target.
            protect=secured_length>=83 or guarding
            if target is None or (protect and not target.get('committed')):
                # 刚复活时自己只有 1—5 格、尾巴还没成型，追尾没有意义（用户第三批第 10 条）。
                # 另外短蛇追尾只是浪费时间：第六批第 9 局零死亡、52% 的帧在追尾，长度仍只有 14。
                # 门槛以下交给巡游去开阔地找钻。
                tail_room=max(TAIL_CHASE_AREA,2*length)
                _tight_now=self.tight_area(head,own,getattr(red,'path',None),limit=tail_room+1)
                _open_enough=(TAIL_CHASE_ONLY_WHEN_TIGHT and _tight_now>tail_room)
                tail=(None if (guarding or (_open_enough and not protect) or length<TAIL_FOLLOW_MIN_LENGTH)
                      else self._safe_tail_target(st,red,head,hard,own))
                if tail is not None:
                    target=tail
                elif protect or length<TAIL_FOLLOW_MIN_LENGTH:
                    survival=self._safe_roam_target(st,red,head,hard,own)
                    if survival is None:
                        survival=self._packing_target(red,head,hard,own)
                    if survival is not None:
                        target=survival
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
        # 2026-10-18 实机取证：这里原来带 `not self.no_boost`，而全程都是 no_boost=True（不按 A 加速），
        # 于是追尾 / 叠层脱困**一次都没有执行过**（第 13 轮 tail_chase=0 帧、stack_escape=0 帧）。
        # 用户第 11 局明确要求"吃完加速道具后就该考虑脱困、自己制造出口"，所以去掉这个多余条件。
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
            # Explore a checked route in open space; packing is reserved for
            # lacking an open route. Both planners simulate the moving body.
            target=self._safe_roam_target(st,red,head,hard,own)
            if target is None:target=self._packing_target(red,head,hard,own)
            if target is None:target=self._safe_block_target(st,red,head,hard,own)
        if self.no_boost and target is not None and target.get('pickup') and target['kind']!='sweep':
            self._remember_pickup(st,target)
        if (not self.no_boost and previous_target and previous_target[0] == 'dia'
                and previous_target[2] <= 2
                and (target is None or target['cell'] != tuple(previous_target[1]))):
            # A newly collected gem can add five body cells.  Keep travelling
            # out of its cramped pickup lane before turning toward the next gem.
            self._pickup_clear_until_frame = max(self._pickup_clear_until_frame, frame + 6)
        requested_target=None if target is None else (target['kind'],target['cell'])
        chosen = None
        precision = False
        waypoint = None
        why = 'travel'
        if self.no_boost and self._final_guard and not recovering and target is None:
            # 末段纯逃避的选步：优先沿用当前方向（转弯就是死亡主因），其次跑道最长的方向。
            _cur = self._sent_dir if self._sent_dir in NEIGH else heading
            _straight = next((s for s in viable if s['d'] == _cur and s['runway'] >= 2
                              and s['risk'] < 0.8), None)
            if _straight is not None:
                chosen = _straight
            else:
                chosen = max(viable, key=lambda s: (s['runway'], -s['risk'], s['area'],
                                                    s['straight']))
            target = None
            self.follower.waypoint = None
            why = 'final_guard'
        # ---- 身体顺序不可信时的保守直行（2026-10-18 死亡回放发现的主因）
        # death_replay.py 用真实帧数据判定 129 次死亡的碰撞对象：**71 次（55%）撞自己的身体**，
        # 撞刺球 0 次、撞敌蛇 0 次；墙 1 次。而其中绝大多数死亡前 10 帧的 `path_trust` 都不是 `ordered`
        # —— 身体无法从头到尾排序时，`own_index` 把所有身体格当成永久的墙，空间推理整个失效，
        # 于是"避障"退化成瞎转，转哪撞哪（用户："避障策略较为愚蠢，时常为了避障自杀"）。
        # 身体看不见的时候最安全的动作是不动方向：保持当前航向，只在直行本身不可行时才允许转弯。
        if (self.no_boost and chosen is not None
                and getattr(red, 'path_trust', '') != 'ordered'):
            _fwd = self._sent_dir if self._sent_dir in NEIGH else heading
            _hold = next((s for s in viable if s['d'] == _fwd and s['risk'] < 0.8), None)
            if _hold is not None:
                chosen = _hold
                target = None
                self.follower.waypoint = None
                why = 'impaired_straight'
        # ---- 可用空间不足时保持直行（用户 10-18 逐帧指正）
        # 第 5 轮 f=570—600 实录：长度 26 的蛇在右下角绕成闭环。f=580 画面过旧之后目标丢失，f=582 起
        # 方向从"左"变成"右"（正对自己身体），f=590 撞上、f=599 trapped。
        # 放行右转的是普通选步排序的第一键 `area >= need` —— 而 need 的上限只有 ESCAPE_AREA_CAP=24，
        # 对 26 格的蛇来说，"24 格的活动面积"根本装不下自己，这个判据无法表达"进去就出不来"。
        # 用户："这里的可用空间根本不足以让红蛇脱困"。所以改用物理判据：
        #   所有方向的 flood 面积都 < 蛇长 + HOLD_AREA_MARGIN  ->  保持直行，不再重新规划。
        # （第 7 轮试过"目标丢失就直行"，触发过度导致只长到 15 格，已删掉那条判据。）
        if self.no_boost and chosen is not None and steps:
            _max_area = max(s['area'] for s in steps)
            _body_room = int(length) + HOLD_AREA_MARGIN
            if _max_area < _body_room or _max_area < need:
                _fwd2 = self._sent_dir if self._sent_dir in NEIGH else heading
                _straight2 = next((s for s in viable if s['d'] == _fwd2), None)
                if _straight2 is None:
                    _straight2 = next((s for s in steps if s['d'] == _fwd2), None)
                if _straight2 is not None:
                    chosen = _straight2
                    target = None
                    self.follower.waypoint = None
                    why = 'hold_straight'
        # ---- 方向连续性（用户："方向控制必须是连续的，不能按了左立刻按右"）
        # 摇杆从 -1 瞬切到 +1，在游戏里既不是合法转向，也表现为原地抖动：上一步左、本步右，蛇头只是
        # 在两格之间打摆。
        # 2026-10-18 单局取证（实机记录/round_001）：两次死亡前都出现了裸的方向反转 ——
        #   第一次 t=54.0 dir=上、t=54.1 dir=下；第二次 t=73.1 dir=左、t=73.2 dir=右、t=73.4 dir=左。
        # 之后 5—6 帧 `trapped`（四面被自己封住）并死亡。原来的这条规则**只在缓冲/脱困帧生效**，
        # 普通吃钻帧完全不受约束，所以打摆就发生在正常行驶里。现在改为**所有帧**都生效：
        #   * 剔除与"上一步真正下发过的方向"相反的那一步（还有其他可行步时才剔除）；
        #   * 同一方向至少保持若干帧（缓冲/脱困 3 帧，普通帧 2 帧）才允许改向。
        # 只从既有的可行步里删，不造新方向，也不会让蛇无路可走。
        if self.no_boost and self._sent_dir in NEIGH:
            _sent = self._sent_dir
            _in_buffer = recovering or red.eye_state == 'buffer'
            _filtered = [s for s in viable if s['d'] != OPPOSITE.get(_sent)]
            if _filtered:
                viable = _filtered
            _hold = BUFFER_DIR_HOLD if _in_buffer else NORMAL_DIR_HOLD
            if frame - self._sent_since < _hold:
                _held = [s for s in viable if s['d'] == _sent]
                if _held:
                    viable = _held
        if target is not None and target.get('kind')=='sweep' and len(target['route'])>1:
            # The sweep has just been simulated with growth and tail movement.
            # Heuristic area/width preferences cannot revoke that witnessed
            # first move. Actual occupancy and immediate hazards still can.
            c=target['route'][1]
            checked=next((step for step in steps if step['cell']==c and step['risk']<.8),None)
            if checked is not None and checked not in viable:viable=viable+[checked]
        pending_step=self._pending_turn_step(red,head,hard,own,steps) if not recovering else None
        if pending_step is not None:
            viable=viable+[pending_step]
            self._future_permission=pending_step['cell']
            if target is None:
                route=pending_step['pending_route'];goal=route[-1]
                target=dict(kind='turn',cell=goal,route=route,dist=len(route)-1,
                            pixel=tuple((v+.5)*CELL for v in goal),extra=True)
        if target is not None and not mash:
            # An anticipated turn executes at a FUTURE corner. Its permission
            # must be checked there, not against the unrelated lateral tile
            # beside the captured head. The checked approach itself remains
            # physically passable throughout the reaction interval.
            future=(None if pending_step is not None else
                    self._future_turn_step(target,red,head,hard,own,viable))
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
        if chosen is None and self.no_boost and not recovering and fresh_native:
            witnessed=self._witnessed_fallback(red,head,hard,own,viable)
            if witnessed is not None:
                chosen=witnessed
                target=None
                why='witnessed_escape'
                waypoint=witnessed['turn_corner']
                self._future_permission=witnessed['future_entry']
                viable=viable+[witnessed]
                if chosen['d']!=heading:
                    self.follower.turn_pending=dict(dir=chosen['d'],cell=chosen['turn_cell'],
                        corner=waypoint,at=st.get('t',0.),learn=False,
                        issued_speed=self.follower.speed_s,**{'from':heading})
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
                # 用户选 B（结构性改造）：长蛇的"主动空间管理"从这里开始 —— 体长 ≥ LONG_SNAKE_LEN 时，
                # 把"前方跑道"提到最前面：宁可多走一段空旷直线，也不要为了一个近钻把自己塞进窄缝。
                # 这就是"不要一直内圈到死"的第一层：不是被围住再脱困，而是根本不进入窄区。
                if length >= LONG_SNAKE_LEN:
                    chosen = max(viable,key=lambda s:(s['area']>=need,s['runway']>=3,-s['risk'],
                                                       s['runway'],s['gap'],s['straight'],
                                                       s['area'],-s['wall']))
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
                         route=None if target is None else target['route'],
                         escape=None if target is None else target.get('escape'))
        # ---- 转头次数上限（用户第六批："2 秒内连续三次以上同方向转头"就是绕圈）。
        # 放在选定之后做一次后置校验：一旦这一步会形成同方向第 4 次（脱困态第 5 次）转头，就改选一个
        # 不违反规则的可行步里最好的那个；若没有，宁可保持原方向也不绕圈。
        if chosen is not None:
            _frm = self._sent_dir if self._sent_dir in NEIGH else heading
            _limit = (TURN_REPEAT_LIMIT_RECOVERING if (recovering or red.eye_state == 'buffer')
                      else TURN_REPEAT_LIMIT)
            _t = st.get('t', 0.) or 0.
            # 只在"空间充足"时限制绕圈：被围困时的连续同向转向是折叠脱困，不是绕圈（见 config 的
            # TURN_LIMIT_MIN_AREA：第七批把围困中的折叠也一起禁掉，死亡从 0.40/局 回升到 1.30/局）。
            _room = self.tight_area(head, own, getattr(red, 'path', None),
                                    limit=TURN_LIMIT_MIN_AREA + 1)
            # A checked bend must not be replaced after route validation by
            # a fixed turn-count cap. Dense packing can require four bends.
            self._record_turn(_frm, chosen['d'], _t)
        self.last.update(opponent_heads=self._opponent_heads,policy_body_path=list(red.path),secured_length=secured_length,planning_hard_cells=[tuple(map(int,c)) for c in sorted(hard)],
                         legal_directions=[step['d'] for step in viable],
                         requested_target=(None if requested_target is None else (requested_target[0],tuple(map(int,requested_target[1])))),
                         opponent_memory_cells=sum(map(len,self._opp_memory.values())))
        if (self.no_boost and not recovering and fresh_native and chosen['d']!=heading
                and self.follower.turn_pending is None):
            self._remember_issued_turn(ds['head_px'],heading,chosen['d'],st.get('t',0.))
        # 记录真正下发过的方向：下一帧的"方向连续性"判断以它为准，而不是以观测航向为准
        # （观测航向在缓冲态里本身就在抖）。
        if chosen is not None and chosen['d'] != self._sent_dir:
            self._sent_dir = chosen['d']
            self._sent_since = frame
        # 记录本帧是否有目标：下一帧若目标突然变 None（常见于 stale_frame 后的识别抖动），
        # 兜底逻辑会重新挑一个方向，往往就是转进自己身体的开始（第 5 轮 f=580）。
        self._had_target = target is not None
        return chosen['d'],accel,mash,why
    def _turn_sign(self, frm, to):
        """+1 = 顺时针（屏幕坐标 y 向下），-1 = 逆时针，0 = 直行或反向。"""
        if frm not in NEIGH or to not in NEIGH:
            return 0
        cross = frm[0] * to[1] - frm[1] * to[0]
        if cross > 0:
            return 1
        if cross < 0:
            return -1
        return 0

    def _recent_turns(self, t):
        """当前 2 秒窗口内、与最近一次转头同方向的连续转向次数。"""
        w = float(TURN_WINDOW_SECONDS)
        seg = [x for x in self._turn_seg if t - x[0] <= w]
        self._turn_seg = seg
        return seg

    def _too_many_turns(self, frm, to, t, limit):
        """这一步会不会形成"同方向连续第 limit+1 次转头"。"""
        sign = self._turn_sign(frm, to)
        if sign == 0:
            return False
        seg = self._recent_turns(t)
        same = sum(1 for _t, s in seg if s == sign)
        return same + 1 > limit

    def _record_turn(self, frm, to, t):
        sign = self._turn_sign(frm, to)
        if sign == 0:
            return
        seg = self._recent_turns(t)
        if seg and seg[-1][1] == sign:
            seg.append((t, sign))
        else:
            seg = [(t, sign)]
        self._turn_seg = seg

    def _observed_heading(self,ds,red,st):
        facing = ds['facing']
        if facing not in NEIGH:
            facing = None
        # The stopped collision animation jitters by a few pixels. It cannot
        # establish a new travel direction or acknowledge an escape command.
        if red.eye_state=='buffer':
            previous=self._last_observation
            native=(getattr(red,'head_from_eyes',False) or getattr(red,'head_from_pixels',False))
            displacement=(sum(abs(ds['head_px'][i]-previous[0][i]) for i in (0,1)) if previous else 0.)
            if (not native or not previous or not 0<st.get('t',0.)-previous[1]<=.3
                    or not 5.5<=displacement<=1.5*CELL):
                return self._heading or facing
        px = ds['head_px']
        prev = self._last_observation
        t = st.get('t',0.0)
        heading = facing
        fresh=(not ds.get('head_px_uncertain') and getattr(red,'head_age',99)==0
               and (getattr(red,'head_from_eyes',False) or getattr(red,'head_from_pixels',False)))
        if not fresh:
            return self._heading or facing
        if prev is not None and 0<t-prev[1]<=0.3:
            dx,dy = px[0]-prev[0][0],px[1]-prev[0][1]
            if 2<=abs(dx)+abs(dy)<=CELL*1.5:
                heading = (1 if dx>0 else -1,0) if abs(dx)>abs(dy) else (0,1 if dy>0 else -1)
                pending=self.follower.turn_pending
                if pending and t-pending['at']<=.6 and pending['from']==self._heading:
                    old=pending['from'];wanted=pending['dir']
                    forward=dx*wanted[0]+dy*wanted[1]
                    backwards=dx*old[0]+dy*old[1]
                    if forward>4 and backwards<-2:
                        heading=wanted  # turn snapped back along the old axis
                if heading==OPPOSITE.get(self._heading):
                    self._opposite_motion_count+=1
                    if self._opposite_motion_count<2:
                        heading=self._heading
                else:
                    self._opposite_motion_count=0
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
        remembered=self.body_history.occupied(length)
        body_cells=set(red.cells)|set(remembered)
        if body_cells-set(path):trust='uncertain'

        # A contiguous sequence of ACTUAL head visits preserves order through
        # the boost flash. Its conservative life-length bound does not claim a
        # hidden pixel was seen or that an unseen tail has already disappeared.
        # The bot constructs a new Policy after real respawn, clearing the bound.
        self._witnessed_body_bound=max(self._witnessed_body_bound,int(length))
        count=self._witnessed_body_bound
        trace=list(self.body_history.trace[:count])
        if (getattr(self,'key_boost',False) and trust!='ordered' and not uncertain_px
                and OwnBodyHistory.verified_head(red) and red.eye_state in ('normal','item')
                and len(trace)==count and count>=3 and trace[0]==head
                and len(set(trace))==count and body_cells.issubset(set(trace))
                and all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(trace,trace[1:]))):
            path=trace
            body_cells=set(trace)
            length=max(int(length),count)
            trust='ordered'
            source='conservative_measured_head_trace'

        return dict(head=head,head_px=px,length=int(length),path=path,path_trust=trust,body_cells=body_cells,
                    facing=facing,head_px_uncertain=uncertain_px,body_source=source,
                    body_coverage=estimate.coverage)

    @staticmethod
    def collision_cells(st, memory=None):
        blocked=set()
        for kind,s in st['snakes'].items():
            if kind!='red':
                blocked.update((int(x),int(y)) for x,y in s.cells)
        # 对手身体的前后帧记忆（用户 10-18 指正 + 机制文档 §4.2）
        # "任何加速态都有一道白光从蛇头沿蛇身扫向蛇尾，周期约 15 帧"，被扫到的段会变白、掉出自己的
        # 颜色掩膜，于是对手在掩膜里周期性"消失"。实录（实机记录/round_006）：
        #   f=190 红蛇该向下拐却向上拐撞上紫蛇；f=655 紫蛇正被白光洗白，红蛇头朝上直撞。
        # 对手身体消失不等于可以走 —— 这里把最近若干帧看过的对手格并回来，宁可多避一格。
        if memory:
            for kind, cells in memory.items():
                if kind == 'red' or not cells:
                    continue
                blocked.update(cells)
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
        # 用户 10-18 第 15 局 f=633—637：长度只有 6 的短蛇在三球在场时被 trapped 5 帧后死亡。
        # 机制文档 §8.5：刺球占 **2×2 格**（核心实测 44—60 × 29—43 px），而原来 radius=0.75*CELL
        # 会把球的占位算成 **3×3** —— 三个球合起来多封 15 格，足以把一条 6 格的蛇按死。
        # 收到 2×2（radius 必须 <= 0.5*CELL，否则 (x*32-8)//32 会落到 x-1）。
        radius=.5*CELL
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
                # 用户第七批：10 局里 4 局"撞刺球了"，其中一次明确是"斜向的刺球"。
                # 斜向球的扫掠线每次反弹都会变，靠代价排序躲不住 —— 它周围的格子直接判为不可通行。
                # 横/纵球只扫固定的一条行/列，站开一格就安全，所以不在这里封（那是代价层的事）。
                if mode=='diagonal' and cell is not None:
                    cx,cy=int(cell[0]),int(cell[1])
                    for dx in range(-BALL_DIAG_HARD_RADIUS,BALL_DIAG_HARD_RADIUS+1):
                        for dy in range(-BALL_DIAG_HARD_RADIUS,BALL_DIAG_HARD_RADIUS+1):
                            n=(cx+dx,cy+dy)
                            if in_board(n):
                                blocked.add(n)
        return blocked

    def _risk_map(self,st,hard):
        risk={}
        if self.no_boost:
            # Exact predicted occupancy, not an obstacle halo.  Adjacent cells
            # remain usable, including the lane directly beside another body.
            current=[]
            hug=set()
            self._hug=hug
            now=st.get('t',0.)
            att=st.get('attention')
            centers=getattr(att,'item_centers',st.get('item_centers',{}))
            # 刺球模式（用户机制）：三个球分别是横向、纵向、斜向各一个，只有斜向的要远离。
            # `ball_tracks` 带拟合出来的 mode；按格坐标就近匹配到这一次检测到的球。
            _b=st.get('belief')
            _modes={}
            for _m,_thr,_cell,_ahead in (getattr(_b,'ball_tracks',[]) or []):
                if _cell is not None:
                    _modes[(int(_cell[0]),int(_cell[1]))]=_m
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
                footprint=set()
                # 用户第七批："计算一下蛇头移速，但凡快一点也不会死"。预测窗口从 0.3 s 延长到 0.6 s：
                # 蛇头 0.6 s 能走 3—6 格，球在这个窗口里扫到的格子必须提前算进风险，而不是撞上才发现。
                for dt in (.1,.2,.3,.45,.6):
                    predicted=tuple(p[i]+velocity[i]*dt for i in (0,1))
                    for c in self.ball_footprint(predicted):
                        footprint.add(c)
                        if c not in hard:
                            risk[c]=2.0
                # No fixed adjacent-cell risk ring. In r059, that legacy ring
                # marked (9,16) dangerous although the moving sphere could not
                # hit the queued turn there, canceling RIGHT and injecting DOWN.
                # Current/predicted footprints remain here; actual head sweeps
                # and timed future-corner contact use _ball_hit_at below.
            self._ball_motion=current+[old for old in unused if 0<=now-old['t']<=.25]
            self._opponent_heads=self._opponent_native.update(st['snakes'],now)
            for kind,s in st['snakes'].items():
                if kind=='red' or not s.head_cell:
                    continue
                facing=getattr(s,'facing',None)
                point=getattr(s,'head_px',None) or tuple((v+.5)*CELL for v in s.head_cell)
                native=self._opponent_heads.get(kind)
                if native is not None:
                    point=tuple((v+.5)*CELL for v in native['cell'])
                    facing=native['direction']
                    for c,value in possible_crossings(native,set(s.cells),hard).items():
                        risk[c]=max(risk.get(c,0.),value)

                head_cell=native['cell'] if native else s.head_cell
                old=self._opponent_motion.get(kind)
                velocity=(0.,0.)
                if old is not None and .025<now-old['t']<=.35:
                    dt=now-old['t']
                    delta=tuple(point[i]-old['p'][i] for i in (0,1))
                    if sum(abs(v) for v in delta)<=3*CELL:
                        velocity=tuple(.5*old['v'][i]+.5*delta[i]/dt for i in (0,1))
                if native is not None:
                    # Grid centroids stay still, then jump 32px. Differencing
                    # each control frame makes a steady 142px/s head oscillate
                    # 73 -> 262 -> 131px/s (r062 f557--559), abandoning a safe
                    # near gem for one frame. Native speed is measured between
                    # changed tiles; use that interval and witnessed heading.
                    velocity=tuple(native['speed']*v for v in native['direction'])
                self._opponent_motion[kind]=dict(p=point,t=now,v=velocity)
                for dt in (.2,.4,.6):
                    predicted=tuple(int((point[i]+velocity[i]*dt)//CELL) for i in (0,1))
                    if in_board(predicted) and predicted not in hard:
                        risk[predicted]=2.0
                if facing in NEIGH:
                    # 用户第五批第 1 局："适当预测一下别的蛇，不要在吃钻的路上一直被别的蛇挡着还最后
                    # 撞死了"。原来只标出敌蛇头前方 1 格，而它的头在下一两个控制帧就会推进 1—2 格，
                    # 于是"看着有空、走过去正好撞上"。这里沿它的朝向把 3 格都按递减代价标出来：
                    # 第 1 格 2.0（等同障碍），第 2 格 1.5，第 3 格 1.0（都会被 0.8 的 low-risk 过滤
                    # 挑出来，但仍是最优路线的候选，不会把整条通道封死）。
                    for k in (1,2,3):
                        n=(head_cell[0]+facing[0]*k,head_cell[1]+facing[1]*k)
                        if in_board(n) and n not in hard:
                            risk[n]=max(risk.get(n,0.),{1:2.0,2:1.5,3:1.0}[k])
                # Running alongside an opponent's body: the user reported it three matches in a row
                # ("不要一直贴着别的蛇的蛇身走").  Collected SEPARATELY from `risk`, because `risk` also
                # gates item approaches through `_first_allowed`; a 0.9 ring in `risk` made every gem that
                # sat beside an enemy body unapproachable and the snake thrashed between far gems with no
                # growth at all for a whole match (live 19:04, M1: peak 7 in 60 s).  `_hug` is applied to
                # the candidate steps only after `_first_allowed` has been fixed, so it re-ranks moves
                # without taking targets off the board.
                for c in s.cells:
                    for dx,dy in NEIGH:
                        n=(int(c[0])+dx,int(c[1])+dy)
                        if in_board(n) and n not in hard:
                            hug.add(n)
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
        # 叠层脱困（用户 10-18 第 11 局 f=430—440 的说明）：
        # "吃完加速道具后就应该考虑脱困，比起现成的、不确定的出口，选择自己制造出口：蛇头应该在 440 帧
        #  左右下拐，然后沿 (27,5)—(26,5)—(26,0)—(27,0)—(27,6) 从蛇尾让开的位置脱困。"
        # 也就是：当**可用的紧凑空间还装不下自己的身体**时，就不要赌"某个方向也许有出口"，
        # 而是把目标设成自己的尾巴 —— 沿着身体边缘平行叠层走过去，让尾巴在移动中把通道腾出来。
        # 原来的门槛是固定的 TAIL_CHASE_AREA=10，对 20 格以上的蛇基本不开放，所以这项战术从没生效过。
        _stacked = (length >= STACK_ESCAPE_MIN_LEN and tight < length + STACK_ESCAPE_MARGIN)
        if length < TAIL_CHASE_MIN_LEN or not (tight <= TAIL_CHASE_AREA or _stacked):
            return None
        if _stacked:
            self.last['stack_escape'] = True
        path = getattr(red, 'path', None) or []
        if len(path) < 2:
            return None
        tail = path[-1]
        return (int(tail[0]), int(tail[1]))

    def _route_food(self,st):
        food={}
        # Use the same authenticated item candidates as ordinary collection.
        for kind,value,c,_bonus in self._candidates(st,set()):
            food[c]=(5 if kind=='dia' else 1 if kind=='bean' else 0,value,kind)
        return food

    def _post_pickup_spacing(self,st,path,segment):
        # A route planned BEFORE a green pickup must also be executable at
        # the speed AFTER it. This is a turn timing constraint, not padding
        # around obstacles. An already active effect is not multiplied twice.
        if st.get('item_accel_until',0)>st.get('t',0):
            return segment
        if not any(self._food.get(c,(0,0,None))[2]=='grn' for c in path[1:]):
            return segment
        speed=max(120.,self.follower.route_speed_s)*(1.25 if getattr(self,'key_boost',False) else 1.8)
        future=max(1,min(3,math.ceil(max(0.,speed*self.follower.lead_s)/CELL)))
        return max(segment,future)

    def _enclosed_route(self,st,red,head,hard,own):
        hard=self._planning_blocked(head,hard)
        ordered=red.path_trust=='ordered' and red.path and tuple(red.path[0])==head
        if not ordered or red.length<8:
            self._sweep=None
            return None
        segment=max(1,min(3,math.ceil(max(0.,self.follower.route_speed_s*self.follower.lead_s)/CELL)))
        saved=self._sweep;self._sweep=None
        route=None
        if saved and head in saved['route'] and segment<=saved['segment']:
            candidate=saved['route'][saved['route'].index(head):]
            if len(candidate)>1:
                first=(candidate[1][0]-head[0],candidate[1][1]-head[1])
                valid=(first!=OPPOSITE.get(self._heading) and self._passable(candidate[1],hard,own,1)
                       and not any(c in hard for c in candidate[1:])
                       and not any(self._risk.get(c,0.)>=.8 for c in candidate[1:4])
                       and walk(red.path,candidate,self._geometry_growth,self._food) is not None)
                if valid: route=candidate
        if route is None:
            region=pocket(head,red.path,hard,max(16,red.length+self._geometry_growth+6))
            if region is None:
                return None
            initial=0
            if self._heading in NEIGH:
                axis=0 if self._heading[0] else 1
                to_center=((head[axis]+.5)*CELL-red.head_px[axis])*self._heading[axis]
                initial=max(0,math.ceil((self.follower.speed_s*self.follower.lead_s-8-to_center)/CELL))
            blocked=set(hard)|{c for c,v in self._risk.items() if v>=.8 and
                                  max(abs(c[i]-head[i]) for i in (0,1))<=3}
            full=plan_sweep(red.path,self._heading,blocked,region,self._food,
                            self._geometry_growth,segment,initial)
            if not full or len(full)<3 or not self._passable(full[1],hard,own,1):
                return None
            exit_index=next((i for i,c in enumerate(full) if c not in region),len(full)-1)
            # Preserve the folded route until the tail-created exit, then
            # another complete leg. Do not turn it back into per-frame greed.
            route=full[:min(len(full),exit_index+segment+2)]
        goal=route[-1]
        self._sweep=dict(route=list(route),segment=segment)
        self._committed_route=None
        target=dict(kind='sweep',cell=goal,route=route,dist=len(route)-1,
                    pixel=tuple((v+.5)*CELL for v in goal),committed=True,extra=True)
        # Item is an intermediate point of the escape, not a reason to stop
        # following it. Feed its real centroid to the corner controller.
        for i,c in enumerate(route):
            item=self._food.get(c)
            if item and item[2] in ('dia','bean'):
                target['pickup']=(c,self._item_pixel(st,item[2],c))
                target['dist']=i
                break
        return target

    def _remember_pickup(self,st,target):
        if target.get('committed'):
            return
        reach=max(3,math.ceil(self.follower.route_speed_s*self.follower.lead_s/CELL)+1)
        if target['dist']>reach:
            return
        route=target['route']
        index=route.index(target['cell'])
        segment=max(1,min(3,math.ceil(max(0.,self.follower.route_speed_s*self.follower.lead_s)/CELL)))
        # Retain the checked exit's upcoming bends, not only two straight
        # cells after the item. A truncated straight hides its exit turn
        # until the head reaches the endpoint and is already too late.
        stop=min(len(route)-1,index+max(8,2*segment+2))
        stop=max(stop,min(len(route)-1,target.get('chain_stop',0)))
        if stop<=index:
            return
        saved=dict(target)
        saved['route']=list(route[:stop+1])
        saved['drive_pixel']=tuple((v+.5)*CELL for v in route[stop])
        saved['committed']=True
        saved['commit_at']=st.get('t',0.)
        self._committed_route=saved

    def _continue_pickup(self,st,red,head,hard,own):
        hard=self._planning_blocked(head,hard)
        saved=self._committed_route
        if saved is None:
            return None
        self._committed_route=None
        # Final stretch while already long: a route committed earlier is not worth the risk of finishing,
        # and `choose_target` has already stopped producing new far targets on its own.
        if self._late_strict and self._late_hold:
            return None
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
            if walk(red.path,route,self._geometry_growth,self._food) is None:
                return None
        elif any(own.get(c,-1)>=i for i,c in enumerate(route[1:],1)):
            return None
        saved=dict(saved,route=route,dist=route.index(saved['cell']) if saved['cell'] in route else 0)
        if saved['cell'] not in route:
            for c in route:
                item=self._food.get(c)
                if item is not None:
                    saved['pickup']=(c,self._item_pixel(st,item[2],c))
                    break
        self._committed_route=saved
        return saved

    def _approach(self,key,head,px,heading,hard,own,speed,delay,*args,**kwargs):
        return self.plans.approach(key,head,px,heading,hard,own,speed,delay,*args,
            pending=self.follower.turn_pending,geometry_speed=self.follower.route_speed_s,**kwargs)

    def _planning_blocked(self,head,hard):
        # r036: a native purple tip will cross (21,12) while our route
        # commits to a U-turn through it. Use the same near-term occupancy
        # evidence for the entire approach AND its exit, not only the next
        # neighbour. Adjacent body lanes remain unpadded.
        return set(hard)|{c for c,v in self._risk.items() if v>=.8
                          and max(abs(c[i]-head[i]) for i in (0,1))<=4}

    def _packing_target(self,red,head,hard,own):
        hard=self._planning_blocked(head,hard)
        segment=max(1,min(3,math.ceil(max(0.,self.follower.route_speed_s*self.follower.lead_s)/CELL)))
        ordered=red.path_trust=='ordered' and red.path and tuple(red.path[0])==head
        if ordered:
            body=red.path;blocked=hard;growth=self._geometry_growth
        else:
            body=[head];blocked=set(hard)|set(red.cells);blocked.discard(head)
            # The unobserved old body remains solid, while the NEW trail
            # cannot disappear before a full body length has passed.
            growth=max(0,red.length-1)+self._geometry_growth
        blocked=set(blocked)|{c for c,value in self._risk.items() if value>=.8
                             and max(abs(c[i]-head[i]) for i in (0,1))<=4}
        initial=0
        if self._heading in NEIGH:
            axis=0 if self._heading[0] else 1
            to_center=((head[axis]+.5)*CELL-red.head_px[axis])*self._heading[axis]
            initial=max(0,math.ceil((self.follower.speed_s*self.follower.lead_s-8-to_center)/CELL))
        route=None
        saved=self._packing_saved
        if saved and head in saved['route'] and segment<=saved['segment']:
            candidate=saved['route'][saved['route'].index(head):]
            if (len(candidate)>=3 and not any(c in blocked for c in candidate[1:])
                    and (candidate[1][0]-head[0],candidate[1][1]-head[1])!=OPPOSITE.get(self._heading)
                    and walk(body,candidate,growth,self._food) is not None):route=candidate
        if route is None:
            route=(plan_sweep(body,self._heading,blocked,{head},self._food,growth,segment,initial)
                   if ordered else None)
        if route is None:
            route=packing_escape(body,self._heading,blocked,growth,segment,initial,partial=True)

        if not route or len(route)<3 or route[1] not in self._first_allowed:
            return None
        if walk(body,route,growth,self._food) is None:
            return None
        # Preserve the witnessed trajectory across frames; revalidate it above.
        self._packing_saved=dict(route=list(route),segment=segment)
        route=route[:9];goal=route[-1]
        return dict(kind='packing',cell=goal,route=route,dist=len(route)-1,
                    pixel=tuple((v+.5)*CELL for v in goal),extra=True)

    def _safe_tail_target(self, st, red, head, hard, own):
        hard=self._planning_blocked(head,hard)
        if red.path_trust!='ordered' or len(red.path)<8:
            return None
        previous=self.last.get('target')
        goal=self._tail_goal if previous and previous[0]=='tail' else None
        reached=goal==head and max(abs(red.head_px[i]-(head[i]+.5)*CELL) for i in (0,1))<=.4*CELL
        if goal is None or reached or goal in hard:
            goal=tuple(red.path[-1])
        self._tail_goal=goal
        path=self._approach(('tail',goal),head,red.head_px,self._heading,hard,own,
            self.follower.speed_s,self.follower.lead_s,red.length+self._geometry_growth,self._risk)
        if not path or (len(path)>1 and path[1] not in self._first_allowed):
            return None
        moved=walk(red.path,path,self._geometry_growth,self._food)
        if moved is None:
            return None
        segment=max(1,min(3,math.ceil(max(0.,self.follower.route_speed_s*self.follower.lead_s)/CELL)))
        escape=self._pickup_escape(path,red,hard,0,segment=segment,require_tail=True,
                                   debt=self._geometry_growth,return_route=True,food=self._food)
        if not escape:
            return None
        route=path+escape[1:1+ESCAPE_ROUTE_CELLS]
        if len(route)<2 or route[1] not in self._first_allowed:
            return None
        return dict(kind='tail',cell=goal,route=route,dist=len(path)-1,
                    pixel=((goal[0]+.5)*CELL,(goal[1]+.5)*CELL),
                    drive_pixel=tuple((v+.5)*CELL for v in route[-1]),extra=True)

    def _remember_issued_turn(self,point,heading,direction,now):
        if heading not in NEIGH or direction==heading or direction==OPPOSITE[heading]:return
        projected=tuple(point[i]+heading[i]*self.follower.speed_s*self.follower.lead_s for i in (0,1))
        cell=tuple(max(0,min((COLS,ROWS)[i]-1,int(projected[i]//CELL))) for i in (0,1))
        corner=tuple((v+.5)*CELL for v in cell)
        self.follower.turn_pending=dict(dir=direction,cell=cell,corner=corner,at=now,
            learn=False,issued_speed=self.follower.speed_s,**{'from':heading})

    def _witnessed_fallback(self,red,head,hard,own,viable):
        """Compare real future turn entries with a full body-turnover witness.

        A forecast beyond the next solid wall cannot be the turn corner.
        The last free lane before it is the only remaining physical exit.
        """
        heading=self._heading
        if (heading not in NEIGH or red.length<12 or self.follower.turn_pending is not None):
            return None
        ordered=red.path_trust=='ordered' and red.path and tuple(red.path[0])==head
        if ordered:
            original=red.path
            blocked=hard
            debt=self._geometry_growth
        else:
            original=[head]
            blocked=set(hard)|set(red.cells)
            blocked.discard(head)
            debt=max(0,red.length-1)+self._geometry_growth
        speed=max(100.,self.follower.speed_s)
        projected=tuple(red.head_px[i]+heading[i]*speed*self.follower.lead_s for i in (0,1))
        axis=0 if heading[0] else 1
        index=max(0,min(4,(int(projected[axis]//CELL)-head[axis])*heading[axis]))
        prefix=[head]
        for i in range(1,index+1):
            cell=(head[0]+heading[0]*i,head[1]+heading[1]*i)
            if not in_board(cell) or not self._passable(cell,blocked,own,i):break
            prefix.append(cell)
        corner=prefix[-1]
        segment=max(1,min(3,math.ceil(self.follower.route_speed_s*self.follower.lead_s/CELL)))
        options=[]
        for step in viable:
            d=step['d']
            if d==OPPOSITE.get(heading):continue
            path=prefix+[(corner[0]+d[0]*i,corner[1]+d[1]*i) for i in range(1,segment+1)]
            if any(not in_board(c) or c in blocked for c in path[1:]):continue
            moved=walk(original,path,debt,self._food)
            if moved is None:continue
            distance=0.;previous=red.head_px;collision=False
            for c in path[1:]:
                point=tuple((v+.5)*CELL for v in c)
                length=sum(abs(point[i]-previous[i]) for i in (0,1))
                count=max(1,int(math.ceil(length/8.)))
                for k in range(1,count+1):
                    q=tuple(previous[i]+(point[i]-previous[i])*k/count for i in (0,1))
                    if self._ball_hit_at(q,(distance+length*k/count)/speed):
                        collision=True;break
                if collision:break
                distance+=length;previous=point
            if collision:continue
            body,growth,_=moved
            proof=packing_escape(body,d,blocked,growth,segment,depth=None)
            if proof is None:continue
            option=dict(step)
            entry=(corner[0]+d[0],corner[1]+d[1])
            option.update(cell=entry,future_entry=entry,turn_corner=tuple((v+.5)*CELL for v in corner),
                          turn_cell=corner,proof_length=len(proof))
            options.append(option)
        return max(options,key=lambda s:(-s['risk'],s['straight'],s['runway'],s['area'])) if options else None

    def _ball_hit_at(self,point,after=0.):
        now=self.follower.previous_t or 0.
        for ball in self._ball_motion:
            age=max(0.,now-ball['t'])+after
            center=tuple(ball['p'][i]+ball['v'][i]*age for i in (0,1))
            # r066 native frames660--669: dark spiked contour reaches34--35px
            # from the measured core centroid. Radius28 clipped the spikes,
            # so a held-item-speed head was sent past them before turning.
            # Include the rendered 33px head and spiked ball outline; this is
            # measured pixel geometry, not an extra empty grid-cell margin.
            dx=max(0.,abs(point[0]-center[0])-16.5)
            dy=max(0.,abs(point[1]-center[1])-16.5)
            if dx*dx+dy*dy<=35.5*35.5:return True
        return False

    def _ball_swept_hit(self,point,d,distance):
        if not self._ball_motion:return False
        speed=max(80.,self.follower.speed_s)
        for travel in range(0,int(math.ceil(distance))+4,4):
            predicted=(point[0]+d[0]*travel,point[1]+d[1]*travel)
            if self._ball_hit_at(predicted,travel/speed):return True
        return False

    def _pending_turn_step(self,red,head,hard,own,steps):
        pending=self.follower.turn_pending
        if not pending or self._heading!=pending.get('from') or not steps:return None
        if not 0<=self.follower.previous_t-pending['at']<=min(.32,self.follower.lead_s+.10):return None
        corner=tuple(pending['cell']);h=self._heading;d=pending['dir']
        distance=(corner[0]-head[0])*h[0]+(corner[1]-head[1])*h[1]
        if not 0<=distance<=3 or (corner[0]-head[0])*h[1]!=(corner[1]-head[1])*h[0]:return None
        path=[(head[0]+i*h[0],head[1]+i*h[1]) for i in range(distance+1)]
        entry=(corner[0]+d[0],corner[1]+d[1]);path.append(entry)
        if self._risk.get(entry,0.)>=.8 or self._ball_hit_at(tuple((v+.5)*CELL for v in entry),
                (distance+1)*CELL/max(80.,self.follower.speed_s)):return None
        if any(not self._passable(c,hard,own,i) for i,c in enumerate(path[1:],1)):return None
        if red.path_trust=='ordered' and walk(red.path,path,self._geometry_growth,self._food) is None:return None
        result=max(steps,key=lambda s:s['area']).copy()
        result.update(d=d,cell=entry,risk=self._risk.get(entry,0.),straight=0,pending_route=path)
        return result

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
        if (not(-.25*CELL<=remaining<=lead) or self._risk.get(entry,0)>=.8
                or self._ball_hit_at(tuple((v+.5)*CELL for v in entry),(j+1)*CELL/max(80.,self.follower.speed_s))):
            return None
        if any(not self._passable(c,hard,own,i) for i,c in enumerate(route[1:j+2],1)):
            return None
        template=max(viable,key=lambda s:s['area']).copy()
        template.update(d=dirs[j],cell=entry,risk=self._risk.get(entry,0),straight=0)
        return template

    def _safe_roam_target(self, st, red, head, hard, own):
        hard=self._planning_blocked(head,hard)
        # Keep an executable multi-cell exit plan while vision cannot prove a
        # tail, instead of choosing an isolated neighbour by flood area.
        dist,_=self.route(head,hard,own)
        goals=[c for c,d in dist.items() if 7<=d<=16]
        h=self._heading or (0,0)
        # User, third and fourth batches: "总在角落转圈缺乏对外探索", "中期在绕圈无收益，然后撞到角落没有
        # 正确拐出".  The old order only asked for "about 11 cells away, preferably forwards", which is
        # satisfied perfectly by a circle in a corner.  Distance to the nearest wall now comes first among
        # the same-radius goals, so roaming drifts toward open floor instead of orbiting the rim.
        # 用户第五批："缺少吃钻"、"富贵险中求"。MEASURED 第九批：28 格的一局里 44% 的帧在吃钻，其余
        # 56% 在"找空间"空转 —— 15 格内没有合格钻石时就完全不再朝钻石移动了。巡游现在也把"离最近钻石
        # 更近"当作一个排序条件，于是空转时间会主动朝下一颗钻石推进，而不是在开阔地绕圈。
        att = st.get('attention')
        # Collection and staging must use the same authenticated targets.
        # gem_best also contains src='extra' for diagnostics; roaming toward
        # every entry bypassed the native/shape checks used by choose_target.
        gems = [tuple(c) for kind,_value,c,_bonus in self._candidates(st,hard)
                if kind=='dia']

        def _gem_gap(c):
            if not gems:
                return 99
            return min(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in gems)

        collecting=bool(gems and min(red.length,int(red.n_px/256))<83)
        if collecting:
            # Stage along an actual executable path around intervening bodies.
            # Manhattan closeness alone can keep choosing the wrong side of a
            # long wall made by another snake, without ever reaching the gem.
            for gem in sorted(gems,key=lambda c:abs(c[0]-head[0])+abs(c[1]-head[1]))[:3]:
                full=self._approach(('stage',gem),head,red.head_px,self._heading,hard,own,
                    self.follower.speed_s,self.follower.lead_s,red.length+self._geometry_growth,self._route_risk)
                if not full or len(full)<7 or not self._pickup_fits_clock(st,len(full)-1):continue
                stop=min(11,len(full)-2)
                while stop>=5 and full[stop] in self._food:stop-=1
                if stop<5:continue
                prefix=full[:stop+1]
                segment=max(1,min(3,math.ceil(self.follower.route_speed_s*self.follower.lead_s/CELL)))
                if prefix[1] not in self._first_allowed:continue
                escape=self._pickup_escape(prefix,red,hard,0,
                    segment=self._post_pickup_spacing(st,prefix,segment),require_tail=red.length>=12,
                    debt=self._geometry_growth,food=self._food,return_route=True)
                if escape:return self._roam_with_exit(prefix,escape,segment)
        if collecting:
            # r036 spent 236 frames roaming while the fixed gems were beyond
            # the 15-cell pickup budget. Stage toward an observed gem with a
            # checked 7--16-cell route, rather than orbiting an old central goal.
            goals.sort(key=lambda c:(_gem_gap(c),abs(dist[c]-11),-self._wall_distance(c)))
        else:
            goals.sort(key=lambda c:(abs(dist[c]-11),-self._wall_distance(c),
                                     -((c[0]-head[0])*h[0]+(c[1]-head[1])*h[1])))
        if (self._roam_goal in dist and 3<=dist[self._roam_goal]<=25
                and (not collecting or _gem_gap(self._roam_goal)<=_gem_gap(head)-3)):
            goals=[self._roam_goal]+[g for g in goals if g!=self._roam_goal]
        for goal in goals[:5]:
            path=self._approach(('roam',goal),head,red.head_px,self._heading,hard,own,
                self.follower.speed_s,self.follower.lead_s,red.length+self._geometry_growth,self._risk)
            if not path or len(path)<2 or path[1] not in self._first_allowed:continue
            segment=max(1,min(3,math.ceil(self.follower.route_speed_s*self.follower.lead_s/CELL)))
            escape=self._pickup_escape(path,red,hard,self._food.get(goal,(0,))[0],
                segment=self._post_pickup_spacing(st,path,segment),require_tail=red.length>=12,
                debt=self._geometry_growth,food=self._food,return_route=True)
            if escape:
                self._roam_goal=goal
                return self._roam_with_exit(path,escape,segment)
        return None

    def _roam_with_exit(self,path,escape,segment):
        # r053: the old roam ended AT (3,24), hiding its necessary up turn.
        # Supply checked bends beyond this non-food waypoint so they can be
        # issued before arrival. Retain the full proof as a revalidated fallback.
        complete=list(path)+list(escape[1:])
        self._packing_saved=dict(route=complete,segment=segment)
        route=list(path)+list(escape[1:1+max(4,2*segment+2)])
        goal=path[-1]
        return dict(kind='roam',cell=goal,route=route,dist=len(path)-1,
                    pixel=tuple((v+.5)*CELL for v in goal),
                    drive_pixel=tuple((v+.5)*CELL for v in route[-1]),extra=True)

    def _loose_gem_target(self, st, red, head, hard, own, need):
        """兜底收集：`choose_target` 全部判不合格之后，仍值得去抢的近处钻石。

        MEASURED 第八批：零死亡的两局都停在 37 格，而 `eat_packing`（纯找空间、不产生长度）占了 35%
        的时间。原因是 `choose_target` 的出口见证检查很严，不少"能吃到、吃完也走得掉"的近钻被拒绝；
        用户第五批也提过"富贵险中求，但是要准"。这一层只要求：路线在 15 格内、第一步合法、前 4 步没有
        硬障碍或高风险格；不要求完整出口见证。它只在"确实没有合格目标"时生效，排在安全兜底之前。
        """
        if not self.no_boost:
            return None
        cands = self._candidates(st, hard)
        if not cands:
            return None
        dist, parent = self.route(head, hard, own)
        best = None
        for kind, val, cell, bonus in cands:
            if kind != 'dia' or cell not in dist:
                continue
            d = dist[cell]
            direct = abs(cell[0] - head[0]) + abs(cell[1] - head[1])
            if d > DIAMOND_MAX_ROUTE and direct > DIAMOND_NEAR_DIRECT:
                continue
            if self._risk.get(cell, 0.) >= 0.8:
                continue
            path = self._route_cells(parent, cell)
            if len(path) < 2 or path[1] not in (self._first_allowed or set()):
                continue
            if any(c in hard or self._risk.get(c, 0.) >= 0.8 for c in path[1:5]):
                continue
            score = val / (d + ITEM_FIXED_COST[kind])
            if best is None or score > best[0]:
                best = (score, dict(kind=kind, cell=cell, route=path, dist=len(path) - 1,
                                    pixel=self._item_pixel(st, kind, cell), loose=True))
        return best[1] if best else None

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
            path=self._approach(('block',goal),head,red.head_px,self._heading,hard,own,
                self.follower.speed_s,self.follower.lead_s,red.length,self._risk)
            if not path or len(path)<2 or path[1] not in self._first_allowed:
                continue
            our_eta=(len(path)-1+.6)*CELL/max(120.,self.follower.speed_s)
            # Assume a fast opponent (ten cells/second); never race its head.
            if our_eta+.12>=.4:
                continue
            if self._pickup_escape(path,red,hard,0,require_tail=True,food=self._food):
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

    def _block_cell_for_gem(self, st, head, gem):
        """The cell in front of an opponent head, when `gem` sits next to that head. -> cell or None.

        User: "钻石在敌蛇旁不仅想办法要抢还要顺便把敌蛇蛇头堵住".  Only used to PREFER an approach that
        already covers that cell (and to make the cell free to cross); it never redirects the snake away
        from the gem and never creates a stand-alone blocking errand, which is what made the old
        `OPP_HEAD_BLOCK` series worse (see config).
        """
        best = None
        for kind, s in (st.get('snakes') or {}).items():
            if kind == 'red':
                continue
            oh = getattr(s, 'head_cell', None)
            if not oh:
                continue
            oh = (int(oh[0]), int(oh[1]))
            near = max(abs(gem[0] - oh[0]), abs(gem[1] - oh[1]))
            if near > BLOCK_GEM_NEAR:
                continue
            facing = getattr(s, 'facing', None)
            if not facing or (facing[0] == 0 and facing[1] == 0):
                dx, dy = gem[0] - oh[0], gem[1] - oh[1]
                if dx == 0 and dy == 0:
                    continue
                facing = (1 if dx > 0 else -1, 0) if abs(dx) >= abs(dy) else (0, 1 if dy > 0 else -1)
            cell = (oh[0] + facing[0] * OPP_BLOCK_AHEAD, oh[1] + facing[1] * OPP_BLOCK_AHEAD)
            if not in_board(cell):
                continue
            score = max(abs(gem[0] - cell[0]), abs(gem[1] - cell[1]))
            if best is None or score < best[0]:
                best = (score, cell)
        return best[1] if best else None

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
                    # 速度约 1.8 倍，等于单位时间能多够到更多钻石。之前 in-late 直接压到 1.5，几乎等于
                    # 放弃它；用户第五批三局都提到加速道具，这里恢复成"仍然值得绕一点"，
                    # 只有已经在加速状态时才降权（避免连着追第二颗）。
                    boosted=st.get('item_accel_until',0) > st.get('t',0)
                    if boosted:continue  # incidental greens remain collectible; no detour or second speed multiplier
                    val = GREEN_LATE_VALUE if late else GREEN_NORMAL_VALUE
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
        hard=self._planning_blocked(head,hard)
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
            if self.no_boost and not self._pickup_fits_clock(st,d):
                continue
            # With a long body, a target on the other side of our own coil can
            # require a huge loop.  Refreshing that route each frame makes the
            # snake orbit the gem and wastes the match; choose another item.
            direct=abs(cell[0]-head[0])+abs(cell[1]-head[1])
            # ---- what the final stretch may still be spent on (see LATE_GUARD_SECONDS in config).
            # A green adds no length and a bean adds one cell; measured, those two were the last target
            # before four of the twelve deaths in the 2026-10-17 evening batch, all inside the last 15 s.
            if self.no_boost and kind=='grn' and direct>GREEN_MAX_DIRECT:
                continue
            if self._late:
                if kind=='grn' and self._late_strict:
                    # 只有最后 6 秒才彻底放弃加速道具：它不加长度，而末段要的是少犯错。45—54 秒仍然抢
                    # （用户第五批："缺少获取加速道具"、"富贵险中求，但是要准"），安全性由取物后的出口
                    # 检查与斜向球环保证。
                    continue
                if kind=='bean' and (self._late_hold or direct>LATE_BEAN_DIST):
                    continue
                if kind=='dia' and self._late_strict and self._late_hold and direct>LATE_STRICT_DIAMOND_DIRECT:
                    continue
                if d==0:
                    # An item reported ON the head cell has no route to steer; keeping it as the target
                    # only blocks the survival fallbacks below (three late deaths held exactly this).
                    continue
            if kind=='dia' and d>DIAMOND_MAX_ROUTE and direct>DIAMOND_NEAR_DIRECT:
                # User: "钻石路线长度上限设到15格".  Measured in the second batch: six of the eight
                # early/mid deaths had a 15-28 cell gem ROUTE as their last target, so the old
                # `d>max(20,direct+14)` let exactly those errands through.  The allowance keeps a CLOSE
                # gem reachable when an obstacle inflates its route (a spike ball covers 3x3 cells and
                # turned a 12-cell gem into a 16-cell route); what it still refuses is a target that is
                # far from the head in its own right.
                continue
            if self.no_boost and d==0:
                # A target reported ON the head cell has no route to steer (`first=head`), so it can only
                # occupy the target slot.  The third batch showed this is not a last-15s problem: four of
                # the thirteen deaths (M2 34.4 s, M6 21.1 s, M7 49.7 s bean, M10 54.2 s) held exactly such
                # a target, and the survival branches could not take over while it sat there.  The snake
                # still crosses the item's centre on its current heading, so giving up the slot does not
                # give up the pickup.
                continue
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
            score=val/(d+ITEM_FIXED_COST[kind]) if self.no_boost else val/(self._route_cost[cell]+ITEM_FIXED_COST[kind])+bonus
            scored.append(dict(kind=kind,cell=cell,dist=d,first=path[1] if d else head,
                               area=area,score=score,route=path,pixel=pixel,value=val,bonus=bonus))
        if not scored:
            # 这一层是"粗筛无候选"的早退。它原来直接清空 _target，导致后面那层跨帧粘性永远看不到
            # 上一帧的目标（实测 kept_target=0 帧）。两条早退都先试着留住原目标。
            keep = self._keep_previous_target(st, red, head, hard, own)
            if keep is not None:
                return keep
            self._target=None
            return None
        # Check the body AFTER the complete approach and growth, not merely
        # the first eight cells or one empty cell beside the pickup.
        if self.no_boost:
            self._candidate_review=[]
            scored.sort(key=lambda s:s['score'], reverse=True)
            # Reserve validation slots for both fixed diamonds and a green
            # item. Several nearby beans must not push a nearer diamond out
            # of the bounded planning budget before it is even checked.
            diamonds=sorted((s for s in scored if s['kind']=='dia'),key=lambda s:s['dist'])
            greens=[s for s in scored if s['kind']=='grn']
            priority=diamonds[:2]+greens[:1]
            scored=priority+[s for s in scored if s not in priority]
            accepted=[]
            segment=max(1,min(3,math.ceil(max(0.,self.follower.route_speed_s*self.follower.lead_s)/CELL)))
            for candidate in scored[:5]:
                # 顺手堵敌蛇头: if this gem sits next to an opponent head, the cell in front of that head
                # costs nothing to cross for THIS approach, so the checked route naturally presses onto it
                # while still finishing at the gem.  Per-candidate map, never a global change.
                risk_map=self._route_risk
                block_cell=(self._block_cell_for_gem(st,head,candidate['cell'])
                            if candidate['kind']=='dia' else None)
                if block_cell is not None and self._route_risk.get(block_cell,0.)>0.:
                    if risk_map is self._route_risk:
                        risk_map=dict(self._route_risk)
                    risk_map[block_cell]=0.
                path=self._approach((candidate['kind'],candidate['cell']),head,red.head_px,
                    self._heading,hard,own,self.follower.speed_s,self.follower.lead_s,
                    length=red.length+self._geometry_growth,risk=risk_map)
                # 接近与选步分层（用户 10-18 第 10 局逐帧评价）
                # 原来第一步必须落在 `_first_allowed` 里 —— 那是经过活动面积、方向连续性筛过之后很窄的
                # 一组方向，于是**侧面/后面的钻石一律被判 approach 拒绝**。实录：
                #   f=345 钻石 (3,5) 就在头 (5,7) 旁 3 格，目标却是 11 格外的 roam(7,10)；
                #   f=585 钻石 (11,7)/(10,6) 就在近处，目标却是 roam(20,7)。
                # 现在只要求第一步"物理可行"（在盘内且不撞硬障碍），面积/连续性留给最终选步。
                if not path:
                    self._candidate_review.append((candidate['kind'], candidate['cell'], 'approach'))
                    continue
                if len(path) > 1:
                    _nxt = path[1]
                    _phys_ok = (in_board(_nxt) and _nxt not in hard) or _nxt in self._first_allowed
                    if not _phys_ok:
                        self._candidate_review.append((candidate['kind'], candidate['cell'], 'approach'))
                        continue
                growth=5 if candidate['kind']=='dia' else 1 if candidate['kind']=='bean' else 0
                exit_segment=self._post_pickup_spacing(st,path,segment)
                escape=self._pickup_escape(path, red, hard, growth,segment=exit_segment,
                    require_tail=red.length>=12,debt=self._geometry_growth,return_route=True,food=self._food)
                if escape:
                    pair=self._paired_exit(st,red,path,hard,segment)
                    if pair is not None:
                        escape,chain_stop=pair
                        candidate['chain_stop']=len(path)-1+chain_stop
                    # 长蛇的主动空间管理（用户选 B）：吃完这颗之后，落点还必须有足够大的活动空间。
                    # 原来的出口检查只证明"吃完能走几步"，不证明"吃完不会被挤进窄区" —— 而今天
                    # 61% 的死亡是 `trapped`，很多正是这样一步步把自己吃进死角的。
                    if (self.no_boost and red.length >= LONG_SNAKE_LEN and escape
                            and len(escape) > 1):
                        _need_room = int(red.length) + GEM_EXIT_ROOM_MARGIN
                        _room_after = self.tight_area(escape[-1], own, getattr(red, 'path', None),
                                                      limit=_need_room + 1)
                        if _room_after < _need_room:
                            self._candidate_review.append((candidate['kind'], candidate['cell'],
                                                           'exit_room'))
                            continue
                    # 用户 10-18 第 12 局："245 帧吃旁边的钻石绕了远路，建议提前规划撤退路线但**不要
                    # 按照撤退路线行动**，仅考虑这个钻石适不适合吃（无法规划撤退路线），然后吃了之后
                    # 按及时策略行动"。
                    # 所以 route 只包含"接近这颗钻石"的路径；撤离段只用来验证"吃完有路可走"，
                    # **不再当作行驶目标**（原来把撤离段拼进 route 并拿 route[-1] 当 drive_pixel，
                    # 等于吃完还朝撤离方向盲开 —— f=510"向下拐去不明位置"就是这么来的）��
                    candidate['route']=path
                    candidate['escape']=list(escape[:1+ESCAPE_ROUTE_CELLS])
                    candidate['pickup']=(candidate['cell'],candidate['pixel'])
                    candidate['drive_pixel']=candidate['pixel']
                    candidate['dist']=len(path)-1
                    candidate['first']=path[1] if len(path)>1 else head
                    # Only reward beans actually crossed by this checked path;
                    # do not interrupt an approach for a sideways bean detour.
                    beans={tuple(map(int,e[:2])) for e in st['items'].get('bean',[])}
                    incidental=len(beans.intersection(path[1:]))
                    # 用户（10-18）第 1 点："多个钻石相距 7 格以内时，规划能吃到所有钻石的合理路径"。
                    # 真正的串珠：这条接近路线**自身覆盖**的其它钻石也算产出（每颗 +5 格），
                    # 于是"一条路串起 2—3 颗钻石"的候选会明显胜出，而不是一颗一颗折返。
                    gems_on_path=0
                    if candidate['kind']=='dia':
                        _gems={tuple(map(int,c)) for _k,_v,c,_b in cands if _k=='dia'}
                        _gems.discard(tuple(candidate['cell']))
                        gems_on_path=len(_gems.intersection(set(path[1:])))
                    candidate['score']=((candidate['value']+5.0*gems_on_path+.45*incidental)
                                        /(len(path)-1+ITEM_FIXED_COST[candidate['kind']]))
                    if gems_on_path:
                        candidate['gems_on_path']=gems_on_path
                    # 用户（10-18）："多个钻石相距 7 格以内时，规划能吃到所有钻石的合理路径"。
                    # 第一步做在评分上：同一片 7 格邻域里钻石越密，这一趟能顺带吃到的就越多，优先级越高。
                    # 只改排序、不改路线本身，所以不会破坏已经通过的全部安全检查。
                    if candidate['kind']=='dia':
                        _cx,_cy=candidate['cell']
                        _near=0
                        for _k,_v,_c,_b in cands:
                            if _k!='dia' or _c==candidate['cell']:
                                continue
                            if max(abs(_c[0]-_cx),abs(_c[1]-_cy))<=GEM_CLUSTER_CELLS:
                                _near+=1
                        if _near:
                            candidate['gem_cluster']=_near
                            candidate['score']*=(1.+GEM_CLUSTER_BONUS*min(3.0,float(_near)))
                    # 顺手堵敌蛇头: when this gem sits near an opponent's head, an approach whose checked
                    # route also covers the cell in front of that head is worth more.  It is a preference
                    # between approaches that already passed every safety check, so the gem is still the
                    # objective and the block costs nothing extra -- the user's "顺便" rather than a detour.
                    if candidate['kind']=='dia':
                        blocked=block_cell if block_cell is not None else self._block_cell_for_gem(
                            st,head,candidate['cell'])
                        if blocked is not None and blocked in path:
                            candidate['score']*=BLOCK_GEM_BONUS
                            candidate['block_head']=blocked
                    self._candidate_review.append((candidate['kind'],candidate['cell'],'accepted'))
                    accepted.append(candidate)
                    if len(accepted)>=3:
                        break
                else:
                    # 近钻降级通道已按用户指示删除（10-18 第 4 轮："堵死蓝蛇后将可用空间视为障碍然后
                    # 撞死在自己身上"，降级接近把风险引了进来）。出口见证仍然是硬要求。
                    self._candidate_review.append((candidate['kind'], candidate['cell'], 'exit'))
            scored=accepted
            if not scored:
                # ---- 目标粘性的跨帧版本（用户 10-18 第 15 局的四段实录）
                # 四段都是同一个模式：目标瞄准正确 -> 某一帧突然没有候选（呼吸灯漏检 / 出口检查 /
                # 方向连续性过滤）-> 兜底逻辑乱转 -> `trapped` -> 死亡：
                #   f=145 瞄准 grn(6,16) d=4 -> f=148 目标丢失
                #   f=345 瞄准 dia(8,21) d=3 -> f=349 丢失 -> f=352 跳到 15 格外的钻石
                #   f=390 瞄准 dia(7,10) d=1 -> f=398 变成 packing
                #   f=633 长度只有 6—9，trapped 7 帧后死亡
                # 所以在"本帧没有候选"时，先别急着清空目标：只要上一帧的目标还在盘面上，
                # 就重新给它规划一次路线（不要求出口见证，只要求物理可达），把它留住。
                keep = self._keep_previous_target(st, red, head, hard, own)
                if keep is not None:
                    return keep
                self._target=None
                return None
        best=max(scored,key=lambda s:s['score'])
        if self.no_boost:
            diamonds=[s for s in scored if s['kind']=='dia']
            if diamonds:
                nearest=min(diamonds,key=lambda s:s['dist'])
                if (best['kind']=='dia' or
                        best['kind']=='grn' and best['dist']+2>=nearest['dist']):
                    best=nearest
        current=next((c for c in scored if (c['kind'],c['cell'])==self._target),None)
        if current and current['score']>=TARGET_STICKY*best['score'] and (
                not self.no_boost or current['kind']==best['kind'] and current['dist']<=best['dist']):
            best=current
        self._target=(best['kind'],best['cell'])
        return best

    def _pickup_fits_clock(self,st,distance):
        elapsed=st.get('match_elapsed')
        if elapsed is None:return True
        speed=self.follower.speed_s or 160.
        return 60.-elapsed >= distance*CELL/max(80.,speed)+.5

    def _keep_previous_target(self, st, red, head, hard, own):
        """A retained goal needs the same moving-body exit proof as a new one."""
        if self._target is None:return None
        kind,cell=self._target
        cell=tuple(map(int,cell))
        if not in_board(cell) or cell in hard:return None
        live={(k,c) for k,_v,c,_bonus in self._candidates(st,hard)}
        if (kind,cell) not in live:return None
        path=self._approach((kind,cell),head,red.head_px,self._heading,hard,own,
            self.follower.speed_s,self.follower.lead_s,red.length+self._geometry_growth,self._route_risk)
        if not path or len(path)<2 or len(path)-1>getattr(self,'target_keep_route',TARGET_KEEP_ROUTE):return None
        if not self._pickup_fits_clock(st,len(path)-1):return None
        segment=max(1,min(3,math.ceil(max(0.,self.follower.route_speed_s*self.follower.lead_s)/CELL)))
        growth=5 if kind=='dia' else 1 if kind=='bean' else 0
        escape=self._pickup_escape(path,red,hard,growth,segment=segment,require_tail=red.length>=12,
            debt=self._geometry_growth,return_route=True,food=self._food)
        if not escape:return None
        pixel=self._item_pixel(st,kind,cell)
        self.last['kept_target']=True
        return dict(kind=kind,cell=cell,route=path,dist=len(path)-1,pixel=pixel,drive_pixel=pixel,
                    pickup=(cell,pixel),escape=list(escape[:1+ESCAPE_ROUTE_CELLS]))

    def _paired_exit(self,st,red,path,hard,segment):
        """Collect a nearby second item before returning to the tail route."""
        if red.path_trust!='ordered' or len(path)<2:
            return None
        moved=walk(red.path,path,self._geometry_growth,self._food)
        if moved is None:
            return None
        body,debt,eaten=moved
        heading=(path[-1][0]-path[-2][0],path[-1][1]-path[-2][1])
        run=0
        for a,b in reversed(list(zip(path,path[1:]))):
            if (b[0]-a[0],b[1]-a[1])!=heading: break
            run+=1
        near=[(c,v) for c,v in self._food.items() if c not in eaten and c not in path
              and v[2] in ('dia','grn') and 0<sum(abs(c[i]-body[0][i]) for i in (0,1))<=4]
        near.sort(key=lambda x:sum(abs(x[0][i]-body[0][i]) for i in (0,1)))
        own={c:len(body)-1-i+debt for i,c in enumerate(body)}
        remaining={c:v for c,v in self._food.items() if c not in eaten and c not in path}
        for cell,item in near[:2]:
            if item[2]=='grn' and st.get('item_accel_until',0)>st.get('t',0)+3:
                continue
            approach=search(body[0],heading,cell,hard,own,segment=segment,initial=max(0,segment-run),
                            length=len(body)+debt,limit=350,risk=self._risk)
            if not approach or len(approach)>7 or any(self._risk.get(c,0)>=.8 for c in approach[1:]):
                continue
            second=walk(body,approach,debt,remaining)
            if second is None:
                continue
            b,g,used=second
            h=(approach[-1][0]-approach[-2][0],approach[-1][1]-approach[-2][1])
            future_segment=self._post_pickup_spacing(st,path+approach[1:],segment)
            run_after=0
            for a,endpoint in reversed(list(zip(approach,approach[1:]))):
                if (endpoint[0]-a[0],endpoint[1]-a[1])!=h:break
                run_after+=1
            exit_route=tail_witness([b[0]],b,h,hard,growth=g,segment=future_segment,
                                    initial=max(0,future_segment-run_after),limit=450)
            if not exit_route:
                continue
            rest={c:v for c,v in remaining.items() if c not in used}
            if walk(b,exit_route,g,rest) is None:
                continue
            return approach+exit_route[1:],len(approach)-1+1
        return None

    @staticmethod
    def _pickup_escape(path, red, hard, growth, segment=1, require_tail=False, debt=0,return_route=False,food=None):
        """Check the complete approach, then return an executable exit witness.

        Reaching a boundary pickup can require a turn AT its centre, queued
        while approaching. The last approach leg determines the next turn's
        spacing; it does not automatically require another straight segment.
        """
        def result(route):
            return route if return_route else bool(route)
        if not path or any(c in hard for c in path[1:]):
            return result(None)
        facing=(path[-1][0]-path[-2][0],path[-1][1]-path[-2][1]) if len(path)>1 else red.facing
        run=0
        for a,b in reversed(list(zip(path,path[1:]))):
            if (b[0]-a[0],b[1]-a[1])!=facing:
                break
            run+=1
        initial=max(0,segment-run) if len(path)>1 else 0
        ordered=getattr(red,'path_trust','')=='ordered' and red.path and tuple(red.path[0])==path[0]
        remaining_food={c:v for c,v in (food or {}).items() if c not in path}
        if ordered:
            approach_food={c:v for c,v in (food or {}).items() if c!=path[-1]}
            moved=walk(red.path,path,debt,approach_food)
            if moved is None:
                return result(None)
            body,remaining,_=moved
            growth+=remaining
            if require_tail:
                route=tail_witness([body[0]],body,facing,hard,growth=growth,
                                   segment=segment,initial=initial)
                if route and walk(body,route,growth,remaining_food) is not None:
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
            growth+=sum(v[0] for c,v in (food or {}).items() if c in path[1:-1])
        route=packing_escape(body,facing,hard,growth,segment,initial,
                             depth=None if require_tail else 12)
        if route and walk(body,route,growth,remaining_food) is None:
            route=None
        if route is None and ordered and remaining_food:
            route=plan_sweep(body,facing,hard,{body[0]},remaining_food,growth,segment,initial)
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
