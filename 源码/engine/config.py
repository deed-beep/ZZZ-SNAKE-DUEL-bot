"""Tunables for the snake-battle bot.

Every number here was measured from recordings luping1..luping7; the comment on
each says which observation it comes from, so a wrong value can be traced back.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------- screen
IS_RETINA = False
CAPTURE_BBOX = (647, 269, 1911, 1369)      # 1276x1100 region on a 2560x1440 screen
FRAME_W, FRAME_H = 928, 800                # what luping.py saves; the bot matches it
WORK_SCALE = 2                             # perception runs at 464x400 (16 px/cell)

# ---------------------------------------------------------------- board
COLS, ROWS = 29, 25
CELL = 32                                  # 928/29 = 800/25 = 32
GAME_HZ = 15.0                             # luping.py recorded at 15 fps

# direction vectors, screen space (y grows downward)
UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)
DIRS = (UP, DOWN, LEFT, RIGHT)

# ---------------------------------------------------------------- timing
START_GRACE = 1.0                          # movement is possible during START
# RESULT_CONFIRM is gone: "whiteout for 0.6 s" was how the end of a match used to
# be detected, and whiteout is the DEATH signature, not the end (see bot.Phase).
# How long our snake can be missing before the board counts as finished rather
# than as a death we are about to respawn from.  A respawn takes 1-2 s (measured
# live: gone at t=85.5, back at t=86.7), so 8 s is a wide margin.
RESULT_ABSENT = 8.0
LOOP_HZ = 30.0                             # control loop; the game steps at 15 Hz

# Hard stop on a single match.  A match is 975 frames at 15 fps = 65.0 s (section
# 0), and the bot enters PLAYING at the START overlay a few seconds before the
# board is live, so 80 s cannot cut a real match short.
#
# This exists because the post-match RESULT screen is NOT reliably detectable from
# one frame: a live capture of it (assets/result_screen.png) has the same cyan
# (H 94, V 204) as the arena floor, no walls at the capture edges, and its four
# coloured avatar squares decode as four snakes of length 4/5/11/14 with the gold
# crowns as 9-11 diamonds.  The old "whiteout and no red snake" rule therefore
# never fired on it and the bot would have gone on steering the score screen and
# its menus.  A static-scene rule was measured and rejected too: the longest run
# of unchanged red head cells inside a real match is 43 frames (luping4), which
# overlaps the "it is a frozen menu" range, so it cannot be told apart that way.
MATCH_MAX_SECONDS = 80.0

# Menu automation (--auto-start).  X starts matchmaking, and the search takes a
# while: retrying sooner than this does nothing useful and risks cancelling it, so
# about 15 s between attempts.  A leaves the settlement page and needs only a beat.
MENU_RETRY_SECONDS = 25.0
MENU_A_TO_X_SECONDS = 1.6

# ---------------------------------------------------------------- perception
OPEN_R = 3                                 # < half the snake width (8 px at 1/2 res)
CLOSE_R = 4
# One cell at working resolution is (32/2)^2 = 256 px, and the opening below leaves
# a clean single-cell block at 240 px.  The old floor of 300 therefore made a
# ONE-CELL SNAKE IMPOSSIBLE TO SEE: a snake reduced to one segment simply vanished,
# was reported as no snake at all, and the bot stopped steering it.  A snake does get
# that short - section 7's head-to-head takes a segment off each snake per exchange.
MIN_SNAKE_PX = 150
# A snake fragment smaller than this is not credited as a cell at all (to_grid's
# last-resort pass).  One cell of body at working resolution is 256 px, so 40 px is
# about a sixth of a cell: below that the "blob" is more likely JPEG noise than a
# snake, and above it the fragment is real evidence that a body is there.  Measured
# on luping1 f507 the fragment was 64 px and the snake was genuinely present.
SHORT_SNAKE_MIN_PX = 40
# An item is a small thing; nothing legitimately exceeds a few cells.  This is the
# safety net that stops a mask which matches the floor from becoming an "item" again
# the way the diamond rule did (~178,000 px at the arena centre, all game long).
ITEM_MAX_PX = 1200
# ---------------------------------------------------------------- item shape gates
# The user's frame report: "多识别了一个左右黄豆 ... 橙蛇尾部颜色与黄豆几乎一致".  The
# bean mask (H 18-35) and the orange snake mask (H 11-29) genuinely OVERLAP at H 18-29,
# so the tail of the orange snake is inside the bean mask - colour cannot separate them.
# Size and shape can.  Measured over luping1, every phantom bean is a piece of a snake's
# own body: area 617-1881 px, elongation 2.4-4.8.  Every real bean is 59-214 px and
# elongation 1.3-2.0.  Same story for the purple/blue bodies (area 1374, elongation
# 4.77, hue 129 - inside the bean band because the mask is a hue WINDOW, not a band).
ITEM_MAX_PX_OF = {"bean": 300, "grn": 400, "ball": 1200}
ITEM_MAX_ELONG_OF = {"bean": 2.2, "grn": 2.5, "ball": 6.0}
# Diamonds.  The gem has a dark hexagonal outline; a brighter patch of background has
# none, and the background brightness varies cyclically (user-observed, and it is the
# reason the count drifts).  Measured over luping1 f1-300: 8 persistent gem locations
# with a dark-ring fraction of 0.18-0.24, and 39 fleeting detections (max run <= 3
# frames) at 0.000.  The ring fraction is measured just outside the blob's own pixels.
DIA_DARK_VMAX = 140
# Ceiling on the gem's red channel.  194 was too tight - see diamonds_full: a spawn or
# collect sparkle is WHITE and lifts R to 200-235 across the gem's face, which broke the
# blob into fragments under DIA_MIN_PX (measured luping1 f124 cell (17,1)).  240 still
# leaves the cyan-dominance tests (B-R>=40, G-R>=20) doing the real separation.
DIA_RMAX = 240
# Erosion applied to the diamond mask before labelling, to break the 1-2 px bridges that
# glue the gem to the arena floor (see diamonds_full).  Measured on luping1 f122 cell
# (17,1): without it the gem has no component of its own at all - it is part of a 666835 px
# blob with the floor; with a 3x3 erosion it is a separate 231 px component and the floor
# is still one huge blob.  Gem area is quoted post-erosion and is ~30-40% under the
# pre-erosion measurement, which DIA_MIN_PX accounts for.
DIA_ERODE = 1   # 1 DISABLES the opening: tune_dia3.py measured that an opening frees
                # the gem from the floor blob but shrinks it enough that its dark-ring
                # fraction reads 0.000, so the ring stops discriminating - and lowering
                # the ring floor to compensate admits drifting background (5/15 frames,
                # up to 6 gems).  See DIA_DARK_MIN.
DIA_DARK_MIN = 0.02
JOIN_TOL = 24
EYE_NEAR_F = 6                             # eye dilation radius, full resolution
EYE_SPACING = 12                           # measured slit separation, full resolution
# --- eye-slit gating (measured over luping1 frames 1-900) ---------------------
# Real slits measure 24-60 px; the ceiling used to be 400, which is below the
# 1326 px blob that the two slits form with their bloom when the acceleration
# flash is over the head (luping1 f444).  EYE_SLIT_MAX_PX lets that through and
# the embedding test below removes what the wider ceiling would otherwise admit:
# the diamond sparkle, the potion star and the arena's static corner squares are
# all white blobs of 40-120 px.
EYE_SLIT_MIN_PX = 8
EYE_SLIT_MAX_PX = 1600
# Of the 4 coarse probes laid out perpendicular to a single slit, this many must fall
# on red/orange body before the blob is even considered.  A real eye is enclosed by its
# head; this only has to be a cheap pre-filter, because the angle it uses comes from the
# blob's own moments and a slit is cut in half by its own highlight.  Measured on
# luping1 f45, the true slit - a 3x17 vertical mark at (574,409) - scores 1 of 4, so a
# floor of 2 discarded it, while the strict white test had already removed the shine on
# the beans that the floor was there to stop.  The ring test below is the real judge.
EYE_WARM_MIN = 1
# The decisive test.  A candidate PAIR is probed on a ring around its midpoint,
# directed by the pair's own separation axis (which is perpendicular to the head
# axis and, unlike one blob's moments, is not disturbed by the highlight).  Measured
# on luping1 f444, the two real eyes are hidden under a 1326 px bloom, and every
# surviving candidate is the white shine on a yellow bean: those score 6-7 of the 40
# probes, while a real eye pair on the head scores 24-34.
EYE_RING_D = (7, 11, 15, 19)               # probe radii, full resolution
EYE_RING_MIN = 14                          # required warm hits out of 40
# Shape bounds for a blob to be half of an eye pair.  Measured real slits are 10-141 px
# and elongated 1.75-3.3 (and 1.04-1.2 in the compact 'x' of a death buffer).  These
# are what stop a pair of stray white marks DEEP INSIDE the body from being chosen: the
# ring test cannot tell them apart from eyes - measured on luping1 f507 such a pair at
# cell (16,16) scores a perfect 40 of 40, better than the real eyes' 26, because being
# inside the body is exactly what the ring measures.  Shape is the discriminator: the
# eye is a small slit, stray marks are either much bigger (the 267-341 px shines and
# potion stars) or rounder.
EYE_PAIR_MIN_PX = 10
EYE_PAIR_MAX_PX = 200
EYE_PAIR_MIN_ELONG = 1.0
# Scale at which an eye pair's geometry stops being believable.  A pair's score is
# 100 * |separation - EYE_SPACING| plus at most 30 of tie-breaks, so 3000 means "30 px
# off the measured spacing" - i.e. not an eye pair at all.
EYE_PAIR_QUALITY_PX = 3000.0
# What a perfect eye pair is worth in player_region's currency.  It has to exceed the
# whole body-fit term (90 overlap + 6 size), because an eye hand and a continuity hand
# can both reach the same body and only the eye quality separates them.
EYE_QUALITY_BONUS = 130.0
# Radius, at working resolution, of the dilation used to re-join the player's body
# across the holes the eyes punch in it and across a flash-washed neck.  Measured
# gaps to bridge: 12 px (the eye spacing) and 6-8 px.  In working-resolution pixels
# that is 6 and 3-4, so 6 is the smallest radius that works; a cell of clear floor
# between two snakes is 16 working px of gap, i.e. a 8-radius dilation, which is why
# this cannot be raised to 8 without risking a merge with a snake beside us.
EYE_BRIDGE_R = 6
# Radius, at working resolution, of the disk marked as body around an eye midpoint
# before the body is re-connected.  The slit pair is EYE_SPACING = 12 full-res px
# apart and each slit is ~8 px tall, so the hole the eyes punch in the mask is about
# 18x14 px = 9x7 working px; 5 working px = 10 full px covers it from the midpoint
# outwards without reaching anything that is not the head.
EYE_HOLE_R = 5
# If the body the eyes land on is smaller than one cell, the head is cut off from the
# neck and the "body" patch is widened by these radii in turn, stopping at the first
# component of at least one cell.  Measured on luping1 f507 (a 11071 px body): radius
# 5/8 reach only the 64 px snout stub, radius 11 reaches the body.  Measured on
# luping1 f862 (a 24155 px body): radius 5-14 reach a 260 px potion star that the eyes
# happen to sit on, radius 17 reaches the body - so the stop-at-one-cell rule is what
# keeps the star from being accepted, and the sequence has to reach 17 for the
# genuine but badly occluded heads.  A wider reach than 17 would have to cross two
# whole cells of clear floor, which is room for an opponent's body to be swallowed.
EYE_HOLE_R_SEQ = (5, 8, 11, 14, 17)
# A snake does not shrink by 99x between two frames at 15 Hz.  When the eyes are false
# they land on a white object that is not a snake at all - measured on luping1 f862, a
# 600 px potion star sitting on our own colour, which put our head at cell (1,1) as a
# 260 px "snake" while the real 28000 px body was elsewhere.  A candidate body smaller
# than this fraction of the previous frame's body is not ours.  Only applies when there
# IS a previous size, so a one-cell respawn (no history) is unaffected.
BODY_MIN_FRAC = 0.12
# When the eyes are lost, a fragment is grown outward until it reaches this fraction of
# the length the previous frame saw.  Below 1.0 on purpose: the flash legitimately
# removes part of the body, so demanding full recovery would over-grow into a neighbour.
BODY_GROW_FRAC = 0.70
# How far the player's orange tail may extend from the red mask and still be called OUR
# colour rather than the orange opponent's.  Half a cell: our gradient tail meets the red
# body directly (measured H 3 at the head -> H 11 at the boundary -> H 15 at the tail), so
# the join is contiguous; an opponent lying alongside has its own body and is not inside
# this sliver.  In working-resolution pixels (half of full resolution).
RED_TAIL_TOL = 5

# ---------------------------------------------------------------- temporal layer
# The user's guidance: "给红蛇蛇头增加高优先级的7乘7的小窗口进行持续追踪".  A 7x7 cell
# window is 224x224 px, i.e. 14 cells of travel around the predicted head - about 20
# frames of a normal-speed snake, so the head is still inside it after a long occlusion.
HEAD_WIN_CELLS = 7
# How strongly a confirmed head position pulls the velocity estimate (0.5 = equal weight
# to the newest sample and the running estimate). Higher tracks fast turns, lower is
# steadier through a covered frame.
HEAD_WIN_WEIGHT = 0.5
# After this many consecutive disagreements the MEASUREMENT wins and becomes the new
# reference.  Without it a single bad reading latches and every later frame is
# "corrected" against a stale number: measured on luping1 the reference stuck at 17-19
# from f83 to f110 while the snake was really 11-15.
# Length continuity limits, used by temporal.py: how far a length may move in one frame
# before it is treated as a misreading, and how fast it may fall.
LEN_MAX_JUMP = 6
LEN_MAX_DROP_PER_FRAME = 3
LEN_REJECT_MAX = 2
# Thresholds are on the MASK AREA, not the cell count: cell quantisation is exactly what
# fragments a body, so the cell count is the noisier of the two.
# Measured on luping1, the orange snake:
#   f624-f629  fragmentation: 9935 -> 5302 -> 4377 -> 3048 -> 1930 -> 427 px, then 3349 px
#              at f630 and it keeps flickering - RECOVERS, so it is not a death;
#   f712-f720  death (user-confirmed): 12231 -> 12503 -> 11119 -> 8591 -> 7846 -> 6778 ->
#              5498 -> 3196 -> 2065 px, monotone and it does not come back.
DEATH_AREA_FRAC = 0.35      # below this fraction of the recent median AREA ...
DEATH_MIN_PX = 2500         # ... and under this absolute area, only then count a run
# How long the collapse must LAST is the discriminator, because the two events differ in
# duration, not in depth: measured on luping1 the orange fragmentation cycles recover after
# 4-5 frames (f656-659 and f685-689), while the purple death (user-labelled as starting at
# f629) holds an unusable body for 13+ consecutive frames - it decays f635-639 and is then
# ABSENT for 60 frames.  DEATH_CONFIRM=8 sits between the two with margin; 3 was measured to
# fire falsely on the orange fragmentation cycles (2 of 5 detections wrong over 900 frames).
DEATH_CONFIRM = 8
# OFF by default: see the measurement note above DEATH_AREA_FRAC.  Four attempts did not
# separate death from fragmentation with the evidence the tracker has, and a WRONG death
# declaration clears the head window and length history - the state the 7x7 tracking depends
# on - so silence is the safe choice.  Nothing downstream needs it: an absent snake already
# publishes no head.  Set True only with new evidence, and re-run
# trace_purple_death.py / death_attrib.py first.
DEATH_DECLARE = False
# Same latch guard for the head window: after this many consecutive "the head jumped"
# rejections the measurement is accepted.  Measured on luping1 f186-f200 the window sat
# at (18,1) while the snake moved on, so every frame was rejected against a stale window.
HEAD_REJECT_MAX = 2
# Eye-state extension across a gap where only one slit is readable.  Gaps measured over
# luping1 f1..900 are 81 in number, median 2 frames, p90 3, max 19; the state must also be
# stable (seen in EYE_STABLE_N of the last EYE_STABLE_M DECODED frames) or it is not
# extended, because 48 of the state runs are a single frame long and carrying those would
# manufacture state that was never there.
EYE_EXTEND_MAX = 3
EYE_STABLE_N = 2
EYE_STABLE_M = 3

# ---------------------------------------------------------------- item persistence
# A real item is a stable object.  A detection is trusted outright when its own quality
# makes it unambiguous (a gem with a strong dark ring, a full-size bean) or when it is
# confirmed within +/-ITEM_TOL_FRAMES of another frame; otherwise it is dropped as a
# flicker.  Measured: the gems at luping1 f107/f124/f125 have a ring of 0.027 against
# bright cyan floor, while the drifting background produces patches seen in exactly one
# frame each (max run <= 3 over 300 frames at ring 0.000).
ITEM_TOL_FRAMES = 2
DIA_RING_TRUST = 0.08
# A gem whose mask fragmented this frame is carried if it was confirmed at the same cell
# within this many frames.  Gem lifetimes are 13-100 consecutive frames, so this is tight
# enough that a drifting-background patch (seen in exactly one frame) cannot survive it,
# and it recovers the f107/f124/f125 class where the gem is visibly on screen.
DIA_PERSIST_FRAMES = 6   # LEGACY: no longer read by temporal.filter_items
# A gem with a weak ring has to be confirmed by seeing it at the same cell in at least
# DIA_CONFIRM_N of the last DIA_CONFIRM_M frames.  Measured: real gems live 13-100
# consecutive frames, the drifting-background patches live 1, so 2-of-4 separates them
# with room to spare, and the small bright gems whose outline never reads as dark
# (luping1 f122 cell (17,1), ring 0.000) are still picked up after two frames.
DIA_CONFIRM_N = 0   # LEGACY: no longer read by temporal.filter_items
DIA_CONFIRM_M = 4   # LEGACY: no longer read by temporal.filter_items
# Ring strength at which a gem needs no confirmation at all.
DIA_RING_STRONG = 0.18   # LEGACY: no longer read by temporal.filter_items
# NOTE: DIA_CONFIRM_N=0 leaves only the ring test, which is what tune_dia3.py selected.
# Requiring confirmation instead of the ring was tried and rejected: a drifting
# background patch visits a new cell every frame, so 'seen here before' is not the
# same as 'still here', and the count went to 2.4-2.9 gems a frame (user counts 2-3).        # above this the ring alone is proof
BEAN_MIN_TRUST_PX = 45       # below this a bean must be confirmed by a neighbour frame

# ---------------------------------------------------------------- item persistence
# A real item is a stable object.  A detection is trusted outright when its own quality
# makes it unambiguous (a gem with a strong dark ring, a full-size bean) or when it is
# confirmed within +/-ITEM_TOL_FRAMES of another frame; otherwise it is dropped as a
# flicker.  Measured: the gems at luping1 f107/f124/f125 have a ring of 0.027 against
# bright cyan floor, while the drifting background produces patches seen in exactly one
# frame each (max run <= 3 over 300 frames at ring 0.000).
ITEM_TOL_FRAMES = 2
DIA_RING_TRUST = 0.08        # above this the ring alone is proof
BEAN_MIN_TRUST_PX = 45       # below this a bean must be confirmed by a neighbour frame

# How far the accepted eye midpoint may sit OUTSIDE the accepted body and still count
# as belonging to it.  The midpoint is inside the white hole the slits punch in the
# head, so it is often just off the body's own pixels; 4 working px = 8 full px is
# half the slit separation.
EYE_GATE_R = 4
# Diamonds: the board never holds more than about five gems (section 6, "diamonds
# respawn within 0.5 s, normally <= 5").  During the respawn whiteout the desaturated
# floor shatters into small cyan fragments that all satisfy the size test: measured on
# luping6 f895, 272 "diamonds" in a single frame.  A count above this is discarded as a
# whole, which is safe because such a frame is not readable anyway.
DIA_MAX_COUNT = 12
# During the same whiteout the blue mask swallows the floor and 8-connectivity turns it
# into one enormous component: measured on luping6 f717, a "blue snake" of 322 cells
# against a real one of 14.  Limit in working-resolution pixels, ~1.3 cells.
MAX_OPP_PX_WHITEOUT = 3400
# Spike-ball mask.  Measured on luping1 f634 at the ball's own cell (4,20): rim
# (170,75,74) H 179 S 0.56 V 0.67, interior (255,109,13) H 13 S 0.95 V 1.0.  The old
# mask had a V<=184 CEILING, which discarded the bright interior and left a 40-167 px
# sliver of a ball that is ~50 px across - so every "ball" read 0.7-1.6 cells wide
# instead of the 2x2 the user confirms it always is.
_BALL_SMIN = 150
_BALL_VMIN = 120
# A spike ball is 2x2 cells; a single-cell red-ish blob is the slowdown item, which is
# harmless and edible.  Reported separately so the policy never treats it as death.
BALL_MIN_CELLS = 2
# Geometry of a spike ball, at working resolution.  Real balls measured 259-419 px in a
# 21-25 px box with fill (area / bbox) 0.56-0.70; the red snake's body, which shares the
# ball's hue family, is 4000-14000 px with fill 0.17-0.30.  So the gates are: compact
# box, high fill, and a size window.  A round blob under BALL_SPIKE_MIN_PX is the
# slowdown item.
# Measured over luping1 f630..975 (815 detections) the ball-core area is BIMODAL with a
# clean gap at about 360 px: 736 detections at 360-479 px (the three real balls, cores
# 394-432) against 66 at 200-359 px and 13 at 480-679 px, which are fragments of a ball -
# its maroon rim or inner shading, clipped when a snake passes over it.  The old window
# (200-900) admitted those fragments, so 34 frames reported 4-5 "balls" where the user says
# there are exactly three.
BALL_MIN_PX, BALL_MAX_PX = 350, 500
BALL_BOX_MIN, BALL_BOX_MAX = 16, 34
BALL_FILL_MIN = 0.45
# Dark-outline fraction in the ring just outside a ball candidate, measured with the 11x11
# ring kernel the code actually uses (a 7x7 ring gave 0.444-0.525 for real balls and 0.151-0.280
# for our own body - the same separation, different scale; always calibrate against the kernel in
# use).  With the real kernel: real spike balls 0.280-0.364, our own coral body 0.185, a rim
# fragment of our body 0.123.  tune_ball.py swept 0.20-0.32 over the ball window and 0.30 is the
# best: it leaves the fewest frames reporting MORE than three balls (1) while keeping 121 frames
# reporting exactly three.
#
# This test REPLACED an earlier "exclude any ball whose cell is inside our body" rule.  That rule
# scored slightly better (142 exact / 1 over) but it was self-inflicted damage: measured on
# luping1 f745 all three balls are present and the third sits at (7,15) with our body adjacent, so
# excluding our cells removed a genuine ball.  The outline test removes the false positives
# without touching real ones, which is why it is preferred despite the lower exact-3 count.
BALL_DARK_MIN = 0.34
# value ceiling for the dark test (the outline and spikes are dark)
BALL_DARK_VMAX = 150
# Width, in working pixels, of the band just outside a ball's bounding box where the dark
# fraction is measured.  Calibrated with ball_band.py: at pad=3 real balls measure 0.444-0.525
# while our own body measures 0.151-0.280 - the widest gap of the widths tried (2, 3, 4, 5).
BALL_BAND_PAD = 3
# The single-cell slowdown item.  Reported as `items["slow"]` and deliberately NOT as a
# ball: a spike ball is 2x2 and contact is fatal, while this is one cell and edible with a
# slight slowdown.  Gates are tighter than the ball's because the red-core mask is noisy -
# measured over luping1 f1..620 there are 232 red-core cells and most are snake highlights,
# so a blob that fails these is reported as nothing rather than as an item.
SLOW_MIN_PX, SLOW_MAX_PX = 60, 230
SLOW_BOX_MIN, SLOW_BOX_MAX = 8, 20
SLOW_FILL_MIN = 0.50
# The slow and ball windows deliberately do NOT meet: slow tops out at 230 px and the ball
# starts at 350, leaving 134 red blobs unfiled over luping1 f1..900.  That gap is on purpose.
# Measured with gap_items.py and confirmed by zooming the cells, those blobs are the RED SNAKE'S
# OWN BODY - its colour gradient runs from saturated red at the head (which is the same hue as
# the ball core) to pale coral at the tail, so a head or neck fragment lands in this size range.
# f12 cell (4,17) is the head with its two white eye slits; f2 cell (8,24) is the neck.
# Widening either window to close the gap files the player's own body as an item - the same
# false positive the ball dark-outline test exists to remove.  If you ever revisit these numbers,
# zoom the cells first.
# 25th-percentile SATURATION floor for the blue snake's mask, in OpenCV's units.  The value
# tested against it is hsv[..., 1], which is 0-255 - the previous 0.85 was on a 0-1 scale, so
# the comparison could never bite and the floor did nothing.  The documented intent is to
# reject the pale bean-drop tail tip, measured at 0.54-0.71 as a FRACTION, i.e. 138-181 on
# this scale; 217 = 0.85 * 255.
BLUE_SAT_P25 = 217
WHITEOUT_SAT = 0.58                        # respawn wash drops arena S from ~0.78
WHITEOUT_VAL = 0.70

# ---------------------------------------------------------------- eye states
BUFFER_ELONG = 1.8                         # 'x' eyes are compact; slits are 2.4-3.9
ITEM_ASYM = 0.45                           # eye-pair asymmetry along the head axis

# ---------------------------------------------------------------- items
BEAN_GAIN = 1                              # a bean adds one segment
DIAMOND_GAIN = 5                           # a diamond adds about five
DIAMOND_MIN = 2                            # diamonds respawn within 0.5 s, normally <=5
BALL_SPAWN_T = 42.0                        # three balls appear together, ~f630
BALL_CELLS = 2                             # each ball covers 2x2 cells
BALL_SPEED_CPS = 0.7                       # cells per second drift

# Fixed diamond pair at match start (user-observed): two diamonds at mid-height,
# one left of centre and one right.  Used only to sanity-check the decoder.
START_DIAMONDS = ((11, 12), (17, 12))      # (col,row) approximate, mid row 12

# ---------------------------------------------------------------- acceleration
# The accelerate key and the item stack.  Measured facts:
#   * holding A leaks beans from the tail: +0.375 beans/s with A held vs
#     +0.200 beans/s without, i.e. roughly one bean every 5.7 s
#   * mashing A drops one bean per press -> this is the intended way to shed
#     length when the tail is long enough to trap you
#   * the green token gives ~10 s (150 frames) of speed WITHOUT the key held,
#     and the key still stacks on top of it
ITEM_ACCEL_SECONDS = 10.0
KEY_LEAK_BEANS_PER_S = 0.175
MASH_MIN_INTERVAL = 0.05

# Steering pulse.  The game samples the pad, so the bot pushes the stick and lets
# go again; a real controller held sideways is what a human does for a long
# straight.  Neither of these has been confirmed against the running game yet
# (GAME_NOTES section 12.3), so the width is a named knob rather than a literal:
# widen it if the snake ignores short taps, and set HOLD=True to keep the stick
# pushed until the direction changes instead of pulsing every frame.
DIR_PULSE_MS = 30
DIR_HOLD = True  # persistent stick; do not delay a new checked bend behind an 83ms pulse lock

# ---------------------------------------------------------------- policy
LEN_HUNT_MIN = 12                          # only pick fights once this long
TAIL_LONG = 30                             # above this, mashing to shed is allowed
FARM_UNTIL_T = BALL_SPAWN_T                # no balls before ~42 s: farm freely

# Item seeking.  The site order of section 13 is a cascade, not a weighting:
# survival first, then shedding, then a safe route, and eating only when the
# route to the item leaves somewhere to go afterwards.
# A diamond grants a RANDOM 3-5 cells (user: "钻石每次实际上提供的是3到5格的随机长度"), not a fixed 5.  The
# value used is the MEAN (4.0) rather than the old 5.0, which over-valued a gem by up to 40%.  The spread is
# recorded so the range can be used wherever it matters (e.g. preferring a guaranteed bean over a risky gem).
DIA_GAIN_MIN, DIA_GAIN_MAX = 3, 5
ITEM_VALUE = {"dia": 4.0, "bean": 1.0, "grn": 4.0}   # value per item kind
ITEM_FIXED_COST = {"dia": 4.0, "bean": 2.0, "grn": 4.0}   # commitment cost, cells
MIN_ESCAPE_AREA = 6                        # never commit to a cell with less room
ESCAPE_AREA_CAP = 24                       # room a long snake needs; stop scaling here
LOW_VALUE_MAX_DETOUR = 8                   # a bean is not worth crossing the board

# Target stickiness.  The policy re-decides every frame, so two items of similar
# value make it turn toward one, then the other, then back: measured live, the
# head thrashed inside a 4x4 cell box for seconds and the snake never arrived
# anywhere, which is how a 5-cell snake dies without eating.  A committed target is
# kept while it is still reachable and worth at least this fraction of the best
# alternative; only a clearly better option takes over.
TARGET_STICKY = 0.75

# Holding the accelerate key during a head-to-head.  Off, deliberately.
#   * the tracker sets st["grind"] on any frame where two heads are within one
#     cell, and keeps it for up to 25 frames, so this fired on mere near-misses:
#     the replay measured 36-252 such frames per match;
#   * a head-to-head is won by LENGTH (section 7: each frame costs both snakes a
#     segment) and holding the key leaks a bean every ~5.7 s (section 4), i.e. it
#     shortens the very thing that decides the fight.
# There is no measurement showing a benefit, and a measured cost, so it is off
# until a live test says otherwise.
GRIND_HOLD_ACCEL = False

# ---------------------------------------------------------------- io
# Kept relative to this file so a copy of the folder stays self-contained: the
# absolute path meant a copy wrote its logs into the original's folder.
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# ====================================================================== attention
# The player's head is the only thing looked at every frame.  Everything else is either inside
# the 7x7-cell window on it (the "fovea") or in the periphery, which is sampled less often and
# carries a lower weight.  See attention.py for why the ordering is what it is.

WINDOW_FIRST = True            # keep the window pass ahead of the peripheral pass

# Periphery: how often the whole frame is re-examined, how long a reading stays usable
# afterwards, and how heavily it counts relative to a window reading.  PERIPHERAL_EVERY=8 at 15
# fps is a 1.9 Hz global refresh - the corner of the eye is for orientation, not for aiming.
PERIPHERAL_EVERY = 8
PERIPHERAL_MAX_AGE = 12        # older than this and a carried reading is dropped
PERIPHERAL_WEIGHT = 0.35       # weight of a peripheral reading next to a window reading

# In-window diamond acceptance.  STRICTER than the global scan, because a false gem in front of
# the head steers the snake into a wall while a missed gem only costs points.  The four tests are
# colour (in diamonds()), size, shape (fill and elongation of the blob), and the dark hexagonal
# border (WIN_DIA_DARK_MIN), which is what the floor texture cannot fake.
WIN_DIA_MIN_PX = 60
WIN_DIA_MAX_PX = 2500
# Shape gates, tuned with tune_window2.py (see that file's docstring for the sweep table).
# The FILL floor must stay low: real gems measure 0.52-0.70 and many sit near the low end, so
# raising this to 0.62 costs three quarters of the recall.  ELONGATION is the useful knob - a
# hexagon is near-isotropic, while the false positives are partial cells cut off by a snake's
# straight edge (confirmed by zooming f299 cell (21,6) and f195 cell (19,8)).  1.5 cuts the
# unconfirmed window claims from 10 to 2 for 1.9 points of recall.
WIN_DIA_MIN_FILL = 0.45
WIN_DIA_MAX_ELONG = 1.5
WIN_DIA_DARK_MIN = 0.18        # a STRONG dark edge: accepted on the spot

# Window obstacles: how far beyond the head the policy should care about a body or wall.
WIN_OBSTACLE_MARGIN = 1        # cells of extra margin for "about to hit"

# ====================================================================== mechanics
# The numbers state_analysis.py uses to judge whether a reading can be true.  All of these are
# measured or stated by the user; see the module docstring for what each one constrains.
BEAN_GROWTH = 1                # a bean adds one cell
DIA_GROWTH = 5                 # a diamond adds five
EAT_ADJACENT_CELLS = 1         # an item that vanished within this many cells was eaten
ITEM_MOVE_TOL_CELLS = 1        # items are stationary; more movement than this is noise
BALL_EXPECTED_COUNT = 3        # the user: spike balls are always exactly three, 2x2

# ================================================================ diamond identity
# Measured / stated: see the block header in attention.WindowDetector.diamonds.
DIA_FIXED_COUNT = 2            # the match opens with exactly this many fixed gems
DIA_FIXED_MIN_RUN = 2          # sightings at one cell needed to call it fixed
# A stationarity test on CONSECUTIVE frames does not survive this detector's gaps: measured on
# luping1 f1..500, cells holding a gem for 65-98 frames have longest consecutive runs of only
# 12-15 frames with 5-6 frame gaps, and 13 of 27 well-established cells are interrupted.  So a
# fixed gem is recognised from its RECORD at a cell, not from an unbroken run.
DIA_FIXED_MEMORY = 20          # frames a sighting still counts toward "this cell is a fixed gem"
DIA_MOVING_MIN_DARK = 0.20     # a moving gem needs a strong edge: it cannot use persistence

# ====================================================================== motion
# The user's rules, which motion.py encodes:
#   * every item BREATHES (its brightness cycles) and the fixed diamonds and beans also WOBBLE
#     slightly in place - so measured position and mask area fluctuate every frame and none of
#     that is travel;
#   * a moving gem travels only vertically or horizontally, never diagonally;
#   * the three spike balls move one horizontally, one vertically, one at 45 degrees, and the
#     45-degree one is the most dangerous.
#
# MOTION_MIN_CPS is the noise floor: slower than this over the fitted track is breathing/wobble,
# not travel.  A ball drifts at 0.7 cells/s, so the floor sits below it with margin.
MOTION_MIN_CPS = 0.25
BALL_DIAG_THREAT = 1.0        # the 45-degree ball crosses the others' lanes: worst case
BALL_AXIS_THREAT = 0.45       # a horizontal/vertical ball sweeps one line, which is avoidable
WOBBLE_TOL_CELLS = 1          # in-place jitter, in cells, that is NOT travel

# ============================================================ snake motion mechanics
# From the user: head and tail travel together; eating a bean pauses the tail briefly, eating a
# diamond pauses it until the head has moved about five cells; the head never stops while alive
# except in the death buffer; at 15 fps the head moves at most 2 cells per frame, typically about
# half a cell, and never stalls longer than 7 frames.
# The user's figure is 2; 3 is used so a reading one frame late is not called a swap.  The
# swaps actually measured on luping3 are 6-22 cells, so the margin costs nothing.
HEAD_MAX_CELLS_PER_FRAME = 3
HEAD_STALL_MAX_FRAMES = 7         # longer than this without moving means the head is lost
# A tail drop from holding accelerate happens about once per 5.7 s (KEY_LEAK_BEANS_PER_S), i.e.
# roughly one cell every 85 frames.  A length that falls faster than this is not a tail drop:
# it is the death dissolve, or the body fragmenting.
TAIL_DROP_MIN_INTERVAL = 20       # frames; a fall inside this window of the last one is suspect
DIAMOND_TAIL_LAG_CELLS = 5        # the tail resumes once the head has moved about this far

# ============================================================ tail-drop mechanics
# From the user: length NEVER falls except through the accelerate key.  One PRESS drops exactly
# one cell, as a bean left on the tail cell - which is also why the tail seems to pause after
# eating, since it cannot advance into a cell the bean occupies.  Holding the key drops at
# KEY_LEAK_BEANS_PER_S (about one per 5.7 s).  Before eating anything the head and tail move
# together and the length is constant.
#
# The cross-check this enables: count the beans that APPEAR from one frame to the next and compare
# with the length that fell.  A known exploit (mashing faster than a limit) makes more beans appear
# than the length lost, stacked on one cell; the sample recordings do not contain it, so the
# detector exists to notice it rather than to expect it.  Beans appearing with no length change is
# normal only in that bug, so it is reported.
TAIL_DROP_BEAN_CHECK = True    # cross-check a length fall against new bean appearances

# Smoothing factor for the head-position estimate that arbitrates between the body's two ends.
# Small enough that one bad frame barely moves it, large enough to follow a snake at half a cell per
# frame.  0.35 follows roughly a three-frame window.
HEAD_EST_ALPHA = 0.35


# ------------------------------------------------------------------ reference only
# The entries below are facts about the game that no perception or analysis module reads yet.  They
# are kept (not deleted) because they are documented from play and the policy layer is about to be
# rewritten - INCREASE/GAIN and the capture flags are exactly what a policy or a capture change will
# want.  Marked here so the next audit does not report them as rot again.
#   IS_RETINA, START_GRACE      capture and menu timing (bot.py / vision.py)
#   BALL_MIN_CELLS              the spike ball's footprint, in cells
#   BEAN_GAIN, DIAMOND_GAIN     item values; DIAMOND_GAIN duplicates DIA_GROWTH
#   DIAMOND_MIN                 how few gems can be on the board
#   ITEM_ACCEL_SECONDS          how long the speed item lasts
#   LEN_HUNT_MIN                the length above which hunting is worth it


# ====================================================================== targeting
# How stale a carried item position may be before it stops being a target.  A peripheral reading
# carried for many frames is a position the item has moved on from - on luping3 the bean list
# contained entries at carried(29), i.e. a coordinate from two seconds earlier.
CARRIED_MAX_AGE = 8
# Tie-breaking bonuses, added to value/(distance+cost).  Small on purpose: they should settle a
# close call between two reachable items, not make a distant one beat a near one.  A gem measured
# inside the 7x7 window is the most trustworthy reading there is; a pair-carried fixed gem has a
# trustworthy position but an unmeasured area, so it earns less.
GEM_BONUS_WINDOW = 0.10
GEM_BONUS_PAIR = 0.08

# ============================================================ strategy: length first
# The objective, in the user's words: "at the end, as long as our snake is the longest, we win".
# So the cascade is LENGTH -> SURVIVAL -> (ignore the opponent), which is the opposite of the old
# survival-first arrangement.
#
# STRATEGY constants:
#   "length"  - the length-first, steady, win-by-being-longest play (this round's target)
#   "survive" - the old survival-first play, kept for A/B
STRATEGY = "length"

# --- what is worth diverging for
# A diamond is the objective.  A bean or the speed item is eaten only when it lies inside the 7x7
# attention window, i.e. "on the way" - which is exactly the user's operational rule.  Setting this
# to a large number restores the old "cross the board for a bean" behaviour.
ONLY_WINDOW_LOW_VALUE = True
HIGH_VALUE_KINDS = ("dia",)

# --- acceleration
# Holding the key costs about one bean per 5.7 s and buys 1.8-2.7x speed, and the user's own
# measurements say the longest snakes are the ones that hold it while running a good route.  So it is
# now the DEFAULT while travelling on a straight, open run, and only released when the position is
# genuinely tight.  Mashing is never routine: it destroys length.
ACCEL_DEFAULT_HOLD = True
ACCEL_MIN_OPEN_AHEAD = 3        # cells of open straight required to keep holding
ACCEL_RELEASE_IF_BOXED = True   # never add speed to a losing position

# --- survival, applied while maximising length (never at length's expense, but always active)
CLEARANCE_MIN = 1               # cells of free space to keep between the head and any obstacle
WALL_MARGIN = 1                 # do not run along the wall: leave a way out
WALL_MARGIN_RELIEF = 4.0        # a target worth this much may justify hugging the wall
# Cells of margin to keep from a snake that is ACCELERATING.  Calibrated with tune_opp.py on
# luping1 f1-400: an accelerating opponent is present on 301 of 400 frames, so this margin decides
# how much of the board the planner refuses.  Measured blocked cells per frame, as a share of the
# 29x25 board:
#     margin 0 ->  2.2%    margin 1 ->  4.7%    margin 2 ->  8.0%    margin 4 -> 16.4%
# At margin 4 the planner had 16% of the board walled off, and 98% of all "trapped" neighbours
# came from that expansion rather than from a real pocket - it was refusing to move.  The user's
# rule is to stay away from an accelerating snake, which ONE cell of margin satisfies: enough that
# a fast diagonal pass cannot connect, without turning a boosting opponent into a wall.
OPP_ACCEL_AVOID = 1
NEAR_DEATH_AREA = 8             # flood area below which the position is treated as losing

# --- shedding
# Only to escape being trapped by our own body.  The old rule also shed whenever the tail was simply
# long (TAIL_LONG), which throws away the objective; that is now off.
SHED_ONLY_WHEN_TRAPPED = True
SHED_TRAPPED_FRAMES = 2         # consecutive frames of being boxed in before shedding

# Score penalty for a route that runs along the map edge, applied in proportion to the fraction of the
# route within WALL_MARGIN cells of it.  Chosen so a diamond (value 5) can pay for hugging while a bean
# (value 1) cannot, which is the user's "acceptable if the reward is high enough".
WALL_HUG_PENALTY = 3.0

# `ramp_span` at or above this marks an accelerating snake.  Documented in perception.ramp_span: the
# acceleration streak shows as a large luminance span along the body, 0.6-2.0 at rest and 20-100 while
# boosting, so the threshold sits well inside that gap.
OPP_ACCEL_RAMP = 12.0


# ============================================================ window mode
# Track the red head with a 13x13-cell window instead of decoding the whole board each frame.
# Six cells to each side is enough for steering, for body reconstruction against OwnBodyHistory, and
# for immediate obstacle avoidance.  See window_perception.py for the measured reason this is a
# SPEED play: the diamond scan and the dark mask scale with area, while the colour masks and the
# connected-component sweep are dominated by per-call overhead and barely shrink when cropped.
WIN_CELLS = 13
# A window cannot see a fixed diamond across the board, notice a respawn, or tell that the match has
# ended, so a full frame still runs this often for those facts.  At 15 fps, 8 means 1.9 global
# sweeps per second against 15 windowed frames.
SWEEP_EVERY = 8
# Use the window path at all.  OFF BY DEFAULT, and the live path is unchanged: bot.py is not wired
# to WindowPerception, so this flag currently governs only the offline tools.  It is set False so
# that turning it on is a deliberate act - the window head solver measures 93.6% exact with a
# worst-case error of 27 cells, and those outliers are what would steer the snake into a wall.
WINDOW_MODE = False

# Largest head movement, in full-resolution pixels, that the window solver will follow in one frame.
# The user's stated limit is "at most 2 cells per frame, usually about half" and one cell is 32 px,
# so 2.5 cells leaves margin for a frame arriving late while still rejecting the detector latching
# onto a different white object.
WIN_HEAD_MAX_STEP_PX = 2.5 * 32.0

# Snap-back for the window head solver.  A motion-fallback head whose velocity has gone wrong centres
# the window in the wrong place, and then the eyes are not inside it to correct the drift - a
# self-sustaining error (measured on luping3 f150-400: 10 of 13 disagreements had the true head
# outside the window).  The escape: if this many consecutive frames reported the EYES and all landed
# near the same cell, while the prediction sits more than WIN_SOLVER_SNAP_CELLS away, then the
# velocity is the thing that is wrong, and the prediction is snapped to the observed cell.
WIN_SOLVER_SNAP_N = 2
WIN_SOLVER_SNAP_CELLS = 2

# How far, in cells, a LONE white slit may be from the previous head and still be accepted as the
# head.  This replaces an on-body proximity test that measured useless: a 2-cell radius around a
# point on a board where our own body is usually within 2 cells made the test almost always true,
# and it cost an extra families() call per frame (exact rate went 94.0% -> 93.6%).  The user's own
# mechanics give the sharper rule - the head moves at most about 2 cells per frame - so a lone
# blob many cells away is not the head however bright it is.  That is the measured failure: on
# luping3 f216/f220 a diamond's sparkle 25 cells away was taken as the head.
WIN_SLIT_MAX_STEP_CELLS = 2.0

# Largest sweep-to-solver correction, in full-resolution pixels, that the re-anchoring will apply.
# A sweep is itself a decode and can be wrong; without a clamp one bad sweep injects an offset that
# compounds through the prediction.  4 cells is far beyond any real accumulated drift between sweeps
# (the head moves about 0.5 cells a frame, so 8 frames is about 4 cells of travel in total).
WIN_ANCHOR_MAX_PX = 4 * 32

# Re-anchor the window solver to the global sweep's head.  OFF because it was measured to HURT, on the
# same frames with only this toggled (luping3 f150-500):
#     OFF  exact 298/349 = 85.4%  median 0  p90  7  max 25
#     ON   exact 271/349 = 77.7%  median 0  p90 10  max 57     (rejected 7 -> 48, eyes 254 -> 228)
# The anchor drags the window onto the sweep's head, the eyes then fall at the window's edge or
# outside it, and the following frames are rejected - so a correction intended to reduce drift was
# creating the "window centred wrong" state it was meant to cure.  Kept as a flag rather than deleted
# because it is cheap to re-test if the solver changes.
WIN_ANCHOR_TO_SWEEP = False

# Global head + window detail.  OFF because the windowed diamond scan was measured to see only 9% of
# the gems on the board (2.00 per frame globally against 0.18 in the window, at a saving of 18 ms), and
# diamonds are the entire objective - the snake sat at 4-7 cells for whole matches while gems were
# visible on the board.  The board-level decode was always global, so turning this off changes only the
# diamond scan's coverage, not the head or the bodies.  The board decode (families, per-family labelling, eye search, path
# extraction) runs on the whole frame, so the head, bodies, opponents and eye state are IDENTICAL to
# the plain global path; only the two per-pixel stages that scale with area - the diamond scan and the
# dark mask for the spike-ball ring test - are restricted to a window around the previous frame's head.
# Measured over luping3 f150-500 (351 frames): 57.53 -> 39.48 ms mean (31.4% saving), head identical on
# 351/351.  This replaces the earlier all-window experiment, whose own head solver measured 84.9% exact
# with a 25-cell worst case and was therefore not safe to steer with.
HYBRID_WINDOW = True   # 10-18 恢复窗口+每3帧全局（关闭后峰值掉到14，反应力优先）；钻石靠跨帧记忆补全
#HYBRID_WINDOW = False  # 10-18 关闭：perception 注释写明"窗口只看到 9% 的钻石(全局 2.00/帧 vs 窗口 0.18)，所以 bot 不再传 box"。实测窗口模式 50% 的帧一颗钻石都看不见

# ============================================================ turn lead
# How early the next direction is sent before a corner, as a multiple of the measured speed.
# Calibrated live: the user reported the steering "略微偏早" (slightly early) after a match that was
# otherwise fine, so these are the previous effective values scaled down by ~20%.  The previous code
# used 2.3 / 1.2 with caps of 2.4 and 0.9 cells, i.e. 2.4 cells of lead at full speed.
#
# Do NOT remove the lead to fix "early": measured on 20 offline approach scenarios with a two-step
# control delay, a direct approach line without lead ate 14/20, a forced 4-cell entry line ate 13/20,
# and locking the direction for 1 / 2 / 3 frames after a turn ate 13 / 10 / 6.  The lead is load-bearing;
# only its size is in question.
TURN_LEAD_SPEED_MULT_FAST = 1.85   # speed >= TURN_LEAD_FAST_SPEED_PX
TURN_LEAD_SPEED_MULT_SLOW = 1.0
TURN_LEAD_MAX_CELLS_FAST = 2.0     # cells
TURN_LEAD_MAX_CELLS_SLOW = 0.75
TURN_LEAD_FAST_SPEED_PX = 25.0

# ============================================================ starvation brake
# Holding the boost costs about one cell per 5.7 s, so it needs roughly +0.18 cells/s of intake just to
# break even.  Measured live in the third match: early (t20-45s) the net was +0.16 cells/s and the snake
# went 5 -> 2 -> 1 -> 0 across three 10 s buckets, i.e. it was LOSING length while boosted; late
# (t45-79s) the net was +0.29 and it grew 8 -> 18 with the boost paying for itself.
#
# So while the snake is shorter than this, the boost is released and the snake keeps the cells.  It is
# re-enabled by itself as soon as the snake grows past the threshold, and the brake NEVER presses the
# key - it only ever stops holding it.
STARVED_LEN = 8
# OFF by default.  MEASURED LIVE and it failed: with the brake active the snake sat at 5-7 cells for the
# whole match (log live_20261012_130655) and finished on 5, the worst of the session.  The premise was
# that releasing the key stops the snake paying for the boost, but releasing does not help it EAT, and
# GAME_NOTES section 4 records that base speed SCALES WITH LENGTH (0.226 -> 0.349 cells/frame going from
# short to 7 cells) - so staying short makes the snake slower, it reaches less food, and it stays short.
# The brake manufactured the starvation it was named after.  Kept as a flag because the release path is
# now tested (`test_accel_routing`) and may be wanted for a different regime.
STARVATION_BRAKE = False

# Cells of margin added around a spike ball's swept forecast when building the danger map.
# A ball is a 2x2 object (user-stated; measured core 44-60 x 29-43 px) that drifts between vision
# samples, and it kills on contact with NO death buffer, so the swept line alone is not a collision
# zone.  Measured in the 4th live match the head repeatedly passed at Chebyshev distance 3 from the
# ball centre with an empty danger list, and then died entering a cell it had itself flagged.
# 1 cell means the ball's own 2x2 body cannot touch the head; 2 (used here) adds a drift allowance.
BALL_DANGER_MARGIN = 2

# How far away a DIAMOND may be and still unlock the wall approach, in cells (manhattan).
# `_prefer_inner_steps` normally refuses any step closer than EDGE_TURN_RESERVE to the wall, and its gem
# exemption only fired for a gem already at wall distance 1-2 AND within 10 cells of the head - so a wall
# gem approached from mid-board could never be unlocked: the step filter would not move the snake closer
# while the exemption was waiting for it to be closer, and the two rules waited on each other.  Diamonds
# are the entire objective (+5 cells each; the win condition is being longest), so the approach is
# unlocked from further out.  Measured live: a diamond by the right wall was never eaten.
GEM_WALL_APPROACH_CELLS = 18

# ============================================================ end-game countdown
# A large pale numeral is drawn over the arena for the final ~10 s.  The user reports it disturbs
# recognition and that the snake then dies in the last few seconds.  It is detected by its dark OUTLINE
# in the interior of the board: a large, roughly numeral-shaped dark component.  Balls are far smaller
# than this and the wall is on the border, so neither is mistaken for it.
# Keyed on the PALE FILL of the numeral (bright and near-neutral), not its dark outline: measured on
# real end-of-match frames the largest dark components are the snake's own outline (up to 1460 px,
# 314 px tall), so a dark test selects the snake.  A pale blob this large in the interior cannot be a
# snake (all snakes are saturated warm/cool) and cannot be a gem (a fraction of a cell).
COUNTDOWN_MIN_PX = 1500         # work-resolution pixels of bright neutral
COUNTDOWN_MIN_H = 40
COUNTDOWN_MIN_W = 20

# ============================================================ control timing
# Extra delay, in milliseconds, added to each loop cycle before the command is issued.  The user asked
# for the control timing to be put back by about 10 ms ("控制时间可以稍微延后10ms左右").  The loop already
# publishes its command at the END of each cycle using the frame captured at the START, so this value
# postpones the stick write within the cycle - the same "how early does the snake commit" axis as
# TURN_LEAD_*, which was shortened from 2.16 to 1.73 cells at full speed earlier in the same session.
# Set to 0 to restore the previous behaviour exactly.
CONTROL_DELAY_MS = 10.0

# How close, in cells, the head must be to an item before the boost is released to land on its centre.
# The user wants the boost held for the whole match and the pickups landed precisely, so the approach is
# made at speed and only the final approach brakes.  Previously the release happened while the item was
# still several cells away (logged as `eat_dia+align`), so the bot crawled up to its food.
ALIGN_BRAKE_CELLS = 2
# Release the boost for a momentary loss of the head reading (head_age > 2) or an uncertain head pixel?
# OFF, because the user's rule is that a BRIEF loss of visual evidence must not toggle the key: measured
# in the last full run, `head_jump` and `no_game_evidence` together released it on 146 of 748 frames
# (a fifth of the match) purely because a detector was unsure.  Vision uncertainty still lowers speed
# through the PATH conditions (runway, risk, wall distance), which is where a bad reading can hurt.
RELEASE_ON_LOW_TRUST = False

# ============================================================ holding the boost
# The user's priority: hold acceleration for the whole match ("尽可能整局保持加速").  Measured, the run
# where the key was held 98.6% of the time was the best of the session (18, won), while runs at 26-46%
# scored 8-9.  These two thresholds were releasing it on path conditions tighter than the risk they guard:
#   * runway: a 3-cell demand, when the planned turn needs about 1-2 cells (TURN_LEAD_MAX_CELLS_FAST 2.0);
#   * risk: 1.5 is a one-cell halo or a ball one cell away, and ball threat now has its own inflated
#     danger footprint plus risk-first ranking in the survival branches, so this guard was double-counting.
# The safety-critical terms are NOT relaxed: `not safe`, the wall reserve, boxed/buffer release and the
# death whiteout all still release the key.
ACCEL_MIN_RUNWAY = 2
ACCEL_MAX_RISK = 2.0

# Hold the boost except for a genuine safety block.  The user's priority, stated three times and finally
# with the reason ("尽量保持长按加速，切换状态代价较大" - hold the boost, switching state is expensive).
# OFF would restore the survival-first behaviour of releasing on alignment and on soft path caution.
#
# Releases that SURVIVE (they can kill the snake):
#   * `not safe`            - the chosen step is not passable
#   * wall reserve          - no room for the planned turn
#   * buffer / mash         - in a crash buffer and turning out
# Releases that DO NOT survive (they only cost speed):
#   * precision/alignment   - the follower steers and arrival is judged by pixel distance, so the snake
#                             can cross an item's centre at speed; releasing here toggled the key on
#                             every single approach, which the log shows frame by frame
#   * runway / risk / low_trust / starved - caution, not danger
HOLD_BOOST_EXCEPT_SAFETY = True

# Cells of wall clearance required before the boost is held.  MEASURED with the new `release_why` log:
# the previous value (EDGE_TURN_RESERVE + 1 = 4) accounted for 291 of 297 releases in a run, i.e. the boost
# was off almost entirely because the snake was within four cells of some edge - a wide band on a 29x25
# board, and exactly where gems "tend to spawn" (see the note in policy.py).  A boosted snake moves about
# 1 cell per control frame and the planned turn needs about 1-2 cells (TURN_LEAD_MAX_CELLS_FAST = 2.0), so
# 2 cells is room for the turn that is actually made.  Set to EDGE_TURN_RESERVE + 1 to restore the old
# behaviour.
ACCEL_WALL_RESERVE = 2

# ============================================================ the final seconds
# Measured (death_tail.py, live): the snake spent ~2.9 s grinding in and out of the crash buffer before
# dying at t=70.1 -- eye=buffer at t=67.7 and t=68.2, balls 3 within 4 cells, length bleeding
# 11 -> 8 -> 6 -> 0.  The existing endgame mode (match_elapsed >= 50 s) only stops item chasing and
# re-ranks steps by risk; it does not stop the snake driving through a crowd, and the 7-frame buffer limit
# makes each fresh contact nearly fatal.  This window adds the situational part.
ENDGAME_WINDOW = 15.0
ENDGAME_MIN_CLEARANCE = 3      # never take a step with fewer free neighbours than this
ENDGAME_BALL_WEIGHT = 1.0      # how strongly to prefer distance from the nearest ball

# Observed match length: 975 frames at 15 fps = 65.0 s (GAME_NOTES section 0).  NOTE that
# MATCH_MAX_SECONDS (80.0) is a hard stop for the bot's loop, NOT the match length - timing the endgame
# from it would place the final window at 68 s, past the real end, making the mode dead code.
MATCH_SECONDS = 65.0

# Floor for arming the final-seconds mode, independent of when the bot recognised the start overlay.
# MEASURED: `match_elapsed` is `now - match_started`, and `match_started` is set when the START overlay is
# recognised, which lagged by seconds in several runs - the bot's clock ran 17.3..71.2 s for a 65 s match,
# so `match_elapsed >= 50` held for under two seconds and the endgame was effectively never armed (the log
# shows `endgame_survive` almost absent, with the snake still on `eat_dia` past t=55).  Arming on the
# earlier of the two signals is safe because this mode only makes the snake cautious.
ENDGAME_ARM_MIN = 38.0

# ============================================================ when the score is taken
# The user: "结算分数即为60秒那一刻红蛇的长度，与中间是否死亡等基本无关" - the score IS the red snake's
# length at the 60 s mark, essentially regardless of whether it died in between.  Consequences, both of
# which reverse assumptions this policy was built on:
#   * the value to maximise is the length AT 60 s, so the last stretch before 60 s is still worth
#     collecting in rather than being given over to pure survival;
#   * DEATH DOES NOT ZERO THE SCORE.  GAME_NOTES records luping2 finishing on 0 after a death, and I took
#     that as "dying zeroes the score", which pushed the whole priority order towards caution.  If the
#     score is the length at 60 s, a death before 60 s costs the cells lost, not the whole result.
MATCH_SCORE_SECONDS = 60.0
# The strict no-risk window now covers only the time AFTER the score is taken, when nothing can be gained
# and a death can no longer be paid for.
ENDGAME_WINDOW = 5.0
# Floor for arming that strict window from the bot's own clock.  It is set ABOVE MATCH_SCORE_SECONDS minus
# ENDGAME_WINDOW on purpose: the bot's clock starts when it recognises the START overlay and that lagged by
# seconds (measured: the clock ran 17.3..71.2 s for a 65 s match), so a floor below 60 would arm the strict
# mode while the score was still being decided.
ENDGAME_ARM_MIN = 55.0

# ============================================================ final-stretch discipline (no-boost)
# MEASURED 2026-10-17 evening, one candidate version, ten matches, settlements 6,16,19,17,6,6,12,8,0,7:
# the snake grew to 37-43 cells in four matches and then died between 45 s and 58 s, wiping the score.
# Twelve deaths in ten matches.  The last item target before six of them sat in that final stretch and was
# a LOSING TRADE rather than collection: a ONE-CELL bean (M8 held 41 cells, chased a bean 4 cells away
# beside a spike ball at 47 s, died at 49 s; M9 held 37 and did the same at 57 s), a gem reported ON the
# head cell with no route to steer (M5, M6), or a 10-cell errand at 55 s (M10).  In no-boost mode the old
# endgame flags only armed at MATCH_SCORE_SECONDS (60 s), i.e. after the score had been taken, so neither
# ever fired.  These gates remove only the losing trades; LOCAL gems -- the main source of length -- are
# still collected through the score moment, which is why PROTECT_LENGTH stays at the goal itself.
LATE_GUARD_SECONDS = 15.0        # discipline starts at MATCH_SCORE_SECONDS - this (45 s)
LATE_GUARD_STRICT_SECONDS = 7.0  # ... strict from 53 s
LATE_HOLD_LENGTH = 32            # from this length the final stretch takes only local gems
LATE_BEAN_DIST = 2               # a bean is worth ONE cell: only this close while late
LATE_STRICT_DIAMOND_DIRECT = 2   # inside the strict window only a gem essentially under the head
# A green item adds NO length directly, but it raises speed by ~1.8x, which multiplies how many gems a
# match can reach -- the user asked for it three matches running in the fifth batch ("缺少获取加速道具",
# "吃钻和加速道具在最后不够有策略，需要富贵险中求，但是要准").  An earlier cut of 6 cells was too timid
# and is reverted: the chase is allowed wider, and the "要准" part is supplied by the ordinary safety
# checks (checked exit after the pickup, diagonal-ball ring, no late-strict errand).
GREEN_MAX_DIRECT = 12
GREEN_LATE_VALUE = 3.5           # green's utility inside the final stretch (speed still pays before 60 s)
GREEN_NORMAL_VALUE = 4.5
# Distance from OUR head at which the survival route replaces an uncommitted item target.  This is the
# user's goal length, deliberately NOT lowered: a 35-cell snake at 49 s must still collect toward 40
# (see test_collection_safety.test_late_thirty_five_still_collects_toward_forty).
PROTECT_LENGTH = 40

# ============================================================ adjacency costs around hazards (no-boost)
# The user, three matches in a row: "不要一直贴着别的蛇的蛇身走", "提高对刺球的警惕！不要长时间贴着别的蛇走".
# In no-boost mode `_risk_map` used to mark ONLY the ball's own swept footprint, so a lane one cell beside a
# ball or an enemy body was a free move and the planner happily ran alongside them for seconds.  Contact is
# what kills (the wall itself is not a hazard), so these are COSTS, not walls: they only re-rank the four
# candidate steps, which keeps a forced tight passage available (the standing "no fixed one-cell margin"
# rule) while making sustained hugging the planner's last choice.
#   * the ball ring goes into the ordinary risk map, because a gem sitting in a spike ball's lane is the
#     exact trade that lost two matches (M8 held 41 cells, M9 held 37);
#   * the body ring does NOT: `risk` also gates item approaches, and at 0.9 in that map every gem beside an
#     enemy body became unapproachable and the snake thrashed between far gems (live 19:04 M1, peak 7).
#     It is applied to the candidate steps after `_first_allowed` instead, so it can only re-rank moves.
BALL_ADJ_RISK = 1.2              # a cell touching a spike ball's footprint (mode unknown)
OPP_BODY_ADJ_RISK = 0.9          # a cell touching an opponent's body (move cost only)

# 用户（第四批后）给的机制：三个刺球一个横向移动、一个纵向移动、一个斜向移动，**只有斜向的必须远离**。
# 横向/纵向只扫一条线，站开一格就安全；斜向的那条线横切退路，才需要大范围躲。
# 这条同时决定了"球旁边的钻石能不能抢"：`_risk` 里 >=0.8 会挡住钻石可接近性，所以
# 横/纵的邻接代价必须压在 0.8 以下（可以抢），斜向的才抬高。
BALL_DIAG_ADJ_RISK = 1.5         # 斜向球：邻接代价，高于闸门线 => 不吃它旁边的钻石
BALL_DIAG_RADIUS = 2             # 斜向球的远离半径（格，代价层）
BALL_DIAG_HARD_RADIUS = 0        # 用户：刺球移速很慢（一局约 20 格），不需要整圈封死；只留球体本身 + 代价层
BALL_AXIS_ADJ_RISK = 1.05         # 横/纵球：只给紧邻那条线一个排序代价，仍可抢它旁边的钻石

# ============================================================ 方向连续性（脱困/死亡缓冲）
# 用户："方向控制必须是连续的，不能按了左立刻按右，按这个规则编写蛇头脱离死亡缓冲态的方案。"
# 摇杆从 -1 瞬切到 +1 既不是合法转向，也会让蛇头在两格之间打摆；死亡缓冲只有几帧，抖动等于全浪费。
# 规则：缓冲/脱困期间禁止与上一步相反的方向（除非它是唯一可行步），且同一方向至少保持这么多帧。
BUFFER_DIR_HOLD = 3

# ============================================================ collection vs. the opponent
# The user, 2026-10-17 evening, after the second batch: "吃钻石的过程不要贴着敌蛇，钻石在敌蛇旁不仅想办法要
# 抢还要顺便把敌蛇蛇头堵住。钻石路线长度上限设到15格。"
# Three separate requirements, deliberately implemented three different ways:
#   * the ROUTE to a gem costs a lane beside an opponent body, so the approach goes around it
#     (`self._route_risk` = risk + hugging lanes).  It is a cost, never a gate: a gem that only has a
#     hugging approach is still reachable, which is what "不仅要抢" asks for -- the live failure of
#     19:04 was exactly the opposite (the ring gated approachability and the snake starved);
#   * when the gem sits near an opponent's HEAD, the approach that also covers the cell in front of that
#     head is preferred, so the gem is taken and the head is boxed in the same manoeuvre ("顺便把敌蛇蛇头
#     堵住").  Only a preference among already-viable approaches, never a reason to detour or to give up
#     the gem;
#   * a diamond whose ROUTE is longer than DIAMOND_MAX_ROUTE is not pursued.  Measured in the second
#     batch: six of the eight early/mid deaths had a 15-28 cell gem route as the last target.
DIAMOND_MAX_ROUTE = 15           # cells of route, not straight-line distance
# 上限限制的是"主动远征"，不该把"绕障多走几步"也算成远征：刺球占 3x3，一颗直线 12 格的钻石路线会顶到
# 16 格，按纯上限会被误杀。因此路线超过 15 格时，只有直线也很远的才放弃；直线在 12 格以内的近钻，
# 允许它因为绕障而多走几格。
DIAMOND_NEAR_DIRECT = 12
BLOCK_GEM_NEAR = 5               # gem within this Chebyshev distance of an opponent head ...
BLOCK_GEM_BONUS = 1.18           # ... multiplies the score of an approach that also covers its front cell

# ============================================================ 复活保护
# The user reported this twice: "复活后不能及时行动（刚复活为1格）" and, for the third batch's last match,
# "结束时追尾巴但没有正确".  The batch logs show the head is re-acquired badly right after a respawn
# (40-110 `head_jump` frames), so for the first few frames the policy must not chase, must not inherit the
# pre-death route and must not try to chase a tail that has not formed yet: open space only.
# A respawn is detected as a large DROP in measured length, because `whiteout` is a bot.py log flag and is
# not part of the state mapping the policy receives.
RESPAWN_GUARD_LEN = 5            # only interesting while the body is this small ...
RESPAWN_GUARD_FRAMES = 8         # ... for this many control frames (~0.4-0.5 s at 15-20 fps)

# ============================================================ 追尾只在长蛇时用
# MEASURED 第六批：六局零死亡，但零死亡局只有 14—40 格。原因是追尾被当成了默认生存方式——
# 第 9 局零死亡、`eat_tail` 占 416/797 帧（52%），长度只有 14：追尾很安全，但它不产生任何长度。
# 追尾原本是"无钻石目标时的兜底"，短蛇用它是纯粹的浪费时间；门槛以下改为巡游（`_safe_roam_target`
# 会朝开阔地走 11 格，路上遇到钻石就会重新进入收集）。
TAIL_FOLLOW_MIN_LENGTH = 0        # 0 = 不设门槛（第九批设 25 后零死亡局反而停在 37 格）

# ============================================================ 转头次数上限（防绕圈）
# 用户第六批："不要让蛇头在短时间内（2秒内）连续三次以上同方向转头（脱离死亡缓冲时放宽到四次）！"
# 同方向连续转头就是绕圈：实测第 8 局"经常无意义的转圈"、第 5 局"吃了加速后直接自杀"，都是这个形状。
# 规则：在 TURN_WINDOW_SECONDS 窗口内，同一旋转方向（顺时针/逆时针）的连续转向不得超过上限；
# 达到上限后，只能保持直行或向反方向转（反向转头会重置计数）。缓冲/脱困态放宽一档。
TURN_WINDOW_SECONDS = 2.0
TURN_REPEAT_LIMIT = 3
TURN_REPEAT_LIMIT_RECOVERING = 4
# 但"连续转向"在被围困时是**求生**而不是绕圈：用户第一批就要求"被身体包围时沿内侧来回折叠"，
# 折叠本身就是连续同向转向。第七批（加了这条限制）死亡从 0.40/局 回升到 1.30/局，四个 0 分局。
# 因此只有头部周围可用面积大于这个阈值时才限制绕圈；空间不够时让蛇自由折叠脱困。
TURN_LIMIT_MIN_AREA = 20

# How close, in cells, the head must be to an item's centre before the follower commits to crossing that
# centre rather than steering for the next waypoint.  1.6 cells was the value already used inside the
# last-mile branch; it is now applied regardless of how many waypoints remain, because gating it on
# `len(route) <= 2` meant it almost never ran for a gem several cells away - and the snake turned away one
# moment before eating (user report).
PICKUP_COMMIT_CELLS = 1.6

# Threat assigned to a spike ball that is not moving.  It used to be 0.0, and `MotionModel.danger()` skips
# any track with `threat <= 0`, so a ball that had not moved yet contributed NOTHING to the danger
# forecast - while being exactly as lethal as a moving one (contact kills with no death buffer).  Measured
# over the recent live logs: on 42 frames a ball was within 3 cells of the head and on 16 of them (38%) the
# forecast was empty or pointed elsewhere, including frames with the ball at distance 0.  Set to the axis
# threat because an unmoving ball still occupies its cells and the head still has to go around it.
BALL_STILL_THREAT = 0.35

# ============================================================ crash-buffer handling
# Consecutive frames of `buffer` eye state required before the escape mode is armed.  MEASURED (series match
# 2): `buffer` appeared in 17 episodes and MOST WERE ONE FRAME, while the death buffer is a STOPPED snake, so
# isolated sightings during normal driving are misreads.  Each sighting armed an 18-frame escape during which
# every frame released the boost (`why='buffer_turn'`), and the log shows 101 frames with the boost off after
# one respawn - the user's "复活后缺少加速".  Requiring a second confirming frame stops a flicker arming it.
EYE_BUFFER_CONFIRM = 2
# How long the escape mode lasts once armed.  GAME_NOTES section 7.1: escapes succeed in 3-5 frames and fail
# at about 7, so 7 is the whole life of the state; the previous value of 18 ran three times longer than the
# buffer can possibly last, which is why a single misread cost so much speed.
BUFFER_MAX_FRAMES = 7

# Radius, in cells, within which the policy refuses a step that increases the perpendicular offset to a
# GEM's centre while that centre is still ahead.  The follower aligns the lateral lane before driving
# forward by design (see test_last_mile_aligns_lateral_lane_before_driving_forward), so on a diagonal
# approach it turns before crossing the gem - measured live as "错过了钻石才转弯".  This guard is in the
# policy, so the follower's documented behaviour and its tests are untouched.  It only ever removes a step
# that is legal, gem-directed and moving AWAY laterally, so it cannot force an unsafe move.
PICKUP_LANE_GUARD_CELLS = 2.2

# ============================================================ control delay by speed state
# The user: "建议根据蛇的速度分三档控制延迟，常态延迟可以较高，按键加速延迟偏小，按键加速同时吃到绿豆延迟最小."
# The delay postpones the command within the loop cycle, so it adds to the age of the frame the command is
# based on; the position error from a stale frame is proportional to speed, which is why the fastest state
# gets the smallest delay.  Measured this session (speed_px, 32 px = 1 cell per control frame):
#     boost off  median 17.3 px      boost on  median 31.2 px      boost + green  ~0.91 cells/game frame
CONTROL_DELAY_MS_NORMAL = 13.0     # no boost: slower snake, so a later command costs least
CONTROL_DELAY_MS_BOOST = 10.0      # A held (the value that used to apply to every state)
CONTROL_DELAY_MS_BOOST_GREEN = 6.0 # A held and a speed item running: fastest, correct soonest

# Half-width, in cells, of the window searched for the eyes.  6 cells = 192 px each way = a 384x384 crop of
# the 928x800 frame (17% of the pixels), and `white_slits` costs 17.7 ms because it runs a connected-component
# pass, a nonzero and an argsort over the whole frame.  The head moves at most about 2 cells per frame
# (user-stated, and measured: 1.00 cell per control frame at boost), so a 6-cell window centred on the
# previous head keeps the eyes inside it.  If the window yields nothing the FULL frame is searched, so this
# can never reduce detection - only the fallback rate changes.
EYE_WINDOW_CELLS = 6

# How close the windowed and full-frame best eye-pair midpoints must be before the windowed result is
# accepted, in pixels.  4 px is a tenth of a cell: the two searches are looking at the same image, so a real
# agreement is exact up to rounding, and anything larger means they found different pairs.
EYE_WINDOW_MATCH_PX = 4.0

# ============================================================ eye search window
# OFF by default: MEASURED and rejected.  Cropping the eye search to a window around the last head does shrink
# `white_slits` (17.7 ms, 31% of the perception pass), but the window's answer CANNOT be used unverified -
# with a 6-cell window it disagreed with the full-frame search on 10 of 41 recorded frames, once by 300 px -
# so the full search has to run anyway as a check.  Swept widths 4..16 cells: head agreement is 30/10 at every
# width and the pass costs 58-67 ms against a 56-60 ms baseline.  The crop is a candidate generator that is
# never cheaper than the search it was meant to replace.
#
# The mechanism is kept (gate and counters included) because it is correct and re-testable, and because the
# counters are what showed it could not work.
EYE_WINDOW = False

# ============================================================ no-boost collection mode
# The user's new target: "红蛇不按键加速且结算长度在50以上", based on their own luping8 run which reached 84 cells
# with the key never pressed.  Measured from that recording with this pipeline: head displacement 0.37
# cells/game frame (base speed, so the boost really was off), length 5 -> 83, and 40% of gem sightings at wall
# distance 0-1.
#
# With NO_BOOST the whole match is spent collecting rather than paying the boost's ~1 cell per 5.7 s, so:
#   * the policy never requests the boost;
#   * the wall-hugging penalty is disabled, because it is what made border gems unattractive - and border
#     gems are 40% of what this match needs. Since 2026-10-17 NO_BOOST also bypasses
#     fixed body/wall clearance; actual occupied cells remain collision obstacles.
NO_BOOST = True
# The gem commit radius, in cells.  A base-speed snake crosses a cell in about 2.7 frames against 1.4 when
# boosted, so the window can be wider without overshooting; this is the "较为精准地吃到钻石" the reference shows.
PICKUP_COMMIT_CELLS_NO_BOOST = 2.0

# ============================================================ neck exemption and border greed
# User: "建议自己蛇脖子那几格不用非得和蛇头避开1格远" (the cells in our own neck do not have to be kept a cell
# away from the head).  `_body_gap` measures the distance to our own body while excluding the neck; the
# exclusion was `delay >= 2 and c not in path[:3]`, and since delay >= 2 already drops the head and the first
# neck cell, the slice was doing almost nothing.  Raising this lets the snake hug its own body when turning -
# the "turn tightly around itself like a mountain road, with no gaps" behaviour the user describes.
NECK_EXEMPT_CELLS = 6
# User: "墙边的钻石增加冒险去吃的策略" (add a strategy that takes the risk to eat border gems).  Border gems are
# 40% of the sightings in their own reference match.  While a gem is within this many cells of the head AND its
# own cell is within WALL_GEM_GREED_RADIUS of the wall, the wall reserve relaxes to 0 for the approach, so the
# head may reach the very edge.  `_passable` still forbids entering a wall cell, an opponent or a ball, so the
# risk is bounded to giving up the one-cell margin, not to driving into anything.
WALL_GEM_GREED = True
WALL_GEM_GREED_CELLS = 6
WALL_GEM_GREED_RADIUS = 1

# ============================================================ turn timing by cell phase
# The user's mechanic: "按键转向实际上仅在蛇头处于未占满当前格的时候有效，0.5格的时候控制转向最佳" - a turn input
# is only effective while the head has not filled the cell it is entering, and half a cell into it is best.
# Turns used to be timed by distance covered (`remaining` and a speed-scaled `lead`), which has no notion of
# the phase, so they were issued at arbitrary points in the cell.
TURN_PHASE_GATE = True
# The fractional position inside the current cell at which a turn is issued, and how far either side of it the
# gate stays open.  0.5 with 0.25 of tolerance covers the middle half of the cell.
TURN_PHASE_BEST = 0.5
TURN_PHASE_TOL = 0.25
# If the phase window has been missed and the head is this far into the cell, issue the turn anyway: the gate
# must never make the snake miss its corner.
TURN_PHASE_MAX_WAIT = 0.85

# ============================================================ obstacle margin is a preference
# User, twice: "隔障碍一格远是为了留生路而不是限制！在逃跑时可以贴着障碍！" and "贴着障碍不会死，只有撞上去才会！
# 放弃远离障碍改为合理规划路线".  Touching is harmless; colliding is fatal.  So while escaping - recovering from a
# crash buffer, or in the endgame - the wall reserve drops to 0 and the snake may hug the wall.  `_passable`
# still forbids entering a wall cell, a body or a ball, so the collision rules are unchanged.
HUG_ON_ESCAPE = True

# A gem must be seen on this many frames before it may be a target.  MEASURED on the reference match: gems
# appeared 268 times while only 22 disappeared at the head, so most appearances are flicker - and the user
# reports "似乎在虚假的钻石处做遍历" (orbiting a gem that was not there).
GEM_MIN_DWELL = 2

# ============================================================ chase the own tail when long
# User: "蛇太长的时候可以追着自己蛇尾跑，这样是非常安全的".  A long snake's body follows the head's own path, so
# a head that stays behind its own tail has nothing to hit.  This is the natural answer to "自己把自己在最后堵死了"
# (trapping itself at the end) and gives the snake somewhere safe to go when no item is reachable.
#
# It is a FALLBACK: a reachable item always wins, and it only engages when the snake is long enough for its own
# body to be the dominant obstacle.
TAIL_CHASE = True
TAIL_CHASE_MIN_LEN = 12          # below this the body is not a serious obstacle
# Engage when the head's reachable area, with the body treated as SOLID, is at most this many cells.  The gate
# must use a body-aware count: `flood()` measures cells the body will VACATE and reports 705 of 725 even on a
# 21-cell coil, which made the gate impossible to satisfy.  The value is kept as measured rather than loosened
# to fire more often - tail chasing gives up collecting, and length is the scoring criterion.  NOT YET
# DEMONSTRATED to change a match: it never engaged in a whole replayed reference match (the policy always had
# an item) and the compact coils tested leave about 11 free cells, just above this threshold.
TAIL_CHASE_AREA = 10

# ============================================================ opponent tail chase and head block
# User: "别的蛇的蛇尾也是安全的，除非它蛇尾不动。追别的蛇尾，堵别的蛇头！" and "建议尝试将其它蛇堵死而非完全避开".
# The opponent splits into two objects with opposite treatments: its TAIL is safe to follow WHILE MOVING (that
# cell is being vacated), and its HEAD is what to block, by occupying the cells ahead of it.
OPP_TAIL_CHASE = True
# MEASURED AND TURNED OFF.  The user asked for head blocking ("堵别的蛇头"), and it was implemented as "drive to
# the cell OPP_BLOCK_AHEAD in front of the opponent's head".  Over a full series it made both deaths and score
# worse - deaths 0.80 -> 1.00, score 14.1 -> 11.4 - and the death cause `trapped` rose from 3 to 8 occurrences in
# the frames preceding the deaths, because driving at an opponent's front is also driving into a cramped region.
# Re-enable only with a version that requires a LARGE escape area behind the blocking cell, so it restricts the
# opponent without restricting us.
OPP_HEAD_BLOCK = False
# A tail is only "moving" if its cell changed within this many frames of our observations.  A stationary tail is
# a static obstacle: following it would drive into a body that is not clearing.
OPP_TAIL_MOVING_FRAMES = 6
# How far ahead of the opponent's head to aim when blocking, in cells.
OPP_BLOCK_AHEAD = 2
# Only consider blocking an opponent within this many cells of our head.  Chasing one across the board would
# waste the match; the point is to restrict a snake that is already near us.
OPP_BLOCK_MAX_DIST = 8

# ============================================================ hug the wall
# User, three times: "可以贴墙走，隔障碍一格远是为了留生路而不是限制！" / "似乎不会贴着左右的墙，修改让红蛇能够贴着墙走！"
# / "贴着障碍不会死，只有撞上去才会！".  The wall is not a collision risk; only entering a wall cell is, and
# `_passable` already forbids that.  The one thing keeping the snake off the wall is `_prefer_inner_steps`, which
# requires candidate steps to be at least `reserve` cells from it.  With this on and NO_BOOST active, the reserve
# becomes 0, so the head may run along the outermost playable line - "贴着墙走".
HUG_WALL = True

# ============================================================ 比赛计时器（时钟对齐）
# 顶部 HUD 的倒计时不在 CAPTURE_BBOX 内（捕获区从 y=269 开始，计时器在 y≈40—90），要单独抓一小块。
# 它才是真正的比赛时钟：昨夜 bot 的 match_elapsed 从"识别 START"起算，实测与真实比赛差 0—14 秒，
# 末段保护窗口与计分时刻因此全部错位（这是"峰值 52 却结算 0"的主要来源）。
TIMER_BBOX = (1152, 42, 1424, 104)   # 七段数码管计时器（含外框），实测位置     # left, top, right, bottom on the 2560x1440 primary screen
TIMER_MIN_ORANGE = 200                # 兼容旧名
TIMER_ORANGE_MIN = 200                # 计时器区域亮像素下限（大厅该区域全黑，实测 0）

# ============================================================ 长蛇的"留在大空间"护栏
# MEASURED 2026-10-18 第一批第 8 局：峰值 60、结束 49，末尾 3 帧被 `trapped`（四个方向全被封），
# 随后 head_jump 死亡，结算 **0**。长蛇真正致命的不是吃不到钻石，而是把自己塞进窄缝后出不来。
# 规则：体长达到 TIGHT_GUARD_LENGTH 后，只要还有别的选择，就不走"落下后活动面积小于 TIGHT_GUARD_AREA"
# 的方向。它只从既有可行步里剔除，若所有方向都会变窄则不过滤（不会把蛇钉死）。
TIGHT_GUARD_LENGTH = 10        # 用户：短蛇也会把自己绕进死角，护栏不能只给长蛇
TIGHT_GUARD_AREA_SHORT = 16    # 体长 < 35 时要求的最小活动面积
TIGHT_GUARD_AREA = 26

# ============================================================ 钻石成组（用户 10-18 第 1 点）
# 用户："多个钻石相距 7 格以内时规划能吃到所有钻石的合理路径，若中途被其余蛇干扰需及时反应。"
GEM_CLUSTER_CELLS = 7      # 切比雪夫距离内视为同一簇
GEM_CLUSTER_BONUS = 0.15   # 每颗邻居钻石的加分系数（最多按 3 颗计，避免极端贪婪）

# ============================================================ 比赛时钟固定偏差（用户 10-18 第 3 点）
# "不要惦记游戏顶部的计时器，知道大致偏差调整实际计时就行"。实测 START overlay 有 2—3 秒动画，加上识别
# 延迟，`match_elapsed` 从识别 START 起算会比真实比赛时间**偏小 2—5 秒**。用这个常量补偿，
# 让末段纪律窗口落在真实 45/53 秒而不是更晚。
MATCH_CLOCK_OFFSET = 3.5

# ============================================================ 追尾只在被围困时用
# MEASURED 今天第三批第 2 局：零死亡，但 `eat_tail` 占 396/760 帧（52%），长度只有 20 ——
# 追尾安全但零产出，开阔时用它等于把比赛时间扔掉。用户第四批也说过"追尾策略未能正确使用"。
# 规则：只有头部周围活动面积 <= TAIL_CHASE_AREA 时（真的被围住了）才允许追尾脱困；
# 开阔时的"没目标"改走巡游/占位，那些至少会朝钻石与开阔地移动。
TAIL_CHASE_ONLY_WHEN_TIGHT = True

# ============================================================ 小圈检测：意识到在绕圈就往外拐
# 用户（10-18）："一定要避免自我围困，当意识到蛇行成一个小圈时一定要往外拐！被自我围困后基本只能等死！
# 短蛇现在时常一直绕圈不作为，要稳准狠地获取钻石并成功脱离险境！"
# MEASURED 今天 28 次死亡里 17 次是 `trapped`，且很多发生在体长 9—21 的短蛇身上 —— 那不是"长蛇缠住自己"，
# 而是蛇在小范围内反复绕圈把出口绕没了。规则：
#   * 头部最近 CIRCLE_WINDOW_FRAMES 帧都落在 CIRCLE_SPAN 格的方框里 -> 判定"在绕小圈"；
#   * 触发后 CIRCLE_BREAK_FRAMES 帧内，剔除所有"朝圈心方向"的步，只往外走；
#   * 短蛇还额外优先朝最近的钻石巡游，避免"没目标就原地绕"。
CIRCLE_SPAN = 6
CIRCLE_WINDOW_FRAMES = 50
CIRCLE_MIN_SAMPLES = 25
CIRCLE_BREAK_FRAMES = 25

# 短蛇（< SHORT_SNAKE_LEN）在没有合格目标时，先朝最近的钻石/开阔地巡游，再考虑"找空间占位"。
# 用户 10-18："短蛇现在时常一直绕圈不作为，要稳准狠的获取钻石并成功脱离险境！"
SHORT_SNAKE_LEN = 25

# 用户 10-18："采用多线程增加红蛇的注意力和反应力"。这里先做的是削减每帧的串行计算：
# 头部窗口把感知 52 ms -> 37 ms，但窗口外看不见钻石（gems/frame 1.8 -> 0.6）。
# 折中：每 HYBRID_GLOBAL_EVERY 帧做一次全局重扫，其余帧只扫头部窗口。
HYBRID_GLOBAL_EVERY = 3

# ============================================================ 末段纯逃避（用户 10-18）
# 用户："很多次蛇吃得很长但是在最后几秒错误操作丢失注意力或者其他原因死得很愚蠢导致得分很低。"
# MEASURED：结算取 60 秒那一刻的长度，那一刻只要在死亡动画里就记 0（多次峰值 49/52 的局结算 0 或 41）。
# 所以最后一段不该再"吃"，只该"活"：不接新目标、不主动转向，只沿已证明可行的最长跑道直行；
# 只有前方被挡时才改向，且改向也要选跑道最长的那个。
FINAL_GUARD_SECONDS = 0.5   # 停用（实测叠加后变差，见 2026-10-18 复盘）： 距计分时刻这么多秒起进入纯逃避（配合 MATCH_CLOCK_OFFSET）
FINAL_GUARD_LEN = 36          # 只对已经够长的蛇生效：35 格以下末段仍要冲 40，不能提前停吃

# ============================================================ 长蛇的主动空间管理（用户选 B）
# 用户 10-18："学会被自己围住之后的脱离方法，不要一直内圈到死；较短和较长时采用不同策略"
# 以及"很多次吃得很长但最后几秒死得很蠢"。核心不是"被围住再脱困"（试过大圈脱出，反而更差），
# 而是**长蛇根本不进窄区**：≥ LONG_SNAKE_LEN 时选步把"前方跑道"放在第一位。
LONG_SNAKE_LEN = 999   # 停用（跑道优先叠加后实测变差）

# 长蛇吃钻的"落点空间"要求：吃完这颗后，落点周围至少还要有这么多的活动空间余量。
GEM_EXIT_ROOM_MARGIN = -999   # 停用（落点空间检查叠加后实测变差）

# 普通帧（非缓冲/脱困）下同一方向至少要连续保持的帧数。取证见 实机记录/round_001：
# 两次死亡前都出现了裸的左右反转，之后 5—6 帧 trapped 并死亡。
NORMAL_DIR_HOLD = 2

# 用户（10-18）："从 290 到 350 这段路径，选择了 3 个钻石中最远的！"
# 根因：**最近的钻石被出口见证检查拒掉**，候选里只剩远处那颗，"最近"就变成了"最远"。
# 近距离钻石的接近路线很短，允许降级为"先接近、不承诺完整出口"（每帧都会重新规划）。
NEAR_GEM_LENIENT_ROUTE = 4

# ============================================================ 死亡检测与等待复活
# 用户（10-18 逐帧指正）："蛇头在 (10,14) 的时候就应该往下走了；识别到 (9,14) 说明已经死了，
# 识别到的是死亡闪光特效！675 到 740 为死亡后的等待复活时间。"
# 也就是说：死亡瞬间到复活之间约 65 帧（4 秒多），红掩膜抓到的是**死亡闪光**，被当成红蛇在盘面上乱跳。
# 策略如果继续按这个假位置决策，既浪费又危险。规则：长度骤降 + 头部跳变 -> 判定死亡并静默等待，
# 直到重新出现长度 >= DEAD_REVIVE_LEN 且位置连续的红蛇（配合原有的复活保护一起冷启动）。
DEAD_LEN_DROP = 5           # 上一次长度 - 当前长度 >= 这个值视为一次死亡
DEAD_LEN_AFTER = 4          # 且当前长度 <= 这个值
DEAD_WAIT_FRAMES = 70       # 最多静默等待这么久
DEAD_REVIVE_LEN = 6         # 长度恢复到这么多且位置连续才算复活

# "可用空间不足就保持直行"的余量：所有方向的 flood 面积都 < 蛇长 + 这个值 -> 保持当前航向。
# 用户 10-18："580 处就应该保持直行，这里的可用空间根本不足以让红蛇脱困"。
# 不能再用 escape_need（上限 24），那对 26 格以上的蛇根本没有意义。
HOLD_AREA_MARGIN = 4       # 第 15 局配置：空间不足时保持直行

# ============================================================ 钻石跨帧记忆（呼吸灯）
# 机制文档 §8.2："所有道具都有呼吸灯效果，亮度周期性变化；尺寸门限必须为这个波动留余量，
# 否则道具有节奏地忽隐忽现；不会动的道具位置应按最近若干帧出现最多的那一格取"。
# 实录（实机记录/round_005 f=575—590）：同一批钻石在 f=577/583/586/589 能检出全部 3 颗，
# 而 f=580 等帧检出 0 颗 —— 画面里钻石一直在。若不补，策略就有一半的帧"看不见目标"，
# 这正是"没有目标只能巡游"的直接来源。
DIA_MEMORY_FRAMES = 6     # 记忆窗口（帧）
DIA_MEMORY_HITS = 2       # 窗口内至少出现这么多次才补回

# 吃完钻石后的撤离路线长度（格）。用户 10-18 第 10 局 f=510：
# "吃完眼前的钻石后，应该去吃 (24,4) 的钻石，但是却选择了向下拐去不明位置，似乎是因为吃完钻石后
#  设置的撤离路线"。原来是 8 格（escape[1:9]），足够把蛇带离下一个目标的方向；缩到 4 格，
# 只保留"别马上撞墙"的作用，交给下一帧重新选目标。
ESCAPE_ROUTE_CELLS = 4

# ============================================================ 叠层脱困（用户 10-18 第 11 局说明）
# "吃完加速道具后就应该考虑脱困，比起现成的、不确定的出口，选择**自己制造出口**：
#  蛇头在 440 帧左右下拐，沿 (27,5)—(26,5)—(26,0)—(27,0)—(27,6) 从蛇尾让开的位置脱困。"
# 实现：头部周围的紧凑可用空间装不下整个身体时（tight < length + 余量），把目标设成自己的尾巴，
# 沿身体边缘平行叠层走过去，让尾巴在移动中把通道腾出来 —— 而不是赌某个方向也许有出口。
# 原来追尾的门槛是固定的 TAIL_CHASE_AREA=10，20 格以上的蛇基本用不上，所以这项战术从未生效。
STACK_ESCAPE_MIN_LEN = 9999   # 停用：第 14 轮打开后峰值掉到 17、死亡 3 次（追尾会把头送进自己身体）
STACK_ESCAPE_MARGIN = 5

# ============================================================ 一格宽的窄路（用户 10-18）
# "始终贴着最近的已占格走且要避开一格宽的窄路；一般不要进入非得使用叠层策略的情况最好。"
# 一格宽通道一旦被对手或自己的尾巴堵住就是死路，也没有转身余地。所以把它作为**预防**：
# 选步时优先避开"左右都被封"或"上下都被封"的格子，只有没有别的选择时才走。
NARROW_LANE_PENALTY = 1

# 目标粘性（用户 10-18 第 15 局）：本帧没有候选时，若上一帧的目标仍有 <= 这么多格的路线，就留住它。
# 四段实录（f=145/345/390/633）都是"目标突然丢失 -> 兜底乱转 -> trapped -> 死亡"。
TARGET_KEEP_ROUTE = 0   # 停用：第 17 轮实测峰值掉到 12（留住旧目标会让蛇锁在不可达目标上绕圈）
