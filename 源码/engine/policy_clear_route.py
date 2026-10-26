"""One checked route owns steering; no later heuristic can replace its bend."""
from copy import copy
import math
from policy_candidate_sweep import Policy as ReferencePolicy, NEIGH, OPPOSITE, in_board
from config import CELL
from causal_planner import packing_escape
from sweep_planner import walk,plan_sweep


class Policy(ReferencePolicy):
    variant='single_checked_route'

    def __init__(self,no_boost=None):
        super().__init__(no_boost=True)
        self.key_boost=True
        self._active_pickup=None
        self._survival_route=None
        self._bootstrap_frames=0
        self._buffer_until=0.
        self._buffer_attempt=None
        self._crossed_low_value={}
        self._pickup_sample=None
        self._low_value_pixels={}
        self.follower.lead_s=.17

    def _route_safe(self,path,red,hard,growth):
        if not path or tuple(path[0])!=red.head_cell:return False
        if any(not in_board(c) or c in hard for c in path[1:]):return False
        ordered=red.path_trust=='ordered' and red.path and tuple(red.path[0])==red.head_cell
        if ordered:
            if walk(red.path,path,growth,self._food) is None:return False
        else:
            if any(c in red.cells for c in path[1:]):return False
            if walk([path[0]],path,max(0,red.length-1)+growth,self._food) is None:return False
        speed=max(100.,self.follower.speed_s)
        previous=red.head_px;distance=0.
        for cell in path[1:]:
            point=tuple((v+.5)*CELL for v in cell)
            length=sum(abs(point[i]-previous[i]) for i in (0,1))
            n=max(1,int(math.ceil(length/8.)))
            for k in range(1,n+1):
                eta=(distance+length*k/n)/speed
                if eta>.6:return True
                q=tuple(previous[i]+(point[i]-previous[i])*k/n for i in (0,1))
                if self._ball_hit_at(q,eta):return False
            distance+=length;previous=point
        return True

    def _survive(self,red,hard):
        head=red.head_cell
        if self._survival_route and head in self._survival_route:
            route=self._survival_route[self._survival_route.index(head):]
            if (len(route)>3 and (route[1][0]-head[0],route[1][1]-head[1])!=OPPOSITE.get(self._heading)
                    and self._route_safe(route,red,hard,self._geometry_growth)):
                return route,'survive_cached'
        ordered=red.path_trust=='ordered' and red.path and tuple(red.path[0])==head
        if ordered:
            body=red.path;blocked=hard;growth=self._geometry_growth
        else:
            body=[head];blocked=set(hard)|set(red.cells);blocked.discard(head)
            growth=max(0,red.length-1)+self._geometry_growth
        spacing=max(1,min(3,math.ceil(self.follower.route_speed_s*self.follower.lead_s/CELL)))
        axis=0 if self._heading and self._heading[0] else 1
        remaining=(((head[axis]+.5)*CELL-red.head_px[axis])*self._heading[axis]) if self._heading else 0.
        initial=max(0,min(4,math.ceil((self.follower.speed_s*self.follower.lead_s-8-remaining)/CELL)))
        route=packing_escape(body,self._heading,blocked,growth,spacing,initial)
        if route is None or walk(body,route,growth,self._food) is None:
            route=plan_sweep(body,self._heading,blocked,{head},self._food,growth,spacing,initial)
        if route is not None and self._route_safe(route,red,hard,self._geometry_growth):
            self._survival_route=route
            return route,'survive_full'
        # Explicit emergency, with all unobserved old cells still occupied.
        route=packing_escape(body,self._heading,blocked,growth,spacing,initial,partial=True)
        if route is None:
            route=packing_escape(body,self._heading,blocked,growth,1,0,partial=True)
        if route and self._route_safe(route,red,hard,self._geometry_growth):
            return route,'emergency_partial'
        return None,'trapped'

    def _buffer_escape(self,red,hard,avoid=None):
        head=red.head_cell;heading=self._heading
        blocked=set(hard)
        axis=getattr(red,'eye_travel_axis',None)
        if (axis in ((1,0),(0,1)) and getattr(red,'head_from_eyes',False)
                and red.head_age==0 and not getattr(red,'eye_extrapolated',False)):
            # A turn can collide inside the same tile before motion shows its
            # new heading. Eye separation still proves its unsigned travel
            # axis: choose a perpendicular escape, never a silent 180 reversal.
            allowed={d for d in NEIGH if d[0]*axis[0]+d[1]*axis[1]==0}
            if heading not in (axis,OPPOSITE[axis]):heading=None
        else:allowed={d for d in NEIGH if d not in (heading,OPPOSITE.get(heading))}
        blocked|={(head[0]+d[0],head[1]+d[1]) for d in NEIGH if d not in allowed}
        # While paused, our currently occupied neighbour has not vacated yet.
        # Do not spend the buffer window betting on an estimated tail release.
        blocked|={(head[0]+d[0],head[1]+d[1]) for d in NEIGH
                  if (head[0]+d[0],head[1]+d[1]) in red.cells or d==avoid}
        ordered=red.path_trust=='ordered' and red.path and tuple(red.path[0])==head
        body=red.path if ordered else [head]
        if not ordered:blocked|=set(red.cells);blocked.discard(head)
        growth=self._geometry_growth+(0 if ordered else max(0,red.length-1))
        for partial in (False,True):
            route=packing_escape(body,heading,blocked,growth,1,0,partial=partial)
            if route and len(route)>1 and self._route_safe(route,red,hard,self._geometry_growth):
                return route
        # Last available buffer window: a measured free perpendicular cell is
        # preferable to holding the colliding direction until dissolution.
        options=[]
        for d in NEIGH:
            if d not in allowed:continue
            cell=(head[0]+d[0],head[1]+d[1]);route=[head,cell]
            if cell not in blocked and self._route_safe(route,red,hard,self._geometry_growth):
                options.append((self._risk.get(cell,0.),route))
        return min(options,key=lambda x:x[0])[1] if options else None

    def _emit(self,red,ds,d,why,route=None,target=None,waypoint=None):
        self._sent_dir=d if d is not None else self._sent_dir
        self.head_hist.append((*red.head_px,0))
        self.last=dict(why=why,heading=self._heading,head=red.head_cell,length=red.length,
            head_px=red.head_px,waypoint=waypoint,route=route,
            target=None if target is None else (target[0],target[1],max(0,len(route or [])-1)),
            path_trust=red.path_trust,body_source=ds['body_source'],body_coverage=ds['body_coverage'],
            policy_body_path=list(red.path),opponent_heads=self._opponent_heads,
            planning_hard_cells=sorted(self._last_hard),planning_growth=self._geometry_growth,
            speed_item_active=self.speed_item_active,area=0,need=red.length,n_steps=0,
            runway=0,clearance=0,gap=0,risk=0,orientation=getattr(red,'orientation',None))
        return d,True,False,why

    def _filter_crossed_pickups(self,candidates,st,red,now):
        # Peripheral item memory can outlive a bean that our head just crossed.
        # A brief exclusion prevents routing a full U-turn back to that stale
        # bean. Diamonds retain their independent disappearance verification.
        self._crossed_low_value={k:t for k,t in self._crossed_low_value.items() if t>now}
        fresh=red.head_age==0 and red.eye_state in ('normal','item')
        point=red.head_px
        old=self._pickup_sample
        if fresh and old and 0<now-old[0]<.3:
            a=old[1];dx=point[0]-a[0];dy=point[1]-a[1];d2=dx*dx+dy*dy
            if 4<d2<96**2 and min(abs(dx),abs(dy))<5:
                for key,pixel in self._low_value_pixels.items():
                    u=((pixel[0]-a[0])*dx+(pixel[1]-a[1])*dy)/d2
                    if 0<=u<=1 and math.dist(pixel,(a[0]+u*dx,a[1]+u*dy))<10:
                        self._crossed_low_value[key]=now+1.
        self._pickup_sample=(now,point) if fresh else None
        self._low_value_pixels={(kind,cell):self._item_pixel(st,kind,cell)
            for kind,value,cell,bonus in candidates if kind in ('bean','grn')}
        return [c for c in candidates if (c[0],c[2]) not in self._crossed_low_value]

    def _rank_actual_approaches(self,candidates,red,hard,own,st):
        """Rank actual detours before proving the chosen pickup's exit."""
        reviewed=[]
        for kind,value,cell,bonus in candidates[:5]:
            path=self._approach((kind,cell),red.head_cell,red.head_px,self._heading,hard,own,
                self.follower.speed_s,self.follower.lead_s,red.length+self._geometry_growth,self._risk)
            if not path or len(path)<2 or not self._pickup_fits_clock(st,len(path)-1):continue
            directions=[(b[0]-a[0],b[1]-a[1]) for a,b in zip(path,path[1:])]
            turns=sum(a!=b for a,b in zip(directions,directions[1:]))
            reward=5. if kind=='dia' else 2.5 if kind=='grn' else 1.
            cost=(len(path)-1+.35*turns)/reward
            if (kind,cell)==self._active_pickup:cost*=.90
            reviewed.append((cost,(kind,value,cell,bonus),path))
        reviewed.sort(key=lambda row:row[0])
        self._candidate_review=[dict(kind=c[0],cell=c[2],steps=len(path)-1,cost=round(cost,3))
                                for cost,c,path in reviewed]
        return reviewed

    def decide(self,st,belief=None):
        raw=st['snakes'].get('red')
        if raw is None or raw.head_cell is None or raw.head_px is None:
            self.last={'why':'no_head','target':None}
            return None,True,False,'no_head'
        if belief is not None:st['belief']=belief
        ds=self._decision_state(st,belief or st.get('belief'))
        red=copy(raw)
        red.head_cell=ds['head'];red.head_px=ds['head_px'];red.facing=ds['facing']
        red.path=list(ds['path']);red.path_trust=ds['path_trust'];red.cells=set(ds['body_cells']);red.length=ds['length']
        now=float(st.get('t',0.))
        if hasattr(self.follower,'set_context'):
            self.follower.set_context(now,red.length,st.get('item_accel_until',0.),
                red.eye_state=='item',native=(red.head_age==0 and not ds['head_px_uncertain'] and red.eye_state in ('normal','item')))
            self._item_accel_until=self.follower.fast_until
        elif red.eye_state=='item':self._item_accel_until=max(self._item_accel_until,now+.3)
        st['item_accel_until']=max(st.get('item_accel_until',0.),self._item_accel_until)
        self.speed_item_active=st['item_accel_until']>now
        self._heading=self._observed_heading(ds,red,st) or red.facing
        if red.head_age==0 and not ds['head_px_uncertain']:
            self.follower.observe(red.head_px,now)
        pending=max(0,min(5,int(getattr(st.get('belief'),'predicted_growth',0))))
        if self._growth_point is not None:
            self._growth_distance+=min(2*CELL,sum(abs(red.head_px[i]-self._growth_point[i]) for i in (0,1)))
            consumed=int(self._growth_distance//CELL)
            self._growth_debt=max(0,self._growth_debt-consumed);self._growth_distance-=consumed*CELL
        self._growth_point=red.head_px
        if pending:self._growth_debt=max(self._growth_debt,pending)
        self._geometry_growth=self._growth_debt+1
        self._food=self._route_food(st)
        hard=self.collision_cells(st)
        self._opp_memory={k:set(s.cells) for k,s in st['snakes'].items() if k!='red'}
        self._risk=self._risk_map(st,hard)
        hard=self._planning_blocked(red.head_cell,hard)
        # START covers the central lanes for the first seconds. Route around
        # the actual occluder instead of driving the head out of observation.
        if (st.get('match_elapsed',99.) or 0.)<2.3:
            hidden={(x,y) for x in range(5,25) for y in range(10,15)}
            if red.head_cell not in hidden:hard=set(hard)|hidden
        self._last_hard=hard
        if self._heading not in NEIGH:
            self._bootstrap_frames+=1
            if self._bootstrap_frames%4==0 and self._sent_dir in NEIGH:
                hard=set(hard)|{(red.head_cell[0]+self._sent_dir[0],red.head_cell[1]+self._sent_dir[1])}
        else:self._bootstrap_frames=0
        own=self.own_index(red,growth=self._geometry_growth)
        self._first_allowed={(red.head_cell[0]+d[0],red.head_cell[1]+d[1]) for d in NEIGH
            if d!=OPPOSITE.get(self._heading) and self._passable((red.head_cell[0]+d[0],red.head_cell[1]+d[1]),hard,own,1)}
        if red.eye_state=='buffer':
            self._buffer_until=now+.35
            self.follower.turn_pending=None
            attempt=self._buffer_attempt
            avoid=(attempt['dir'] if attempt and now-attempt['at']>=.18
                   and math.dist(red.head_px,attempt['point'])<6 else None)
            route=self._buffer_escape(red,hard,avoid)
            if route:
                self._survival_route=route
                d=(route[1][0]-route[0][0],route[1][1]-route[0][1])
                if not attempt or d!=attempt['dir'] or math.dist(red.head_px,attempt['point'])>=6:
                    self._buffer_attempt=dict(dir=d,at=now,point=red.head_px)
                return self._emit(red,ds,d,'buffer_immediate_escape',route)
        else:self._buffer_attempt=None
        # Finish a sent bend when its actual future corridor remains free.
        queued=self.follower.turn_pending
        if queued and self._heading!=queued['dir'] and now-queued['at']<=.35:
            head=red.head_cell;corner=tuple(queued['cell']);h=queued['from'];d=queued['dir']
            n=(corner[0]-head[0])*h[0]+(corner[1]-head[1])*h[1]
            if -2<=n<=4 and (corner[0]-head[0])*h[1]==(corner[1]-head[1])*h[0]:
                # A sampled input can still be in flight when the capture
                # crosses its intended corner. Do not cancel it on that basis,
                # or replace the latency measurement with a different corner.
                actual=corner if n>=0 else head
                route=[(head[0]+i*h[0],head[1]+i*h[1]) for i in range(max(0,n)+1)]
                route.extend((actual[0]+k*d[0],actual[1]+k*d[1]) for k in (1,2))
                if self._route_safe(route,red,hard,self._geometry_growth):
                    return self._emit(red,ds,d,'queued_checked_turn',route,waypoint=queued['corner'])
            self.follower.turn_pending=None
        candidates=self._candidates(st,hard)
        candidates=self._filter_crossed_pickups(candidates,st,red,now)
        candidates.sort(key=lambda c:sum(abs(c[2][i]-red.head_cell[i]) for i in (0,1))/(5. if c[0]=='dia' else 2.5 if c[0]=='grn' else 1.))
        target=None;route=None
        collecting=(st.get('match_elapsed',0.) or 0.)<60. and now>=self._buffer_until
        for _,(kind,value,cell,bonus),path in self._rank_actual_approaches(candidates,red,hard,own,st) if collecting else []:
            if not self._route_safe(path,red,hard,self._geometry_growth):continue
            segment=max(1,min(3,math.ceil(self.follower.route_speed_s*self.follower.lead_s/CELL)))
            segment=self._post_pickup_spacing(st,path,segment)
            growth=5 if kind=='dia' else 1 if kind=='bean' else 0
            exit_route=self._pickup_escape(path,red,hard,growth,segment=segment,require_tail=red.length>=10,
                debt=self._geometry_growth,food=self._food,return_route=True)
            if not exit_route:continue
            target=(kind,cell);route=path;self._active_pickup=target
            break
        if route is None:
            self._active_pickup=None
            route,why=self._survive(red,hard)
            target=None
        else:why='pickup_'+target[0]
        if not route or len(route)<2:return self._emit(red,ds,None,'trapped')
        pixel=self._item_pixel(st,*target) if target else tuple((v+.5)*CELL for v in route[-1])
        if self._heading not in NEIGH:
            d=(route[1][0]-route[0][0],route[1][1]-route[0][1])
            return self._emit(red,ds,d,'spawn_bootstrap_'+why,route,target,pixel)
        key=target if target else ('survival',route[-1])
        d,_,waypoint=self.follower.command(red.head_px,self._heading,route,key,pixel,
            set(NEIGH)-{OPPOSITE.get(self._heading)},pickup=(target[1],pixel) if target else None)
        return self._emit(red,ds,d,why,route,target,waypoint)
