"""Long-body policy hooks; the r179 pixel steering/timing stays unchanged."""
from collections import deque
from policy_clear_route import Policy as BaselinePolicy, OPPOSITE
import time
from long_strategy_geometry import chamber,head_crossing


class Policy(BaselinePolicy):
    variant=BaselinePolicy.variant+'+long_escape_fallback_and_late_protect'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.long_active=False
        self._long_lengths=deque(maxlen=3)
        self._long_hints=[]
        self._strategy_now=0.
        self._strategy_elapsed=0.
        self._long_priority=False
        self._long_head_conflicts=0

    def decide(self,st,belief=None):
        self._strategy_now=float(st.get('t',0.))
        self._strategy_elapsed=float(st.get('match_elapsed')or 0.)
        result=super().decide(st,belief)
        self.last.update(long_strategy_active=bool(self.long_active),
            long_survival_priority=bool(self._long_priority),long_escape_hints=len(self._long_hints),
            long_head_conflicts=int(self._long_head_conflicts))
        return result

    def _decision_state(self,st,belief):
        ds=super()._decision_state(st,belief)
        raw=st['snakes'].get('red')
        if raw is not None and raw.head_age==0:
            self._long_lengths.append(int(ds['length']))
            if sum(n>40 for n in self._long_lengths)>=2:
                self.long_active=True
            # Actual losses can occur in head contact; do not keep long-only
            # strategy on a measured short snake after a sustained collapse.
            if len(self._long_lengths)==3 and max(self._long_lengths)<35:
                self.long_active=False
        self._long_priority=self.long_active and self._strategy_elapsed>=45.
        return ds

    def _rank_actual_approaches(self,*args,**kwargs):
        if self._long_priority:
            self._candidate_review=[]
            return []
        ranked=super()._rank_actual_approaches(*args,**kwargs)
        if not self.long_active or not ranked:return ranked
        red,hard=args[1],args[2]
        budget=time.perf_counter()+.003
        adjusted=[];details=[]
        for cost,candidate,path in ranked:
            area=exits=None;tail=False;penalty=0.
            if time.perf_counter()<budget:
                area,exits,tail=chamber(path,red,hard,self._geometry_growth,self._food)
                if area<12 and exits<=1:penalty=1.8
                elif area<24 and exits<=1:penalty=.5
                if tail:penalty*=.35
            adjusted.append((cost+penalty,candidate,path))
            details.append(dict(kind=candidate[0],cell=candidate[2],steps=len(path)-1,
                cost=round(float(cost+penalty),3),chamber=area,outlets=exits,tail_contact=bool(tail)))
        adjusted.sort(key=lambda row:row[0]);self._candidate_review=details
        return adjusted

    def _route_safe(self,path,red,hard,growth):
        if not super()._route_safe(path,red,hard,growth):return False
        if self.long_active and head_crossing(path,red.head_px,self.follower.speed_s,
                                            self._opponent_heads,self._strategy_now):
            self._long_head_conflicts+=1
            return False
        return True

    def _pickup_escape(self,path,red,hard,growth,**kwargs):
        result=super()._pickup_escape(path,red,hard,growth,**kwargs)
        if self.long_active and result and kwargs.get('return_route'):
            # This is a fallback, not a command to follow the entire witness.
            # A fresh safe pickup still takes priority outside the final window.
            hint=[tuple(c)for c in result]
            self._long_hints=[(t,r)for t,r in self._long_hints
                              if self._strategy_now-t<4. and r!=hint][-3:]
            self._long_hints.append((self._strategy_now,hint))
        return result

    def _survive(self,red,hard):
        if self.long_active:
            for at,hint in reversed(self._long_hints):
                if self._strategy_now-at>4. or red.head_cell not in hint:continue
                route=hint[hint.index(red.head_cell):]
                if len(route)<4:continue
                first=(route[1][0]-route[0][0],route[1][1]-route[0][1])
                if first==OPPOSITE.get(self._heading):continue
                if self._route_safe(route,red,hard,self._geometry_growth):
                    self._survival_route=route
                    return route,'long_escape_rechecked'
        return super()._survive(red,hard)
