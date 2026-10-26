import importlib.util,math,time,traceback
from app_settings import validate_rules

def context_from(st):
    red=st.get('snakes',{}).get('red');belief=st.get('belief')
    head=getattr(red,'head_cell',None)
    length=getattr(belief,'length',0)or getattr(red,'length',0)or 0
    balls=st.get('items',{}).get('ball',[])
    nearest=min((math.hypot(head[0]-b[0],head[1]-b[1])for b in balls),default=999.)if head else 999.
    wall=min(head[0],head[1],28-head[0],24-head[1])if head else 999.
    return dict(length=int(length),elapsed=float(st.get('match_elapsed')or 0.),nearest_ball=nearest,
                near_wall=wall,eye=getattr(red,'eye_state',None),path_trust=getattr(red,'path_trust',None),
                head_cell=list(head)if head else None,phase='PLAYING')

def matches(when,context):
    for key,value in when.items():
        if key.endswith('_gte'):
            if context.get(key[:-4],-float('inf'))<value:return False
        elif key.endswith('_lte'):
            if context.get(key[:-4],float('inf'))>value:return False
        elif context.get(key)!=value:return False
    return True

class RuleEngine:
    def __init__(self,profile):
        self.profile=profile;self.plugin=None;self.error=None
        if profile['strategies']['plugin_enabled']:
            spec=importlib.util.spec_from_file_location('snake25_user_strategy',profile['plugin_path'])
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            self.plugin=getattr(module,'plan')
    def evaluate(self,st):
        context=context_from(st);actions={}
        if self.profile['strategies']['rules_enabled']:
            for rule in self.profile['rules']:
                if rule.get('enabled',True)and matches(rule.get('when',{}),context):actions.update(rule.get('then',{}))
        if self.plugin:
            try:
                start=time.perf_counter();value=self.plugin(context,st)
                if value is not None:
                    validate_rules([{'when':{},'then':value}]);actions.update(value)
                elapsed=time.perf_counter()-start
                if elapsed>.02:
                    print('[策略插件] 单次执行超过20毫秒，已停用本次插件以维持反应速度。',flush=True);self.plugin=None
            except Exception:
                self.error=traceback.format_exc();print('[策略插件] 已退回内置策略：\n'+self.error,flush=True);self.plugin=None
        return actions

def configure_entry(entry,profile):
    flags=profile['strategies']
    default_bools=all(flags[k]for k in('long_strategy','long_protect','chamber','head_crossings','escape_hints','food_dia','food_bean','food_grn'))
    if default_bools and not flags['rules_enabled']and not flags['plugin_enabled']:return entry.Policy
    from policy_clear_route import Policy as Base
    from policy_long_strategy import Policy as Long
    original=entry.Policy
    class ConfiguredPolicy(original):
        variant=original.variant+'+desktop_strategy_settings'
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.app_actions={};self.app_rules=RuleEngine(profile)
        def decide(self,st,belief=None):
            self.app_actions=self.app_rules.evaluate(st)
            return super().decide(st,belief)
        def _decision_state(self,st,belief):
            if not flags['long_strategy']:
                self.long_active=False;self._long_priority=False
                return Base._decision_state(self,st,belief)
            result=super()._decision_state(st,belief)
            if not flags['long_protect']:self._long_priority=False
            return result
        def _candidates(self,st,hard):
            items=super()._candidates(st,hard)
            disabled=set(self.app_actions.get('disable_food',[]))
            return[c for c in items if flags.get('food_'+c[0],True)and c[0]not in disabled]
        def _rank_actual_approaches(self,*args,**kwargs):
            if self.app_actions.get('prefer_survival')or self._long_priority:return []
            ranked=super()._rank_actual_approaches(*args,**kwargs)if flags['chamber']and flags['long_strategy']else Base._rank_actual_approaches(self,*args,**kwargs)
            weights=self.app_actions.get('food_weights',{})
            target=self.app_actions.get('target_cell')
            if weights:
                base={'dia':5.,'bean':1.,'grn':2.5}
                ranked=[(cost*base[c[0]]/weights.get(c[0],base[c[0]]),c,path)for cost,c,path in ranked]
            if target:ranked.sort(key=lambda r:(tuple(r[1][2])!=tuple(target),r[0]))
            else:ranked.sort(key=lambda r:r[0])
            return ranked
        def _route_safe(self,*args,**kwargs):
            if not flags['head_crossings']or not flags['long_strategy']:return Base._route_safe(self,*args,**kwargs)
            return super()._route_safe(*args,**kwargs)
        def _pickup_escape(self,*args,**kwargs):
            if not flags['escape_hints']or not flags['long_strategy']:return Base._pickup_escape(*args,**kwargs)
            return super()._pickup_escape(*args,**kwargs)
        def _survive(self,*args,**kwargs):
            if not flags['escape_hints']or not flags['long_strategy']:return Base._survive(self,*args,**kwargs)
            return super()._survive(*args,**kwargs)
    return ConfiguredPolicy
