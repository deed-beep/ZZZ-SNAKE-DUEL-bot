"""Enumerate config constants and code literals; specialize only overridden values at import."""
from pathlib import Path
import ast,hashlib,json,runpy

FIXED={'CAPTURE_BBOX','FRAME_W','FRAME_H','WORK_SCALE','COLS','ROWS','CELL','UP','DOWN','LEFT','RIGHT','DIRS','LOG_DIR','FRAME_DUMP'}
GROUPS={'config':'原始设置','latency_precision':'控制精度','navigation_candidate_sweep':'转弯与吃钻',
 'input_pad':'手柄与复活','policy_clear_route':'基础选路','policy_long_strategy':'长蛇策略',
 'long_strategy_geometry':'窄口与蛇头交汇','perception':'识别','perception_candidate_precise':'识别',
 'perception_latency':'高速观察','motion':'运动估计','state_analysis':'状态判断',
 'endgame_strategy':'局末松A','run_precise_candidate':'运行资源','body_history':'身体历史',
 'causal_planner':'路径规划','sweep_planner':'脱困规划','bot':'运行与识别门槛'}

def literal_key(module,node):return '%s:%d:%d'%(module,node.lineno,node.col_offset)

def serializable(v):
    try:json.dumps(v,allow_nan=False);return True
    except (TypeError,ValueError):return False

def build_catalog(engine):
    engine=Path(engine);items=[]
    cfg=runpy.run_path(str(engine/'config.py'))
    lines=(engine/'config.py').read_text(encoding='utf8').splitlines()
    assignments={}
    for node in ast.walk(ast.parse('\n'.join(lines))):
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name):assignments[target.id]=node.lineno
    for key,value in cfg.items():
        if not key.isupper()or not serializable(value)or not isinstance(value,(int,float,bool,str,tuple,list,dict,type(None))):continue
        line=assignments.get(key,1);context=' '.join(s.strip('# ')for s in lines[max(0,line-3):line])
        items.append(dict(id='config.'+key,module='config',function='全局',category='原始设置',
            name=key,default=value,kind='config',type=type(value).__name__,line=line,
            context=context,editable=key not in FIXED and not key.endswith('_DIR')))
    for path in sorted(engine.glob('*.py')):
        module=path.stem
        if module=='config':continue
        text=path.read_text(encoding='utf-8-sig');source_lines=text.splitlines()
        class Visitor(ast.NodeVisitor):
            def __init__(self):self.stack=[]
            def visit_FunctionDef(self,node):
                self.stack.append(node.name);self.generic_visit(node);self.stack.pop()
            visit_AsyncFunctionDef=visit_FunctionDef
            def visit_ClassDef(self,node):
                self.stack.append(node.name);self.generic_visit(node);self.stack.pop()
            def visit_Constant(self,node):
                if isinstance(node.value,(int,float,bool)):
                    function='.'.join(self.stack)or'全局'
                    context=source_lines[node.lineno-1].strip()
                    items.append(dict(id=literal_key(module,node),module=module,function=function,
                        category=GROUPS.get(module,module),name=function+' · '+str(node.value),
                        default=node.value,type=type(node.value).__name__,kind='literal',line=node.lineno,
                        context=context,editable=True))
        Visitor().visit(ast.parse(text))
    ident=hashlib.sha256(''.join(hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(engine.glob('*.py'))).encode()).hexdigest()
    return dict(schema=1,engine_id=ident,parameters=items)

def alias_overrides(profile,catalog):
    controls=profile['controls'];opts=profile['strategy_options'];out={}
    matches=[('run_precise_candidate','全局',4,controls['cv_threads']),
        ('latency_precision',None,.17,controls['turn_lead_ms']/1000.),
        ('endgame_strategy','EndgameSchedule.__init__',45.,controls['release_seconds']),
        ('input_pad','Pad.rearm_accel',.7,controls['respawn_release_seconds']),
        ('input_pad','Pad.confirm_respawn_motion',.7,controls['respawn_release_seconds']),
        ('policy_long_strategy','Policy._decision_state',40,opts['long_threshold']),
        ('policy_long_strategy','Policy._decision_state',35,opts['long_exit_threshold']),
        ('policy_long_strategy','Policy._decision_state',45.,opts['long_protect_at']),
        ('policy_long_strategy','Policy._rank_actual_approaches',.003,opts['topology_budget_ms']/1000.),
        ('policy_long_strategy','Policy._pickup_escape',4.,opts['escape_hint_seconds']),
        ('policy_long_strategy','Policy._survive',4.,opts['escape_hint_seconds'])]
    for item in catalog['parameters']:
        if item['kind']!='literal':continue
        for module,function,old,new in matches:
            if item['module']==module and(function is None or item['function']==function)and type(item['default'])is not bool and item['default']==old and new!=old:
                if function in('Policy._pickup_escape','Policy._survive')and 'strategy_now' not in item['context']:continue
                out[item['id']]=int(new)if item['type']=='int'else float(new)
    out.update(profile.get('literal_overrides',{}))
    return out

def validate_overrides(profile,catalog):
    import math
    by_id={item['id']:item for item in catalog['parameters']}
    for prefix,values in [('config.',profile.get('config_overrides',{})),('',profile.get('literal_overrides',{}))]:
        for key,value in values.items():
            ident=prefix+key;item=by_id.get(ident)
            if not item:raise ValueError('参数不属于当前2.5版本：'+ident)
            if not item['editable']:raise ValueError('此项由屏幕适配或固定棋盘规格管理：'+ident)
            expected=item['type']
            valid=(isinstance(value,bool)if expected=='bool'else
                   isinstance(value,int)and not isinstance(value,bool)if expected=='int'else
                   isinstance(value,(int,float))and not isinstance(value,bool)and math.isfinite(value)if expected=='float'else
                   isinstance(value,(list,tuple))if expected in('list','tuple')else
                   isinstance(value,dict)if expected=='dict'else isinstance(value,str)if expected=='str'else value is None)
            if not valid:raise ValueError('参数类型不正确：%s，应为%s'%(ident,expected))
