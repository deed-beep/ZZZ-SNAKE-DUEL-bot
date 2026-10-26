from copy import deepcopy
from pathlib import Path
import math,json
from app_paths import data_dir,write_json,read_json

STRATEGIES={
 'late_release':'到时松开加速键','long_strategy':'长蛇策略总开关','long_protect':'长蛇局末保长',
 'chamber':'长蛇窄口评分','head_crossings':'预测蓝紫蛇头交汇','escape_hints':'复用已验证的逃生路线',
 'food_dia':'主动追钻石','food_bean':'主动追黄豆','food_grn':'主动追加速道具',
 'rules_enabled':'启用自定义规则','plugin_enabled':'启用本地Python策略插件'}
DEFAULT={
 'schema':1,'name':'2.5原版默认',
 'run':{'matches':1,'infinite':False,'pause_seconds':1.,'start_delay':3.,'waiting_timeout_minutes':0.,
        'after_start_seconds':75.,'save_frames':3,'pause_on_focus_loss':True,'controller_wakeup':True,
        'controller_confirm_seconds':.35,'controller_retry_seconds':5.,'controller_max_attempts':3},
 'capture':{'mode':'window','target_title':'绝区零','monitor':1,'manual_board':None,
            'fit_16_9':True,'viewport':None,'target_hwnd':0,'lobby_word_threshold':.88,'lobby_x_threshold':.82},
 'controls':{'accel_mode':'hold','cv_threads':4,'capture_threads':1,'turn_lead_ms':170.,
             'release_seconds':45.,'respawn_release_seconds':.7},
 'strategy_options':{'long_threshold':40,'long_exit_threshold':35,'long_protect_at':45.,
                     'topology_budget_ms':3.,'escape_hint_seconds':4.},
 'strategies':{k:k not in ('rules_enabled','plugin_enabled')for k in STRATEGIES},
 'config_overrides':{},'literal_overrides':{},'rules':[],'plugin_path':''}

SAMPLE_RULES=[{'name':'长蛇遇近刺球时保长','enabled':False,
    'when':{'length_gte':40,'elapsed_gte':45,'nearest_ball_lte':3},
    'then':{'prefer_survival':True,'release_accel':True}},
    {'name':'后半局不主动追加速道具','enabled':False,'when':{'elapsed_gte':45},'then':{'disable_food':['grn']}}]

def merge(base,overrides):
    out=deepcopy(base)
    for key,value in overrides.items():
        out[key]=merge(out[key],value)if key in out and isinstance(out[key],dict)and isinstance(value,dict)else deepcopy(value)
    return out

def defaults():return deepcopy(DEFAULT)

def validate(profile):
    p=merge(DEFAULT,profile)
    def number(section,key,lo,hi,integer=False):
        v=p[section][key]
        if isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or not lo<=v<=hi or(integer and int(v)!=v):
            raise ValueError('%s：请输入%s到%s之间的%s。'%(key,lo,hi,'整数'if integer else'数值'))
        p[section][key]=int(v)if integer else float(v)
    for item in [('run','matches',1,100000,True),('run','pause_seconds',0,3600,False),
        ('run','start_delay',0,120,False),('run','waiting_timeout_minutes',0,100000,False),
        ('run','after_start_seconds',20,600,False),('run','save_frames',0,10000,True),
        ('controls','cv_threads',1,32,True),('controls','turn_lead_ms',0,1000,False),
        ('controls','release_seconds',0,600,False),('controls','respawn_release_seconds',.667,10,False),
        ('strategy_options','long_threshold',1,724,True),('strategy_options','long_exit_threshold',1,724,True),
        ('strategy_options','long_protect_at',0,600,False),('strategy_options','topology_budget_ms',0,100,False),
        ('strategy_options','escape_hint_seconds',0,60,False)]:number(*item)
    for item in [('run','controller_confirm_seconds',.1,5,False),('run','controller_retry_seconds',2,60,False),
                 ('run','controller_max_attempts',1,10,True),('capture','lobby_word_threshold',.5,1.,False),
                 ('capture','lobby_x_threshold',.5,1.,False)]:number(*item)
    if p['controls']['capture_threads']not in(1,2):raise ValueError('截图线程只能为1或2。')
    if p['controls']['accel_mode']not in('hold','off','adaptive'):raise ValueError('加速模式无效。')
    if p['capture']['mode']not in('window','monitor','manual'):raise ValueError('截图方式无效。')
    for section,key in [('run','infinite'),('run','pause_on_focus_loss'),('run','controller_wakeup'),('capture','fit_16_9')]:
        if not isinstance(p[section][key],bool):raise ValueError(key+'必须为布尔值。')
    number('capture','monitor',1,32,True)
    if p['capture']['mode']=='manual'and not p['capture']['manual_board']:raise ValueError('请选择手动棋盘区域，或使用自动游戏窗口模式。')
    box=p['capture']['manual_board']
    if box is not None and(not isinstance(box,list)or len(box)!=4 or not all(isinstance(v,(int,float))for v in box)
                         or not all(not isinstance(v,bool)and math.isfinite(v)and 0<=v<=1 for v in box)
                         or box[2]<=box[0]or box[3]<=box[1]):raise ValueError('手动区域无效。')
    viewport=p['capture'].get('viewport')
    if viewport is not None and(not isinstance(viewport,list)or len(viewport)!=4 or not all(isinstance(v,(int,float))and not isinstance(v,bool)and math.isfinite(v)and 0<=v<=1 for v in viewport)
        or viewport[2]<=0 or viewport[3]<=0 or viewport[0]+viewport[2]>1.001 or viewport[1]+viewport[3]>1.001):raise ValueError('游戏内容区域无效。')
    for key in STRATEGIES:
        if not isinstance(p['strategies'][key],bool):raise ValueError('策略开关必须为开或关：'+key)
    if not isinstance(p['config_overrides'],dict)or not isinstance(p['literal_overrides'],dict):raise ValueError('高级参数格式必须为JSON对象。')
    validate_rules(p['rules'])
    if p['strategies']['plugin_enabled']and not Path(p['plugin_path']).is_file():raise ValueError('请选择存在的Python策略插件文件。')
    json.dumps(p,allow_nan=False)
    return p

CONDITIONS={'length_gte','length_lte','elapsed_gte','elapsed_lte','nearest_ball_lte','nearest_ball_gte','near_wall_lte','eye','path_trust'}
ACTIONS={'prefer_survival','release_accel','disable_food','food_weights','target_cell'}
def validate_rules(rules):
    if not isinstance(rules,list):raise ValueError('规则应为JSON数组。')
    for r in rules:
        if not isinstance(r,dict)or not isinstance(r.get('when',{}),dict)or not isinstance(r.get('then',{}),dict):raise ValueError('规则需要when和then对象。')
        if not isinstance(r.get('enabled',True),bool):raise ValueError('规则enabled必须为true或false。')
        if set(r.get('when',{}))-CONDITIONS:raise ValueError('存在不支持的规则条件。')
        if set(r.get('then',{}))-ACTIONS:raise ValueError('存在不支持的规则动作。')
        for k,v in r.get('when',{}).items():
            if k not in('eye','path_trust')and(isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)):
                raise ValueError('规则数值条件不正确：'+k)
        for k in('prefer_survival','release_accel'):
            if k in r.get('then',{})and not isinstance(r['then'][k],bool):raise ValueError(k+'必须为布尔值。')
        disabled=r.get('then',{}).get('disable_food',[])
        if not isinstance(disabled,list)or any(x not in('dia','bean','grn')for x in disabled):raise ValueError('disable_food只接受dia、bean、grn。')
        weights=r.get('then',{}).get('food_weights',{})
        if not isinstance(weights,dict)or any(k not in('dia','bean','grn')or isinstance(v,bool)or not isinstance(v,(int,float))or not math.isfinite(v)or v<=0 for k,v in weights.items()):raise ValueError('食物权重必须为正数。')
        target=r.get('then',{}).get('target_cell')
        if target is not None and(not isinstance(target,list)or len(target)!=2 or not all(isinstance(x,int)and not isinstance(x,bool)for x in target)or not(0<=target[0]<29 and 0<=target[1]<25)):
            raise ValueError('target_cell应为棋盘范围内的[x,y]。')

def load_profile():return validate(read_json(data_dir()/'profile.json',{}))
def save_profile(p):
    p=validate(p);write_json(data_dir()/'profile.json',p);return p
