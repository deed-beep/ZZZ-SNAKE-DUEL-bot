"""Offline and native lifecycle acceptance. Never creates a virtual controller."""
import ast,copy,hashlib,io,json,os,subprocess,sys,tempfile,time,unittest,uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2,numpy as np
from app_paths import BASE,ENGINE,read_json,write_json
from app_settings import defaults,validate,validate_rules
from app_catalog import validate_overrides,alias_overrides
from app_engine_loader import install,TuneLoader

CATALOG=read_json(BASE/'parameter_catalog.json')
_scratch=None

class SoftwareTests(unittest.TestCase):
    def test_notice_signature_rejects_missing_and_modified_declaration(self):
        import shutil
        from app_notice import verify_at,NOTICE_NAME,NoticeIntegrityError
        folder=_scratch/'notice_copy';resources=folder/'resources';(resources/'app_assets').mkdir(parents=True)
        for name in('notice_manifest.json','notice_manifest.sig'):shutil.copy2(BASE/'app_assets'/name,resources/'app_assets'/name)
        original=(BASE.parent/NOTICE_NAME if getattr(sys,'frozen',False)else BASE/NOTICE_NAME).read_bytes()
        # In a portable build the visible declaration is next to the exe.
        if getattr(sys,'frozen',False):original=(Path(sys.executable).parent/NOTICE_NAME).read_bytes()
        target=folder/NOTICE_NAME;target.write_bytes(original)
        self.assertIn('101010',verify_at(folder,resources,False))
        target.unlink()
        with self.assertRaises(NoticeIntegrityError):verify_at(folder,resources,False)
        target.write_bytes(original+b'\nchanged')
        with self.assertRaises(NoticeIntegrityError):verify_at(folder,resources,False)
        target.write_bytes(original);signature=resources/'app_assets/notice_manifest.sig';data=signature.read_bytes();signature.write_bytes(bytes([data[0]^1])+data[1:])
        with self.assertRaises(NoticeIntegrityError):verify_at(folder,resources,False)

    def test_windows_signature_verification_and_protected_resources(self):
        from app_notice import signature_valid,verify_distribution
        payload=(BASE/'app_assets/notice_manifest.json').read_bytes();signed=(BASE/'app_assets/notice_manifest.sig').read_bytes()
        self.assertTrue(signature_valid(payload,signed));self.assertFalse(signature_valid(payload+b' ',signed));self.assertFalse(signature_valid(payload,signed[:-1]))
        self.assertIn('软件作者：101010',verify_distribution())

    def test_notice_watchdog_stops_controller_without_waiting_for_focus(self):
        import app_notice,app_runtime_control as control
        old=control.notice_watchdog
        with patch.object(app_notice,'verify_distribution',side_effect=app_notice.NoticeIntegrityError('removed')):
            guard=app_notice.NoticeWatchdog(interval=.01)
            try:
                deadline=time.monotonic()+2
                while not guard.error and time.monotonic()<deadline:time.sleep(.01)
                self.assertEqual(guard.error,'removed');control.notice_watchdog=guard
                pad=SimpleNamespace(reset=__import__('unittest').mock.Mock())
                with self.assertRaises(app_notice.NoticeIntegrityError):control.loop_tick(pad)
                pad.reset.assert_called_once()
            finally:control.notice_watchdog=old;guard.close()

    def test_supplied_icon_has_windows_sizes(self):
        from PIL import Image
        with Image.open(BASE/'app.ico')as icon:
            self.assertTrue({(16,16),(32,32),(48,48),(256,256)}.issubset(icon.ico.sizes()))

    def test_baseline_source_integrity(self):
        baseline=read_json(BASE/'baseline.json')
        for relative,digest in baseline['sha256'].items():
            self.assertEqual(hashlib.sha256((ENGINE/relative).read_bytes()).hexdigest(),digest,relative)

    def test_default_profile_preserves_original_policy_and_timing(self):
        import run_precise_candidate as entry
        from app_rules import configure_entry
        from endgame_strategy import EndgameSchedule
        self.assertIs(configure_entry(entry,defaults()),entry.Policy)
        self.assertEqual(alias_overrides(defaults(),CATALOG),{})
        self.assertEqual(entry.Policy().follower.lead_s,.17)
        schedule=EndgameSchedule();self.assertEqual(schedule.mode(44.9,0),'hold');self.assertEqual(schedule.mode(45.,0),'off')
        self.assertEqual(schedule.mode(46.,None),'off')

    def test_profile_round_trip_and_validation(self):
        p=defaults();p['run']['matches']=12;p['config_overrides']['LOOP_HZ']=35.
        p['rules']=[{'when':{'length_gte':40},'then':{'food_weights':{'dia':8.}}}]
        path=_scratch/'配置.json';write_json(path,p);self.assertEqual(validate(read_json(path)),p)
        validate_overrides(p,CATALOG)
        for section,key,value in [('run','matches',0),('run','matches',True),('controls','turn_lead_ms',float('nan')),('controls','respawn_release_seconds',.1),('run','pause_on_focus_loss','false')]:
            invalid=defaults();invalid[section][key]=value
            with self.assertRaises(ValueError):validate(invalid)
        bad=defaults();bad['config_overrides']['CAPTURE_BBOX']=[0,0,1,1]
        with self.assertRaises(ValueError):validate_overrides(bad,CATALOG)

    def test_unwritable_directory_falls_back_without_retries_or_losing_settings(self):
        import app_paths
        primary=_scratch/'readonly_config';primary.mkdir();write_json(primary/'profile.json',defaults())
        original_open=Path.open;old_path=app_paths._data_path;old_notice=app_paths.data_notice
        attempts=[]
        def guarded_open(path,*args,**kwargs):
            if path.parent==primary and '.write_check_'in path.name:
                attempts.append(path);raise PermissionError('simulated directory restriction')
            return original_open(path,*args,**kwargs)
        app_paths._data_path=None;app_paths.data_notice=''
        try:
            with patch.dict(os.environ,{'SNAKE25_DATA_DIR':str(primary)}),patch.object(app_paths.tempfile,'gettempdir',return_value=str(_scratch)),patch.object(Path,'open',guarded_open):
                chosen=app_paths.data_dir();self.assertEqual(chosen,_scratch/'SnakeAssistant25');self.assertEqual(len(attempts),1)
                self.assertTrue(app_paths.data_notice);self.assertEqual(read_json(chosen/'profile.json'),defaults())
                write_json(chosen/'profile.json',{'updated':True});self.assertEqual(read_json(chosen/'profile.json'),{'updated':True})
            target=_scratch/'atomic_original.json';write_json(target,{'old':True})
            with patch.object(app_paths.os,'replace',side_effect=PermissionError('locked destination')):
                with self.assertRaises(PermissionError):write_json(target,{'new':True})
            self.assertEqual(read_json(target),{'old':True});self.assertFalse(list(_scratch.glob('.atomic_original_*.tmp')))
        finally:app_paths._data_path=old_path;app_paths.data_notice=old_notice
        bad=defaults();bad['literal_overrides']['not-a-parameter']=1
        with self.assertRaises(ValueError):validate_overrides(bad,CATALOG)

    def test_catalog_specializes_functions_and_menu_hooks(self):
        p=defaults();p['controls']['release_seconds']=41.;p['controls']['turn_lead_ms']=195.
        values=alias_overrides(p,CATALOG)
        loader=TuneLoader('endgame_strategy',ENGINE/'endgame_strategy.py',p,values,_scratch)
        ns={};exec(loader.source_to_code(loader.path.read_bytes(),str(loader.path)),ns)
        self.assertEqual(ns['EndgameSchedule']().release_at,41.)
        loader=TuneLoader('latency_precision',ENGINE/'latency_precision.py',p,values,_scratch)
        ns={};exec(loader.source_to_code(loader.path.read_bytes(),str(loader.path)),ns)
        self.assertEqual(ns['PrecisionFollower']().lead_s,.195)
        for module in('bot','run_series'):
            item=next(x for x in CATALOG['parameters']if x['module']==module and x['kind']=='literal'and type(x['default'])is int)
            loader=TuneLoader(module,ENGINE/(module+'.py'),p,{item['id']:item['default']+1},_scratch)
            code=loader.source_to_code(loader.path.read_bytes(),str(loader.path))
            self.assertIn('app_runtime_control',code.co_names)
        p['strategy_options']['escape_hint_seconds']=6.
        aliases=alias_overrides(p,CATALOG)
        self.assertFalse(any(x['id']in aliases and 'len(route)<4'in x['context']for x in CATALOG['parameters']))

    def test_all_runtime_modules_compile_with_hooks(self):
        for path in ENGINE.glob('*.py'):
            TuneLoader(path.stem,path,defaults(),{},_scratch).source_to_code(path.read_bytes(),str(path))
        import bot,run_series
        self.assertIn('loop_tick',bot.main.__code__.co_names)
        self.assertIn('match_completed',run_series.main.__code__.co_names)

    def test_unicode_images(self):
        from app_image_io import imread,imwrite
        folder=_scratch/'中文截图';folder.mkdir(exist_ok=True);path=folder/'棋盘.png'
        image=np.zeros((40,70,3),np.uint8);image[:,:,1]=180
        self.assertTrue(imwrite(path,image));self.assertTrue(np.array_equal(imread(path),image))
        from game_start import GameDetector
        self.assertIsNotNone(GameDetector().tpl)

    def test_resolution_transform_and_normalized_capture(self):
        from app_capture import BOARD,LOBBY,fit_viewport,transform_box,normalize
        ref=cv2.cvtColor(__import__('app_image_io').imread(ENGINE/'assets'/'lobby_footer.png'),cv2.COLOR_BGR2GRAY)
        scores={}
        for width,height in((1280,720),(1920,1080),(2560,1440),(3840,2160)):
            viewport=fit_viewport((0,0,width,height))
            box=transform_box(viewport,BOARD)
            for actual,expected in zip(box,BOARD):self.assertLessEqual(abs(actual*2560/width-expected),1.)
            l,t,r,b=transform_box(viewport,LOBBY)
            small=cv2.resize(ref,(r-l,b-t),interpolation=cv2.INTER_AREA)
            restored=cv2.resize(small,(420,125),interpolation=cv2.INTER_AREA)
            score=float(cv2.matchTemplate(restored[20:110,40:230],ref[20:110,40:230],cv2.TM_CCOEFF_NORMED)[0,0])
            scores[str(width)+'x'+str(height)]=score;self.assertGreaterEqual(score,.90)
            raw=np.zeros((box[3]-box[1],box[2]-box[0],3),np.uint8);raw[:]=[12,34,56]
            normalized=normalize(raw);self.assertEqual(normalized.shape,(800,928,3));self.assertTrue(normalized.flags.c_contiguous)
            self.assertEqual(tuple(normalized[200,200]),(56,34,12))
        self.assertEqual(transform_box((-1920.,0.,1920.,1080.),(0,0,2560,1440)),(-1920,0,0,1080))
        viewport=fit_viewport((100,200,1920,1200));self.assertEqual(viewport,(100.,260.,1920.,1080.))
        write_json(_scratch/'resolution_scores.json',scores)

    def test_adaptive_lobby_uses_scaled_external_footer(self):
        from app_capture import AdaptiveLobby,LOBBY,transform_box
        from app_image_io import imread
        lobby=AdaptiveLobby.__new__(AdaptiveLobby)
        reference=imread(ENGINE/'assets'/'lobby_footer.png');lobby.ref=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)[20:110,40:230]
        from app_lobby import LobbyEvidence
        lobby.evidence=LobbyEvidence(cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY))
        class FakeCapture:
            def grab(self,box):
                return cv2.resize(reference,(box['width'],box['height']),interpolation=cv2.INTER_AREA)
        lobby.sct=FakeCapture();lobby.locator=SimpleNamespace(viewport=lambda s:(0,0,1280,720))
        self.assertTrue(lobby.visible())
        lobby.locator=SimpleNamespace(viewport=lambda s:None);self.assertFalse(lobby.visible())

    def test_lobby_word_and_x_are_independent_across_resolutions(self):
        from app_lobby import LobbyEvidence
        from app_image_io import imread
        ref=imread(ENGINE/'assets'/'lobby_footer.png',cv2.IMREAD_GRAYSCALE);detector=LobbyEvidence(ref)
        for width,height in((210,62),(315,94),(420,125),(630,188)):
            scaled=cv2.resize(cv2.resize(ref,(width,height),interpolation=cv2.INTER_AREA),(420,125),interpolation=cv2.INTER_AREA)
            evidence=detector.observe(scaled)
            self.assertTrue(evidence['word_visible'],str((width,evidence)))
            self.assertTrue(evidence['x_visible'],str((width,evidence)))
        no_x=ref.copy();no_x[50:110,32:100]=0
        evidence=detector.observe(no_x);self.assertTrue(evidence['word_visible']);self.assertFalse(evidence['x_visible'])
        no_word=ref.copy();no_word[50:110,175:280]=0
        evidence=detector.observe(no_word);self.assertFalse(evidence['word_visible']);self.assertTrue(evidence['x_visible'])
        evidence=detector.observe(np.zeros_like(ref));self.assertFalse(evidence['word_visible']);self.assertFalse(evidence['x_visible'])
        keyboard=imread(BASE/'app_assets'/'lobby_keyboard_footer.png',cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(keyboard)
        evidence=detector.observe(keyboard);self.assertTrue(evidence['word_visible']);self.assertFalse(evidence['x_visible'])

    def test_controller_wakeup_requires_stable_missing_x_and_bounded_retries(self):
        from app_lobby import ControllerModeGate
        missing={'word_visible':True,'x_visible':False};ready={'word_visible':True,'x_visible':True}
        g=ControllerModeGate()
        self.assertIsNone(g.action(missing,0,False));self.assertIsNone(g.action(missing,.2,False))
        self.assertEqual(g.action(missing,.4,False),'connect');g.connected(.4)
        self.assertIsNone(g.action(missing,.8,True));self.assertEqual(g.action(missing,1.11,True),'start');g.sent(1.11)
        self.assertIsNone(g.action(ready,1.2,True));self.assertIsNone(g.action(missing,1.3,True))
        # Keep sampling while waiting. Missing-X does not bypass the cooldown.
        for tick in np.arange(1.4,6.1,.2):self.assertIsNone(g.action(missing,float(tick),True))
        self.assertEqual(g.action(missing,6.12,True),'start');g.sent(6.12)
        for tick in np.arange(6.2,11.12,.2):g.action(missing,float(tick),True)
        self.assertEqual(g.action(missing,11.13,True),'start');g.sent(11.13)
        for tick in np.arange(11.2,30.,.2):self.assertIsNone(g.action(missing,float(tick),True))
        off=ControllerModeGate(enabled=False)
        for tick in(0,.4,.8):self.assertIsNone(off.action(missing,tick,True))

    def test_controller_wakeup_sends_start_only_after_fresh_confirmation(self):
        from app_lobby import ControllerModeGate,wake_controller
        missing={'word_visible':True,'x_visible':False};ready={'word_visible':True,'x_visible':True}
        class Pad:
            def __init__(self):self.calls=[]
            def reset(self):self.calls.append('reset')
            def tap(self,name,hold_ms):self.calls.append((name,hold_ms));return True
        g=ControllerModeGate();p=Pad()
        wake_controller(g,missing,0,p,lambda:p,lambda:missing)
        self.assertFalse(wake_controller(g,missing,.4,p,lambda:p,lambda:ready));self.assertEqual(p.calls,[])
        self.assertTrue(wake_controller(g,missing,.5,p,lambda:p,lambda:missing));self.assertEqual(p.calls,['reset',('start',160)])
        p=Pad();g=ControllerModeGate()
        for tick in(0,.5,1.):self.assertFalse(wake_controller(g,missing,tick,p,lambda:p,lambda:missing,dry=True))
        self.assertEqual(p.calls,[])

    def test_strategy_rule_actions_and_errors(self):
        from app_rules import RuleEngine,matches
        p=defaults();p['strategies']['rules_enabled']=True
        p['rules']=[{'when':{'length_gte':40,'elapsed_gte':45},'then':{'prefer_survival':True,'release_accel':True}}]
        red=SimpleNamespace(length=45,head_cell=(4,5),eye_state='normal',path_trust='ordered')
        state={'snakes':{'red':red},'items':{'ball':[(5,5,20,.1)]},'match_elapsed':50}
        self.assertTrue(RuleEngine(p).evaluate(state)['release_accel'])
        state['match_elapsed']=20;self.assertEqual(RuleEngine(p).evaluate(state),{})
        for bad in([{'when':{'oops':1},'then':{}}],[{'when':{},'then':{'target_cell':[29,0]}}],[{'when':{},'then':{'food_weights':{'dia':-1}}}]):
            with self.assertRaises(ValueError):validate_rules(bad)
        plugin=_scratch/'插件.py';plugin.write_text('def plan(context,state):\n    raise ValueError("test")\n',encoding='utf8')
        p['strategies']['plugin_enabled']=True;p['plugin_path']=str(plugin);r=RuleEngine(p)
        self.assertEqual(r.evaluate(state),{});self.assertIsNone(r.plugin);self.assertIn('ValueError',r.error)

    def test_strategy_toggles_keep_safe_pickup_proof(self):
        from app_rules import configure_entry
        import run_precise_candidate as entry
        from perception import SnakeObs
        head=(10,12);path=[(10-i,12)for i in range(5)]
        red=SnakeObs(family='red',n_px=1200,cells=set(path),length=5,head_px=(336.,400.),head_cell=head,
            tail_cell=path[-1],path=path,facing=(1,0),head_from_eyes=True,orientation='eyes',path_trust='ordered',eye_state='normal')
        belief=SimpleNamespace(head_cell=head,length=5,facing=(1,0),items={'dia':[]},danger_cells=[],ball_tracks=[])
        att=SimpleNamespace(gem_best=[(12,12,300,.2,'window',True)],item_centers={('dia',12,12):(400.,400.)},window_box=(0,0,928,800))
        def state():return dict(snakes={'red':red},items={'dia':[(12,12,300,.2)],'bean':[],'grn':[],'ball':[]},frame=1,t=1.,whiteout=False,player_alive=True,belief=belief,attention=att,item_centers=att.item_centers,match_elapsed=1.)
        for flag in('long_strategy','long_protect','chamber','head_crossings','escape_hints','food_dia','food_bean','food_grn'):
            p=defaults();p['strategies'][flag]=False;policy=configure_entry(entry,p)()
            decision=policy.decide(state());self.assertEqual(len(decision),4,flag)
            self.assertFalse(policy._route_safe([head,path[1]],red,set(),0),flag)

    def test_stop_focus_and_acceleration_release(self):
        import app_runtime_control as control
        class Event:
            def is_set(self):return self.set
            set=False
        class Pad:
            resets=0
            def reset(self):self.resets+=1
        event=Event();pad=Pad();control.event=event;control.profile=defaults();control.locator=SimpleNamespace(hwnd=1)
        control._paused=False
        with patch('app_windows.client_rect',return_value=(0,0,1000,1000)),patch('app_windows.foreground',return_value=False):
            self.assertEqual(control.loop_tick(pad),'paused');self.assertEqual(control.loop_tick(pad),'paused');self.assertEqual(pad.resets,1)
        with patch('app_windows.client_rect',return_value=(0,0,1000,1000)),patch('app_windows.foreground',return_value=True):self.assertEqual(control.loop_tick(pad),'resumed')
        event.set=True
        with self.assertRaises(KeyboardInterrupt):control.loop_tick(pad)
        control.profile=None;control.event=None

    def test_match_count_and_stop_do_not_start_extra_matches(self):
        import run_series,bot,app_runtime_control as control
        folder=_scratch/'series_count';(folder/'logs').mkdir(parents=True,exist_ok=True)
        played=[]
        def fake_match(**kwargs):
            played.append(kwargs);number=len(played)
            row=json.dumps({'phase':'PLAYING','whiteout':False,'red_len':12,'dia_cells':[],'accel':True,'why':'test'})+'\n'
            (folder/'logs'/('live_test_%d.jsonl'%number)).write_text(row*12,encoding='utf8')
            (folder/'logs'/('result_test_%d.png'%number)).write_bytes(b'completed-result-placeholder')
        argv=['run_series','--matches','2','--pause','0','--save-frames','0']
        with patch.object(run_series,'HERE',str(folder)),patch.object(bot,'main',side_effect=fake_match),patch.object(sys,'argv',argv),patch.object(control,'stop_requested',return_value=False),patch.object(control,'match_completed')as completed:
            with patch('sys.stdout',io.StringIO()):run_series.main()
            self.assertEqual(len(played),2);self.assertEqual(completed.call_count,2)
            self.assertEqual(completed.call_args[0][0],2)
        with patch.object(run_series,'HERE',str(folder)),patch.object(bot,'main')as main,patch.object(sys,'argv',argv),patch.object(control,'stop_requested',side_effect=[False,True]),patch.object(control,'match_completed')as completed:
            with patch('sys.stdout',io.StringIO()):run_series.main()
            self.assertEqual(main.call_count,1);completed.assert_not_called()

    def test_respawn_release_and_final_reset_with_fake_controller(self):
        from input_pad import Pad
        class Backend:
            def __init__(self):self.buttons=set();self.stick=(0,0)
            def press_button(self,button):self.buttons.add(button)
            def release_button(self,button):self.buttons.discard(button)
            def left_joystick_float(self,x_value_float,y_value_float):self.stick=(x_value_float,y_value_float)
            def reset(self):self.buttons.clear();self.stick=(0,0)
            def update(self):pass
        backend=Backend();clock=SimpleNamespace(now=0.)
        buttons=SimpleNamespace(**{'XUSB_GAMEPAD_'+s:s for s in('A','B','X','Y','START','BACK','LEFT_SHOULDER','RIGHT_SHOULDER','DPAD_RIGHT','DPAD_LEFT','DPAD_UP','DPAD_DOWN')})
        pad=Pad(backend=backend,vg=SimpleNamespace(XUSB_BUTTON=buttons),clock=lambda:clock.now)
        pad.set_accel(True);self.assertIn('A',backend.buttons);pad.rearm_accel(wait_for_motion=True)
        for t,x in((0.,100),(.08,110),(.16,120)):
            clock.now=t;pad.confirm_respawn_motion(SimpleNamespace(alive=True,head_px=(x,400.),head_age=0,eye_state='normal'),True,t)
        self.assertGreaterEqual(pad._rearm_at-.16,10/15)
        clock.now=.82;pad.service();self.assertNotIn('A',backend.buttons)
        clock.now=.87;pad.service();self.assertIn('A',backend.buttons)
        pad.reset();self.assertEqual(backend.buttons,set());self.assertEqual(backend.stick,(0,0))

    def test_native_tray_hotkey_and_worker_stop(self):
        from app_windows import StopEvent,Job,u
        from app_tray import Tray
        events=[];tray=Tray(events.append)
        try:
            self.assertIsNone(tray.failed);self.assertTrue(tray.hotkey)
            u.PostMessageW(tray.hwnd,0x0312,1,0)
            deadline=time.monotonic()+2
            while not events and time.monotonic()<deadline:time.sleep(.02)
            self.assertIn('stop',events)
        finally:tray.close()
        name='Local\\Snake25Test_'+uuid.uuid4().hex;event=StopEvent(name);job=Job()
        path=_scratch/'probe.json'
        command=([sys.executable]if getattr(sys,'frozen',False)else[sys.executable,str(BASE/'app.py')])+['--probe-worker','--event',name,'--status',str(path)]
        proc=subprocess.Popen(command,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            job.assign(proc);deadline=time.monotonic()+12
            while read_json(path,{}).get('state')!='probe_ready'and proc.poll()is None and time.monotonic()<deadline:time.sleep(.05)
            self.assertEqual(read_json(path,{}).get('state'),'probe_ready');event.set();proc.wait(timeout=5)
            self.assertEqual(proc.returncode,0);self.assertEqual(read_json(path)['state'],'probe_stopped')
        finally:
            job.close();event.close()
            if proc.poll()is None:proc.kill()

def self_test(output=None):
    global _scratch
    with tempfile.TemporaryDirectory(prefix='snake25_checks_')as temp:
        _scratch=Path(temp)
        install(ENGINE,defaults(),CATALOG,_scratch)
        stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(SoftwareTests)
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        report={'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
                'details':stream.getvalue(),'resolution_lobby_scores':read_json(_scratch/'resolution_scores.json',{})}
        if output:write_json(output,report)
        if sys.stdout:print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0 if result.wasSuccessful()else 1
