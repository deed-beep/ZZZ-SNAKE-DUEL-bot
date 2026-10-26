import ast,importlib.abc,importlib.util,sys
from pathlib import Path
from app_catalog import literal_key,alias_overrides

def patch_runtime_source(name,text):
    if name=='bot':
        text=text.replace('import argparse\n','import argparse\nimport app_runtime_control as _desktop\n',1)
        text=text.replace('import app_runtime_control as _desktop\n','import app_runtime_control as _desktop\nimport app_lobby as _app_lobby\n',1)
        anchor='    lobby = LobbyDetector()'
        assert anchor in text
        text=text.replace(anchor,anchor+'\n    _controller_mode = _app_lobby.ControllerModeGate(**_desktop.menu_options())\n',1)
        anchor='    def return_to_lobby():'
        helper=('    def _wake_controller(now, evidence):\n'
                '        return _app_lobby.wake_controller(_controller_mode, evidence, now, pad, ensure_pad, lobby.observe, args.dry)\n\n')
        text=text.replace(anchor,helper+anchor,1)
        anchor=("            if ph == 'WAITING' and not phase.last_info.get('arena_like'):\n"
                '                if lobby.visible():\n'
                "                    phase.last_info['menu_kind'] = 'lobby'\n"
                "                    phase.last_info['lobby_score'] = round(lobby.score(), 3)")
        replacement=("            if ph == 'WAITING' and not phase.last_info.get('arena_like'):\n"
                '                _lobby_evidence = lobby.observe()\n'
                "                phase.last_info['controller_lobby'] = _lobby_evidence\n"
                "                if _lobby_evidence['word_visible'] and _lobby_evidence['x_visible']:\n"
                "                    phase.last_info['menu_kind'] = 'lobby'\n"
                "                    phase.last_info['lobby_score'] = round(_lobby_evidence['score'], 3)\n"
                "                elif _lobby_evidence['word_visible']:\n"
                "                    phase.last_info['menu_kind'] = None\n"
                '                    _wake_controller(now, _lobby_evidence)')
        assert anchor in text;text=text.replace(anchor,replacement,1)
        anchor="            if lobby.visible():\n                print('[auto] matching lobby confirmed')"
        replacement=("            _lobby_evidence = lobby.observe()\n"
            "            if _lobby_evidence['word_visible'] and not _lobby_evidence['x_visible']:\n"
            '                _wake_controller(tick, _lobby_evidence)\n'
            "            if _lobby_evidence['word_visible'] and _lobby_evidence['x_visible']:\n"
            "                print('[auto] matching lobby confirmed')")
        assert anchor in text;text=text.replace(anchor,replacement,1)
        text=text.replace('            args.dry = True','            raise RuntimeError("虚拟手柄创建失败，请检查ViGEmBus驱动。") from e',1)
        anchor='        while True:\n            now = control_time()'
        replacement=('        while True:\n            _desktop_state = _desktop.loop_tick(pad)\n'
                     '            if _desktop_state == "paused":\n                time.sleep(.1)\n                continue\n'
                     '            if _desktop_state == "resumed":\n                fresh_tracking()\n'
                     '            now = control_time()')
        assert anchor in text;text=text.replace(anchor,replacement,1)
        text=text.replace('            match_accel_mode = endgame_schedule.mode(now, match_started)',
            '            match_accel_mode = endgame_schedule.mode(now, match_started)\n'
            '            match_accel_mode = _desktop.effective_mode(match_accel_mode,policy,pad)',1)
        text=text.replace('            ph = phase.update(img, snakes, flags, now)',
            '            ph = phase.update(img, snakes, flags, now)\n            _desktop.phase_update(ph,red,now,match_started)',1)
        text=text.replace('                dec = fixed_accel_decision(dec, match_accel_mode)',
            '                dec = fixed_accel_decision(dec, match_accel_mode)\n                dec = _desktop.filter_decision(dec,policy)',1)
        text=text.replace('        while control_time() < deadline:\n            tick = control_time()',
            '        while control_time() < deadline:\n'
            '            if _desktop.loop_tick(pad) == "paused":\n                time.sleep(.1)\n                continue\n'
            '            tick = control_time()',1)
        text=text.replace("        if hasattr(vision,'close'):\n            vision.close()",
            "        if hasattr(vision,'close'):\n            vision.close()\n        if hasattr(lobby,'close'):\n            lobby.close()",1)
    if name=='run_series':
        text=text.replace('import argparse\n','import argparse\nimport app_runtime_control as _desktop\n',1)
        text=text.replace('    for i in range(len(results)+1, args.matches + 1):',
            '    for i in range(len(results)+1, args.matches + 1):\n'
            '        if _desktop.stop_requested():break\n        _desktop.match_started(i,args.matches)',1)
        text=text.replace('        elapsed = time.time() - t0','        if _desktop.stop_requested():break\n        elapsed = time.time() - t0',1)
        text=text.replace('        results.append(d)','        results.append(d)\n        _desktop.match_completed(i,d)',1)
        text=text.replace("        d['result_image']=fresh_shots[-1] if fresh_shots else None",
            "        d['result_image']=fresh_shots[-1] if fresh_shots else None\n"
            "        if not d['result_image']:raise RuntimeError('未确认本局结算，已停止；本局未计入完成局数。')",1)
        text=text.replace('            time.sleep(args.pause)','            if _desktop.wait_pause(args.pause):break',1)
    return text

class TuneLoader(importlib.abc.SourceLoader):
    def __init__(self,name,path,profile,literals,run_dir):
        self.name=name;self.path=path;self.profile=profile;self.literals=literals;self.run_dir=run_dir
    def get_filename(self,fullname):return str(self.path)
    def get_data(self,path):return Path(path).read_bytes()
    def source_to_code(self,data,path,**kwargs):
        data=data.replace(b'\r\n',b'\n')
        source=patch_runtime_source(self.name,data.decode('utf-8-sig'))
        tree=ast.parse(source,filename=path)
        # Literal ids refer to pristine 2.5 source lines, so apply before runtime
        # hook insertion changes line positions. Reparse transformed original first.
        original=ast.parse(data.decode('utf-8-sig'),filename=path)
        literal_values=self.literals;module=self.name
        class Literals(ast.NodeTransformer):
            def visit_Constant(self,node):
                key=literal_key(module,node)
                if key in literal_values:return ast.copy_location(ast.Constant(value=literal_values[key]),node)
                return node
        if any(k.startswith(module+':')for k in literal_values):
            transformed=Literals().visit(original);ast.fix_missing_locations(transformed)
            # Use source text edits, not ast.unparse (Python 3.9) for source hooks.
            if module not in('bot','run_series'):
                source=ast.unparse(transformed)
            # ast.unparse normalizes whitespace and defeats textual hooks. Hook
            # modules are instead specialized by offset replacement below.
            if module in('bot','run_series'):
                lines=data.decode('utf-8-sig').splitlines(keepends=True)
                edits=[]
                for node in ast.walk(original):
                    if isinstance(node,ast.Constant)and literal_key(module,node)in literal_values:
                        edits.append((node.lineno,node.col_offset,node.end_col_offset,repr(literal_values[literal_key(module,node)])))
                for line,start,end,value in sorted(edits,reverse=True):
                    # AST columns are UTF-8 byte offsets, not Unicode positions.
                    raw=lines[line-1].encode('utf8');lines[line-1]=(raw[:start]+value.encode()+raw[end:]).decode('utf8')
                source=patch_runtime_source(module,''.join(lines))
            tree=ast.parse(source,filename=path)
        class UnicodeIO(ast.NodeTransformer):
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Attribute)and isinstance(node.func.value,ast.Name)and node.func.value.id in('cv2','_cv2')and node.func.attr in('imread','imwrite'):
                    node.func.value=ast.Name(id='_unicode_image_io',ctx=ast.Load())
                return node
        tree=UnicodeIO().visit(tree)
        insert=0
        while insert<len(tree.body)and(isinstance(tree.body[insert],ast.Expr)and isinstance(tree.body[insert].value,ast.Constant)and isinstance(tree.body[insert].value.value,str)
              or isinstance(tree.body[insert],ast.ImportFrom)and tree.body[insert].module=='__future__'):insert+=1
        tree.body.insert(insert,ast.Import(names=[ast.alias(name='app_image_io',asname='_unicode_image_io')]))
        ast.fix_missing_locations(tree)
        return compile(tree,path,'exec',dont_inherit=True)
    def exec_module(self,module):
        code=self.source_to_code(self.path.read_bytes(),str(self.path));exec(code,module.__dict__)
        if self.name=='config':
            for key,value in self.profile.get('config_overrides',{}).items():
                old=getattr(module,key)
                setattr(module,key,tuple(value)if isinstance(old,tuple)else value)
            module.LOG_DIR=str(self.run_dir/'logs');module.FRAME_DUMP=str(self.run_dir/'frames_dump')
        elif self.name=='run_series':module.HERE=str(self.run_dir)
        elif self.name=='bot':
            module.FRAME_DUMP=str(self.run_dir/'frames_dump')
            module.TIMER_DUMP=str(self.run_dir/'timer_dump')

class EngineFinder(importlib.abc.MetaPathFinder):
    def __init__(self,engine,profile,catalog,run_dir):
        self.engine=Path(engine);self.profile=profile;self.literals=alias_overrides(profile,catalog);self.run_dir=Path(run_dir)
    def find_spec(self,fullname,path=None,target=None):
        if '.'in fullname:return None
        candidate=self.engine/(fullname+'.py')
        if not candidate.is_file():return None
        return importlib.util.spec_from_loader(fullname,TuneLoader(fullname,candidate,self.profile,self.literals,self.run_dir))

def install(engine,profile,catalog,run_dir):
    finder=EngineFinder(engine,profile,catalog,run_dir);sys.meta_path.insert(0,finder);return finder
