"""Small, non-blocking control hooks. All disk status writing is off the game loop."""
import threading,time
from app_paths import write_json

event=None;profile=None;locator=None;status_path=None
_lock=threading.Lock();_data={};_done=threading.Event();_paused=False;_last_phase_update=0.;_thread=None;notice_watchdog=None

def configure(stop_event,options,screen,status_file):
    global event,profile,locator,status_path,_data,_paused,_thread
    event=stop_event;profile=options;locator=screen;status_path=status_file;_paused=False
    _data=dict(state='running',phase='等待大厅',completed=0,current=0,total=0,length=0,message='',updated=time.time())
    _done.clear();_thread=threading.Thread(target=_writer,daemon=True,name='status-writer');_thread.start()

def update(**values):
    with _lock:_data.update(values,updated=time.time())

def _writer():
    while not _done.is_set():
        with _lock:value=dict(_data)
        try:write_json(status_path,value)
        except OSError:pass
        _done.wait(.4)

def finish(state,message=''):
    update(state=state,message=message)
    _done.set()
    if _thread and _thread is not threading.current_thread():_thread.join(timeout=2.)
    with _lock:value=dict(_data)
    write_json(status_path,value)

def stop_requested():return bool(event and event.is_set())
def check_stop():
    if stop_requested():raise KeyboardInterrupt()

def menu_options():
    options=profile['run']if profile else{}
    return dict(enabled=options.get('controller_wakeup',True),confirm_seconds=options.get('controller_confirm_seconds',.35),
                retry_seconds=options.get('controller_retry_seconds',5.),max_attempts=options.get('controller_max_attempts',3))

def loop_tick(pad):
    global _paused
    if notice_watchdog:
        try:notice_watchdog.check()
        except Exception:
            if pad is not None:pad.reset()
            raise
    check_stop()
    if profile is None:return 'run'
    if profile['run']['pause_on_focus_loss']:
        from app_windows import foreground,find_target,client_rect
        hwnd=locator.hwnd
        if not client_rect(hwnd):
            hwnd=find_target(profile['capture']);locator.hwnd=hwnd
        if not foreground(hwnd):
            if not _paused:
                if pad is not None:pad.reset()
                update(phase='等待游戏前台',message='请将游戏窗口置于前台，恢复后自动继续。')
            _paused=True;return 'paused'
    if _paused:_paused=False;update(message='');return 'resumed'
    return 'run'

def effective_mode(mode,policy,pad):
    actions=getattr(policy,'app_actions',{})
    if actions.get('release_accel'):
        if pad is not None:pad.stop_accel_schedule()
        return 'off'
    return mode

def filter_decision(decision,policy):
    if getattr(policy,'app_actions',{}).get('release_accel'):
        return(decision[0],False,False,decision[3]+'+rule_release')
    return decision

def phase_update(phase,red,now,started):
    global _last_phase_update
    if now-_last_phase_update<.5 and _data.get('engine_phase')==phase:return
    _last_phase_update=now
    names={'WAITING':'等待大厅 / 匹配','PLAYING':'正在游戏','RESULT':'结算 / 返回大厅'}
    update(phase=names.get(phase,phase),engine_phase=phase,length=int(getattr(red,'length',0)or 0),
           elapsed=round(now-started,1)if started is not None else 0.,message='')

def match_started(number,total):update(current=number,total=total,phase='等待大厅 / 匹配')
def match_completed(number,digest):update(completed=number,length=0,phase='本局结束',result_image=digest.get('result_image'))
def wait_pause(seconds):return event.wait(seconds)if event else False
