from pathlib import Path
import json,sys,time,traceback
from app_paths import ENGINE,BASE,read_json

def run_worker(profile_path,event_name,status_path,run_dir):
    from app_windows import StopEvent,driver_installed
    from app_settings import validate
    from app_catalog import validate_overrides
    from app_engine_loader import install
    from app_capture import Locator,AdaptiveVision,AdaptiveLobby
    import app_runtime_control as control
    run_dir=Path(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    stream=(run_dir/'console.log').open('a',encoding='utf8',buffering=1)
    sys.stdout=stream;sys.stderr=stream
    event=StopEvent(event_name,create=False)
    profile=validate(read_json(profile_path,{}));catalog=read_json(BASE/'parameter_catalog.json')
    validate_overrides(profile,catalog);locator=Locator(profile['capture'])
    control.configure(event,profile,locator,Path(status_path))
    from app_notice import verify_distribution,NoticeWatchdog
    watchdog=NoticeWatchdog();control.notice_watchdog=watchdog
    try:
        verify_distribution()
        if not driver_installed():raise RuntimeError('未检测到ViGEmBus虚拟手柄驱动，请先安装驱动。')
        control.update(phase='正在加载2.5引擎')
        install(ENGINE,profile,catalog,run_dir)
        import bot
        import run_precise_candidate as entry
        from app_rules import configure_entry
        bot.Policy=configure_entry(entry,profile)
        bot.Perception=entry.Perception;bot.StateAnalysis=entry.StateAnalysis
        bot.Vision=lambda:AdaptiveVision(locator)
        bot.LobbyDetector=lambda:AdaptiveLobby(locator)
        if not profile['strategies']['late_release']:
            original=bot.EndgameSchedule
            class NoLateRelease(original):
                def __init__(self,requested='hold',release_at=45.):super().__init__(requested,1.e12)
            bot.EndgameSchedule=NoLateRelease
        delay=profile['run']['start_delay']
        control.update(phase='准备开始',message='即将识别大厅')
        if event.wait(delay):control.finish('stopped','已停止');return 0
        if event.is_set():control.finish('stopped','已停止');return 0
        watchdog.check();verify_distribution()
        import run_series
        options=profile['run'];matches=1000000000 if options['infinite']else options['matches']
        sys.argv=['run_series.py','--matches',str(matches),'--pause',str(options['pause_seconds']),
                  '--minutes',str(options['waiting_timeout_minutes']or 1.e8),
                  '--after-start',str(options['after_start_seconds']),'--save-frames',str(options['save_frames']),
                  '--capture-threads',str(profile['controls']['capture_threads']),
                  '--accel-mode',profile['controls']['accel_mode']]
        run_series.main()
        control.finish('stopped'if event.is_set()else'completed','已停止，手柄已释放。'if event.is_set()else'已完成所选局数。')
        return 0
    except KeyboardInterrupt:
        control.finish('stopped','已停止，手柄已释放。');return 0
    except BaseException:
        error=traceback.format_exc();print(error,flush=True)
        control.finish('error',error.splitlines()[-1]);return 1
    finally:watchdog.close();control.notice_watchdog=None;event.close();stream.flush()

def probe_worker(event_name,status_path):
    """Exercise packaged worker and cooperative stop without creating a controller."""
    from app_windows import StopEvent
    from app_paths import write_json
    event=StopEvent(event_name,False)
    try:
        write_json(status_path,{'state':'probe_ready','pid':__import__('os').getpid()})
        event.wait(30.)
        write_json(status_path,{'state':'probe_stopped'})
        return 0
    finally:event.close()
