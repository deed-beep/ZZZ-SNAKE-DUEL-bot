import sys,os,argparse,json,traceback
from pathlib import Path

def main():
    from app_windows import enable_dpi
    enable_dpi()
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--profile');parser.add_argument('--event');parser.add_argument('--status');parser.add_argument('--run-dir')
    parser.add_argument('--probe-worker',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--self-test-output')
    parser.add_argument('--ui-smoke');parser.add_argument('--acceptance-run',action='store_true')
    parser.add_argument('--acceptance-matches',type=int,default=1)
    parser.add_argument('--notice-check',action='store_true')
    args=parser.parse_args()
    from app_notice import verify_distribution,NoticeIntegrityError
    try:verify_distribution()
    except NoticeIntegrityError as exc:
        if args.notice_check:return 3
        raise
    if args.notice_check:return 0
    if args.probe_worker:
        from app_worker import probe_worker
        return probe_worker(args.event,args.status)
    if args.worker:
        from app_worker import run_worker
        return run_worker(args.profile,args.event,args.status,args.run_dir)
    if args.self_test:
        from app_tests import self_test
        return self_test(args.self_test_output)
    from app_gui import App
    from app_windows import SingleInstance
    instance=SingleInstance()
    try:
        if instance.already_running:
            __import__('ctypes').windll.user32.MessageBoxW(None,'软件已经在运行，请从右下角托盘打开。','蛇对蛇助手',64)
            return 0
        app=App(smoke_path=args.ui_smoke,acceptance=args.acceptance_run,acceptance_matches=args.acceptance_matches)
        app.run();return 0
    finally:instance.close()

if __name__=='__main__':
    try:
        import multiprocessing
        multiprocessing.freeze_support()
        raise SystemExit(main())
    except SystemExit:raise
    except BaseException:
        error=traceback.format_exc()
        try:
            from app_paths import data_dir
            (data_dir()/'startup_error.log').write_text(error,encoding='utf8')
        finally:
            if any(flag in sys.argv for flag in('--worker','--self-test','--probe-worker')):
                try:
                    from app_paths import write_json
                    flag='--self-test-output'if'--self-test'in sys.argv else'--status'
                    if flag in sys.argv:write_json(sys.argv[sys.argv.index(flag)+1],{'state':'error','passed':False,'message':error.splitlines()[-1],'details':error})
                finally:raise SystemExit(1)
            try:
                from app_notice import NoticeIntegrityError
                message=(str(sys.exc_info()[1])+'\n请从作者提供的完整原始软件包恢复声明与校验清单。')if isinstance(sys.exc_info()[1],NoticeIntegrityError)else error[-2500:]
                __import__('ctypes').windll.user32.MessageBoxW(None,message,'蛇对蛇助手启动失败',16)
            except Exception:pass
        raise
