"""Play several matches back to back, pausing after each so the user can review it.

The user's request: "现在继续改进验证脚本，至少迭代5局以上，每局结束后稍作等待，我会给出我对本局游戏的
表现评价" - run at least five matches, wait a moment after each, and they will comment on each one.

So this is a loop around `bot.main` rather than a new controller:

  * each match is one `bot.main` call, which already handles the whole cycle (lobby X, START, steering, the
    result screen and A back to the lobby), so the cross-match logic belongs here and not in `bot.py`;
  * after each match it waits `--pause` seconds, prints the score screen path and a one-line digest of the
    match just played, so the user's comment lands against specific numbers;
  * it stops after `--matches` matches or on Ctrl-C, and prints a summary table at the end.

The per-match digest reuses `run_summary.py`'s numbers rather than inventing new ones, and reads the log
that match just wrote, so it reports what happened rather than what was intended.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def newest_log(before):
    """The newest live log that did not exist before this match started."""
    now = set(glob.glob(os.path.join(HERE, "logs", "live_*.jsonl")))
    new = sorted(now - before)
    return new[-1] if new else None


def digest(path):
    rows = []
    for line in io.open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    P = [r for r in rows if r.get("phase") == "PLAYING"]
    if not P:
        return dict(frames=0)
    live = [r for r in P if r.get("whiteout") is not True]
    accel = sum(1 for r in live if r.get("accel") is True)
    lens = [r["red_len"] for r in P if isinstance(r.get("red_len"), (int, float))]
    gems = [len(r.get("dia_cells") or []) for r in P if isinstance(r.get("dia_cells"), list)]
    deaths = 0
    in_wo = False
    for r in P:
        w = r.get("whiteout") is True
        if w and not in_wo:
            deaths += 1
        in_wo = w
    modes = {}
    for r in P:
        k = str(r.get("why") or "").split("+")[0]
        modes[k] = modes.get(k, 0) + 1
    return dict(frames=len(P),
                duty=(100.0 * accel / len(live)) if live else 0.0,
                peak=max(lens) if lens else 0,
                final=lens[-1] if lens else 0,
                gems=(sum(gems) / len(gems)) if gems else 0.0,
                deaths=deaths,
                top=sorted(modes.items(), key=lambda kv: -kv[1])[:4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=6, help="how many matches to play")
    ap.add_argument("--pause", type=float, default=20.0,
                    help="seconds to wait after each match so the user can review it")
    ap.add_argument("--after-start", type=float, default=75.0)
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--save-frames", type=int, default=0)
    ap.add_argument('--capture-threads',type=int,choices=(1,2),default=1)
    ap.add_argument('--resume-manifest',help='continue an existing completed-match prefix')
    ap.add_argument('--accel-mode',choices=('hold','off','adaptive'),default='hold')
    args = ap.parse_args()

    import bot
    results = []
    os.makedirs(os.path.join(HERE,'logs'),exist_ok=True)
    manifest=os.path.join(HERE,'logs','batch_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    if args.resume_manifest:
        manifest=os.path.abspath(args.resume_manifest)
        with io.open(manifest,encoding='utf-8') as stream:
            results=json.load(stream)
        if (not isinstance(results,list) or len(results)>=args.matches or
                [r.get('match') for r in results]!=list(range(1,len(results)+1)) or
                any(not r.get('result_image') or not r.get('log') for r in results)):
            raise ValueError('resume requires a contiguous completed-match prefix')
        print(f'[resume] preserving {len(results)} completed matches; next is {len(results)+1}',flush=True)
    for i in range(len(results)+1, args.matches + 1):
        before = set(glob.glob(os.path.join(HERE, "logs", "live_*.jsonl")))
        shots_before=set(glob.glob(os.path.join(HERE,'logs','result_*.png')))
        print(f"\n{'=' * 78}\n  MATCH {i} of {args.matches}\n{'=' * 78}", flush=True)
        t0 = time.time()
        # bot.main() parses sys.argv itself (the test suite wraps it in patch('sys.argv', ...)), so the
        # runner's own flags must be hidden from it or it exits on "unrecognized arguments".
        saved_argv = sys.argv
        sys.argv = ["bot.py",'--save-frames',str(args.save_frames),
                    '--capture-threads',str(args.capture_threads)]
        try:
            # NO BOOST: the current target is the user's luping8 style - the key never pressed, length from
            # collection alone ("红蛇不按键加速且结算长度在50以上").  `off` makes bot.py pass fixed_accel=False
            # and policy.Policy() reads NO_BOOST=True, so A is never written down.  Pass --accel-mode hold to
            # restore the boosted mode.
            bot.main(log_on=True, auto_start=True, in_match=True,
                     after_start=args.after_start, minutes=args.minutes, accel_mode=args.accel_mode)
        except KeyboardInterrupt:
            print("\ninterrupted by the user - stopping the series")
            break
        except Exception as exc:
            print(f"match {i} raised {type(exc).__name__}: {exc}")
            raise  # Do not silently start further matches after a control crash.
        finally:
            sys.argv = saved_argv
        elapsed = time.time() - t0
        log = newest_log(before)
        d = digest(log) if log else dict(frames=0)
        if d.get('frames',0)<10:
            raise RuntimeError('No completed match evidence; series stopped')
        d['match']=i
        d['controller_variant']=getattr(bot.Policy,'variant','batch4_reference')
        d['log']=log
        fresh_shots=sorted(set(glob.glob(os.path.join(HERE,'logs','result_*.png')))-shots_before)
        d['result_image']=fresh_shots[-1] if fresh_shots else None
        results.append(d)
        with io.open(manifest,'w',encoding='utf-8') as out:
            json.dump(results,out,ensure_ascii=False,indent=2)
        shots = sorted(glob.glob(os.path.join(HERE, "logs", "result_*.png")))
        print(f"\n  match {i}: {elapsed:.0f}s   log={os.path.basename(log) if log else 'none'}")
        print(f"    boost duty {d.get('duty', 0):.1f}%   peak len {d.get('peak', 0)}   "
              f"visual end {d.get('final', 0)} (not settlement)   deaths {d.get('deaths', 0)}   "
              f"gems/frame {d.get('gems', 0):.2f}")
        print(f"    modes {d.get('top', [])}")
        if shots:
            print(f"    score screen: {shots[-1]}")
        if i < args.matches:
            print(f"\n  pausing {args.pause:.0f}s for review ...", flush=True)
            time.sleep(args.pause)

    print(f"\n{'=' * 78}\n  SERIES SUMMARY ({len(results)} matches)\n{'=' * 78}")
    print(f"  {'#':>3s} {'duty%':>7s} {'peak':>5s} {'visual':>6s} {'whiteout':>8s} {'gems/f':>7s}")
    for i, d in enumerate(results, 1):
        print(f"  {i:3d} {d.get('duty', 0):7.1f} {d.get('peak', 0):5d} {d.get('final', 0):5d} "
              f"{d.get('deaths', 0):7d} {d.get('gems', 0):7.2f}")
    if results:
        print(f"\n  mean peak {sum(d.get('peak', 0) for d in results) / len(results):.1f}   "
              f"mean deaths {sum(d.get('deaths', 0) for d in results) / len(results):.2f}   "
              f"mean duty {sum(d.get('duty', 0) for d in results) / len(results):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
