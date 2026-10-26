"""One capture worker, one latest-frame slot; the control thread owns all input.

MSS is constructed and closed inside its worker because its handles are local
to a thread. Frames carry the time capture STARTED, never their dequeue time.
"""
from threading import Condition, Event, Thread
import time


class LatestCapture:
    def __init__(self, factory, hz=40):
        self.factory=factory
        self.period=1./hz
        self.condition=Condition()
        self.stop=Event()
        self.latest=None
        self.error=None
        self.consumed=0
        self.skipped=0
        self.capture_ms=0.
        self.worker=Thread(target=self._run,name='snake-capture',daemon=True)
        self.worker.start()

    def _run(self):
        source=None
        try:
            source=self.factory()
            sequence=0
            while not self.stop.is_set():
                started=time.perf_counter()
                frame=source.grab()
                duration=time.perf_counter()-started
                sequence+=1
                with self.condition:
                    self.latest=(sequence,frame,started,duration,time.perf_counter())
                    self.condition.notify_all()
                self.stop.wait(max(0.,self.period-duration))
        except BaseException as exc:
            with self.condition:
                self.error=exc
                self.condition.notify_all()
        finally:
            if source is not None and hasattr(source,'close'):
                source.close()

    def grab_sample(self):
        cutoff=time.perf_counter()-.004
        with self.condition:
            ready=self.condition.wait_for(lambda: self.error is not None or
                self.stop.is_set() or (self.latest is not None and self.latest[0]>self.consumed
                                      and self.latest[4]>=cutoff),2.)
            if self.error is not None:
                raise RuntimeError('capture worker failed') from self.error
            if not ready or self.latest is None or self.latest[0]<=self.consumed:
                raise RuntimeError('no fresh screen frame')
            sequence,frame,started,duration,_=self.latest
            self.skipped+=max(0,sequence-self.consumed-1)
            self.consumed=sequence
            self.capture_ms=duration*1000
            return frame,started

    def grab(self):
        return self.grab_sample()[0]

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        self.worker.join(timeout=1.)
