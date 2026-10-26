"""Bounded diagnostic worker. It never observes menus or sends pad commands."""
from pathlib import Path
from queue import Queue,Empty,Full
from threading import Thread,Event
import cv2


class DiagnosticWriter:
    def __init__(self,folder,perception_factory):
        self.folder=Path(folder)
        self.factory=perception_factory
        self.queue=Queue(maxsize=2)
        self.stop=Event()
        self.dropped=0
        self.error=None
        self.worker=Thread(target=self._run,name='snake-diagnostics',daemon=True)
        self.worker.start()

    def submit(self,frame_number,image):
        if self.error is not None:
            self.dropped+=1
            return False
        try:
            self.queue.put_nowait((frame_number,image.copy()))
            return True
        except Full:
            self.dropped+=1
            return False

    def _run(self):
        try:
            per=self.factory()
            self.folder.mkdir(parents=True,exist_ok=True)
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    number,image=self.queue.get(timeout=.05)
                except Empty:
                    continue
                try:
                    stem=self.folder/f'pf{number:05d}'
                    cv2.imwrite(str(stem)+'.png',cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                    w=per.work(image);fam,*_=per.families(w);vis=w.copy()
                    for key,color in [('red',(0,255,0)),('org',(255,0,255)),
                                      ('pur',(255,255,0)),('blu',(0,128,255))]:
                        vis[per.clean(fam[key])]=color
                    cv2.imwrite(str(stem)+'_mask.png',cv2.cvtColor(vis,cv2.COLOR_RGB2BGR))
                finally:
                    self.queue.task_done()
        except Exception as exc:
            self.error=exc

    def close(self):
        self.stop.set()
        self.worker.join(timeout=2.)
