"""Remove stale adjacent copies of a ball that was measured again this frame."""
from motion import MotionModel as ReferenceMotionModel
from state_analysis import StateAnalysis as ReferenceStateAnalysis


class MotionModel(ReferenceMotionModel):
    def update_balls(self,items,frame):
        super().update_balls(items,frame)
        fresh=[t for t in self.balls if t.frames and t.frames[-1]==frame]
        self.balls=[t for t in self.balls if t in fresh or not any(
            max(abs(t.cells[-1][i]-seen.cells[-1][i]) for i in (0,1))<=1
            for seen in fresh)]
        self.notes=[t.mode for t in self.balls]


class StateAnalysis(ReferenceStateAnalysis):
    def __init__(self):
        super().__init__()
        self.motion=MotionModel()
