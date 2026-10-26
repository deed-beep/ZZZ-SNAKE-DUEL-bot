"""Explicit candidate entry with a three-ball continuity roster."""
import cv2
cv2.setNumThreads(4)
import bot
from policy_long_strategy import Policy as CollectionPolicy
from perception_latency import Perception
from latency_precision import PrecisionFollower
from motion_candidate_precise import StateAnalysis
from run_series import main


class Policy(CollectionPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.follower=PrecisionFollower(no_boost=True)
        self.key_boost=True
        self.target_keep_route=6

    variant=CollectionPolicy.variant+'+r188_long_chamber_head_crossing_20261025'


if __name__=='__main__':
    bot.Policy=Policy
    bot.Perception=Perception
    bot.StateAnalysis=StateAnalysis
    print('[candidate] nearest food, checked pairs, buffer direction, three-ball roster; A held until START+45s, then released',flush=True)
    main()
