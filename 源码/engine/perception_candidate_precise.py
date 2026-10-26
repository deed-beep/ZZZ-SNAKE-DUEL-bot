"""Keep the three measured balls distinct from transient round snake snouts."""
import math
import cv2
import numpy as np
from perception import Perception as ReferencePerception
from perception import _pale_red_mask
from config import CELL,WORK_SCALE
from config import WHITEOUT_SAT,WHITEOUT_VAL
from temporal_flash_body import FlashBodySupport


class Perception(ReferencePerception):
    def __init__(self):
        super().__init__()
        self.flash_body=FlashBodySupport()
        self._flash_whiteout=False
        self._flash_recovered_px=0
        self._reacquire_eye=None

    def reset(self):
        super().reset()
        if hasattr(self,'flash_body'):self.flash_body.reset()
        self._reacquire_eye=None

    def families(self,img):
        result=super().families(img)
        self._flash_whiteout=(float(result[2].mean())/255.<WHITEOUT_SAT and
                              float(result[3].mean())/255.>WHITEOUT_VAL)
        return result

    def player_region(self,img_full,cleaned,slits,pairs):
        native=cleaned['red']|cleaned['org']
        self._flash_recovered_px=0
        if self._flash_whiteout:
            self.flash_body.reset()
            return None,'whiteout-identity-unavailable',None
        # Eye geometry alone can fit two highlights on the orange snake. Require
        # actual coral head pixels locally, independent of the connected body.
        verified=[]
        for entry in pairs:
            x,y=map(lambda v:int(round(v)),entry[1])
            crop=img_full[max(0,y-14):y+15,max(0,x-14):x+15]
            if not crop.size:continue
            hue=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)[...,0]
            coral=_pale_red_mask(crop)&((hue<11)|(hue>172))
            if int(coral.sum())>=40:verified.append(entry)
        pairs=verified
        old=self.prev_head_px.get('red')
        stale=self.frame_no-self.prev_eye_frame.get('red',-999)>6
        candidate=next((e for e in verified if e[0]<=120 and old is not None
                        and math.dist(e[1],old)>3*CELL),None)
        if stale and candidate is not None:
            last=self._reacquire_eye
            point=tuple(candidate[1])
            if (last and 0<self.frame_no-last[1]<=3 and math.dist(point,last[0])<=CELL):
                # Repeated real coral eyes may invalidate an eye-less orange
                # continuation (r039: actual stationary respawn at 17,14).
                for cache in (self.prev_cells,self.prev_head_px,self.prev_head_frame,
                              self.prev_head_vel,self.prev_eye_frame):cache.pop('red',None)
                self.prev_red_px=0
                self.player_body_hint=frozenset()
                self.flash_body.reset()
                self._eye_win_head=None
                self._reacquire_eye=None
            else:self._reacquire_eye=(point,self.frame_no)
        else:self._reacquire_eye=None
        pale=_pale_red_mask(self.work(img_full))
        excluded=cleaned['pur']|cleaned['blu']
        patch=self.flash_body.reconnect(native,pale,excluded,self.match_t)
        # The verified head trail supplies ownership even when a flash joins
        # several touching parts into one global component. Current pale
        # body pixels are still required; empty old tail tiles are not filled.
        hints=np.zeros_like(native)
        cell_work=CELL//WORK_SCALE
        for x,y in getattr(self,'player_body_hint',()):
            if 0<=x<29 and 0<=y<25:
                hints[y*cell_work:(y+1)*cell_work,x*cell_work:(x+1)*cell_work]=True
        patch |= pale & hints & ~excluded & ~native

        adjusted=cleaned
        if patch.any():
            adjusted=dict(cleaned,red=cleaned['red']|patch)
        result=super().player_region(img_full,adjusted,slits,pairs)
        if result[0] is not None:
            # A blue/purple crossing can split our currently visible warm body
            # into multiple components. The measured still-in-body head trail
            # witnesses those CURRENT native pixels even without a warm bridge.
            # No floor pixels or unrelated whole orange component is added.
            owned_now=(native|patch)&hints&~excluded
            joined=result[0]|owned_now
            if np.any(joined&~result[0]):
                result=(joined,result[1]+'+trailparts',result[2])
            # Only ORIGINAL native observations refresh memory. Reconstructed
            # pixels cannot keep themselves alive across frames.
            self.flash_body.remember(result[0]&native,self.match_t)
            self._flash_recovered_px=int((patch&result[0]).sum())
        return result

    def run(self,*args,**kwargs):
        snakes,items,flags=super().run(*args,**kwargs)
        flags['red_flash_recovered_px']=int(self._flash_recovered_px)
        return snakes,items,flags

    def _choose_balls(self,candidates):
        recent=[r for r in self._ball_recent if self.frame_no-r['frame']<=20 and r['hits']>=5]
        # Deduplicate history by physical position before reserving identities.
        stable=[]
        for r in sorted(recent,key=lambda r:(-r['hits'],-r['frame'])):
            if all(math.dist(r['px'],s['px'])>1.5*CELL for s in stable):
                stable.append(r)
        rejected=[]
        if len(stable)>=3:
            kept=[]
            for c in candidates:
                matches=any(math.dist(c['px'],r['px'])<=.75*CELL+2*(self.frame_no-r['frame'])
                            for r in stable[:3])
                if matches:kept.append(c)
                else:rejected.append(dict(c,reason='three established balls; unrelated new blob'))
            candidates=kept
        chosen=super()._choose_balls(candidates)
        self.ball_dropped.extend(rejected)
        return chosen
