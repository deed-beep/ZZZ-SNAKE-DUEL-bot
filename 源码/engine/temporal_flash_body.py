"""Reconnect a current pale body gap using recent native ownership evidence."""
from collections import deque
import cv2
import numpy as np


class FlashBodySupport:
    def __init__(self,ttl=.40,max_patch=768):
        self.ttl=ttl;self.max_patch=max_patch
        self.history=deque(maxlen=4)

    def reset(self):
        self.history.clear()

    def remember(self,native_owner,stamp):
        self.history.append((float(stamp),native_owner.copy()))

    def reconnect(self,native,pale,excluded,stamp):
        out=np.zeros_like(native,dtype=bool)
        prior=np.zeros_like(native,dtype=bool)
        for t,mask in self.history:
            if mask.shape==native.shape and 0<float(stamp)-t<=self.ttl:
                prior|=mask
        candidates=prior&pale&~native&~excluded
        if int(candidates.sum())<8:
            return out
        _n,native_labels=cv2.connectedComponents(native.astype('uint8'),8)
        n,labels,stats,_=cv2.connectedComponentsWithStats(candidates.astype('uint8'),8)
        kernel=np.ones((5,5),np.uint8)
        for k in range(1,n):
            if not 8<=int(stats[k,cv2.CC_STAT_AREA])<=self.max_patch:
                continue
            patch=labels==k
            contact=cv2.dilate(patch.astype('uint8'),kernel)>0
            anchors=np.unique(native_labels[contact&native])
            # A previous tail on empty floor and a lone pale sparkle are not
            # bridges. Both current native fragments must still be visible.
            if len(anchors)>=2:
                out|=patch
        return out
