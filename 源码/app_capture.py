"""Adapt external pixels only; canonical 928x800 perception stays unchanged."""
import time
from pathlib import Path
import cv2,numpy as np
from app_paths import ENGINE
from app_image_io import imread
from app_windows import client_rect,find_target

REFERENCE=(2560.,1440.)
BOARD=(647.,269.,1911.,1369.)
LOBBY=(2080.,1290.,2500.,1415.)

def fit_viewport(rect,preserve=True):
    x,y,w,h=rect
    if not preserve:return(float(x),float(y),float(w),float(h))
    scale=min(w/REFERENCE[0],h/REFERENCE[1]);width=REFERENCE[0]*scale;height=REFERENCE[1]*scale
    return(x+(w-width)/2.,y+(h-height)/2.,width,height)

def transform_box(viewport,box):
    x,y,w,h=viewport;sx=w/REFERENCE[0];sy=h/REFERENCE[1]
    return tuple(int(round(v))for v in(x+box[0]*sx,y+box[1]*sy,x+box[2]*sx,y+box[3]*sy))

def normalize(raw,size=(928,800)):
    if raw.shape[:2]!=(size[1],size[0]):raw=cv2.resize(raw,size,interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB))

class Locator:
    def __init__(self,settings):self.settings=settings;self.hwnd=0;self.last=0.;self.rect=None;self.error=''
    def viewport(self,sct):
        now=time.perf_counter()
        mode=self.settings['mode']
        if now-self.last>.4 or not self.hwnd:
            self.hwnd=find_target(self.settings);self.last=now
        if mode in('window','manual'):
            self.rect=client_rect(self.hwnd)
            if not self.rect:self.error='等待游戏窗口';return None
        else:
            index=int(self.settings.get('monitor',1));monitors=sct.monitors
            if not 1<=index<len(monitors):self.error='所选显示器不存在';return None
            m=monitors[index];self.rect=(m['left'],m['top'],m['width'],m['height'])
        self.error=''
        if self.settings.get('viewport'):
            a=self.settings['viewport'];x,y,w,h=self.rect
            return(x+a[0]*w,y+a[1]*h,a[2]*w,a[3]*h)
        return fit_viewport(self.rect,self.settings.get('fit_16_9',True))
    def board_box(self,sct):
        viewport=self.viewport(sct)
        if viewport is None:return None
        custom=self.settings.get('manual_board')
        if self.settings['mode']=='manual'and custom:
            x,y,w,h=self.rect
            return tuple(int(round(v))for v in(x+custom[0]*w,y+custom[1]*h,x+custom[2]*w,y+custom[3]*h))
        return transform_box(viewport,BOARD)

class AdaptiveVision:
    def __init__(self,locator):
        import mss
        self.sct=mss.mss();self.locator=locator;self.size=(928,800);self.last_box=None
    def grab(self):
        box=self.locator.board_box(self.sct);self.last_box=box
        if box is None:return np.zeros((800,928,3),np.uint8)
        l,t,r,b=box
        try:raw=np.asarray(self.sct.grab(dict(left=l,top=t,width=r-l,height=b-t)))[:,:,:3]
        except Exception:
            self.locator.error='游戏窗口暂不可见';return np.zeros((800,928,3),np.uint8)
        return normalize(raw)
    def close(self):self.sct.close()

class AdaptiveLobby:
    def __init__(self,locator):
        import mss
        self.sct=mss.mss();self.locator=locator
        ref=imread(ENGINE/'assets'/'lobby_footer.png',cv2.IMREAD_GRAYSCALE)
        if ref is None:raise RuntimeError('缺少大厅识别模板。')
        self.ref=ref[20:110,40:230]
        from app_lobby import LobbyEvidence
        self.evidence=LobbyEvidence(ref,locator.settings.get('lobby_word_threshold',.88),locator.settings.get('lobby_x_threshold',.82))
    def observe(self):
        viewport=self.locator.viewport(self.sct)
        if viewport is None:return self.evidence.observe(None)
        l,t,r,b=transform_box(viewport,LOBBY)
        try:
            raw=np.asarray(self.sct.grab(dict(left=l,top=t,width=r-l,height=b-t)))[:,:,:3]
            if raw.shape[:2]!=(125,420):raw=cv2.resize(raw,(420,125),interpolation=cv2.INTER_AREA)
            gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
            return self.evidence.observe(gray)
        except Exception:return self.evidence.observe(None)
    def score(self):return self.observe()['score']
    def visible(self):
        evidence=self.observe();return evidence['word_visible']and evidence['x_visible']
    def close(self):self.sct.close()
