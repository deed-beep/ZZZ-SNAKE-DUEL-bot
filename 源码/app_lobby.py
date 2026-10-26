"""Independent lobby evidence and a bounded controller-mode wake-up gate."""
import cv2,numpy as np

class LobbyEvidence:
    def __init__(self,reference,word_threshold=.88,x_threshold=.82):
        self.reference=reference
        self.word=reference[62:102,190:264]
        self.x=reference[68:97,51:80]
        self.word_threshold=float(word_threshold);self.x_threshold=float(x_threshold)

    @staticmethod
    def correlate(search,template):
        if search.size==0 or search.shape[0]<template.shape[0]or search.shape[1]<template.shape[1]:return 0.
        if np.std(template)<1:return 0.
        return float(cv2.minMaxLoc(cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED))[1])

    def observe(self,gray):
        if gray is None or gray.shape!=(125,420):return dict(word_visible=False,x_visible=False,word_score=0.,x_score=0.,score=0.)
        # Keyboard mode centers the word about 16 px farther left. Search both
        # layouts, while excluding a far-right suffix such as “取消匹配”.
        word=self.correlate(gray[54:110,154:280],self.word)
        x=self.correlate(gray[55:108,36:97],self.x)
        score=self.correlate(gray[20:110,40:230],self.reference[20:110,40:230])
        return dict(word_visible=word>=self.word_threshold,x_visible=x>=self.x_threshold,
                    word_score=word,x_score=x,score=score)


class ControllerModeGate:
    def __init__(self,enabled=True,confirm_seconds=.35,retry_seconds=5.,max_attempts=3,warmup_seconds=.7):
        self.enabled=enabled;self.confirm_seconds=float(confirm_seconds);self.retry_seconds=float(retry_seconds)
        self.max_attempts=int(max_attempts);self.warmup_seconds=float(warmup_seconds)
        self.first=None;self.samples=0;self.last=None;self.ready_at=0.;self.next_at=0.;self.attempts=0

    def action(self,evidence,now,connected):
        missing=bool(evidence.get('word_visible')and not evidence.get('x_visible'))
        if not self.enabled or not missing:
            self.first=None;self.samples=0;self.last=now;return None
        if self.first is None or self.last is None or now-self.last>.6:self.first=now;self.samples=0
        self.last=now;self.samples+=1
        if self.samples<2 or now-self.first<self.confirm_seconds or self.attempts>=self.max_attempts:return None
        if not connected:return 'connect'
        if now<self.ready_at or now<self.next_at:return None
        return 'start'

    def connected(self,now):self.ready_at=now+self.warmup_seconds
    def sent(self,now):
        self.attempts+=1;self.next_at=now+self.retry_seconds;self.first=None;self.samples=0


def wake_controller(gate,evidence,now,pad,ensure_pad,verify,dry=False):
    """Only a fresh word-without-X observation authorizes Xbox START/Menu."""
    if dry:return False
    action=gate.action(evidence,now,pad is not None)
    if action=='connect':
        if ensure_pad()is not None:gate.connected(now)
        return False
    if action!='start':return False
    fresh=verify()
    if not fresh.get('word_visible')or fresh.get('x_visible'):return False
    pad.reset()
    if not pad.tap('start',hold_ms=160):return False
    gate.sent(now)
    print('[auto] 识别到“匹配”但缺少X提示；按手柄菜单键切换手柄模式（%d/%d）'%(gate.attempts,gate.max_attempts),flush=True)
    return True
