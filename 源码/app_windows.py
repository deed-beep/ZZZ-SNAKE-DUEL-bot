import ctypes,os,time
from ctypes import wintypes as W

u=ctypes.windll.user32;k=ctypes.windll.kernel32
u.GetForegroundWindow.restype=W.HWND
u.GetAncestor.argtypes=[W.HWND,W.UINT];u.GetAncestor.restype=W.HWND
u.IsWindow.argtypes=[W.HWND];u.IsWindowVisible.argtypes=[W.HWND]
u.GetClientRect.argtypes=[W.HWND,ctypes.POINTER(W.RECT)]
u.ClientToScreen.argtypes=[W.HWND,ctypes.POINTER(W.POINT)]
u.GetWindowTextLengthW.argtypes=[W.HWND]
u.GetWindowTextW.argtypes=[W.HWND,W.LPWSTR,ctypes.c_int]
u.SetForegroundWindow.argtypes=[W.HWND];u.ShowWindow.argtypes=[W.HWND,ctypes.c_int]

def enable_dpi():
    try:u.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except Exception:
        try:ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:pass

def client_rect(hwnd):
    if not hwnd or not u.IsWindow(hwnd):return None
    rect=W.RECT();point=W.POINT(0,0)
    if not u.GetClientRect(hwnd,ctypes.byref(rect))or not u.ClientToScreen(hwnd,ctypes.byref(point)):return None
    if rect.right<100 or rect.bottom<100:return None
    return(point.x,point.y,rect.right,rect.bottom)

def title(hwnd):
    n=u.GetWindowTextLengthW(hwnd);buf=ctypes.create_unicode_buffer(n+1)
    u.GetWindowTextW(hwnd,buf,n+1);return buf.value

def windows():
    result=[]
    @ctypes.WINFUNCTYPE(W.BOOL,W.HWND,W.LPARAM)
    def callback(hwnd,_):
        name=title(hwnd);rect=client_rect(hwnd)
        if u.IsWindowVisible(hwnd)and name and rect and rect[2]>=400 and rect[3]>=300 and name not in('Program Manager','Windows 输入体验','NVIDIA GeForce Overlay'):
            result.append(dict(hwnd=int(hwnd),title=name,rect=rect))
        return True
    u.EnumWindows(callback,0);return result

def find_target(capture):
    hwnd=int(capture.get('target_hwnd')or 0)
    expected=capture.get('target_title','')
    if hwnd and client_rect(hwnd)and(not expected or title(hwnd)==expected):return hwnd
    entries=windows()
    for row in entries:
        if expected and row['title']==expected:return row['hwnd']
    for row in entries:
        if any(h.lower()in row['title'].lower()for h in('绝区零','ZenlessZoneZero','Zenless Zone Zero')):return row['hwnd']
    return 0

def foreground(hwnd):
    if not hwnd:return False
    current=u.GetForegroundWindow()
    return bool(current==hwnd or u.GetAncestor(current,2)==hwnd)

def activate(hwnd):
    if hwnd:
        u.ShowWindow(hwnd,9);return bool(u.SetForegroundWindow(hwnd))
    return False

class StopEvent:
    def __init__(self,name,create=True):
        k.CreateEventW.argtypes=[ctypes.c_void_p,W.BOOL,W.BOOL,W.LPCWSTR];k.CreateEventW.restype=W.HANDLE
        k.OpenEventW.argtypes=[W.DWORD,W.BOOL,W.LPCWSTR];k.OpenEventW.restype=W.HANDLE
        k.SetEvent.argtypes=[W.HANDLE];k.WaitForSingleObject.argtypes=[W.HANDLE,W.DWORD]
        k.CloseHandle.argtypes=[W.HANDLE]
        self.handle=k.CreateEventW(None,True,False,name)if create else k.OpenEventW(0x00100002,False,name)
        if not self.handle:raise ctypes.WinError()
    def set(self):k.SetEvent(self.handle)
    def is_set(self):return k.WaitForSingleObject(self.handle,0)==0
    def wait(self,seconds):return k.WaitForSingleObject(self.handle,int(max(0,seconds)*1000))==0
    def close(self):
        if self.handle:k.CloseHandle(self.handle);self.handle=None

class Job:
    """A background worker cannot outlive its owning desktop application."""
    def __init__(self):
        class BASIC(ctypes.Structure):
            _fields_=[('PerProcessUserTimeLimit',ctypes.c_longlong),('PerJobUserTimeLimit',ctypes.c_longlong),
                ('LimitFlags',W.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),('MaximumWorkingSetSize',ctypes.c_size_t),
                ('ActiveProcessLimit',W.DWORD),('Affinity',ctypes.c_size_t),('PriorityClass',W.DWORD),('SchedulingClass',W.DWORD)]
        class IO(ctypes.Structure):_fields_=[(n,ctypes.c_ulonglong)for n in('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
        class EXT(ctypes.Structure):_fields_=[('BasicLimitInformation',BASIC),('IoInfo',IO),('ProcessMemoryLimit',ctypes.c_size_t),('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),('PeakJobMemoryUsed',ctypes.c_size_t)]
        k.CreateJobObjectW.argtypes=[ctypes.c_void_p,W.LPCWSTR];k.CreateJobObjectW.restype=W.HANDLE
        k.SetInformationJobObject.argtypes=[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD]
        k.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE]
        self.handle=k.CreateJobObjectW(None,None)
        info=EXT();info.BasicLimitInformation.LimitFlags=0x2000
        if not self.handle or not k.SetInformationJobObject(self.handle,9,ctypes.byref(info),ctypes.sizeof(info)):raise ctypes.WinError()
    def assign(self,process):
        if not k.AssignProcessToJobObject(self.handle,W.HANDLE(int(process._handle))):raise ctypes.WinError()
    def close(self):
        if self.handle:k.CloseHandle(self.handle);self.handle=None

def driver_installed():
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'SYSTEM\CurrentControlSet\Services\ViGEmBus'):return True
    except FileNotFoundError:return False

class SingleInstance:
    def __init__(self):
        k.CloseHandle.argtypes=[W.HANDLE]
        k.CreateMutexW.argtypes=[ctypes.c_void_p,W.BOOL,W.LPCWSTR];k.CreateMutexW.restype=W.HANDLE
        self.handle=k.CreateMutexW(None,False,'Local\\SnakeAssistant25Desktop')
        self.already_running=k.GetLastError()==183
        if not self.handle:raise ctypes.WinError()
    def close(self):
        if self.handle:k.CloseHandle(self.handle);self.handle=None
