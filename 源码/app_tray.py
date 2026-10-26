"""Native Windows notification icon and stop hotkey, no extra GUI runtime."""
import ctypes,os,threading
from ctypes import wintypes as W
from app_windows import u,k
from app_paths import APP_NAME,BASE

LRESULT=ctypes.c_ssize_t
WNDPROC=ctypes.WINFUNCTYPE(LRESULT,W.HWND,W.UINT,W.WPARAM,W.LPARAM)
class WNDCLASS(ctypes.Structure):
    _fields_=[('style',W.UINT),('lpfnWndProc',WNDPROC),('cbClsExtra',ctypes.c_int),('cbWndExtra',ctypes.c_int),
        ('hInstance',W.HINSTANCE),('hIcon',W.HANDLE),('hCursor',W.HANDLE),('hbrBackground',W.HANDLE),
        ('lpszMenuName',W.LPCWSTR),('lpszClassName',W.LPCWSTR)]
class GUID(ctypes.Structure):_fields_=[('a',W.DWORD),('b',W.WORD),('c',W.WORD),('d',ctypes.c_ubyte*8)]
class NOTIFY(ctypes.Structure):
    _fields_=[('cbSize',W.DWORD),('hWnd',W.HWND),('uID',W.UINT),('uFlags',W.UINT),
        ('uCallbackMessage',W.UINT),('hIcon',W.HANDLE),('szTip',W.WCHAR*128),('dwState',W.DWORD),
        ('dwStateMask',W.DWORD),('szInfo',W.WCHAR*256),('uVersion',W.UINT),('szInfoTitle',W.WCHAR*64),
        ('dwInfoFlags',W.DWORD),('guidItem',GUID),('hBalloonIcon',W.HANDLE)]

class Tray:
    def __init__(self,callback):
        self.callback=callback;self.hwnd=None;self.ready=threading.Event();self.failed=None
        self.tooltip=APP_NAME;self.thread=threading.Thread(target=self._run,daemon=True,name='tray-menu')
        self.thread.start();self.ready.wait(3.)
    def _run(self):
        try:
            k.GetModuleHandleW.argtypes=[W.LPCWSTR];k.GetModuleHandleW.restype=W.HINSTANCE
            u.DefWindowProcW.argtypes=[W.HWND,W.UINT,W.WPARAM,W.LPARAM];u.DefWindowProcW.restype=LRESULT
            u.CreateWindowExW.argtypes=[W.DWORD,W.LPCWSTR,W.LPCWSTR,W.DWORD,ctypes.c_int,ctypes.c_int,
                ctypes.c_int,ctypes.c_int,W.HWND,W.HMENU,W.HINSTANCE,ctypes.c_void_p]
            u.CreateWindowExW.restype=W.HWND
            u.RegisterClassW.argtypes=[ctypes.POINTER(WNDCLASS)];u.RegisterClassW.restype=W.ATOM
            u.LoadImageW.argtypes=[W.HINSTANCE,W.LPCWSTR,W.UINT,ctypes.c_int,ctypes.c_int,W.UINT];u.LoadImageW.restype=W.HANDLE
            u.LoadIconW.argtypes=[W.HINSTANCE,ctypes.c_void_p];u.LoadIconW.restype=W.HANDLE
            u.CreatePopupMenu.restype=W.HMENU
            u.AppendMenuW.argtypes=[W.HMENU,W.UINT,ctypes.c_size_t,W.LPCWSTR]
            u.TrackPopupMenu.argtypes=[W.HMENU,W.UINT,ctypes.c_int,ctypes.c_int,ctypes.c_int,W.HWND,ctypes.c_void_p]
            u.TrackPopupMenu.restype=W.UINT
            u.DestroyMenu.argtypes=[W.HMENU]
            u.PostMessageW.argtypes=[W.HWND,W.UINT,W.WPARAM,W.LPARAM]
            u.DestroyWindow.argtypes=[W.HWND];u.RegisterHotKey.argtypes=[W.HWND,ctypes.c_int,W.UINT,W.UINT]
            u.DestroyIcon.argtypes=[W.HANDLE];u.UnregisterHotKey.argtypes=[W.HWND,ctypes.c_int]
            u.RegisterWindowMessageW.argtypes=[W.LPCWSTR];u.RegisterWindowMessageW.restype=W.UINT
            u.GetMessageW.argtypes=[ctypes.POINTER(W.MSG),W.HWND,W.UINT,W.UINT]
            u.DispatchMessageW.argtypes=[ctypes.POINTER(W.MSG)];u.DispatchMessageW.restype=LRESULT
            shell=ctypes.windll.shell32;shell.Shell_NotifyIconW.argtypes=[W.DWORD,ctypes.POINTER(NOTIFY)]
            self.shell=shell;self._taskbar=u.RegisterWindowMessageW('TaskbarCreated')
            self._proc=WNDPROC(self._window_proc)
            instance=k.GetModuleHandleW(None);name='SnakeAssistant25Tray_%d'%os.getpid()
            wc=WNDCLASS();wc.lpfnWndProc=self._proc;wc.hInstance=instance;wc.lpszClassName=name
            if not u.RegisterClassW(ctypes.byref(wc)):raise ctypes.WinError()
            self.hwnd=u.CreateWindowExW(0,name,name,0,0,0,0,0,None,None,instance,None)
            if not self.hwnd:raise ctypes.WinError()
            icon=u.LoadImageW(None,str(BASE/'app.ico'),1,32,32,0x10)
            self._owned_icon=bool(icon)
            if not icon:icon=u.LoadIconW(None,ctypes.c_void_p(32512))
            self.icon=NOTIFY();self.icon.cbSize=ctypes.sizeof(NOTIFY);self.icon.hWnd=self.hwnd
            self.icon.uID=1;self.icon.uFlags=7;self.icon.uCallbackMessage=0x8001;self.icon.hIcon=icon
            self.icon.szTip=self.tooltip[:127]
            if not shell.Shell_NotifyIconW(0,ctypes.byref(self.icon)):raise ctypes.WinError()
            self.hotkey=bool(u.RegisterHotKey(self.hwnd,1,0x4003,0x77)) # Ctrl+Alt+F8, no autorepeat
            self.ready.set();message=W.MSG()
            while u.GetMessageW(ctypes.byref(message),None,0,0)>0:
                u.TranslateMessage(ctypes.byref(message));u.DispatchMessageW(ctypes.byref(message))
        except Exception as exc:self.failed=str(exc);self.ready.set()
    def _window_proc(self,hwnd,message,wparam,lparam):
        try:
            if message==0x8001:
                if lparam in(0x0202,0x0203):self.callback('show')
                elif lparam==0x0205:
                    menu=u.CreatePopupMenu()
                    for ident,text in((1,'显示主窗口'),(2,'停止游戏控制'),(3,'退出软件')):u.AppendMenuW(menu,0,ident,text)
                    point=W.POINT();u.GetCursorPos(ctypes.byref(point));u.SetForegroundWindow(hwnd)
                    choice=u.TrackPopupMenu(menu,0x102,point.x,point.y,0,hwnd,None)
                    u.DestroyMenu(menu);u.PostMessageW(hwnd,0,0,0)
                    if choice:self.callback({1:'show',2:'stop',3:'quit'}[choice])
                return 0
            if message==0x0312:self.callback('stop');return 0
            if message==0x8002:
                self.icon.uFlags=4;self.icon.szTip=self.tooltip[:127]
                self.shell.Shell_NotifyIconW(1,ctypes.byref(self.icon));return 0
            if message==getattr(self,'_taskbar',-1):
                self.icon.uFlags=7;self.shell.Shell_NotifyIconW(0,ctypes.byref(self.icon));return 0
            if message==0x0010:u.DestroyWindow(hwnd);return 0
            if message==0x0002:
                self.shell.Shell_NotifyIconW(2,ctypes.byref(self.icon));u.UnregisterHotKey(hwnd,1)
                if self._owned_icon:u.DestroyIcon(self.icon.hIcon)
                u.PostQuitMessage(0);return 0
        except Exception:pass
        return u.DefWindowProcW(hwnd,message,wparam,lparam)
    def set_status(self,text):
        self.tooltip=APP_NAME+' · '+text
        if self.hwnd:u.PostMessageW(self.hwnd,0x8002,0,0)
    def close(self):
        if self.hwnd:u.PostMessageW(self.hwnd,0x0010,0,0)
        self.thread.join(timeout=1.)
