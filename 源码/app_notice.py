"""Offline signed distribution check. Private signing material is never shipped."""
import ctypes,hashlib,json,struct,sys,threading
from pathlib import Path
from ctypes import wintypes as W
from app_paths import BASE
from app_notice_anchor import PUBLIC_MODULUS_HEX,PUBLIC_EXPONENT,MANIFEST_SHA256

NOTICE_NAME='使用声明与免责声明.txt'
AUTHOR='101010'

class NoticeIntegrityError(RuntimeError):pass

def signature_valid(payload,signature):
    """Windows CNG verifies RSA-4096 / PKCS#1 v1.5 / SHA-256."""
    native=ctypes.WinDLL('bcrypt.dll')
    handle=ctypes.c_void_p;ptr=ctypes.c_void_p;ulong=W.ULONG
    native.BCryptOpenAlgorithmProvider.argtypes=[ctypes.POINTER(handle),W.LPCWSTR,W.LPCWSTR,ulong]
    native.BCryptImportKeyPair.argtypes=[handle,handle,W.LPCWSTR,ctypes.POINTER(handle),ptr,ulong,ulong]
    native.BCryptVerifySignature.argtypes=[handle,ptr,ptr,ulong,ptr,ulong,ulong]
    native.BCryptDestroyKey.argtypes=[handle];native.BCryptCloseAlgorithmProvider.argtypes=[handle,ulong]
    for name in('BCryptOpenAlgorithmProvider','BCryptImportKeyPair','BCryptVerifySignature','BCryptDestroyKey','BCryptCloseAlgorithmProvider'):
        getattr(native,name).restype=ctypes.c_long
    modulus=bytes.fromhex(PUBLIC_MODULUS_HEX);exponent=PUBLIC_EXPONENT.to_bytes((PUBLIC_EXPONENT.bit_length()+7)//8,'big')
    if len(signature)!=len(modulus):return False
    blob=struct.pack('<6I',0x31415352,len(modulus)*8,len(exponent),len(modulus),0,0)+exponent+modulus
    key=handle();algorithm=handle()
    class Padding(ctypes.Structure):_fields_=[('algorithm',W.LPCWSTR)]
    try:
        if native.BCryptOpenAlgorithmProvider(ctypes.byref(algorithm),'RSA',None,0)!=0:return False
        raw=ctypes.create_string_buffer(blob)
        if native.BCryptImportKeyPair(algorithm,None,'RSAPUBLICBLOB',ctypes.byref(key),raw,len(blob),0)!=0:return False
        digest=hashlib.sha256(payload).digest();hashed=ctypes.create_string_buffer(digest);signed=ctypes.create_string_buffer(signature)
        padding=Padding('SHA256')
        return native.BCryptVerifySignature(key,ctypes.byref(padding),hashed,len(digest),signed,len(signature),2)==0
    finally:
        if key:native.BCryptDestroyKey(key)
        if algorithm:native.BCryptCloseAlgorithmProvider(algorithm,0)

def distribution_root():return Path(sys.executable).resolve().parent if getattr(sys,'frozen',False)else BASE

def verify_at(root,resources,check_assets=True):
    root=Path(root);resources=Path(resources)
    try:
        payload=(resources/'app_assets/notice_manifest.json').read_bytes()
        signed=(resources/'app_assets/notice_manifest.sig').read_bytes()
        if hashlib.sha256(payload).hexdigest()!=MANIFEST_SHA256 or not signature_valid(payload,signed):
            raise NoticeIntegrityError('声明签名无效或校验材料被修改。')
        manifest=json.loads(payload.decode('utf8'))
        if manifest.get('author')!=AUTHOR or manifest.get('notice_file')!=NOTICE_NAME:raise NoticeIntegrityError('作者或声明信息不一致。')
        notice=(root/NOTICE_NAME).read_bytes()
        if hashlib.sha256(notice).hexdigest()!=manifest['notice_sha256']:raise NoticeIntegrityError('使用声明与免责声明已被修改。')
        if check_assets:
            for name,expected in manifest['resources'].items():
                path=resources/name
                if not path.resolve().is_relative_to(resources.resolve()):raise NoticeIntegrityError('校验材料中的文件位置无效。')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise NoticeIntegrityError('受保护的程序资源已被修改：'+name)
        return notice.decode('utf-8-sig')
    except NoticeIntegrityError:raise
    except (OSError,ValueError,KeyError,TypeError)as exc:
        raise NoticeIntegrityError('使用声明或签名材料缺失、损坏或无法读取，请恢复完整原始软件包。')from exc

def verify_distribution(check_assets=True):return verify_at(distribution_root(),BASE,check_assets)

class NoticeWatchdog:
    """Read-only checks stay off the steering loop; only that loop releases input."""
    def __init__(self,interval=1.5):
        self.error=None;self.stop=threading.Event();self.interval=interval
        self.thread=threading.Thread(target=self._run,daemon=True,name='notice-integrity')
        self.thread.start()
    def _run(self):
        while not self.stop.wait(self.interval):
            try:verify_distribution(check_assets=False)
            except NoticeIntegrityError as exc:self.error=str(exc);return
    def check(self):
        if self.error:raise NoticeIntegrityError(self.error)
    def close(self):self.stop.set();self.thread.join(timeout=2.)
