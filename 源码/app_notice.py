"""Offline distribution integrity check based on a plain SHA-256 manifest.

The manifest (app_assets/notice_manifest.json) binds the declaration text and
the protected engine resources to recorded hashes. There is no signature and no
private key: this detects accidental or casual modification, not a determined
attacker who edits the program as well.
"""
import hashlib,json,sys,threading
from pathlib import Path
from app_paths import BASE

NOTICE_NAME='使用声明与免责声明.txt'
AUTHOR='101010'

class NoticeIntegrityError(RuntimeError):pass

def distribution_root():return Path(sys.executable).resolve().parent if getattr(sys,'frozen',False)else BASE

def verify_at(root,resources,check_assets=True):
    root=Path(root);resources=Path(resources)
    try:
        manifest=json.loads((resources/'app_assets/notice_manifest.json').read_text(encoding='utf8'))
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
        raise NoticeIntegrityError('使用声明或校验清单缺失、损坏或无法读取，请恢复完整原始软件包。')from exc

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
