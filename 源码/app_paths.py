from pathlib import Path
import os,sys,json,tempfile,uuid

APP_NAME='蛇对蛇助手 2.5'
APP_VERSION='1.0.3'
BASE=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parent))
ENGINE=BASE/'engine'

_data_path=None
data_notice=''

def data_dir():
    global _data_path,data_notice
    if _data_path is not None:return _data_path
    override=os.environ.get('SNAKE25_DATA_DIR')
    p=Path(override)if override else Path(os.environ.get('LOCALAPPDATA',str(Path.home()))) / 'SnakeAssistant25'
    fallback=Path(tempfile.gettempdir())/'SnakeAssistant25'
    original=p
    for candidate in dict.fromkeys((p,fallback)):
        probe=candidate/('.write_check_'+uuid.uuid4().hex)
        try:
            candidate.mkdir(parents=True,exist_ok=True)
            with probe.open('xb')as stream:stream.write(b'1')
            probe.unlink();_data_path=candidate
            if candidate!=original:
                data_notice='默认配置目录无法写入，已使用备用目录：'+str(candidate)
                old=original/'profile.json';new=candidate/'profile.json'
                if old.is_file()and not new.exists():
                    try:new.write_bytes(old.read_bytes())
                    except OSError:pass
            return candidate
        except OSError:
            try:probe.unlink(missing_ok=True)
            except OSError:pass
    raise PermissionError('无法写入用户配置目录或临时目录，请检查目录访问权限。')

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)
    temporary=None
    try:
        temporary=path.with_name('.'+path.stem+'_'+uuid.uuid4().hex+'.tmp')
        with temporary.open('x',encoding='utf8')as stream:
            stream.write(payload);stream.flush()
        os.replace(str(temporary),str(path));temporary=None
    finally:
        if temporary:
            try:temporary.unlink()
            except OSError:pass

def read_json(path,default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):return default
