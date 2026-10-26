"""Run in a Python 3.9 x64 environment with requirements installed."""
import json,subprocess,sys,os,ast,importlib.util,shutil
from pathlib import Path
from app_catalog import build_catalog
from app_paths import write_json

def main():
    root=Path(__file__).resolve().parent
    write_json(root/'parameter_catalog.json',build_catalog(root/'engine'))
    from sign_notice import sign_release_assets
    sign_release_assets(root)
    args=[sys.executable,'-m','PyInstaller','--noconfirm','--onedir','--windowed','--noupx',
          '--name','SnakeAssistant25','--distpath',str(root/'dist'),'--workpath',str(root/'build'),
          '--specpath',str(root),'--icon',str(root/'app.ico'),'--manifest',str(root/'app.manifest'),
          '--add-data',str(root/'engine')+';engine','--add-data',str(root/'parameter_catalog.json')+';.',
          '--add-data',str(root/'baseline.json')+';.','--add-data',str(root/'app.ico')+';.',
          '--add-data',str(root/'app_assets')+';app_assets',
          '--copy-metadata','mss','--copy-metadata','vgamepad',
          '--hidden-import','cv2','--hidden-import','numpy','--hidden-import','PIL.Image',
          '--hidden-import','PIL.ImageTk','--hidden-import','PIL.ImageGrab',
          '--hidden-import','mss.windows','--hidden-import','vgamepad',str(root/'app.py')]
    vgroot=Path(importlib.util.find_spec('vgamepad').origin).parent
    args[-1:-1]=['--add-binary',str(vgroot/'win'/'vigem'/'client'/'x64'/'ViGEmClient.dll')+';vgamepad/win/vigem/client/x64']
    # Conda keeps these beside Library/bin instead of Python/DLLs. Include
    # them explicitly so the program never depends on the build machine PATH.
    native=Path(sys.prefix)/'Library'/'bin'
    for name in('ffi.dll','tcl86t.dll','tk86t.dll'):
        if(native/name).is_file():args[-1:-1]=['--add-binary',str(native/name)+';.']
    local={f.stem for f in(root/'engine').glob('*.py')};dependencies=set()
    for path in(root/'engine').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            if isinstance(node,ast.Import):dependencies.update(a.name for a in node.names)
            if isinstance(node,ast.ImportFrom)and node.module:dependencies.add(node.module)
    for name in sorted(dependencies):
        if name.split('.')[0]not in local|{'__future__'}:args[-1:-1]=['--hidden-import',name]
    env=os.environ.copy()
    if native.is_dir():env['PATH']=str(native)+os.pathsep+env.get('PATH','')
    subprocess.run(args,cwd=str(root),env=env,check=True)
    shutil.copy2(root/'使用声明与免责声明.txt',root/'dist/SnakeAssistant25/使用声明与免责声明.txt')
    driver=next((p for p in(root/'driver_bundle',root.parent/'驱动安装包')if p.is_dir()),None)
    if driver:shutil.copytree(driver,root/'dist/SnakeAssistant25/驱动安装包',dirs_exist_ok=True)
    print('Built:',root/'dist'/'SnakeAssistant25'/'SnakeAssistant25.exe')

if __name__=='__main__':main()
