"""Copy a new portable release; never modifies the original 2.5 installation."""
import hashlib,json,shutil,sys
from pathlib import Path

def release(destination):
    root=Path(__file__).resolve().parent;destination=Path(destination)
    if destination.exists():raise FileExistsError('目标目录已存在，请使用新的交付目录：'+str(destination))
    report=json.loads((root/'final_packaged_report.json').read_text(encoding='utf8'))
    if not report.get('passed'):raise RuntimeError('最终独立程序验收未通过。')
    shutil.copytree(root/'dist'/'SnakeAssistant25',destination)
    for name in('使用说明.md','策略与参数说明.md','runtime_versions.json','使用声明与免责声明.txt','声明保护说明.md'):
        shutil.copy2(root/name,destination/name)
    shutil.copytree(root/'第三方许可',destination/'第三方许可')
    source=destination/'源码';source.mkdir()
    for path in root.iterdir():
        if path.suffix in('.py','.json','.manifest','.ico')and not path.name.endswith('_report.json'):
            if path.name not in('runtime_versions.json',):shutil.copy2(path,source/path.name)
    shutil.copy2(root/'requirements.txt',source/'requirements.txt')
    shutil.copy2(root/'使用声明与免责声明.txt',source/'使用声明与免责声明.txt')
    shutil.copytree(root/'engine',source/'engine',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for folder in('examples','app_assets'):
        if(root/folder).exists():shutil.copytree(root/folder,source/folder)
    checks=destination/'验收记录';checks.mkdir()
    shutil.copy2(root/'final_packaged_report.json',checks/'独立程序检查.json')
    shutil.copy2(root/'final_packaged_ui.png',checks/'主界面.png')
    shutil.copytree(root/'ui_checks',checks/'设置页面')
    for run in(root/'acceptance_data'/'records').glob('*'):
        target=checks/'实机自动流程'/run.name;target.mkdir(parents=True)
        for path in run.glob('*'):
            if path.is_file():shutil.copy2(path,target/path.name)
        (target/'logs').mkdir()
        for path in(run/'logs').glob('*'):
            if path.suffix in('.json','.jsonl')or path.name.startswith('result_'):shutil.copy2(path,target/'logs'/path.name)
    return destination

def manifest(destination):
    destination=Path(destination)
    hashes={str(p.relative_to(destination)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(destination.rglob('*'))if p.is_file()and p.name!='文件校验.json'}
    (destination/'文件校验.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf8')
    return hashes

if __name__=='__main__':
    result=release(sys.argv[1]);manifest(result);print(result)
