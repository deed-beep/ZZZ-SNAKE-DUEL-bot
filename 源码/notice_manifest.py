"""Generate the SHA-256 manifest that binds the declaration and engine files.

Replaces the old RSA signing step: no private key, no OpenSSL, no DPAPI. Anyone
who edits a protected file can regenerate the manifest, so the check only guards
against accidental or casual modification.
"""
import hashlib,json
from pathlib import Path

AUTHOR='101010'
NOTICE_NAME='使用声明与免责声明.txt'

def build_manifest(root):
    root=Path(root)
    resources={}
    paths=(list((root/'engine').glob('*.py'))+list((root/'engine'/'assets').glob('*'))
        +[root/'parameter_catalog.json',root/'app.ico'])
    for path in sorted(paths):
        if path.is_file():resources[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={'schema':1,'author':AUTHOR,'notice_file':NOTICE_NAME,
        'notice_sha256':hashlib.sha256((root/NOTICE_NAME).read_bytes()).hexdigest(),'resources':resources}
    destination=root/'app_assets/notice_manifest.json';destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,separators=(',',':')),encoding='utf8')
    return manifest

if __name__=='__main__':
    import sys
    target=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
    build_manifest(target)
    print('哈希清单已更新：',target/'app_assets'/'notice_manifest.json')
