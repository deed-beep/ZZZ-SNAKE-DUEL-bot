from pathlib import Path
import cv2,numpy as np

def imread(path,flags=cv2.IMREAD_COLOR):
    try:return cv2.imdecode(np.fromfile(str(path),np.uint8),flags)
    except (OSError,ValueError):return None

def imwrite(path,image,params=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    ok,data=cv2.imencode(p.suffix or'.png',image,params or[])
    if ok:data.tofile(str(p))
    return bool(ok)
