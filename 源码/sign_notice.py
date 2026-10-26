"""Owner-only signing step. No private key is copied into the portable release."""
import base64,hashlib,json,os,subprocess,ctypes,struct
from pathlib import Path

def protect_private(data,decode=False):
    """Windows DPAPI binds the at-rest signing key to this Windows account."""
    from ctypes import wintypes as W
    class Blob(ctypes.Structure):_fields_=[('size',W.DWORD),('data',ctypes.c_void_p)]
    crypt=ctypes.WinDLL('crypt32.dll',use_last_error=True);kernel=ctypes.WinDLL('kernel32.dll')
    raw=ctypes.create_string_buffer(data);source=Blob(len(data),ctypes.cast(raw,ctypes.c_void_p));destination=Blob()
    function=crypt.CryptUnprotectData if decode else crypt.CryptProtectData
    function.argtypes=[ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,W.DWORD,ctypes.POINTER(Blob)]
    function.restype=W.BOOL;kernel.LocalFree.argtypes=[ctypes.c_void_p]
    if not function(ctypes.byref(source),None,None,None,None,1,ctypes.byref(destination)):raise ctypes.WinError(ctypes.get_last_error())
    try:return ctypes.string_at(destination.data,destination.size)
    finally:kernel.LocalFree(destination.data)

def tlv(data,offset=0):
    tag=data[offset];length=data[offset+1];start=offset+2
    if length&128:
        n=length&127;length=int.from_bytes(data[start:start+n],'big');start+=n
    return tag,data[start:start+length],start+length

def public_numbers(der):
    _,outer,_=tlv(der);_,modulus,next_at=tlv(outer);_,exponent,_=tlv(outer,next_at)
    return int.from_bytes(modulus,'big'),int.from_bytes(exponent,'big')

def sign_memory(pem,payload,openssl):
    """Import an in-memory RSA private blob into Windows CNG; never write PEM."""
    from ctypes import wintypes as W
    der=subprocess.run([openssl,'rsa','-traditional','-outform','DER'],input=pem,check=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL).stdout
    _,outer,_=tlv(der);values=[];at=0
    while at<len(outer):
        tag,value,at=tlv(outer,at)
        if tag!=2:raise ValueError('Unexpected private key format')
        values.append(int.from_bytes(value,'big'))
    if len(values)!=9:raise ValueError('Expected a two-prime RSA private key')
    _,n,e,_,p,q,*_=values
    width=(n.bit_length()+7)//8;pw=(p.bit_length()+7)//8;qw=(q.bit_length()+7)//8;ew=(e.bit_length()+7)//8
    blob=struct.pack('<6I',0x32415352,width*8,ew,width,pw,qw)+e.to_bytes(ew,'big')+n.to_bytes(width,'big')+p.to_bytes(pw,'big')+q.to_bytes(qw,'big')
    native=ctypes.WinDLL('bcrypt.dll');handle=ctypes.c_void_p;ptr=ctypes.c_void_p
    native.BCryptOpenAlgorithmProvider.argtypes=[ctypes.POINTER(handle),W.LPCWSTR,W.LPCWSTR,W.ULONG]
    native.BCryptImportKeyPair.argtypes=[handle,handle,W.LPCWSTR,ctypes.POINTER(handle),ptr,W.ULONG,W.ULONG]
    native.BCryptSignHash.argtypes=[handle,ptr,ptr,W.ULONG,ptr,W.ULONG,ctypes.POINTER(W.ULONG),W.ULONG]
    native.BCryptDestroyKey.argtypes=[handle];native.BCryptCloseAlgorithmProvider.argtypes=[handle,W.ULONG]
    for name in('BCryptOpenAlgorithmProvider','BCryptImportKeyPair','BCryptSignHash'):getattr(native,name).restype=ctypes.c_long
    class Padding(ctypes.Structure):_fields_=[('algorithm',W.LPCWSTR)]
    algorithm=handle();key=handle()
    try:
        if native.BCryptOpenAlgorithmProvider(ctypes.byref(algorithm),'RSA',None,0)!=0:raise RuntimeError('CNG RSA unavailable')
        raw=ctypes.create_string_buffer(blob)
        if native.BCryptImportKeyPair(algorithm,None,'RSAPRIVATEBLOB',ctypes.byref(key),raw,len(blob),0)!=0:raise RuntimeError('CNG private key import failed')
        digest=hashlib.sha256(payload).digest();hashed=ctypes.create_string_buffer(digest);padding=Padding('SHA256');size=W.ULONG()
        if native.BCryptSignHash(key,ctypes.byref(padding),hashed,len(digest),None,0,ctypes.byref(size),2)!=0:raise RuntimeError('CNG signature size failed')
        output=ctypes.create_string_buffer(size.value)
        if native.BCryptSignHash(key,ctypes.byref(padding),hashed,len(digest),output,size.value,ctypes.byref(size),2)!=0:raise RuntimeError('CNG signature failed')
        return output.raw[:size.value]
    finally:
        if key:native.BCryptDestroyKey(key)
        if algorithm:native.BCryptCloseAlgorithmProvider(algorithm,0)

def sign_release_assets(root):
    root=Path(root)
    keydir=Path(os.environ.get('SNAKE25_SIGNING_DIR',str(root.parent/'.snake25_signing')))
    keydir.mkdir(parents=True,exist_ok=True);private=keydir/'notice_private.dpapi';legacy=keydir/'notice_private.pem'
    openssl=os.environ.get('SNAKE25_OPENSSL',str(Path(__import__('sys').prefix)/'Library/bin/openssl.exe'))
    if not Path(openssl).is_file():raise RuntimeError('签名构建需要OpenSSL，请设置SNAKE25_OPENSSL。')
    if private.exists():material=protect_private(private.read_bytes(),True)
    else:
        if legacy.exists():material=legacy.read_bytes()
        else:
            if(root/'app_notice_anchor.py').exists():raise RuntimeError('未找到原作者私钥，拒绝自动更换签名身份。请设置SNAKE25_SIGNING_DIR。')
            material=subprocess.run([openssl,'genpkey','-algorithm','RSA','-pkeyopt','rsa_keygen_bits:4096'],check=True,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL).stdout
        encrypted=protect_private(material);assert protect_private(encrypted,True)==material
        private.write_bytes(encrypted)
        if legacy.exists()and keydir.resolve()==(root.parent/'.snake25_signing').resolve():legacy.unlink()
    public=keydir/'notice_public.der'
    subprocess.run([openssl,'rsa','-RSAPublicKey_out','-outform','DER','-out',str(public)],input=material,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    modulus,exponent=public_numbers(public.read_bytes())
    resources={}
    paths=list((root/'engine').glob('*.py'))+list((root/'engine/assets').glob('*'))+[root/'parameter_catalog.json',root/'app.ico']
    for path in sorted(paths):
        if path.is_file():resources[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest={'schema':1,'author':'101010','notice_file':'使用声明与免责声明.txt',
        'notice_sha256':hashlib.sha256((root/'使用声明与免责声明.txt').read_bytes()).hexdigest(),'resources':resources}
    destination=root/'app_assets/notice_manifest.json';destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,separators=(',',':')),encoding='utf8')
    (root/'app_assets/notice_manifest.sig').write_bytes(sign_memory(material,destination.read_bytes(),openssl))
    anchor=("# Generated public verification anchor; no private material.\n"
        "PUBLIC_MODULUS_HEX = %r\nPUBLIC_EXPONENT = %r\nMANIFEST_SHA256 = %r\n"%
        (format(modulus,'x'),exponent,hashlib.sha256(destination.read_bytes()).hexdigest()))
    (root/'app_notice_anchor.py').write_text(anchor,encoding='utf8')
    return keydir

if __name__=='__main__':
    sign_release_assets(Path(__file__).resolve().parent)
    print('Owner notice signed. Private key is outside the software project.')
