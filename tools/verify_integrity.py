#!/usr/bin/env python3
"""校验仓库里受哈希保护的文件，确认检出的字节与清单完全一致。

用法：
    python tools/verify_integrity.py

检查两项内容：
    1. 源码/baseline.json 中登记的引擎模块 SHA-256；
    2. 源码/app_assets/notice_manifest.json 中登记的资源与声明正文 SHA-256。

为什么要单独检查：.gitattributes 一旦对上述文件做了换行转换（CRLF ↔ LF），
哈希就会变化，程序启动时会判定“引擎或声明被修改”并拒绝运行。这个脚本在
本地和 CI 上都能在提交/合并前发现问题，不需要安装 cv2、vgamepad 等依赖。
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "源码"

# CI 上的 Python 3.9 输出管道默认用系统代码页，中文会编码失败
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load(path):
    # 清单由 Windows 侧脚本生成，允许带 UTF-8 BOM
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compare(errors, relative, expected, checked):
    path = SRC / relative
    checked.append(relative)
    if not path.is_file():
        errors.append(f"缺少文件：{relative}")
        return
    actual = digest(path)
    if actual.lower() != str(expected).lower():
        errors.append(f"哈希不符：{relative}\n  期望 {expected}\n  实际 {actual}")


def main():
    errors = []
    checked = []

    baseline = load(SRC / "baseline.json")
    for relative, expected in sorted(baseline["sha256"].items()):
        compare(errors, "engine/" + relative, expected, checked)

    manifest = load(SRC / "app_assets" / "notice_manifest.json")
    for relative, expected in sorted(manifest["resources"].items()):
        compare(errors, relative, expected, checked)
    compare(errors, manifest["notice_file"], manifest["notice_sha256"], checked)

    # 参数目录是程序运行的必需文件，顺便确认它是合法 JSON
    try:
        catalog = load(SRC / "parameter_catalog.json")
        if not catalog.get("parameters"):
            errors.append("parameter_catalog.json 中没有 parameters 字段")
    except (OSError, ValueError) as exc:
        errors.append(f"parameter_catalog.json 无法解析：{exc}")

    if errors:
        print("完整性校验失败：\n")
        for item in errors:
            print("  - " + item)
        print("\n提示：检查 .gitattributes 是否对受保护文件启用了换行转换。")
        return 1

    print(f"完整性校验通过：{len(checked)} 个受保护文件全部匹配。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
