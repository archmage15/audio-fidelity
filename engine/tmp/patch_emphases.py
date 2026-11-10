# -*- coding: utf-8 -*-
"""自动 patch emphases 源码（3 处），让它在「无 penn/torbi」环境下能 import 推理。

背景（见 问题清单.md 问题 4）：
  emphases 依赖 penn→torbi，torbi 只编译到 pt211cpu；且 emphases 的
  __init__.py 默认 import baselines/train/annotate/evaluate/partition/plot
  等子模块，其中 baselines 依赖 penn。本项目只用 neural 主模型做推理，
  PITCH_FEATURE / PERIODICITY_FEATURE / LOUDNESS_FEATURE 默认全 False，
  penn 在推理时**根本不会被执行到** —— 所以只需把「会触发 penn 加载」的
  import 注释掉即可（顶层 import penn + __init__ 的子模块 import），
  **不需要**往函数内补惰性 import（feature 全 False，函数内 penn 永不调用）。

幂等：重复运行不会报错（发现已 patch 就跳过）。

用法：python patch_emphases.py
退出码 0 = 成功（或已 patch 过）；非 0 = 失败。
"""
import io
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def find_emphases_dir():
    """定位 emphases 的 site-packages 安装目录。"""
    import site
    for base in site.getsitepackages() + [site.getusersitepackages()]:
        d = os.path.join(base, "emphases")
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "__init__.py")):
            return d
    return None


def comment_top_import(path, module_name="penn"):
    """把文件顶层的 `import <module>`（行首、未注释）注释掉。返回是否改动。"""
    if not os.path.isfile(path):
        return False
    with io.open(path, encoding="utf-8") as f:
        src = f.read()
    if src.startswith("import %s" % module_name):
        line = "import %s\n" % module_name
        src = ("# " + line.rstrip("\n") +
               "  # 已由 patch_emphases.py 注释（推理不需要）\n" + src[len(line):])
        with io.open(path, "w", encoding="utf-8") as f:
            f.write(src)
        return True
    return False


def patch_init(d):
    """注释掉 __init__.py 里会拉进 penn/重依赖的子模块 import。"""
    p = os.path.join(d, "__init__.py")
    with io.open(p, encoding="utf-8") as f:
        src = f.read()

    submodules = ["baselines", "train", "annotate", "evaluate", "partition", "plot"]
    changed = False
    out_lines = []
    for line in src.splitlines():
        stripped = line.strip()
        hit = any(
            stripped.startswith("from . import %s" % m) or
            stripped.startswith("from .%s import" % m) or
            stripped.startswith("from . import %s," % m)
            for m in submodules
        )
        if hit and not stripped.startswith("#"):
            out_lines.append("# " + line + "  # 已由 patch_emphases.py 注释（推理不需要）")
            changed = True
        else:
            out_lines.append(line)
    if changed:
        with io.open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + "\n")
    return changed


def main():
    d = find_emphases_dir()
    if not d:
        print("[ERROR] 找不到 emphases 安装目录。请先：pip install emphases")
        return 1

    print("emphases 目录：%s" % d)

    c1 = patch_init(d)
    print("  __init__.py 子模块注释：%s" % ("已改" if c1 else "已 patch（跳过）"))

    c2 = comment_top_import(os.path.join(d, "data", "preprocess", "core.py"))
    print("  core.py 顶层 import penn 注释：%s" % ("已改" if c2 else "已 patch（跳过）"))

    c3 = comment_top_import(os.path.join(d, "data", "preprocess", "loudness.py"))
    print("  loudness.py 顶层 import penn 注释：%s" % ("已改" if c3 else "已 patch（跳过）"))

    try:
        import emphases
        print("  验证 import emphases：OK")
    except Exception as e:
        print("[ERROR] import emphases 仍失败：%s" % e)
        return 1

    print("patch 完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
