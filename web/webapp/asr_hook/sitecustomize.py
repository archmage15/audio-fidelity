"""网站给引擎子进程注入的启动钩子（经 PYTHONPATH 加载，只在选用通义听悟时生效）。

引擎里有两处讯飞调用，都在这里改道到 afd_tingwu：
  1. analyze_reference.py 以子进程运行 scripts/xunfei_transcribe.py <音频>
     → 本进程直接用听悟完成转写、写出同名结果文件后退出，不执行讯飞脚本；
  2. compare_long.py 里 `import xunfei_transcribe` 后调用 transcribe_audio
     → 导入钩子在模块加载后替换该函数。
AFD_ASR_PROVIDER 不是 tingwu 时什么都不做。最后照常加载系统自带的 sitecustomize。
"""
import os
import sys


def _install():
    if os.environ.get('AFD_ASR_PROVIDER', '').strip().lower() != 'tingwu':
        return
    import afd_tingwu
    script = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ''
    if script == 'xunfei_transcribe.py':
        code = afd_tingwu.cli_main(sys.argv)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    afd_tingwu.install_import_hook()
    if script == 'analyze_reference.py':
        afd_tingwu.allow_missing_xunfei_config(sys.argv[0])


def _chain_system_sitecustomize():
    import importlib.machinery
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    paths = [p for p in sys.path if os.path.abspath(p or '.') != here]
    spec = importlib.machinery.PathFinder.find_spec('sitecustomize', paths)
    if spec and spec.loader and spec.origin and os.path.abspath(spec.origin) != os.path.abspath(__file__):
        spec.loader.exec_module(importlib.util.module_from_spec(spec))


try:
    _install()
except Exception as error:  # noqa: BLE001 - 不能让钩子悄悄退回讯飞
    print('[通义听悟] 钩子初始化失败：%s: %s' % (type(error).__name__, error), file=sys.stderr, flush=True)
    if os.path.basename(sys.argv[0] if sys.argv else '') == 'xunfei_transcribe.py':
        os._exit(1)
try:
    _chain_system_sitecustomize()
except Exception:  # noqa: BLE001
    pass
