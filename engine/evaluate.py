"""Evaluate arbitrary audio paths after their reference has been cached."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'scripts'))
from path_jobs import evaluate


def main():
    from console_runtime import prevent_console_pause
    prevent_console_pause()
    parser = argparse.ArgumentParser(description='评测任意路径的两段音频；请先缓存原音频。')
    parser.add_argument('reference_path', nargs='?', help='原音频路径，也可使用 --reference')
    parser.add_argument('imitation_path', nargs='?', help='模仿音频路径，也可使用 --imitation')
    parser.add_argument('--reference', dest='reference_option', help='原音频路径')
    parser.add_argument('--imitation', dest='imitation_option', help='模仿音频路径')
    parser.add_argument('--output', help='本次报告的空目录；默认自动创建独立目录')
    parser.add_argument('--cache-dir', help='原音频缓存库；默认项目 tmp/reference_cache')
    args = parser.parse_args()
    if (args.reference_option and args.reference_path) or (args.imitation_option and args.imitation_path):
        parser.error('位置参数和同名选项不要重复提供。')
    reference = args.reference_option or args.reference_path
    imitation = args.imitation_option or args.imitation_path
    if not reference or not imitation:
        parser.error('必须提供原音频和模仿音频两个路径。')
    try:
        result = evaluate(reference, imitation, output_dir=args.output, cache_dir=args.cache_dir)
    except Exception as error:
        print(f'[ERROR] {error}', file=sys.stderr, flush=True)
        return 1
    print(f"\n综合还原度：{result['score']} / 100\n报告：{result['report']}", flush=True)
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    sys.exit(main())
