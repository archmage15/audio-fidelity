"""Cache one reference audio file, or all supported files in a folder."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'scripts'))
from path_jobs import ROOT, FORMATS, prepare_reference, run_id, write_json


def main():
    from console_runtime import prevent_console_pause
    prevent_console_pause()
    parser = argparse.ArgumentParser(description='一键缓存某个文件夹的原音频，也支持单个音频文件。')
    parser.add_argument('path', help='音频文件夹路径或单个音频路径')
    parser.add_argument('--recursive', action='store_true', help='同时处理子文件夹；默认仅当前文件夹')
    parser.add_argument('--confirm-text', action='store_true', help='逐个人工核对转写；默认自动采用转写')
    parser.add_argument('--cache-dir', help='缓存库位置；评测时使用相同 --cache-dir')
    args = parser.parse_args()
    source = Path(args.path).expanduser().resolve()
    if source.is_dir():
        candidates = source.rglob('*') if args.recursive else source.iterdir()
        files = sorted((p for p in candidates if p.is_file() and p.suffix.lower() in FORMATS), key=str)
    elif source.is_file():
        files = [source]
    else:
        parser.error(f'路径不存在：{source}')
    if not files:
        parser.error('未找到支持的音频：wav / mp3 / flac / ogg；m4a 请先转换。')
    batch = ROOT / 'output/cache_batches' / run_id()
    batch.mkdir(parents=True)
    records = []
    for index, audio in enumerate(files, 1):
        print(f'\n[{index}/{len(files)}] 缓存原音频：{audio}', flush=True)
        try:
            item = prepare_reference(audio, cache_dir=args.cache_dir,
                                     confirm_text=args.confirm_text,
                                     logs_dir=batch / f'{index:04d}')
            print(f"  {item['status']} → {item['cache']}", flush=True)
        except Exception as error:
            item = {'reference': str(audio), 'status': 'failed', 'error': str(error)}
            print(f'  [ERROR] {error}', file=sys.stderr, flush=True)
        records.append(item)
        write_json(batch / '缓存清单.json', records)
    failures = sum(item['status'] == 'failed' for item in records)
    print(f'\n缓存完成：成功/可复用 {len(records)-failures}，失败 {failures}。\n清单：{batch / "缓存清单.json"}', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    sys.exit(main())
