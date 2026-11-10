"""Check runtime requirements without printing service credentials."""

from cpu_runtime import configure_cpu_runtime
configure_cpu_runtime()
import argparse
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description='Linux 运行环境检查')
    parser.add_argument('--skip-models', action='store_true')
    parser.add_argument('--skip-config', action='store_true')
    args = parser.parse_args()
    failures = []
    if sys.version_info < (3, 12):
        failures.append('需要 Python 3.12 以上')
    for name in ['numpy','soundfile','filelock','torch','torchaudio','transformers','matplotlib','emphases']:
        try:
            module = importlib.import_module(name)
            print(f'[OK] {name} {getattr(module, "__version__", "")}', flush=True)
        except Exception as error:
            failures.append(f'{name}: {error}')
    from gen_ref_phonemes import find_espeak, raw_phonemize
    executable, version = find_espeak()
    if not executable:
        failures.append('找不到 espeak-ng；请安装系统软件包')
    else:
        code, ipa, error = raw_phonemize(executable, 'Hello world')
        if code or not ipa.strip():
            failures.append(f'espeak-ng 音素转换失败：{error}')
        else:
            print(f'[OK] {version}\n[OK] IPA {ipa.strip()}', flush=True)
    if not args.skip_models:
        import os
        from download_models import valid_json_files
        directory = Path(os.environ.get('PHONEME_MODEL_DIR') or ROOT / 'models/wav2vec2-espeak')
        weight_files = [directory / 'pytorch_model.bin', directory / 'model.safetensors']
        if (not valid_json_files(directory, ['config.json','preprocessor_config.json','vocab.json'])
                or not any(p.is_file() and p.stat().st_size for p in weight_files)):
            failures.append('音素模型缺失：请运行 bash linux.sh models')
        checkpoint = ROOT / 'models/emphases/model.pt'
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            failures.append('重音模型缺失：请运行 bash linux.sh models')
    if not args.skip_config:
        try:
            config = json.loads((ROOT / '讯飞配置.json').read_text(encoding='utf-8'))
            if not config.get('app_id') or not config.get('secret_key'):
                raise ValueError('app_id/secret_key 未填写')
            print('[OK] 讯飞配置字段齐全（未输出凭据）', flush=True)
        except (OSError, ValueError) as error:
            failures.append(f'讯飞配置：{error}')
    for failure in failures:
        print('[ERROR]', failure, file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
