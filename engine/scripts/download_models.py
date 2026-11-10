"""Prepare portable model data; no Windows virtual environment is needed."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def valid_json_files(directory, names):
    try:
        for name in names:
            json.loads((directory / name).read_text(encoding='utf-8'))
        return True
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description='下载或复用音素/重音模型；可选下载标点模型。')
    parser.add_argument('--punctuation', action='store_true', help='同时准备可选标点模型')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download, hf_hub_download
    phoneme = Path(os.environ.get('PHONEME_MODEL_DIR') or ROOT / 'models/wav2vec2-espeak')
    ready = valid_json_files(phoneme, ['config.json','preprocessor_config.json','vocab.json'])
    weights = [phoneme / 'pytorch_model.bin', phoneme / 'model.safetensors']
    if not ready or not any(p.is_file() and p.stat().st_size > 0 for p in weights):
        print('下载音素模型（约 1.26 GB）：', phoneme, flush=True)
        snapshot_download(repo_id='facebook/wav2vec2-lv-60-espeak-cv-ft', local_dir=str(phoneme),
                          allow_patterns=['*.json','pytorch_model.bin','model.safetensors'])
    if not valid_json_files(phoneme, ['config.json','preprocessor_config.json','vocab.json']):
        raise RuntimeError('音素模型配置/词表不完整。')
    if not any(p.is_file() and p.stat().st_size > 0 for p in weights):
        raise RuntimeError('缺少音素模型权重。')
    stress = ROOT / 'models/emphases'
    if not (stress / 'model.pt').is_file() or (stress / 'model.pt').stat().st_size == 0:
        print('下载重音模型：', stress, flush=True)
        hf_hub_download(repo_id='maxrmorrison/emphases', filename='model.pt', local_dir=str(stress))
    if args.punctuation:
        punct = Path(os.environ.get('PUNCT_MODEL_DIR') or ROOT / 'models/punct-multilang')
        print('准备可选标点模型：', punct, flush=True)
        snapshot_download(repo_id='oliverguhr/fullstop-punctuation-multilingual-base',
                          local_dir=str(punct), allow_patterns=['*.json','*.model','*.safetensors','pytorch_model.bin'])
    print('模型文件已准备完成。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
