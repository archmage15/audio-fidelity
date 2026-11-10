"""通义听悟（Tingwu OpenAPI 2023-09-30）离线转写，只用标准库。

在引擎的 Python 进程里运行（经 sitecustomize 钩子加载），替换引擎里的讯飞转写：
  - transcribe_audio(src) 与引擎 xunfei_transcribe.transcribe_audio 同签名，
    返回 (整段文本, 句子列表, 任务号)；
  - cli_main(argv) 等价于 `python xunfei_transcribe.py <音频>`，
    写 <AUDIO_TEST_WORK_DIR>/转写文本.txt 与 转写结果.json。

听悟离线任务只接受音频 URL：音频以随机文件名放进 AFD_ASR_MEDIA_DIR，
经网站 GET /api/asr-media/<name> 临时提供（1 小时后失效），任务结束即删除。

环境变量（由网站 Worker 设置）：
  AFD_TINGWU_CONFIG   听悟配置.json 路径：{"access_key_id","access_key_secret","app_key"}
                      可选 "endpoint"（默认 tingwu.cn-beijing.aliyuncs.com）、"language"（默认 en）
  AFD_ASR_MEDIA_DIR   临时音频目录
  AFD_PUBLIC_ORIGIN   网站对外地址，如 https://audio-fidelity.example.com
  AFD_TINGWU_TIMEOUT  等待转写完成的最长秒数（默认 600）
"""
import datetime
import hashlib
import hmac
import importlib.machinery
import json
import os
import secrets
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

PROVIDER_NAME = '通义听悟'
API_VERSION = '2023-09-30'
DEFAULT_ENDPOINT = 'tingwu.cn-beijing.aliyuncs.com'
REQUIRED_FIELDS = ('access_key_id', 'access_key_secret', 'app_key')
MEDIA_EXTENSIONS = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus', '.aac', '.webm', '.mp4')
NO_SPACE_BEFORE = set(',.!?;:%)]}，。！？；：、）】》…')
NO_SPACE_AFTER = set('([{（【《')


class TingwuError(RuntimeError):
    pass


def _log(message):
    print('[%s] %s' % (PROVIDER_NAME, message), flush=True)


# ---- config ------------------------------------------------------------------

def load_config(path=None):
    path = path or os.environ.get('AFD_TINGWU_CONFIG', '')
    if not path or not os.path.isfile(path):
        raise TingwuError('找不到听悟配置文件：%s' % (path or '（未设置 AFD_TINGWU_CONFIG）'))
    try:
        with open(path, encoding='utf-8-sig') as stream:
            raw = json.load(stream)
    except (OSError, ValueError) as error:
        raise TingwuError('听悟配置文件无法读取：%s' % type(error).__name__) from None
    if not isinstance(raw, dict):
        raise TingwuError('听悟配置文件应为 JSON 对象')
    config = {key: str(raw.get(key) or '').strip() for key in REQUIRED_FIELDS}
    missing = [key for key in REQUIRED_FIELDS if not config[key]]
    if missing:
        raise TingwuError('听悟配置缺少字段：%s' % ', '.join(missing))
    config['endpoint'] = str(raw.get('endpoint') or DEFAULT_ENDPOINT).strip()
    config['language'] = str(raw.get('language') or 'en').strip()
    return config


# ---- signed OpenAPI calls (ACS3-HMAC-SHA256) --------------------------------

def _pct(value):
    return urllib.parse.quote(str(value), safe='-_.~')


def sign_headers(config, method, path, action, query, payload, *, host, content_type=None,
                 date=None, nonce=None, version=API_VERSION):
    """ACS3-HMAC-SHA256：返回含 Authorization 的完整请求头（纯函数，便于对照官方示例验证）。"""
    headers = {
        'host': host,
        'x-acs-action': action,
        'x-acs-version': version,
        'x-acs-date': date or datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'x-acs-signature-nonce': nonce or uuid.uuid4().hex,
        'x-acs-content-sha256': hashlib.sha256(payload).hexdigest(),
    }
    if content_type:
        headers['content-type'] = content_type
    canonical_uri = urllib.parse.quote(path, safe='/-_.~')
    canonical_query = '&'.join('%s=%s' % (_pct(k), _pct(v)) for k, v in sorted((query or {}).items()))
    signed_names = sorted(headers)
    canonical_headers = ''.join('%s:%s\n' % (name, headers[name].strip()) for name in signed_names)
    signed_headers = ';'.join(signed_names)
    canonical_request = '\n'.join([method, canonical_uri, canonical_query, canonical_headers,
                                   signed_headers, headers['x-acs-content-sha256']])
    string_to_sign = 'ACS3-HMAC-SHA256\n' + hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
    signature = hmac.new(config['access_key_secret'].encode('utf-8'), string_to_sign.encode('utf-8'),
                         hashlib.sha256).hexdigest()
    headers['Authorization'] = ('ACS3-HMAC-SHA256 Credential=%s,SignedHeaders=%s,Signature=%s'
                                % (config['access_key_id'], signed_headers, signature))
    return headers, canonical_uri, canonical_query


def _base_url(endpoint):
    # endpoint 可写成 host，或带协议的 http(s)://host:port（本地测试用）。
    if '://' in endpoint:
        parsed = urllib.parse.urlsplit(endpoint)
        return '%s://%s' % (parsed.scheme, parsed.netloc), parsed.netloc
    return 'https://' + endpoint, endpoint


def _call(config, method, path, action, query=None, body=None):
    payload = b'' if body is None else json.dumps(body, ensure_ascii=False).encode('utf-8')
    base, host = _base_url(config['endpoint'])
    headers, canonical_uri, canonical_query = sign_headers(
        config, method, path, action, query, payload, host=host,
        content_type='application/json; charset=utf-8' if body is not None else None)
    url = base + canonical_uri + ('?' + canonical_query if canonical_query else '')
    request = urllib.request.Request(url, data=payload if body is not None else None,
                                     method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as error:
        text = error.read().decode('utf-8', 'replace')
        try:
            detail = json.loads(text)
        except ValueError:
            detail = {'Message': text[:300]}
        raise TingwuError('%s 失败（HTTP %s，%s）：%s' % (action, error.code, detail.get('Code', ''),
                                                    detail.get('Message', ''))) from None
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise TingwuError('%s 请求失败：%s' % (action, error)) from None
    if str(data.get('Code', '0')) != '0':
        raise TingwuError('%s 失败（%s）：%s' % (action, data.get('Code'), data.get('Message')))
    return data


def create_task(config, file_url, task_key):
    body = {
        'AppKey': config['app_key'],
        'Input': {'SourceLanguage': config['language'], 'FileUrl': file_url, 'TaskKey': task_key},
        'Parameters': {'Transcription': {
            'DiarizationEnabled': False,
            # 保留原话：不做口语顺滑（会删掉重复/语气词），不过滤敏感词（会打星号）。
            'DisfluencyEnabled': False,
            'ProfanityFilterEnabled': False,
        }},
    }
    data = _call(config, 'PUT', '/openapi/tingwu/v2/tasks', 'CreateTask',
                 query={'type': 'offline'}, body=body)
    task_id = (data.get('Data') or {}).get('TaskId')
    if not task_id:
        raise TingwuError('CreateTask 没有返回 TaskId')
    return task_id


def get_task(config, task_id):
    data = _call(config, 'GET', '/openapi/tingwu/v2/tasks/%s' % task_id, 'GetTaskInfo')
    return data.get('Data') or {}


def wait_task(config, task_id, timeout):
    deadline = time.monotonic() + timeout
    delay = 2.0
    while True:
        info = get_task(config, task_id)
        status = info.get('TaskStatus')
        if status == 'COMPLETED':
            return info
        if status in ('FAILED', 'INVALID'):
            raise TingwuError('转写失败（%s %s）：%s' % (status, info.get('ErrorCode') or '',
                                                   info.get('ErrorMessage') or ''))
        if time.monotonic() + delay > deadline:
            raise TingwuError('等待转写超时（%d 秒），任务号 %s' % (timeout, task_id))
        time.sleep(delay)
        delay = min(delay * 1.5, 10.0)


def _download_json(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.loads(response.read().decode('utf-8'))
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise TingwuError('下载转写结果失败：%s' % error) from None


# ---- result parsing ------------------------------------------------------------

def _join_words(texts):
    if any(t != t.strip() for t in texts):
        return ' '.join(''.join(texts).split())
    out = ''
    for text in texts:
        if not text:
            continue
        if out and text[0] not in NO_SPACE_BEFORE and out[-1] not in NO_SPACE_AFTER \
                and not (_is_cjk(out[-1]) and _is_cjk(text[0])):
            out += ' '
        out += text
    return out


def _is_cjk(char):
    return '　' <= char <= '鿿' or '＀' <= char <= '￯'


def parse_transcription(document):
    """听悟 Transcription JSON → 句子列表（与引擎讯飞结果同结构）。"""
    body = document.get('Transcription', document) if isinstance(document, dict) else {}
    sentences, order = {}, []
    for paragraph in body.get('Paragraphs') or []:
        speaker = str(paragraph.get('SpeakerId', '0'))
        for word in paragraph.get('Words') or []:
            key = (paragraph.get('ParagraphId'), word.get('SentenceId'))
            if key not in sentences:
                sentences[key] = {'words': [], 'start': None, 'end': None, 'speaker': speaker}
                order.append(key)
            item = sentences[key]
            item['words'].append(str(word.get('Text') or ''))
            start, end = word.get('Start'), word.get('End')
            if start is not None:
                item['start'] = start if item['start'] is None else min(item['start'], start)
            if end is not None:
                item['end'] = end if item['end'] is None else max(item['end'], end)
    result = []
    for key in order:
        item = sentences[key]
        text = _join_words(item['words']).strip()
        if text:
            result.append({'文本': text, '开始ms': int(item['start'] or 0),
                           '结束ms': int(item['end'] or 0), '说话人': item['speaker']})
    return result


# ---- media publishing ----------------------------------------------------------

def publish_media(src):
    media_dir = os.environ.get('AFD_ASR_MEDIA_DIR', '')
    origin = os.environ.get('AFD_PUBLIC_ORIGIN', '').rstrip('/')
    if not media_dir or not origin:
        raise TingwuError('未设置 AFD_ASR_MEDIA_DIR / AFD_PUBLIC_ORIGIN，无法向听悟提供音频')
    extension = os.path.splitext(src)[1].lower()
    if extension not in MEDIA_EXTENSIONS:
        raise TingwuError('听悟不支持的音频格式：%s' % (extension or '无扩展名'))
    os.makedirs(media_dir, exist_ok=True)
    name = secrets.token_hex(16) + extension
    target = os.path.join(media_dir, name)
    # 复制而不是硬链接：网站按文件 mtime 判断链接是否过期，硬链接会继承原文件的旧 mtime。
    shutil.copyfile(src, target)
    return target,'%s/api/asr-media/%s' % (origin, name)


# ---- public API ----------------------------------------------------------------

def transcribe_audio(src):
    """与引擎 xunfei_transcribe.transcribe_audio 同签名：返回 (整段文本, 句子列表, 任务号)。"""
    if not src or not os.path.exists(src):
        raise RuntimeError('找不到音频文件' + ('：' + str(src) if src else ''))
    config = load_config()
    timeout = float(os.environ.get('AFD_TINGWU_TIMEOUT') or 600)
    media_path, url = publish_media(src)
    started = time.monotonic()
    try:
        task_id = create_task(config, url, 'afd-' + uuid.uuid4().hex[:16])
        _log('已提交转写任务 %s：%s' % (task_id, os.path.basename(src)))
        info = wait_task(config, task_id, timeout)
        result_url = (info.get('Result') or {}).get('Transcription')
        if not result_url:
            raise TingwuError('任务完成但没有转写结果地址，任务号 %s' % task_id)
        sentences = parse_transcription(_download_json(result_url))
    finally:
        try:
            os.remove(media_path)
        except OSError:
            pass
    full_text = ' '.join(s['文本'] for s in sentences).strip()
    if not full_text:
        raise RuntimeError('转写结果为空（音频里可能没有可识别的语音）')
    _log('转写完成，用时 %.1f 秒，%d 句' % (time.monotonic() - started, len(sentences)))
    return full_text, sentences, task_id


def cli_main(argv):
    """替代 `python scripts/xunfei_transcribe.py <音频>`；返回退出码。"""
    script = os.path.abspath(argv[0])
    engine_root = os.path.dirname(os.path.dirname(script))
    work_dir = os.environ.get('AUDIO_TEST_WORK_DIR') or os.path.join(engine_root, 'audio')
    args = [a for a in argv[1:] if not a.startswith('-')]
    if not args:
        _log('X 缺少音频路径参数（网站模式下必须显式传入）')
        return 1
    src = args[0]
    _log('使用通义听悟转写（替代讯飞）：%s' % os.path.basename(src))
    try:
        full_text, sentences, task_id = transcribe_audio(src)
    except Exception as error:  # noqa: BLE001 - 引擎只看退出码，原因写进日志
        _log('X 转写失败：%s' % error)
        return 1
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, '转写文本.txt'), 'w', encoding='utf-8') as stream:
        stream.write(full_text + '\n')
    with open(os.path.join(work_dir, '转写结果.json'), 'w', encoding='utf-8') as stream:
        json.dump({'文本': full_text, '句子': sentences, '任务号': task_id,
                   '来源音频': os.path.basename(src), '转写服务': PROVIDER_NAME},
                  stream, ensure_ascii=False, indent=2)
    _log('识别文本：%s' % (full_text[:90] + ('...' if len(full_text) > 90 else '')))
    return 0


# ---- engine integration --------------------------------------------------------

class _EngineModuleFinder:
    """import xunfei_transcribe 时，加载完原模块后按当时的 AFD_ASR_PROVIDER 改道。

    常驻推理进程里每个任务都会重新导入引擎模块，所以每次加载都重新判断：
    tingwu → 换成听悟；disabled → 直接报错（预热时不调用付费转写，评测会降级跳过）。
    """

    def find_spec(self, name, path=None, target=None):
        if name != 'xunfei_transcribe':
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        original = spec.loader.exec_module

        def exec_module(module):
            original(module)
            provider = os.environ.get('AFD_ASR_PROVIDER', '').strip().lower()
            if provider not in ('tingwu', 'disabled'):
                return
            if not hasattr(module, 'transcribe_audio'):
                raise ImportError('引擎 xunfei_transcribe 缺少 transcribe_audio，无法切换转写服务')
            module.transcribe_audio = transcribe_audio if provider == 'tingwu' else _transcription_disabled
            module.AFD_ASR_PROVIDER = provider

        spec.loader.exec_module = exec_module
        return spec


def _transcription_disabled(src):
    raise RuntimeError('本次运行不调用转写服务（预热）')


def install_import_hook():
    if not any(isinstance(f, _EngineModuleFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _EngineModuleFinder())


def allow_missing_xunfei_config(script_path):
    """analyze_reference.py 转写前会检查 讯飞配置.json 是否存在；听悟模式下不需要它。"""
    target = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(script_path))), '讯飞配置.json')
    original = os.path.exists

    def exists(path):
        try:
            if os.path.abspath(path) == target:
                return True
        except TypeError:
            pass
        return original(path)

    os.path.exists = exists
