'use strict';

/* ---------- helpers ---------- */
const $ = (selector) => document.querySelector(selector);
const STORE_JOB = 'afd.currentJob';
const STORE_REF = 'afd.reference';
const PART_LABELS = { pronunciation: '发音', sound_changes: '音变', rhythm: '节奏', stress: '重音位置' };
const STATUS_TEXT = {
  pending_enqueue: '等待入队', queued: '排队中', running: '处理中',
  succeeded: '已完成', failed: '失败', timed_out: '超时',
};
const STAGE_ORDER = ['queued', 'normalizing', 'evaluating', 'saving'];
const RING = 326.7;

const state = {
  user: null, refs: [], refId: null, mode: 'upload', imitation: null, idemKey: null,
  job: null, jobFetchedAt: 0, pollTimer: null, tickTimer: null, historyBefore: null, pendingTimer: null,
};

class ApiError extends Error {
  constructor(status, code, message, extra) { super(message); this.status = status; this.code = code; this.extra = extra || {}; }
}

function storeGet(key) { try { return localStorage.getItem(key); } catch { return null; } }
function storeSet(key, value) { try { value == null ? localStorage.removeItem(key) : localStorage.setItem(key, value); } catch { /* ignore */ } }

function uuid() {
  if (window.crypto?.randomUUID) return crypto.randomUUID();
  const bytes = new Uint8Array(16);
  (window.crypto || {}).getRandomValues ? crypto.getRandomValues(bytes) : bytes.forEach((_, i) => { bytes[i] = Math.random() * 256; });
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
}

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat()) if (child != null) node.append(child);
  return node;
}

function fmtSeconds(value, digits = 1) {
  if (value == null || Number.isNaN(value)) return '—';
  if (value < 60) return `${value.toFixed(digits)} 秒`;
  const m = Math.floor(value / 60);
  return `${m} 分 ${Math.round(value - m * 60)} 秒`;
}
function fmtClock(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}
function fmtTime(iso) {
  if (!iso) return '';
  return new Date(iso).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}
function withParam(url, key, value) { return url + (url.includes('?') ? '&' : '?') + `${key}=${encodeURIComponent(value)}`; }

let toastTimer;
function toast(message, isError = false) {
  const node = $('#toast');
  node.textContent = message;
  node.className = 'toast' + (isError ? ' error' : '');
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { node.hidden = true; }, isError ? 6000 : 3000);
}

async function api(path, options = {}) {
  const headers = { 'X-Requested-With': 'fetch', ...(options.headers || {}) };
  let body = options.body;
  if (body && !(body instanceof FormData)) { headers['Content-Type'] = 'application/json'; body = JSON.stringify(body); }
  let response;
  try {
    response = await fetch(path, { method: options.method || 'GET', headers, body, credentials: 'same-origin' });
  } catch {
    throw new ApiError(0, 'network', '网络连接失败，请检查网络后重试。');
  }
  const isJson = (response.headers.get('content-type') || '').includes('json');
  const data = isJson ? await response.json().catch(() => ({})) : await response.text();
  if (!response.ok) {
    const err = (isJson && data.error) || {};
    if (response.status === 401 && !options.allow401) showLogin();
    throw new ApiError(response.status, err.code || `http_${response.status}`, err.message || `请求失败（${response.status}）`, err);
  }
  return data;
}

function uploadFile(file, { kind = 'upload', purpose = 'imitation', filename } = {}) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append('file', file, filename || file.name || 'audio');
    form.append('kind', kind);
    form.append('purpose', purpose);
    const xhr = new XMLHttpRequest();
    const bar = $('#upload-progress');
    xhr.open('POST', '/api/assets');
    xhr.setRequestHeader('X-Requested-With', 'fetch');
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      bar.hidden = false;
      bar.firstElementChild.style.width = `${(event.loaded / event.total) * 100}%`;
    };
    xhr.onloadend = () => { bar.hidden = true; bar.firstElementChild.style.width = '0'; };
    xhr.onerror = () => reject(new ApiError(0, 'network', '上传失败，请检查网络后重试。'));
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText || '{}'); } catch { /* non-JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) return resolve(data);
      if (xhr.status === 401) showLogin();
      const err = data.error || {};
      reject(new ApiError(xhr.status, err.code || `http_${xhr.status}`, err.message || `上传失败（${xhr.status}）`));
    };
    xhr.send(form);
  });
}

/* ---------- session ---------- */
function showLogin() {
  stopPolling();
  $('#app-view').hidden = true;
  $('#login-view').hidden = false;
  $('#invite-code').focus();
}

async function showApp(user) {
  state.user = user;
  $('#login-view').hidden = true;
  $('#app-view').hidden = false;
  $('#user-name').textContent = user.name;
  $('#admin-badge').hidden = !user.is_admin;
  $('#admin-ref').hidden = !user.is_admin;
  const lim = user.limits;
  $('#upload-limits').textContent = `WAV / MP3 / FLAC / OGG / M4A / WebM，≤ ${Math.round(lim.max_upload_bytes / 1048576)} MiB，${lim.min_audio_seconds}–${lim.max_audio_seconds} 秒。推荐 16 kHz 单声道 WAV。`;
  $('#rec-max').textContent = ` / ${fmtClock(lim.max_audio_seconds)}`;
  $('#invites').hidden = !user.is_admin;
  $('#asr-settings').hidden = !user.is_admin;
  await loadRefs();
  if (user.is_admin) { loadPendingRefs(); loadUsers(); loadAsr(); }
  loadHistory(true);
  const saved = storeGet(STORE_JOB);
  if (saved) openJob(saved, { quiet: true });
}

$('#login-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const error = $('#login-error');
  error.hidden = true;
  try {
    const data = await api('/api/session', { method: 'POST', body: { invite_code: $('#invite-code').value.trim() }, allow401: true });
    $('#invite-code').value = '';
    showApp(data.user);
  } catch (err) {
    error.textContent = err.message;
    error.hidden = false;
  }
});

$('#logout').addEventListener('click', async () => {
  try { await api('/api/session', { method: 'DELETE' }); } catch { /* ignore */ }
  storeSet(STORE_JOB, null);
  location.reload();
});

/* ---------- references ---------- */
const player = $('#ref-player');
let playingRefId = null;

async function loadRefs() {
  const data = await api('/api/references');
  state.refs = data.items;
  $('#ref-empty').hidden = state.refs.length > 0;
  $('#ref-count').textContent = state.refs.length ? `共 ${state.refs.length} 段` : '';
  $('#ref-filter-wrap').hidden = state.refs.length <= 6;
  const wanted = state.refId || storeGet(STORE_REF);
  const keep = state.refs.find((r) => r.reference_id === wanted) || (state.refs.length === 1 ? state.refs[0] : null);
  state.refId = keep?.reference_id || null;
  renderRefs();
  renderTarget();
  updateSubmit();
}

function refDuration(ref) {
  return ref.duration_seconds ? `${ref.duration_seconds.toFixed(1)} 秒` : '';
}

function renderRefs() {
  const query = ($('#ref-filter').value || '').trim().toLowerCase();
  const visible = state.refs.filter((r) => !query
    || r.display_name.toLowerCase().includes(query) || (r.transcript || '').toLowerCase().includes(query));
  $('#ref-list').replaceChildren(...visible.map((ref) => {
    const selected = ref.reference_id === state.refId;
    const play = el('button', { type: 'button', class: 'play-btn', 'aria-label': `播放 ${ref.display_name}`,
      onclick: (event) => { event.stopPropagation(); playRef(ref); } }, el('span', { class: 'icon-play' }));
    const goImitate = el('button', { type: 'button', class: 'btn btn-primary btn-sm ref-check', text: '去模仿 ↓',
      onclick: (event) => { event.stopPropagation(); $('#step-imit').scrollIntoView({ behavior: 'smooth', block: 'start' }); } });
    return el('li', {
      class: 'ref-item', role: 'radio', tabindex: '0', 'aria-checked': String(selected), 'data-ref': ref.reference_id,
      onclick: () => selectRef(ref.reference_id),
      onkeydown: (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectRef(ref.reference_id); } },
    }, play,
    el('div', { class: 'ref-main' },
      el('div', { class: 'ref-title' }, el('span', { text: ref.display_name }), el('span', { class: 'muted', text: refDuration(ref) })),
      el('p', { class: ref.transcript ? 'ref-text' : 'ref-text none', text: ref.transcript || '暂无原文' }),
      el('div', { class: 'progress-line' }, el('span'))),
    el('div', { class: 'ref-side' }, el('span', { class: 'ref-pick', text: '选择' }), goImitate));
  }));
  if (query && !visible.length) $('#ref-list').append(el('li', { class: 'empty', text: '没有匹配的原音频。' }));
  updatePlayUi();
}
$('#ref-filter').addEventListener('input', renderRefs);

function selectRef(refId) {
  if (refId !== state.refId) state.idemKey = null;
  state.refId = refId;
  storeSet(STORE_REF, refId);
  document.querySelectorAll('#ref-list .ref-item').forEach((li) => li.setAttribute('aria-checked', String(li.dataset.ref === refId)));
  renderTarget();
  updateSubmit();
  if (state.autoSubmitPending && state.imitation) submitEvaluation({ auto: true });
}

function renderTarget() {
  const ref = state.refs.find((r) => r.reference_id === state.refId);
  $('#target').hidden = !ref;
  $('#target-none').hidden = !!ref;
  if (!ref) return;
  $('#target-name').textContent = ref.display_name;
  $('#target-dur').textContent = refDuration(ref);
  $('#target-text').textContent = ref.transcript || '暂无原文';
  updatePlayUi();
}
$('#target-play').addEventListener('click', () => {
  const ref = state.refs.find((r) => r.reference_id === state.refId);
  if (ref) playRef(ref);
});

function playRef(ref) {
  if (playingRefId === ref.reference_id && !player.paused) { player.pause(); return; }
  if (playingRefId !== ref.reference_id) {
    player.src = ref.audio_url;
    playingRefId = ref.reference_id;
  }
  player.play().catch((err) => { if (err.name !== 'AbortError') toast(`无法播放：${err.message}`, true); });
}

function updatePlayUi() {
  const active = !player.paused && !player.ended;
  const ratio = player.duration ? Math.min(1, player.currentTime / player.duration) : 0;
  const paint = (button, line, id) => {
    const isCurrent = id === playingRefId;
    button?.classList.toggle('playing', isCurrent && active);
    button?.setAttribute('aria-label', `${isCurrent && active ? '暂停' : '播放'}原音频`);
    if (line) {
      line.classList.toggle('on', isCurrent && (active || player.currentTime > 0));
      line.firstElementChild.style.width = isCurrent ? `${ratio * 100}%` : '0';
    }
  };
  document.querySelectorAll('#ref-list .ref-item').forEach((li) =>
    paint(li.querySelector('.play-btn'), li.querySelector('.progress-line'), li.dataset.ref));
  paint($('#target-play'), $('#target-progress').parentElement, state.refId);
}
['play', 'pause', 'ended', 'timeupdate', 'emptied'].forEach((type) => player.addEventListener(type, updatePlayUi));
player.addEventListener('ended', () => { player.currentTime = 0; updatePlayUi(); });
// Only one thing plays at a time: any other <audio> starting pauses the list player and vice versa.
document.addEventListener('play', (event) => {
  document.querySelectorAll('audio').forEach((audio) => { if (audio !== event.target) audio.pause(); });
}, true);

async function loadPendingRefs() {
  clearTimeout(state.pendingTimer);
  try {
    const data = await api('/api/references?include_pending=1');
    const pending = data.items.filter((r) => r.status !== 'ready');
    const list = $('#pending-refs');
    list.replaceChildren(...pending.map((r) => el('li', {},
      el('span', { text: r.display_name }),
      el('span', { class: r.status === 'failed' ? 'small' : 'muted small', text: r.status === 'preparing' ? '缓存准备中…' : `失败：${r.error || ''}` }),
      r.status === 'failed' && r.job_id ? el('button', { class: 'btn btn-ghost btn-sm', type: 'button', text: '重试', onclick: () => retryJob(r.job_id).then(loadPendingRefs) }) : null)));
    const known = new Set(state.refs.map((r) => r.reference_id));
    if (data.items.some((r) => r.status === 'ready' && !known.has(r.reference_id))) await loadRefs();
    if (pending.some((r) => r.status === 'preparing')) state.pendingTimer = setTimeout(loadPendingRefs, 4000);
  } catch (err) { if (err.status !== 401) console.warn(err); }
}

$('#ref-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = $('#ref-file').files[0];
  if (!file) return;
  const button = event.submitter || event.target.querySelector('button');
  button.disabled = true;
  try {
    const asset = await uploadFile(file, { purpose: 'reference' });
    await api('/api/references', {
      method: 'POST', headers: { 'Idempotency-Key': uuid() },
      body: { asset_id: asset.asset_id, display_name: $('#ref-name').value.trim(), is_public: $('#ref-public').checked },
    });
    event.target.reset();
    toast('已提交，原音频缓存准备中。');
    loadPendingRefs();
  } catch (err) {
    toast(err.message, true);
  } finally {
    button.disabled = false;
  }
});

/* ---------- imitation: upload ---------- */
function setMode(mode) {
  state.mode = mode;
  for (const tab of document.querySelectorAll('.segmented button')) tab.setAttribute('aria-selected', String(tab.dataset.mode === mode));
  $('#pane-upload').hidden = mode !== 'upload' || !!state.imitation;
  $('#pane-record').hidden = mode !== 'record' || !!state.imitation;
}
for (const tab of document.querySelectorAll('.segmented button')) tab.addEventListener('click', () => setMode(tab.dataset.mode));

function setImitation(asset) {
  state.imitation = asset;
  state.idemKey = null;
  state.autoSubmitPending = false;
  $('#imit-ready').hidden = !asset;
  if (asset) {
    $('#imit-name').textContent = asset.kind === 'recording' ? '我的录音' : asset.display_name;
    $('#imit-info').textContent = [asset.duration_seconds ? `${asset.duration_seconds.toFixed(1)} 秒` : null,
      `${(asset.size_bytes / 1024).toFixed(0)} KB`, '已上传'].filter(Boolean).join(' · ');
    $('#imit-audio').src = asset.audio_url;
  } else {
    $('#imit-audio').removeAttribute('src');
  }
  setMode(state.mode);
  updateSubmit();
}

$('#imit-file').addEventListener('change', async (event) => {
  const file = event.target.files[0];
  event.target.value = '';
  if (!file) return;
  if (file.size > state.user.limits.max_upload_bytes) { toast('文件超过大小上限。', true); return; }
  try {
    setImitation(await uploadFile(file));
    afterImitationUpload('上传成功');
  } catch (err) { toast(err.message, true); }
});
const dropzone = $('#dropzone');
['dragenter', 'dragover'].forEach((type) => dropzone.addEventListener(type, () => dropzone.classList.add('drag')));
['dragleave', 'drop'].forEach((type) => dropzone.addEventListener(type, () => dropzone.classList.remove('drag')));

$('#imit-remove').addEventListener('click', async () => {
  const asset = state.imitation;
  if (!asset) return;
  try { await api(`/api/assets/${asset.asset_id}`, { method: 'DELETE' }); }
  catch (err) { if (err.code !== 'asset_in_use') { toast(err.message, true); return; } }
  setImitation(null);
});

/* ---------- imitation: recording ---------- */
const rec = { recorder: null, stream: null, chunks: [], blob: null, url: null, timer: null, started: 0, failed: false, audioCtx: null, raf: null };

function stopStream() {
  rec.stream?.getTracks().forEach((track) => track.stop());
  rec.stream = null;
  cancelAnimationFrame(rec.raf);
  rec.audioCtx?.close().catch(() => {});
  rec.audioCtx = null;
  $('#rec-level').style.width = '0';
}

function recordingUi(active) {
  const button = $('#rec-btn');
  button.classList.toggle('recording', active);
  button.setAttribute('aria-label', active ? '停止录音' : '开始录音');
  $('#rec-hint').textContent = active ? '录音中… 再次点击停止' : '点击红色按钮开始录音';
}

function micError(err) {
  const name = err?.name || '';
  if (name === 'NotAllowedError' || name === 'SecurityError') return '麦克风权限被拒绝，可在浏览器设置中允许，或改用“上传文件”。';
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return '没有检测到麦克风，请改用“上传文件”。';
  if (name === 'NotReadableError') return '麦克风被其他程序占用，请关闭后重试。';
  return err?.message || '无法开始录音，请改用“上传文件”。';
}

function startMeter(stream) {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    rec.audioCtx = new Ctx();
    const analyser = rec.audioCtx.createAnalyser();
    analyser.fftSize = 512;
    rec.audioCtx.createMediaStreamSource(stream).connect(analyser);
    const data = new Uint8Array(analyser.fftSize);
    const loop = () => {
      analyser.getByteTimeDomainData(data);
      let peak = 0;
      for (const v of data) peak = Math.max(peak, Math.abs(v - 128));
      $('#rec-level').style.width = `${Math.min(100, (peak / 128) * 160)}%`;
      $('#rec-time').textContent = fmtClock((performance.now() - rec.started) / 1000);
      rec.raf = requestAnimationFrame(loop);
    };
    loop();
  } catch { /* meter is optional */ }
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    throw new Error(window.isSecureContext ? '当前浏览器不支持录音，请上传音频文件。' : '需要通过 HTTPS 访问才能录音。');
  }
  document.querySelectorAll('audio').forEach((audio) => audio.pause());
  rec.stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: { ideal: 1 }, echoCancellation: { ideal: false }, noiseSuppression: { ideal: false }, autoGainControl: { ideal: false } },
  });
  try {
    const type = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4'].find((t) => MediaRecorder.isTypeSupported(t));
    const recorder = new MediaRecorder(rec.stream, type ? { mimeType: type } : {});
    rec.recorder = recorder;
    rec.chunks = []; rec.blob = null; rec.failed = false;
    recorder.ondataavailable = (event) => { if (event.data.size) rec.chunks.push(event.data); };
    recorder.onstop = () => {
      clearTimeout(rec.timer);
      stopStream();
      recordingUi(false);
      if (rec.failed) return;
      const mime = recorder.mimeType || rec.chunks[0]?.type || 'application/octet-stream';
      rec.blob = new Blob(rec.chunks, { type: mime });
      if (rec.url) URL.revokeObjectURL(rec.url);
      rec.url = URL.createObjectURL(rec.blob);
      $('#rec-audio').src = rec.url;
      $('#rec-preview').hidden = false;
      const seconds = (performance.now() - rec.started) / 1000;
      if (seconds < state.user.limits.min_audio_seconds) toast(`录音太短（${seconds.toFixed(1)} 秒），建议重录。`, true);
    };
    recorder.onerror = () => {
      clearTimeout(rec.timer);
      rec.failed = true; rec.blob = null;
      stopStream();
      recordingUi(false);
      $('#rec-preview').hidden = true;
      toast('录音过程中出错，请重录。', true);
    };
    recorder.start(1000);
    rec.started = performance.now();
    rec.timer = setTimeout(stopRecording, state.user.limits.max_audio_seconds * 1000);
    recordingUi(true);
    $('#rec-preview').hidden = true;
    startMeter(rec.stream);
  } catch (error) {
    stopStream();
    throw error;
  }
}

function stopRecording() {
  if (rec.recorder?.state === 'recording') rec.recorder.stop();
}

function discardRecording() {
  stopRecording();
  rec.blob = null; rec.chunks = [];
  if (rec.url) URL.revokeObjectURL(rec.url);
  rec.url = null;
  $('#rec-audio').removeAttribute('src');
  $('#rec-preview').hidden = true;
  $('#rec-time').textContent = '00:00';
}

$('#rec-btn').addEventListener('click', async () => {
  if (rec.recorder?.state === 'recording') { stopRecording(); return; }
  try { await startRecording(); } catch (err) { toast(micError(err), true); recordingUi(false); }
});
$('#rec-redo').addEventListener('click', async () => {
  discardRecording();
  try { await startRecording(); } catch (err) { toast(micError(err), true); }
});
$('#rec-discard').addEventListener('click', discardRecording);
$('#rec-upload').addEventListener('click', async (event) => {
  if (!rec.blob?.size) { toast('请先完成录音。', true); return; }
  const type = rec.blob.type;
  const ext = type.includes('mp4') ? 'm4a' : type.includes('ogg') ? 'ogg' : type.includes('webm') ? 'webm' : 'bin';
  event.target.disabled = true;
  try {
    const asset = await uploadFile(rec.blob, { kind: 'recording', filename: `recording.${ext}` });
    discardRecording();
    setImitation(asset);
    afterImitationUpload('录音已上传');
  } catch (err) {
    toast(`${err.message}（录音已保留，可重试）`, true);
  } finally { event.target.disabled = false; }
});
window.addEventListener('pagehide', () => { stopRecording(); stopStream(); });

/* ---------- submit & job polling ---------- */
function pairJob() {
  return state.job && state.job.reference?.reference_id === state.refId
    && state.job.imitation?.asset_id === state.imitation?.asset_id ? state.job : null;
}

function updateSubmit() {
  const job = pairJob();
  const busy = state.submitting || (job && ['pending_enqueue', 'queued', 'running'].includes(job.status));
  const button = $('#submit-btn');
  button.disabled = !(state.refId && state.imitation) || busy;
  const rerun = Boolean(job && !busy);
  button.textContent = rerun ? '重新评测' : '开始评测';
  button.classList.toggle('btn-primary', !rerun);
  $('#submit-hint').textContent = !state.refId
    ? (state.imitation ? '选择原音频后会自动开始评测。' : '先选择原音频，上传或录制模仿后会自动开始评测。')
    : !state.imitation ? '上传或录制模仿音频后会自动开始评测。'
    : busy ? '这组音频正在评测中。'
    : job ? '这组音频已评测过；可以换一段原音频或重新上传模仿。' : '可以开始评测。';
}

// Uploading an imitation starts the evaluation right away (or as soon as a reference is chosen).
function afterImitationUpload(message) {
  if (!state.refId) {
    state.autoSubmitPending = true;
    toast(`${message}，选择原音频后会自动开始评测。`);
    updateSubmit();
    $('#step-ref').scrollIntoView({ behavior: 'smooth', block: 'start' });
    return;
  }
  toast(`${message}，开始评测`);
  submitEvaluation({ auto: true });
}

async function submitEvaluation({ auto = false } = {}) {
  if (!state.refId || !state.imitation || state.submitting) return;
  state.autoSubmitPending = false;
  state.idemKey ||= uuid();
  state.submitting = true;
  updateSubmit();
  try {
    const data = await api('/api/evaluations', {
      method: 'POST', headers: { 'Idempotency-Key': state.idemKey },
      body: { reference_id: state.refId, imitation_asset_id: state.imitation.asset_id },
    });
    await openJob(data.job_id);
    loadHistory(true);
    if (auto) $('#step-submit').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    toast(auto ? `没能自动开始评测：${err.message}` : err.message, true);
    if (err.extra?.job_id) openJob(err.extra.job_id);
  } finally {
    state.submitting = false;
    updateSubmit();
  }
}

$('#submit-btn').addEventListener('click', () => {
  if (pairJob()) state.idemKey = null;  // explicit re-run of the same pair → a new job
  submitEvaluation();
});

function stopPolling() {
  clearTimeout(state.pollTimer);
  clearInterval(state.tickTimer);
  state.pollTimer = null;
}

async function openJob(jobId, { quiet = false } = {}) {
  stopPolling();
  try {
    const job = await api(`/api/jobs/${jobId}`);
    renderJob(job, true);
  } catch (err) {
    if (err.status === 404) storeSet(STORE_JOB, null);
    if (!quiet) toast(err.message, true);
  }
}

async function poll() {
  if (!state.job) return;
  try {
    renderJob(await api(`/api/jobs/${state.job.job_id}`));
  } catch (err) {
    if (err.status === 401) return;
    state.pollTimer = setTimeout(poll, 5000);
  }
}

function renderJob(job, scroll = false) {
  state.job = job;
  state.jobFetchedAt = performance.now();
  storeSet(STORE_JOB, job.job_id);
  const active = ['pending_enqueue', 'queued', 'running'].includes(job.status);
  $('#job-status').hidden = false;
  const chip = $('#job-chip');
  chip.className = `chip ${job.status}`;
  chip.textContent = job.status === 'queued' && job.queue_ahead ? `排队中 · 前面 ${job.queue_ahead} 个` : STATUS_TEXT[job.status] || job.status;
  $('#job-id').textContent = `任务 ${job.job_id}${job.attempt > 1 ? ` · 第 ${job.attempt} 次尝试` : ''}`;
  const current = job.status === 'succeeded' ? STAGE_ORDER.length : active && job.status !== 'running' ? 0
    : Math.max(0, STAGE_ORDER.indexOf(job.stage === 'preparing_reference' ? 'evaluating' : job.stage));
  document.querySelectorAll('#stages li').forEach((li, index) => {
    li.classList.toggle('done', index < current || job.status === 'succeeded');
    li.classList.toggle('active', active && index === current);
  });
  renderTimers();
  clearInterval(state.tickTimer);
  if (active) state.tickTimer = setInterval(renderTimers, 1000);

  const errorBox = $('#job-error');
  errorBox.hidden = !job.error;
  if (job.error) {
    errorBox.replaceChildren(
      el('div', { text: `${job.error.message || '任务失败'}（任务编号 ${job.job_id}）` }),
      el('div', { class: 'row', style: 'margin-top:8px' },
        el('button', { class: 'btn btn-sm', type: 'button', text: '重试', onclick: () => retryJob(job.job_id) })),
      job.error.detail ? el('details', {}, el('summary', { text: '错误详情（管理员）' }), el('pre', { text: job.error.detail })) : null);
  }
  if (job.status === 'succeeded' && job.result) renderResult(job, scroll);
  else $('#step-result').hidden = true;
  if (!active) {
    state.idemKey = null;
    markHistory(job.job_id);
    if (scroll) $('#step-submit').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } else {
    state.pollTimer = setTimeout(poll, job.poll_after_ms || 2000);
  }
  updateSubmit();
  if (!active && document.querySelector(`#history [data-job="${job.job_id}"] .h-sub`)?.dataset.status !== job.status) loadHistory(true);
}

function renderTimers() {
  const job = state.job;
  if (!job) return;
  const drift = (performance.now() - state.jobFetchedAt) / 1000;
  const running = job.status === 'running';
  $('#t-elapsed').textContent = job.elapsed_seconds == null ? '—' : fmtSeconds(job.elapsed_seconds + (running ? drift : 0), 0);
  const waiting = ['pending_enqueue', 'queued'].includes(job.status);
  const queueSeconds = waiting ? (Date.now() - new Date(job.queued_at || job.created_at).getTime()) / 1000 : job.queue_wait_seconds;
  $('#t-queue').textContent = queueSeconds == null ? '—' : fmtSeconds(queueSeconds, 0);
}

async function retryJob(jobId) {
  try {
    const data = await api(`/api/jobs/${jobId}/retry`, { method: 'POST' });
    if (state.job?.job_id === jobId || !state.user.is_admin) await openJob(data.job_id);
    toast('已重新提交');
  } catch (err) { toast(err.message, true); }
}

/* ---------- result ---------- */
function renderResult(job, scroll) {
  const result = job.result;
  $('#step-result').hidden = false;
  $('#score-overall').textContent = result.overall.toFixed(1);
  requestAnimationFrame(() => { $('#ring-fg').style.strokeDashoffset = String(RING * (1 - Math.min(100, result.overall) / 100)); });
  $('#parts').replaceChildren(...Object.entries(PART_LABELS).map(([key, label]) => {
    const value = result.parts?.[key];
    const bar = el('span');
    requestAnimationFrame(() => { bar.style.width = value == null ? '0' : `${Math.max(0, Math.min(100, value))}%`; });
    return el('div', { class: 'part-row' }, el('span', { text: label }), el('div', { class: 'part-bar' }, bar),
      el('span', { class: value == null ? 'part-val na' : 'part-val', text: value == null ? '暂无数据' : value.toFixed(1) }));
  }));
  $('#result-grade').replaceChildren(...(result.grade ? ['等级：', el('strong', { text: result.grade })] : []));
  $('#result-warnings').replaceChildren(...(result.warnings || []).map((w) => el('div', { class: 'warning', text: w })));
  $('#advice-wrap').hidden = !(result.advice || []).length;
  $('#advice').replaceChildren(...(result.advice || []).map((tip) => el('li', { text: tip })));
  const m = job.metrics || {};
  $('#result-metrics').replaceChildren(
    el('span', {}, '原音频 ', el('strong', { text: fmtSeconds(m.reference_duration_seconds) })),
    el('span', {}, '模仿音频 ', el('strong', { text: fmtSeconds(m.imitation_duration_seconds) })),
    el('span', {}, '推理耗时 ', el('strong', { text: fmtSeconds(m.inference_elapsed_seconds) })),
    el('span', {}, '处理耗时 ', el('strong', { text: fmtSeconds(job.elapsed_seconds) })));
  setAudio('#res-ref-audio', job.reference?.audio_url);
  setAudio('#res-imit-audio', job.imitation?.audio_url);
  const captions = { rhythm: '节奏对比', pitch: '音高对比（语调仅作参考，不计分）' };
  $('#charts').replaceChildren(...Object.entries(result.chart_urls || {}).map(([name, url]) =>
    el('figure', {}, el('img', { src: url, alt: captions[name] || name, loading: 'lazy' }), el('figcaption', { text: captions[name] || name }))));
  const available = result.files_available !== false && result.report_url;
  $('#report-actions').hidden = !available;
  $('#report-text').hidden = true;
  $('#structured-download').hidden = true;
  $('#report-toggle').textContent = '查看详细报告';
  if (available) {
    $('#report-download').href = withParam(result.report_url, 'download', 1);
    const detail = result.report_urls?.detail;
    $('#detail-download').hidden = !detail;
    if (detail) $('#detail-download').href = withParam(detail, 'download', 1);
    const structured = result.report_urls?.json;
    $('#structured-download').hidden = !structured;
    if (structured) $('#structured-download').href = withParam(structured, 'download', 1);
  } else {
    $('#result-warnings').append(el('div', { class: 'warning', text: '详细报告和图表已超过保留期限被清理，分数仍可查看。' }));
  }
  if (scroll) $('#step-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setAudio(selector, url) {
  const audio = $(selector);
  audio.parentElement.hidden = !url;
  if (url && audio.getAttribute('src') !== url) audio.src = url;
}

$('#report-toggle').addEventListener('click', async () => {
  const pre = $('#report-text');
  if (!pre.hidden) { pre.hidden = true; $('#report-toggle').textContent = '查看详细报告'; return; }
  try {
    pre.textContent = await api(state.job.result.report_url);
    pre.hidden = false;
    $('#report-toggle').textContent = '收起报告';
  } catch (err) { toast(err.message, true); }
});

/* ---------- history ---------- */
async function loadHistory(reset = false) {
  if (reset) state.historyBefore = null;
  const query = `/api/jobs?type=evaluate&limit=10${state.historyBefore ? `&before=${encodeURIComponent(state.historyBefore)}` : ''}`;
  try {
    const data = await api(query);
    const items = data.items.map((job) => {
      const score = job.status === 'succeeded' && job.result ? job.result.overall.toFixed(1) : '';
      const sub = el('span', { class: 'h-sub', text: `${fmtTime(job.created_at)} · ${STATUS_TEXT[job.status] || job.status}` });
      sub.dataset.status = job.status;
      return el('li', { 'data-job': job.job_id },
        el('button', { type: 'button', class: state.job?.job_id === job.job_id ? 'current' : null, onclick: () => openJob(job.job_id) },
          el('span', { class: 'h-title', text: job.reference?.display_name || '原音频' }),
          el('span', { class: 'h-score', text: score }), sub));
    });
    if (reset) $('#history').replaceChildren(...items);
    else $('#history').append(...items);
    if (reset && !items.length) $('#history').append(el('li', { class: 'empty', text: '还没有评测记录。' }));
    state.historyBefore = data.next_before;
    $('#history-more').hidden = !data.next_before;
  } catch (err) { if (err.status !== 401) console.warn(err); }
}
function markHistory(jobId) {
  document.querySelectorAll('#history button').forEach((b) => b.classList.toggle('current', b.parentElement.dataset.job === jobId));
}
$('#history-refresh').addEventListener('click', () => loadHistory(true));
$('#history-more').addEventListener('click', () => loadHistory(false));

/* ---------- invites (admin) ---------- */
let lastInvite = null;

async function copyText(text, label) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const area = el('textarea', { style: 'position:fixed;opacity:0' });
    area.value = text;
    document.body.append(area);
    area.select();
    const ok = document.execCommand('copy');
    area.remove();
    if (!ok) { toast('复制失败，请手动选择文字复制。', true); return; }
  }
  toast(`${label}已复制`);
}

function inviteMessage(invite) {
  const origin = location.origin;
  return `你好，邀请你试用「音频还原度测试」：\n${origin}/#invite=${invite.code}\n（或打开 ${origin} 输入邀请码 ${invite.code}）\n邀请码即你的个人身份，请勿转发。`;
}

function showInvite(user, code, reset = false) {
  lastInvite = { name: user.name, code };
  $('#invite-for').textContent = `${reset ? '已重置' : '邀请已创建'} · ${user.name}${user.is_admin ? '（管理员）' : ''}`;
  $('#invite-code-out').textContent = code;
  $('#invite-result').hidden = false;
  $('#invite-result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function loadUsers() {
  try {
    const data = await api('/api/admin/users');
    $('#user-list').replaceChildren(...data.items.map((u) => {
      const me = u.id === state.user.id;
      const sub = [`创建于 ${fmtTime(u.created_at)}`, u.last_login_at ? `最近登录 ${fmtTime(u.last_login_at)}` : '尚未登录',
        `评测 ${u.evaluations} 次`].join(' · ');
      const actions = me ? [] : [
        el('button', { type: 'button', class: 'btn btn-ghost btn-sm', text: '重置邀请码', onclick: () => resetInvite(u) }),
        el('button', { type: 'button', class: 'btn btn-ghost btn-sm', text: u.disabled ? '启用' : '停用', onclick: () => toggleUser(u) }),
      ];
      return el('li', { class: `user-row${u.disabled ? ' off' : ''}` },
        el('div', {},
          el('div', { class: 'user-name' }, u.name,
            u.is_admin ? el('span', { class: 'tag tag-accent', text: '管理员' }) : null,
            u.disabled ? el('span', { class: 'tag tag-warn', text: '已停用' }) : null,
            me ? el('span', { class: 'tag', text: '你' }) : null),
          el('div', { class: 'user-sub', text: sub })),
        el('div', { class: 'user-actions' }, actions));
    }));
  } catch (err) { if (err.status !== 401) toast(err.message, true); }
}

async function resetInvite(user) {
  if (!confirm(`重置「${user.name}」的邀请码？旧邀请码立即失效，对方已登录的设备会被退出。`)) return;
  try {
    const data = await api(`/api/admin/users/${user.id}/reset-invite`, { method: 'POST' });
    showInvite(data.user, data.invite_code, true);
    loadUsers();
  } catch (err) { toast(err.message, true); }
}

async function toggleUser(user) {
  if (!user.disabled && !confirm(`停用「${user.name}」？对方将无法登录，已登录的设备会被退出。`)) return;
  try {
    await api(`/api/admin/users/${user.id}`, { method: 'PATCH', body: { disabled: !user.disabled } });
    toast(user.disabled ? '已启用' : '已停用');
    loadUsers();
  } catch (err) { toast(err.message, true); }
}

$('#invite-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = event.submitter || event.target.querySelector('button');
  button.disabled = true;
  try {
    const data = await api('/api/admin/users', {
      method: 'POST', body: { name: $('#invite-name').value.trim(), is_admin: $('#invite-is-admin').checked },
    });
    event.target.reset();
    showInvite(data.user, data.invite_code);
    loadUsers();
  } catch (err) {
    toast(err.message, true);
  } finally { button.disabled = false; }
});
$('#copy-code').addEventListener('click', () => lastInvite && copyText(lastInvite.code, '邀请码'));
$('#copy-message').addEventListener('click', () => lastInvite && copyText(inviteMessage(lastInvite), '邀请消息'));
$('#invite-dismiss').addEventListener('click', () => {
  lastInvite = null;
  $('#invite-result').hidden = true;
  $('#invite-code-out').textContent = '';
});
$('#users-refresh').addEventListener('click', loadUsers);

// ---- transcription provider (admin) ----
function renderAsr(data) {
  $('#asr-updated').textContent = data.updated_at ? `${data.updated_by || '管理员'} 于 ${fmtTime(data.updated_at)} 设置` : '使用默认设置';
  $('#asr-options').replaceChildren(...data.providers.map((p) => {
    const current = p.id === data.asr_provider;
    const state = p.configured ? `已配置（${p.config_file}）` : p.problem;
    return el('button', {
      type: 'button', role: 'radio', class: 'asr-option', 'aria-checked': String(current),
      disabled: !current && !p.configured, onclick: () => { if (!current) switchAsr(p); },
    },
    el('span', { class: 'asr-name' }, p.name, current ? el('span', { class: 'tag tag-accent', text: '使用中' }) : null),
    el('span', { class: `asr-state ${p.configured ? 'ok' : 'bad'}`, text: state }));
  }));
}

async function loadAsr() {
  try { renderAsr(await api('/api/admin/settings')); } catch (err) { toast(err.message, true); }
}

async function switchAsr(provider) {
  if (!confirm(`切换到「${provider.name}」转写？之后开始的原音频准备和评测都会使用它。`)) return;
  try {
    renderAsr(await api('/api/admin/settings', { method: 'PUT', body: { asr_provider: provider.id } }));
    toast(`已切换到${provider.name}`);
  } catch (err) { toast(err.message, true); loadAsr(); }
}

/* ---------- boot ---------- */
// Invite links (#invite=CODE) prefill the login form; the code never reaches server logs.
const linkInvite = new URLSearchParams(location.hash.slice(1)).get('invite');
if (linkInvite) {
  $('#invite-code').value = linkInvite;
  history.replaceState(null, '', location.pathname + location.search);
}

(async () => {
  try {
    const data = await api('/api/me', { allow401: true });
    await showApp(data.user);
  } catch (err) {
    if (err.status === 401) showLogin();
    else { showLogin(); toast(err.message, true); }
  }
})();
