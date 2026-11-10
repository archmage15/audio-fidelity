# HTTP 接口

- 所有接口位于 `/api/`，JSON 编码；会话为 HttpOnly Cookie `afd_session`（SameSite=Lax，Secure）。
- **写接口（POST/PATCH/DELETE）**必须带 `X-Requested-With: fetch`；如带 `Origin`，必须等于站点来源。否则 `403`。
- 错误统一为 `{"error": {"code": "...", "message": "中文说明", ...}}`。查不到或无权访问他人资源一律 `404`。

## 会话

| 方法与地址 | 请求 | 返回 |
| --- | --- | --- |
| `POST /api/session` | `{"invite_code": "XXXX-XXXX-XXXX"}` | `{"user": {...}}`，设置 Cookie；错误 `401 invalid_invite`，同 IP 10 分钟超过 10 次 `429` |
| `DELETE /api/session` | — | 退出 |
| `GET /api/me` | — | `{"user": {"id","name","is_admin","limits": {...}}}` |

## 音频文件

| 方法与地址 | 请求 | 返回 |
| --- | --- | --- |
| `POST /api/assets` | multipart：`file`；可选 `kind=upload\|recording`、`purpose=imitation\|reference`（reference 仅管理员） | `201` 资产信息，`status=uploaded` |
| `GET /api/assets/{id}` | — | 资产信息 |
| `GET /api/assets/{id}/audio` | 支持 `Range` | 转码后的 WAV（尚未转码时返回原文件） |
| `DELETE /api/assets/{id}` | — | 删除；被原音频素材或活动任务引用时 `409 asset_in_use` |

上传错误：`413 file_too_large`、`415 unsupported_format`（无音轨/非音频容器）、`422 duration_out_of_range | empty_file | decode_failed`、`503 storage_low`。

## 原音频

| 方法与地址 | 请求 | 返回 |
| --- | --- | --- |
| `GET /api/references` | 管理员可加 `?include_pending=1` | `{"items": [{reference_id, display_name, status, is_public, transcript, duration_seconds, audio_url, job_id, error}]}` |
| `GET /api/references/{id}` | — | `preparing / ready / failed / archived`、转写文本 |
| `POST /api/references` | 管理员；`{"asset_id","display_name","is_public":true}`，可带 `Idempotency-Key` | `202 {"reference_id","job_id","status":"queued"}` |
| `PATCH /api/references/{id}` | 管理员；`display_name` / `is_public` / `archived:true` | 更新后的原音频 |

## 评测与任务

```http
POST /api/evaluations
Idempotency-Key: <浏览器生成的 UUID，重试时复用>
{"reference_id": "ref_...", "imitation_asset_id": "asset_..."}
→ 202 {"job_id": "job_...", "status": "queued", "poll_after_ms": 2000}
```

错误：`409 reference_not_ready`、`409 idempotency_conflict`（同键不同参数）、`422 decode_failed`、`429 user_quota`、`503 queue_full | queue_unavailable | storage_low`。`queue_unavailable` 时任务已记录（返回体含 `job_id`），会被自动补投。

| 方法与地址 | 说明 |
| --- | --- |
| `GET /api/jobs/{id}` | 状态、阶段、耗时、指标、结果摘要 |
| `GET /api/jobs?limit=20&before=<created_at>&type=evaluate` | 当前用户最近任务，`next_before` 翻页 |
| `POST /api/jobs/{id}/retry` | 失败/超时任务新建一次尝试（准备原音频的重试仅管理员） |
| `GET /api/jobs/{id}/report?kind=summary\|detail[&download=1]` | 路径脱敏后的文本报告；超过保留期 `410` |
| `GET /api/jobs/{id}/charts/{rhythm\|pitch}` | 节奏 / 音高对比 PNG |

任务状态 `pending_enqueue → queued → running → succeeded | failed | timed_out`；阶段 `normalizing / preparing_reference / evaluating / saving / done`。

```json
{
  "job_id": "job_...", "type": "evaluate", "status": "succeeded", "stage": "done", "attempt": 1,
  "created_at": "...Z", "queued_at": "...Z", "started_at": "...Z", "finished_at": "...Z",
  "queue_wait_seconds": 0.4, "elapsed_seconds": 63.2,
  "reference": {"reference_id": "ref_...", "display_name": "L1.5.1_05", "audio_url": "/api/assets/.../audio"},
  "imitation": {"asset_id": "asset_...", "display_name": "recording.webm", "audio_url": "/api/assets/.../audio"},
  "metrics": {"reference_duration_seconds": 4.63, "imitation_duration_seconds": 4.63, "inference_elapsed_seconds": 61.8},
  "result": {
    "overall": 94.4, "grade": "优",
    "parts": {"pronunciation": 87.8, "sound_changes": 100.0, "rhythm": 96.6, "stress": 100.0},
    "advice": ["..."], "warnings": [],
    "report_url": "/api/jobs/job_.../report",
    "report_urls": {"summary": "...", "detail": "...?kind=detail"},
    "chart_urls": {"rhythm": "...", "pitch": "..."}, "files_available": true
  },
  "error": null
}
```

- 分项缺失为 `null`（页面显示“暂无数据”），不按 0 分处理；不提供语调分。
- 排队中返回 `queue_ahead`（前面的任务数）；失败时 `error: {code, message}`，管理员额外看到 `detail`（堆栈/引擎错误）。
- `elapsed_seconds` 为 Worker 开始处理到结束的“处理耗时”（含转码）；`inference_elapsed_seconds` 仅为 `evaluate()` 调用耗时。

## 管理员

| 地址 | 说明 |
| --- | --- |
| `GET /api/admin/inference-runs?start=&end=&task_type=&format=json\|csv` | 每次引擎调用一条：两段时长、推理耗时、起止时间、状态、`cache_hit`、引擎版本 |
| `GET /api/admin/stats?start=&end=` | 评测尝试按状态计数、成功率、推理耗时均值/P50/P95（仅成功且耗时非空） |
| `GET /api/admin/settings` | 转写服务：`asr_provider`（`xunfei`/`tingwu`）、`updated_at/by`、各服务配置是否就绪（只给状态，不给值） |
| `PUT /api/admin/settings` `{"asr_provider"}` | 切换转写服务，对之后开始的任务生效；目标服务未配置好时 409 `provider_not_configured` |
| `GET /api/asr-media/{name}`（无需登录） | 给通义听悟拉取的临时音频：随机 128 位文件名，1 小时有效，任务结束即删除 |
| `GET /api/admin/health` | Redis、引擎目录/解释器、模型完整性、转写服务配置、磁盘余量、版本；`engine_server`：常驻推理进程是否就绪、PID、已处理任务数、预热结果、模型加载记录、内存 |
| `GET /api/admin/users` | 用户列表：`id,name,is_admin,disabled,created_at,last_login_at,evaluations`（不含邀请码/哈希） |
| `POST /api/admin/users` `{"name","is_admin"}` | 201，`{"user", "invite_code"}`；邀请码只在此返回一次 |
| `POST /api/admin/users/{id}/reset-invite` | 生成新邀请码并注销该用户所有会话；不能重置自己（409 `self_reset`） |
| `PATCH /api/admin/users/{id}` `{"disabled": bool}` | 停用（同时注销会话）/恢复；不能停用自己（409 `self_disable`） |
| `GET /api/health` | 公开健康检查（db、redis） |
