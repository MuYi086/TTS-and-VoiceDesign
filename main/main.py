#!/usr/bin/env python3
"""用于共享存储和运行时诊断的轻量控制面 API。

模型推理不在这里执行。各模型服务自行管理推理生命周期；本进程只保留
周边 WebUI 使用的 8300 端口健康检查、上传、文件检查和 CPU 音频导出工具。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from gpu_runtime import cuda_status
from spatial_audio import (
    SPATIAL_EXPORT_FORMATS,
    SPATIAL_EXPORT_PROFILES,
    SpatialAudioExportError,
    SpatialAudioProcessor,
)
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

from unitale_runtime import (
    AudioReferenceStore,
    AudioUploadError,
    StagedUpload,
    UploadPolicy,
    stage_audio_upload,
    storage_disk_status,
)

MAIN_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MAIN_DIR.parent


def expand_path(path: str) -> str:
    """展开环境变量和用户目录，统一得到可用于存储配置的绝对路径。"""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


STORAGE_DIR = expand_path(os.getenv("STORAGE_DIR", str(PROJECT_DIR / "storage")))
TIMBRE_STORAGE_DIR = expand_path(os.getenv("TIMBRE_STORAGE_DIR", str(Path(STORAGE_DIR) / "timbre")))
TIMBRE_REFERENCE_DIR = str(Path(TIMBRE_STORAGE_DIR) / ".references")
SOUNDEFFECT_STORAGE_DIR = expand_path(
    os.getenv("SOUNDEFFECT_STORAGE_DIR", str(Path(STORAGE_DIR) / "soundEffect"))
)
CLONE_STORAGE_DIR = expand_path(os.getenv("CLONE_STORAGE_DIR", str(Path(STORAGE_DIR) / "clone")))
PROMPTS_DIR = expand_path(os.getenv("PROMPTS_DIR", CLONE_STORAGE_DIR))
RUNTIME_CACHE_DIR = expand_path(
    os.getenv("RUNTIME_CACHE_DIR", str(Path(STORAGE_DIR) / ".cache/runtime"))
)
SPATIAL_EXPORT_CACHE_DIR = expand_path(
    os.getenv(
        "SPATIAL_EXPORT_CACHE_DIR",
        str(Path(RUNTIME_CACHE_DIR) / "spatial_exports"),
    )
)
SPATIAL_EXPORT_MAX_BYTES = int(os.getenv("SPATIAL_EXPORT_MAX_BYTES", str(512 * 1024 * 1024)))
SPATIAL_EXPORT_TIMEOUT = float(os.getenv("SPATIAL_EXPORT_TIMEOUT", "600"))
SPATIAL_EXPORT_FFMPEG_BIN = os.getenv("SPATIAL_EXPORT_FFMPEG_BIN", "ffmpeg")
if SPATIAL_EXPORT_MAX_BYTES <= 0:
    raise ValueError("SPATIAL_EXPORT_MAX_BYTES 必须为正整数。")
if SPATIAL_EXPORT_TIMEOUT <= 0:
    raise ValueError("SPATIAL_EXPORT_TIMEOUT 必须大于 0。")
GPU_LOCK_FILE = expand_path(
    os.getenv("GPU_LOCK_FILE", str(Path(RUNTIME_CACHE_DIR) / "gpu-runtime.lock"))
)
API_HOST = os.getenv("HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", "8300"))
MIMO_TTS_PROXY_URL = os.getenv(
    "MIMO_TTS_PROXY_URL",
    f"http://127.0.0.1:{os.getenv('MIMO_TTS_PORT', '8303')}/v1/mimo/timbre",
)
MIMO_TTS_PROXY_TIMEOUT = float(os.getenv("MIMO_TTS_PROXY_TIMEOUT", "310"))
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}

for directory in (
    TIMBRE_STORAGE_DIR,
    TIMBRE_REFERENCE_DIR,
    SOUNDEFFECT_STORAGE_DIR,
    CLONE_STORAGE_DIR,
    PROMPTS_DIR,
    RUNTIME_CACHE_DIR,
    SPATIAL_EXPORT_CACHE_DIR,
):
    os.makedirs(directory, exist_ok=True)

reference_store = AudioReferenceStore(PROMPTS_DIR, TIMBRE_STORAGE_DIR)
spatial_audio_processor = SpatialAudioProcessor(
    SPATIAL_EXPORT_CACHE_DIR,
    ffmpeg_bin=SPATIAL_EXPORT_FFMPEG_BIN,
    timeout_seconds=SPATIAL_EXPORT_TIMEOUT,
)


def spatial_export_available() -> bool:
    """检查配置的 FFmpeg 命令是否能从当前控制面环境解析。"""
    return shutil.which(SPATIAL_EXPORT_FFMPEG_BIN) is not None


app = FastAPI(title="Unitale AI Control Plane")


class ForceCORS(BaseHTTPMiddleware):
    """为 WebUI 请求补充宽松的跨域响应头，并快速处理预检请求。"""

    async def dispatch(self, request, call_next):
        if request.method == "OPTIONS":
            return Response(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Credentials": "false",
                },
            )
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Expose-Headers"] = (
            "Content-Disposition, X-Audio-Sample-Rate, X-Audio-Profile, X-Audio-Format"
        )
        return response


app.add_middleware(ForceCORS)


def forward_mimo_design_request(body: bytes, accept: str) -> tuple[int, bytes, str]:
    """将控制面最终保留的 MiMo 路由转发到独立服务。"""
    headers = {
        "Content-Type": "application/json",
        "Accept": accept or "*/*",
    }
    upstream_request = urllib.request.Request(
        MIMO_TTS_PROXY_URL,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            upstream_request,
            timeout=MIMO_TTS_PROXY_TIMEOUT,
        ) as upstream:
            response_body = upstream.read()
            content_type = upstream.headers.get_content_type() or "application/octet-stream"
            return upstream.status, response_body, content_type
    except urllib.error.HTTPError as exc:
        response_body = exc.read()
        content_type = exc.headers.get_content_type() if exc.headers else "application/json"
        return exc.code, response_body, content_type or "application/json"
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 MiMo 独立服务 {MIMO_TTS_PROXY_URL}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"连接 MiMo 独立服务超时: {MIMO_TTS_PROXY_URL}") from exc


def hash_filename(filename: str) -> str:
    """将 WebUI 的逻辑路径映射为稳定文件名，避免直接使用用户输入作路径。"""
    return reference_store.clone_path(filename).name


def prompt_audio_path(filename: str) -> Path:
    """解析普通克隆上传，或解析预览时引用的音色设计音频。"""
    return reference_store.prompt_audio_path(filename)


def store_uploaded_audio(staged: StagedUpload, full_path: str) -> dict[str, object]:
    """在线程池中原子提交流式暂存的参考音频。"""
    return reference_store.commit_staged_upload(staged, full_path)


@app.get("/v1/health")
@app.get("/v1/control")
def health():
    """返回控制面、存储目录和 GPU 可见性的诊断信息。"""
    cuda = cuda_status()
    return {
        "code": 200,
        "paths": {
            "storage_dir": STORAGE_DIR,
            "timbre_storage_dir": TIMBRE_STORAGE_DIR,
            "soundeffect_storage_dir": SOUNDEFFECT_STORAGE_DIR,
            "clone_storage_dir": CLONE_STORAGE_DIR,
            "prompts_dir": PROMPTS_DIR,
            "runtime_cache_dir": RUNTIME_CACHE_DIR,
            "spatial_export_cache_dir": SPATIAL_EXPORT_CACHE_DIR,
            "gpu_lock_file": GPU_LOCK_FILE,
            "mimo_tts_proxy_url": MIMO_TTS_PROXY_URL,
        },
        "available": {
            "cuda": cuda["available"],
        },
        "cuda": cuda,
        "storage": storage_disk_status(STORAGE_DIR),
        "last_errors": {},
        "offline": {
            "local_files_only": LOCAL_FILES_ONLY,
            "hf_hub_offline": os.getenv("HF_HUB_OFFLINE"),
            "transformers_offline": os.getenv("TRANSFORMERS_OFFLINE"),
        },
        "runtime": {
            "service_role": "control_plane",
            "model_inference": "delegated to standalone services",
        },
        "audio_export": {
            "available": spatial_export_available(),
            "ffmpeg_bin": SPATIAL_EXPORT_FFMPEG_BIN,
            "sample_rate": 48000,
            "profiles": list(SPATIAL_EXPORT_PROFILES),
            "formats": list(SPATIAL_EXPORT_FORMATS),
            "max_bytes": SPATIAL_EXPORT_MAX_BYTES,
            "timeout_seconds": SPATIAL_EXPORT_TIMEOUT,
        },
    }


@app.post("/v1/mimo/timbre")
async def mimo_design_proxy(request: Request):
    """在 MiMo 推理位于 ``mimo_tts`` 时保留 8300 控制面路由。"""
    body = await request.body()
    try:
        status_code, response_body, content_type = await asyncio.to_thread(
            forward_mimo_design_request,
            body,
            request.headers.get("accept", "*/*"),
        )
    except RuntimeError as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    return Response(
        content=response_body,
        status_code=status_code,
        headers={"content-type": content_type},
    )


@app.post("/v1/upload_audio")
async def upload_audio(audio: UploadFile = File(...), full_path: str = Form(...)):
    """上传克隆参考音频，但不复制已有的音色设计资产。"""
    try:
        staged = await stage_audio_upload(audio, Path(RUNTIME_CACHE_DIR) / "uploads")
        return await run_in_threadpool(store_uploaded_audio, staged, full_path)
    except AudioUploadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


@app.post("/v1/audio/export")
async def export_audio(
    audio: UploadFile = File(...),
    profile: Literal["standard", "balanced", "immersive"] = Form("balanced"),
    output_format: Literal["wav", "mp3"] = Form("wav"),
):
    """流式接收 WebUI 混音，经 CPU FFmpeg 处理后返回 48 kHz 成品。"""
    upload_policy = UploadPolicy(max_bytes=SPATIAL_EXPORT_MAX_BYTES)
    try:
        staged = await stage_audio_upload(
            audio,
            Path(SPATIAL_EXPORT_CACHE_DIR) / "uploads",
            policy=upload_policy,
        )
    except AudioUploadError as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    try:
        result = await run_in_threadpool(
            spatial_audio_processor.export,
            staged,
            profile=profile,
            output_format=output_format,
        )
    except SpatialAudioExportError as exc:
        staged.path.unlink(missing_ok=True)
        return JSONResponse(status_code=503, content={"detail": str(exc)})
    except Exception:
        staged.path.unlink(missing_ok=True)
        raise

    return FileResponse(
        path=result.path,
        media_type=result.media_type,
        filename=result.download_name,
        headers={
            "X-Audio-Sample-Rate": "48000",
            "X-Audio-Profile": profile,
            "X-Audio-Format": output_format,
        },
        background=BackgroundTask(result.cleanup),
    )


@app.get("/v1/check/audio")
def check_audio_exists(file_name: str):
    exists = prompt_audio_path(file_name).is_file()
    return {"code": 200 if exists else 404, "exists": exists}


if __name__ == "__main__":
    print("==================================================")
    print("   Unitale AI control plane (no model inference)")
    print("==================================================")
    print(f"[配置] storage: {STORAGE_DIR}")
    print(f"[配置] prompts: {PROMPTS_DIR}")
    print(f"[配置] host={API_HOST}, port={API_PORT}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)
