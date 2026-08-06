"""Step-Audio-EditX 音频编辑请求模型与一次性 worker 调度。"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from local_worker import LocalWorkerConfig, run_local_worker


API_DIR = Path(__file__).resolve().parent
HF_MIRROR_DIR = Path(os.path.expandvars(os.path.expanduser(os.getenv("HF_MIRROR_DIR", "~/hf-mirror")))).resolve()
STEP_AUDIO_EDITX_CONDA_ENV = os.getenv("STEP_AUDIO_EDITX_CONDA_ENV", "Step-Audio-EditX")
STEP_AUDIO_EDITX_MODEL_DIR = Path(
    os.path.expandvars(
        os.path.expanduser(
            os.getenv("STEP_AUDIO_EDITX_MODEL_DIR", str(HF_MIRROR_DIR / "stepfun-ai/Step-Audio-EditX"))
        )
    )
).resolve()
STEP_AUDIO_TOKENIZER_PATH = Path(
    os.path.expandvars(
        os.path.expanduser(
            os.getenv("STEP_AUDIO_TOKENIZER_PATH", str(HF_MIRROR_DIR / "stepfun-ai/Step-Audio-Tokenizer"))
        )
    )
).resolve()
STEP_AUDIO_EDITX_CODE_PATH = Path(
    os.path.expandvars(
        os.path.expanduser(os.getenv("STEP_AUDIO_EDITX_CODE_PATH", "~/tts-depency/Step-Audio-EditX"))
    )
).resolve()
STEP_AUDIO_EDITX_WORKER_SCRIPT = API_DIR / "step_audio_editx_worker.py"
STEP_AUDIO_EDITX_WORKER_TMP_DIR = Path(
    os.path.expandvars(
        os.path.expanduser(
            os.getenv("STEP_AUDIO_EDITX_WORKER_TMP_DIR", str(API_DIR / ".cache/runtime/step_audio_editx_worker"))
        )
    )
).resolve()
STEP_AUDIO_EDITX_REQUEST_TIMEOUT = float(os.getenv("STEP_AUDIO_EDITX_REQUEST_TIMEOUT", "900"))
STEP_AUDIO_EDITX_DTYPE = os.getenv("STEP_AUDIO_EDITX_DTYPE", "bfloat16")
STEP_AUDIO_EDITX_MAX_MODEL_LEN = int(os.getenv("STEP_AUDIO_EDITX_MAX_MODEL_LEN", "3072"))
STEP_AUDIO_EDITX_GPU_MEMORY_UTILIZATION = float(
    os.getenv("STEP_AUDIO_EDITX_GPU_MEMORY_UTILIZATION", "0.5")
)
STEP_AUDIO_EDITX_MAX_NUM_SEQS = int(os.getenv("STEP_AUDIO_EDITX_MAX_NUM_SEQS", "1"))
STEP_AUDIO_EDITX_COSYVOICE_DTYPE = os.getenv("STEP_AUDIO_EDITX_COSYVOICE_DTYPE", "bfloat16")
STEP_AUDIO_EDITX_ENFORCE_EAGER = os.getenv("STEP_AUDIO_EDITX_ENFORCE_EAGER", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
STEP_AUDIO_EDITX_COSYVOICE_CUDA_GRAPH = os.getenv(
    "STEP_AUDIO_EDITX_COSYVOICE_CUDA_GRAPH", "0"
).strip().lower() in {"1", "true", "yes", "on"}

STEP_AUDIO_EDITX_EDIT_TYPES = frozenset({
    "emotion", "style", "paralinguistic", "denoise", "vad", "speed"
})


class StepAudioEditXEditRequest(BaseModel):
    """`/v1/step-audio-editx/edit` 的稳定 JSON 请求契约。"""

    prompt_text: Optional[str] = None
    prompt_audio: str = Field(min_length=1, description="经 /v1/upload_audio 上传后的 prompt 音频路径")
    generated_text: Optional[str] = None
    edit_type: Literal["emotion", "style", "paralinguistic", "denoise", "vad", "speed"]
    edit_info: str = ""

    @model_validator(mode="after")
    def validate_edit_contract(self):
        self.prompt_audio = self.prompt_audio.strip()
        self.prompt_text = self.prompt_text.strip() if self.prompt_text else None
        self.generated_text = self.generated_text.strip() if self.generated_text else None
        self.edit_info = self.edit_info.strip()
        if not self.prompt_audio:
            raise ValueError("prompt_audio 不能为空。")
        if self.edit_type not in {"denoise", "vad"} and not self.prompt_text:
            raise ValueError(f"edit_type={self.edit_type} 需要与 prompt_audio 一致的 prompt_text。")
        if self.edit_type not in {"denoise", "vad"} and not self.generated_text:
            self.generated_text = self.prompt_text
        if self.edit_type in {"emotion", "style", "speed"} and not self.edit_info:
            raise ValueError(f"edit_type={self.edit_type} 需要 edit_info。")
        return self


STEP_AUDIO_EDITX_WORKER = LocalWorkerConfig(
    conda_env=STEP_AUDIO_EDITX_CONDA_ENV,
    worker_script=str(STEP_AUDIO_EDITX_WORKER_SCRIPT),
    model_dir=str(STEP_AUDIO_EDITX_MODEL_DIR),
    temp_dir=str(STEP_AUDIO_EDITX_WORKER_TMP_DIR),
    timeout=STEP_AUDIO_EDITX_REQUEST_TIMEOUT,
    label="Step-Audio-EditX",
    file_prefix="step_audio_editx",
)


class StepAudioEditXWorkerManager:
    """将轻量 HTTP 请求转换为官方 Step-Audio-EditX 推理负载。"""

    def __init__(self):
        self.lock = threading.RLock()
        self.last_error: Optional[str] = None

    def build_worker_payload(
        self,
        request: StepAudioEditXEditRequest,
        prompt_audio_path: Path,
        *,
        local_files_only: bool,
    ) -> dict:
        if not prompt_audio_path.is_file():
            raise FileNotFoundError(f"Step-Audio-EditX prompt 音频不存在：{prompt_audio_path}")
        return {
            "prompt_wav_path": str(prompt_audio_path),
            "prompt_text": request.prompt_text,
            "generated_text": request.generated_text,
            "edit_type": request.edit_type,
            "edit_info": request.edit_info,
            "model_path": str(STEP_AUDIO_EDITX_MODEL_DIR),
            "tokenizer_path": str(STEP_AUDIO_TOKENIZER_PATH),
            "code_path": str(STEP_AUDIO_EDITX_CODE_PATH),
            "dtype": STEP_AUDIO_EDITX_DTYPE,
            "max_model_len": STEP_AUDIO_EDITX_MAX_MODEL_LEN,
            "gpu_memory_utilization": STEP_AUDIO_EDITX_GPU_MEMORY_UTILIZATION,
            "max_num_seqs": STEP_AUDIO_EDITX_MAX_NUM_SEQS,
            "cosyvoice_dtype": STEP_AUDIO_EDITX_COSYVOICE_DTYPE,
            "enforce_eager": STEP_AUDIO_EDITX_ENFORCE_EAGER,
            "cosyvoice_cuda_graph": STEP_AUDIO_EDITX_COSYVOICE_CUDA_GRAPH,
            "local_files_only": local_files_only,
        }

    def run_worker(self, payload: dict) -> bytes:
        try:
            audio = run_local_worker(payload, STEP_AUDIO_EDITX_WORKER)
            self.last_error = None
            return audio
        except Exception as exc:
            self.last_error = str(exc)
            raise


manager = StepAudioEditXWorkerManager()


def step_audio_editx_is_ready() -> bool:
    """无需加载模型即可判断本机 EditX 所需的文件是否已配置。"""

    return (
        STEP_AUDIO_EDITX_MODEL_DIR.is_dir()
        and STEP_AUDIO_TOKENIZER_PATH.is_dir()
        and (STEP_AUDIO_EDITX_CODE_PATH / "tts.py").is_file()
        and (STEP_AUDIO_EDITX_CODE_PATH / "tokenizer.py").is_file()
        and STEP_AUDIO_EDITX_WORKER_SCRIPT.is_file()
    )
