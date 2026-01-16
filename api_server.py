# -*- coding: utf-8 -*-
import base64
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from model import FunASRNano


class AppConfig(BaseModel):
    model_dir: str = "FunAudioLLM/Fun-ASR-Nano-2512"
    vllm_model_dir: Optional[str] = None
    device: str = "cuda:0"
    gpu_memory_utilization: float = 0.4
    vllm_max_tokens: int = 500
    vllm_top_p: float = 0.001
    vllm_max_model_len: int = 8192  # 限制模型最大序列长度


class AsrRequest(BaseModel):
    audio_b64: str
    audio_format: Optional[str] = None
    language: Optional[str] = None
    itn: bool = True
    hotwords: List[str] = []


app = FastAPI(title="Fun-ASR vLLM API")
app_state = {}


def _read_config() -> AppConfig:
    return AppConfig(
        model_dir=os.getenv("FUNASR_MODEL_DIR", "FunAudioLLM/Fun-ASR-Nano-2512"),
        vllm_model_dir=os.getenv("FUNASR_VLLM_MODEL_DIR"),
        device=os.getenv("FUNASR_DEVICE", "cuda:0"),
        gpu_memory_utilization=float(
            os.getenv("FUNASR_VLLM_GPU_MEM", "0.4")
        ),
        vllm_max_tokens=int(os.getenv("FUNASR_VLLM_MAX_TOKENS", "500")),
        vllm_top_p=float(os.getenv("FUNASR_VLLM_TOP_P", "0.001")),
        vllm_max_model_len=int(os.getenv("FUNASR_VLLM_MAX_MODEL_LEN", "8192")),
    )


def _save_temp_bytes(content: bytes, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _get_audio_duration_sec(audio_path: Path) -> Optional[float]:
    try:
        info = torchaudio.info(str(audio_path))
        if info.num_frames and info.sample_rate:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        return None
    return None


def _sanitize_meta(meta: dict) -> dict:
    """Remove non-serializable values from meta_data"""
    import torch
    result = {}
    for k, v in meta.items():
        if isinstance(v, torch.Tensor):
            continue  # skip tensors
        if isinstance(v, (str, int, float, bool, type(None))):
            result[k] = v
        elif isinstance(v, (list, tuple)):
            result[k] = [x for x in v if not isinstance(x, torch.Tensor)]
        elif isinstance(v, dict):
            result[k] = _sanitize_meta(v)
    return result


def _run_inference(
    audio_path: Path,
    language: Optional[str],
    itn: bool,
    hotwords: List[str],
) -> dict:
    model = app_state["model"]
    base_kwargs = app_state["base_kwargs"]
    start = time.perf_counter()
    results, meta_data = model.inference(
        data_in=[str(audio_path)],
        language=language,
        itn=itn,
        hotwords=hotwords,
        **base_kwargs,
    )
    elapsed = time.perf_counter() - start
    duration = _get_audio_duration_sec(audio_path)
    rtf = elapsed / duration if duration else None
    first = results[0] if results else {}
    return {
        "text": first.get("text", ""),
        "text_tn": first.get("text_tn", ""),
        "label": first.get("label", ""),
        "latency_sec": elapsed,
        "audio_duration_sec": duration,
        "rtf": rtf,
        "meta": _sanitize_meta(meta_data),
    }


@app.on_event("startup")
def _startup() -> None:
    config = _read_config()
    model, kwargs = FunASRNano.from_pretrained(
        model=config.model_dir,
        device=config.device,
    )
    model.eval()

    if config.vllm_model_dir:
        from vllm import LLM, SamplingParams

        vllm = LLM(
            model=config.vllm_model_dir,
            enable_prompt_embeds=True,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.vllm_max_model_len,
            dtype="bfloat16",
        )
        sampling_params = SamplingParams(
            top_p=config.vllm_top_p,
            max_tokens=config.vllm_max_tokens,
        )
        model.vllm = vllm
        model.vllm_sampling_params = sampling_params

    app_state["model"] = model
    app_state["base_kwargs"] = kwargs
    app_state["config"] = config


@app.get("/health")
def health() -> dict:
    config: AppConfig = app_state.get("config")
    return {
        "status": "ok",
        "model_dir": getattr(config, "model_dir", None),
        "vllm_model_dir": getattr(config, "vllm_model_dir", None),
    }


@app.post("/asr")
def asr_json(payload: AsrRequest) -> dict:
    if not payload.audio_b64:
        raise HTTPException(status_code=400, detail="audio_b64 is required")
    try:
        audio_bytes = base64.b64decode(payload.audio_b64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64: {exc}") from exc

    suffix = ".wav"
    if payload.audio_format:
        suffix = f".{payload.audio_format.lstrip('.')}"

    tmp_path = _save_temp_bytes(audio_bytes, suffix)
    try:
        return _run_inference(
            tmp_path, payload.language, payload.itn, payload.hotwords
        )
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


@app.post("/asr/file")
def asr_file(
    file: UploadFile = File(...),
    language: Optional[str] = Form(default=None),
    itn: bool = Form(default=True),
    hotwords: Optional[str] = Form(default=None),
) -> dict:
    if not file:
        raise HTTPException(status_code=400, detail="file is required")
    content = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="empty file")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp_path = _save_temp_bytes(content, suffix)
    try:
        hotword_list = (
            [w.strip() for w in hotwords.split(",") if w.strip()]
            if hotwords
            else []
        )
        return _run_inference(tmp_path, language, itn, hotword_list)
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
