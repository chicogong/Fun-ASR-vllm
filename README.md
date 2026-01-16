# Fun-ASR vLLM Acceleration

This repository provides an accelerated implementation of [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) using [vLLM](https://github.com/vllm-project/vllm). By leveraging vLLM's efficient attention mechanisms and memory management, this project significantly boosts the inference performance of Fun-ASR models while maintaining accuracy.

## Environment Setup 🐍

To get started, clone the repository and install the required dependencies:

```shell
git clone https://github.com/yuekaizhang/Fun-ASR-vllm.git
cd Fun-ASR-vllm
apt-get install -y ffmpeg
uv pip install -r requirements.txt
```

<a name="usage-tutorial"></a>

## Features 📝
- Support vLLM
- Support batch > 1 Inference
- Support [FunAudioLLM/Fun-ASR-MLT-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512) and [FunAudioLLM/Fun-ASR-Nano-2512](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512)
- Integration with [NVIDIA Triton Inference Server](./triton_server/)

## Usage 🛠️

### Python API Inference

You can run inference directly using the Python API:

```python
from model import FunASRNano
from vllm import LLM, SamplingParams

def main():
    model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"
    # Load the base model
    m, kwargs = FunASRNano.from_pretrained(model=model_dir, device="cuda:0")
    m.eval()
    
    # Initialize vLLM
    vllm = LLM(model="yuekai/Fun-ASR-Nano-2512-vllm", enable_prompt_embeds=True, gpu_memory_utilization=0.4)
    sampling_params = SamplingParams(
        top_p=0.001,
        max_tokens=500,
    )
    
    # Attach vLLM to the model
    m.vllm = vllm
    m.vllm_sampling_params = sampling_params

    # Run inference
    wav_path = f"{kwargs['model_path']}/example/zh.mp3"
    res = m.inference(data_in=[wav_path], **kwargs)
    print(res)
    text = res[0][0]["text"]
    print(text)


if __name__ == "__main__":
    main()
```

### External API (FastAPI)

We provide a lightweight HTTP API in `api_server.py` for external callers.

Start the server:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8080
```

Environment variables:

- `FUNASR_MODEL_DIR` (default: `FunAudioLLM/Fun-ASR-Nano-2512`)
- `FUNASR_VLLM_MODEL_DIR` (optional, enable vLLM)
- `FUNASR_DEVICE` (default: `cuda:0`)
- `FUNASR_VLLM_GPU_MEM` (default: `0.4`)
- `FUNASR_VLLM_MAX_TOKENS` (default: `500`)
- `FUNASR_VLLM_TOP_P` (default: `0.001`)

Example (multipart upload):

```bash
curl -X POST "http://localhost:8080/asr/file" \
  -F "file=@/path/to/audio.wav" \
  -F "language=中文" \
  -F "itn=true" \
  -F "hotwords=腾讯,语音"
```

Example (base64 JSON):

```bash
python - <<'PY'
import base64, json, requests
audio = open("/path/to/audio.wav","rb").read()
payload = {
  "audio_b64": base64.b64encode(audio).decode(),
  "audio_format": "wav",
  "language": "中文",
  "itn": True,
  "hotwords": ["腾讯","语音"]
}
print(requests.post("http://localhost:8080/asr", json=payload).json())
PY
```


For multilingual [FunAudioLLM/Fun-ASR-MLT-Nano-2512](http://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512):

```python
from model import FunASRNano
from vllm import LLM, SamplingParams

def main():
    model_dir = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
    # Load the base model
    m, kwargs = FunASRNano.from_pretrained(model=model_dir, device="cuda:0")
    m.eval()
    
    # Initialize vLLM
    vllm = LLM(model="yuekai/Fun-ASR-MLT-Nano-2512-vllm", enable_prompt_embeds=True, gpu_memory_utilization=0.4)
    sampling_params = SamplingParams(
        top_p=0.001,
        max_tokens=500,
    )
    
    # Attach vLLM to the model
    m.vllm = vllm
    m.vllm_sampling_params = sampling_params

    # Run inference
    wav_path = f"{kwargs['model_path']}/example/en.mp3"
    # 中文、英文、日文 for Fun-ASR-Nano-2512
    # 中文、英文、粤语、日文、韩文、越南语、印尼语、泰语、马来语、菲律宾语、阿拉伯语、
    # 印地语、保加利亚语、克罗地亚语、捷克语、丹麦语、荷兰语、爱沙尼亚语、芬兰语、希腊语、
    # 匈牙利语、爱尔兰语、拉脱维亚语、立陶宛语、马耳他语、波兰语、葡萄牙语、罗马尼亚语、
    # 斯洛伐克语、斯洛文尼亚语、瑞典语 for Fun-ASR-MLT-Nano-2512
    res = m.inference(data_in=[wav_path], language="英文", **kwargs)
    print(res)
    text = res[0][0]["text"]
    print(text)


if __name__ == "__main__":
    main()
```

### Running Benchmarks

To evaluate performance on a dataset (e.g., SpeechIO):

```bash
dataset_name="yuekai/speechio"
subset_name="SPEECHIO_ASR_ZH00007"
split_name="test"

uv run python \
    infer.py \
    --model_dir FunAudioLLM/Fun-ASR-Nano-2512 \
    --huggingface_dataset $dataset_name \
    --subset_name $subset_name \
    --split_name $split_name \
    --batch_size 16 \
    --log_dir ./logs_vllm_$dataset_name_$subset_name \
    --vllm_model_dir yuekai/Fun-ASR-Nano-2512-vllm
```

To evaluate multilingual performance of FunAudioLLM/Fun-ASR-MLT-Nano-2512:

```bash
dataset_name="google/fleurs"
subset_name="en_us"
split_name="test"

uv run python \
    infer.py \
    --model_dir FunAudioLLM/Fun-ASR-MLT-Nano-2512 \
    --huggingface_dataset $dataset_name \
    --subset_name $subset_name \
    --split_name $split_name \
    --batch_size 16 \
    --language "英文" \
    --log_dir ./logs_mlt_${batch_size}_${dataset_name}_${subset_name} \
    --vllm_model_dir yuekai/Fun-ASR-MLT-Nano-2512-vllm
```

## Performance 🚀

We compared the performance of the standard HuggingFace PyTorch implementation against our vLLM-accelerated version.

**Benchmark Details:**
- **Dataset:** [SPEECHIO_ASR_ZH00007](https://github.com/SpeechColab/Leaderboard) (approx. 1 hour of audio)
- **Hardware:** Single NVIDIA H20 GPU

| Mode | Decoding Time | RTF | RTFx | CER | Note |
|------|---------------|-----|------|-----|------|
| Huggingface PyTorch | 211.40 Secs | 0.0587 | 17.03 | 7.02% | batch_size=1 |
| Huggingface PyTorch | 41.6 Secs | 0.0116 | 86.54 | 8.53% | batch_size=16 |
| vLLM (Qwen3-0.6B) | 132.78 Secs | 0.0369 | 27.11 | 6.99% | batch_size=1 |
| **vLLM (Qwen3-0.6B)** | **19.9 Secs** | **0.0055** | **180.90** | **7.03%** | batch_size=16 |

*Note: RTF (Real Time Factor) - lower is better; RTFx (Speedup factor) - higher is better.*

## Triton Inference Server Deployment 🚀

For production deployment with high concurrency, we provide integration with NVIDIA Triton Inference Server.

### Quick Start

```bash
cd triton_server

# Using Docker Compose (recommended)
docker compose up

# Or pull and run the pre-built image
docker pull soar97/triton-fun-asr:25.06
```

### Triton Performance
- **Dataset:** [SPEECHIO_ASR_ZH00007](https://github.com/SpeechColab/Leaderboard) (approx. 1 hour of audio)
- **Hardware:** Single NVIDIA H20 GPU

| Concurrency | CER | Processing Time | P50 Latency | RTF |
|-------------|-----|-----------------|-------------|-----|
| 8 | 7.04% | 44.56s | 450.99ms | 0.0126 |
| 16 | 7.00% | 27.96s | 533.36ms | 0.0079 |
| 32 | 7.07% | 24.51s | 952.93ms | 0.0069 |

For detailed setup instructions, see [triton_server/README.md](./triton_server/README.md).
