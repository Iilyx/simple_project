# Kandinsky 5.0 T2I — FastAPI (Windows + CUDA)

`POST /generate` — промпт → PNG. Модель: `models/t2i-lite`. Только NVIDIA CUDA.

## Установка

```bat
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

```bat
huggingface-cli download kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers --local-dir .\models\t2i-lite
```

Проверка GPU:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Нужно: `2.7.x+cu124` и `True`.

## Запуск (быстро, RTX 4070 16GB)

```bat
set OFFLOAD=0
uvicorn main:app --host 0.0.0.0 --port 8000
```

Быстрый запрос (768, 20 steps):

```bat
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{\"prompt\":\"кот в красной шапке\",\"width\":768,\"height\":768,\"steps\":20}" --output out.png
```

Ещё быстрее: `"width":640,"height":640,"steps":16`.

Опционально `torch.compile` (первый старт дольше, потом быстрее):

```bat
set OFFLOAD=0
set COMPILE=1
uvicorn main:app --host 0.0.0.0 --port 8000
```

Если OOM — только тогда:

```bat
set OFFLOAD=1
```

**Что тормозит сильнее всего:** `OFFLOAD=1`, `steps=50`, `1024x1024`, холодный первый запуск.

## API

```bat
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{\"prompt\":\"кот в красной шапке\"}" --output out.png
```

```json
{
  "prompt": "кот в красной шапке",
  "width": 768,
  "height": 768,
  "steps": 28,
  "guidance": 3.5,
  "seed": 42
}
```

`GET /health` — CUDA / имя GPU.
