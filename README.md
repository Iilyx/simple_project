# FLUX.2 klein 4B

Сервер `app.py` — пример с [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B). Веса уже лежат в `models/FLUX.2-klein-4B` и поднимаются один раз при старте. Нужны [uv](https://docs.astral.sh/uv/), Python 3.12 и NVIDIA GPU.

`POST /generate` принимает JSON: `prompt`, `width`, `height`, `steps`, `guidance`, `seed`. `POST /edit` принимает те же поля формой и файл `image` (можно несколько). Ответ — PNG.

## Windows

```bat
uv venv --python 3.12 && uv pip install --system-certs -r requirements.txt
.venv\Scripts\python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

## Linux

```bash
uv venv --python 3.12 && uv pip install --system-certs -r requirements.txt
.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
