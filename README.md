# FLUX.2 klein 4B

Скрипт `generate.py` — пример с [Hugging Face](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B). Веса уже лежат в `models/FLUX.2-klein-4B`, скрипт их не скачивает. Нужны [uv](https://docs.astral.sh/uv/), Python 3.12 и NVIDIA GPU. Картинка пишется в `flux-klein.png`.

## Windows

```bat
uv venv --python 3.12 && uv pip install --system-certs -r requirements.txt
.venv\Scripts\python generate.py "A cat holding a sign that says hello world"
```

## Linux

```bash
uv venv --python 3.12 && uv pip install --system-certs -r requirements.txt
.venv/bin/python generate.py "A cat holding a sign that says hello world"
```
