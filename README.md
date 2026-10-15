# Kandinsky 5.0 T2I Lite — CUDA (Windows)

Каждый новый `python generate.py -p ...` **заново грузит веса** (~десятки ГБ с диска в VRAM) — это и есть главный тормоз.

**Держи процесс живым:** interactive CLI или FastAPI.

## Установка

```bat
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

```bat
huggingface-cli download kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers --local-dir .\models\t2i-lite
```

## Вариант 1 — interactive (веса 1 раз)

```bat
python generate.py --interactive --width 640 --height 1408 --steps 16
```

Потом вводишь промпты; `quit` — выход. Загрузка только при старте.

## Вариант 2 — FastAPI (веса 1 раз, HTTP)

```bat
uvicorn main:app --host 0.0.0.0 --port 8000
```

Offload **выключен по умолчанию**. Включать только при OOM: `set OFFLOAD=1`.

```bat
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{\"prompt\":\"кот в красной шапке\",\"width\":640,\"height\":1408,\"steps\":16}" --output out.png
```

Второй и следующие запросы — уже без перезагрузки модели. Смотри заголовок `X-Gen-Seconds`.

## Разрешения (только эти)

`640×1408`, `1408×640`, `768×1280`, `1280×768`, `896×1152`, `1152×896`, `1024×1024`

## One-shot CLI (медленно на каждый запуск)

```bat
python generate.py -p "кот в красной шапке" --width 640 --height 1408 --steps 16
```
