# Kandinsky 5.0 T2I Lite — CUDA (Windows)

Каждый новый `python generate.py -p ...` **заново грузит веса** — главный тормоз.  
Держи процесс живым: `--interactive` или FastAPI. Настройки в `.env`.

## Установка

```bat
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

```bat
huggingface-cli download kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers --local-dir .\models\t2i-lite
```

## FastAPI (рекомендуется)

```bat
uvicorn main:app --host 0.0.0.0 --port 8000
```

```bat
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{\"prompt\":\"кот в красной шапке\"}" --output out.png
```

Дефолт из `.env`: **1024×1024**, `steps=16`, `OFFLOAD=0`.

Картинка также сохраняется в `outputs/k5_reqN_..._WxH.png`.  
В логе смотри `[req #N]` — если за один клик три `#1 #2 #3`, клиент шлёт три запроса.

`640×1408` — очень высокий кадр; модель часто рисует «комикс» из нескольких сцен в **одном** PNG. Для одной сцены используй `1024×1024`.

## Interactive CLI

```bat
python generate.py --interactive
```

## Разрешения

`1024×1024`, `640×1408`, `1408×640`, `768×1280`, `1280×768`, `896×1152`, `1152×896`
