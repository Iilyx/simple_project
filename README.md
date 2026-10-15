# Kandinsky 5.0 Image Lite (local)

Локальная генерация картинок через Diffusers:  
`kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers`

## Требования

- Python **3.11+**
- [uv](https://docs.astral.sh/uv/)
- Mac (MPS) / Windows+NVIDIA (CUDA) / CPU

## Установка uv

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Проверка:

```bash
uv --version
```

## Окружение и зависимости

```bash
git clone https://github.com/Iilyx/simple_project.git
cd simple_project

uv venv --python 3.11
```

Активация:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Установка пакетов:

```bash
uv pip install -r requirements.txt
```

## Скачать модель

```bash
huggingface-cli download kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers \
  --local-dir ./models/t2i-lite
```

## Запуск

```bash
python generate.py --model ./models/t2i-lite -p "кот в красной шапке"
```

Картинка сохранится в `outputs/`.

### Полезные флаги

```bash
# Mac ~24 ГБ RAM — меньше разрешение
python generate.py --model ./models/t2i-lite -p "..." --width 512 --height 512

# Windows + NVIDIA с запасом VRAM
python generate.py --model ./models/t2i-lite -p "..." --device cuda --width 1024 --height 1024 --no-offload
```

| Флаг | По умолчанию | Зачем |
|------|--------------|--------|
| `--device` | `auto` | `mps` / `cuda` / `cpu` |
| `--offload` | включён | экономия памяти (на Mac почти всегда нужен) |
| `--width` / `--height` | `768` | на слабой машине ставь `512` |
| `--steps` | `50` | качество / скорость |
| `--seed` | `42` | воспроизводимость |
