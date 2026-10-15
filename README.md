# Kandinsky 5.0 T2I Lite — CUDA (Windows)

Только **NVIDIA CUDA**. CLI: промпт → PNG.

## Установка

```bat
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

Проверка:

```bat
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Нужно: `2.7.x+cu124` и `True`.

## Модель

```bat
huggingface-cli download kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers --local-dir .\models\t2i-lite
```

## Запуск (RTX 4070 16GB — быстро)

Разрешения **только** из списка (512 / 768×768 нет):

| size | заметка |
|------|---------|
| **640×1408** / **1408×640** | меньше пикселей → быстрее |
| 768×1280 / 1280×768 | |
| 896×1152 / 1152×896 | |
| 1024×1024 | квадрат |

```bat
python generate.py --model .\models\t2i-lite -p "кот в красной шапке" --width 640 --height 1408 --steps 16
```

По умолчанию уже: `640×1408`, `steps=16`, **без** offload и attention-slicing.

Квадрат:

```bat
python generate.py --model .\models\t2i-lite -p "..." --width 1024 --height 1024 --steps 20
```

Если OOM:

```bat
python generate.py ... --offload
```
