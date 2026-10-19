# FLUX.2 klein 4B, локально

Генерация и редактирование на NVIDIA GPU. Окружение собирает [uv](https://docs.astral.sh/uv/). Веса [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) скачиваются на ту машину, где запускается скрипт, и в git не попадают. Лицензия весов — Apache 2.0.

Нужны драйвер NVIDIA и uv. На RTX 4070 Ti Super (16 ГБ) модель помещается примерно в 13 ГБ. PyTorch берётся с канала CUDA 12.8.

## Подготовка

На компьютере с видеокартой:

```bash
git clone https://github.com/Iilyx/simple_project.git
cd simple_project
uv run setup.py
```

Если uv ещё нет: https://docs.astral.sh/uv/getting-started/installation/

`setup.py` делает `uv sync` (создаёт `.venv` и ставит зависимости) и скачивает модель в `models/FLUX.2-klein-4B` (около 24 ГБ). Другой канал PyTorch: `uv run setup.py --cuda cu126`.

## Генерация

```bash
uv run generate.py --prompt "a red fox in an autumn forest, golden hour"
```

Картинка пишется в `output/`. Klein дистиллирован: 4 шага и guidance 1.0.

## Редактирование

```bash
uv run generate.py --image input.png --prompt "change the background to snow, keep the subject"
```

Несколько референсов: повтори `--image`. Если 16 ГБ не хватает под другие процессы, добавь `--offload`.
