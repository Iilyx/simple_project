# FLUX.2 klein 4B, локально

Генерация и редактирование на NVIDIA GPU. Веса [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) скачиваются на ту машину, где запускается скрипт, и в git не попадают. Лицензия весов — Apache 2.0.

Нужны Python 3.10+ и драйвер NVIDIA. На RTX 4070 Ti Super (16 ГБ) модель помещается примерно в 13 ГБ.

## Подготовка

На компьютере с видеокартой:

```bash
git clone https://github.com/Iilyx/simple_project.git
cd simple_project
python setup.py
```

`setup.py` создаёт `.venv`, ставит PyTorch с CUDA 12.8 и скачивает модель в `models/FLUX.2-klein-4B` (около 24 ГБ). Если нужен другой канал PyTorch: `python setup.py --cuda cu126`.

## Генерация

Windows:

```bat
.venv\Scripts\python generate.py --prompt "a red fox in an autumn forest, golden hour"
```

Linux:

```bash
.venv/bin/python generate.py --prompt "a red fox in an autumn forest, golden hour"
```

Картинка пишется в `output/`. Klein дистиллирован: 4 шага и guidance 1.0.

## Редактирование

```bash
.venv/bin/python generate.py --image input.png --prompt "change the background to snow, keep the subject"
```

Несколько референсов: повтори `--image`. Если 16 ГБ не хватает под другие процессы, добавь `--offload`.
