# FLUX.2 klein 4B, локально

Генерация и редактирование на NVIDIA GPU. Зависимости ставятся руками через `uv` и `pip`, из `requirements.txt`. Веса [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) скачиваются на ту машину, где выполняются команды, и в git не попадают. Лицензия весов — Apache 2.0.

Нужны драйвер NVIDIA и [uv](https://docs.astral.sh/uv/getting-started/installation/). На RTX 4070 Ti Super (16 ГБ) модель помещается примерно в 13 ГБ.

`pip-system-certs` подключает хранилище сертификатов системы. Его ставят отдельной командой до остальных пакетов.

PyTorch с CUDA ставится последней командой и только с индекса `cu128`. Если поставить его из `requirements.txt` вместе с PyPI, pip выбирает сборку CPU: у версии с PyPI нет суффикса `+cu128`, и она считается новее.

## Windows

```bat
uv venv --python 3.12
.venv\Scripts\python -m pip install pip-system-certs
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv\Scripts\python -c "from huggingface_hub import snapshot_download; snapshot_download('black-forest-labs/FLUX.2-klein-4B', local_dir='models/FLUX.2-klein-4B')"
```

В выводе версии должно быть `+cu128` и `True`. Если версия `+cu128`, а `False`, драйвер NVIDIA старее, чем нужно CUDA 12.8: `nvidia-smi` показывает версию драйвера. Тогда повтори последнюю установку с `cu126` вместо `cu128`.

Если сам PyPI не открывается, пока сертификаты ещё не подключены:

```bat
.venv\Scripts\python -m pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pip-system-certs
```

После этого повтори установку из `requirements.txt`.

Генерация:

```bat
.venv\Scripts\python generate.py --prompt "a red fox in an autumn forest, golden hour"
```

Редактирование:

```bat
.venv\Scripts\python generate.py --image input.png --prompt "change the background to snow, keep the subject"
```

## Linux

```bash
uv venv --python 3.12
.venv/bin/python -m pip install pip-system-certs
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('black-forest-labs/FLUX.2-klein-4B', local_dir='models/FLUX.2-klein-4B')"
.venv/bin/python generate.py --prompt "a red fox in an autumn forest, golden hour"
.venv/bin/python generate.py --image input.png --prompt "change the background to snow, keep the subject"
```

Картинка пишется в `output/`. Klein дистиллирован: 4 шага и guidance 1.0. Несколько референсов: повтори `--image`. Если 16 ГБ не хватает под другие процессы, добавь `--offload`. Модель занимает около 24 ГБ в `models/FLUX.2-klein-4B`.
