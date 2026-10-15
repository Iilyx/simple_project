# Kandinsky 5.0 T2I Lite — локально (CUDA)

Тот же подход, что в [карточке модели](https://huggingface.co/kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers):

```python
pipe = Kandinsky5T2IPipeline.from_pretrained(path, torch_dtype=torch.float16, local_files_only=True)
pipe.to(device="cuda", dtype=torch.float16)
```

`bfloat16` из карточки HF у тебя даёт `mat1 Float / mat2 BFloat16` (Qwen), поэтому **float16**.

## Установка

```bat
uv venv --python 3.11
.venv\Scripts\activate
uv pip install -r requirements.txt
```

```bat
huggingface-cli download kandinskylab/Kandinsky-5.0-T2I-Lite-sft-Diffusers --local-dir .\models\t2i-lite
```

## FastAPI

```bat
uvicorn main:app --host 0.0.0.0 --port 8000
```

```bat
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d "{\"prompt\":\"A cat in a red hat\"}" --output out.png
```

Дефолты в `.env`: `1024×1024`, `steps=50`, `guidance=3.5`, `OFFLOAD=0`.

## CLI

```bat
python generate.py -p "A cat in a red hat"
```
