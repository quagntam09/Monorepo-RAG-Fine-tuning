# Monorepo RAG + Fine-tuning

Monorepo nay gom 2 phan:
- `rag-ft-*`: fine-tune DistilBERT extractive QA va export artifact ONNX
- `rag-chatbox` / `rag-eval`: RAG chatbox + danh gia end-to-end

## Cai dat

```bash
cd /home/quagntam/Projects/Monorepo-RAG-Fine-tuning
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

## Workflow chuan

1. Train model QA
2. Export artifact ONNX
3. Chay chatbox
4. Chay eval end-to-end tren bo cau hoi that

## Train

```bash
rag-ft-train --config config/defaults.yaml
```

Output chinh:
- `outputs/checkpoints/best_model/`
- `outputs/checkpoints/training_config.json`

## Eval checkpoint

```bash
rag-ft-eval --config config/defaults.yaml --checkpoint-dir outputs/checkpoints/best_model
```

## Export artifact

```bash
rag-ft-export --config config/defaults.yaml --checkpoint-dir outputs/checkpoints/best_model --artifact-dir artifacts/readers/run_best
```

Artifact export ra se co:
- `model.onnx` hoac `model_quantized.onnx`
- `tokenizer.json`
- `tokenizer_config.json`
- `model_metadata.json`

## Chatbox

`rag-chatbox` hien chay theo luong:
- FAISS retrieval
- DistilBERT ONNX reader
- LLM synthesis cuoi cung
- source filtering trong `Nguồn:`

Trong `.env`, giu:

```env
RAG_READER_ARTIFACT_DIR=./artifacts/readers/run_best
RAG_LLM_MODEL=qwen2.5:0.5b
RAG_QUERY_REWRITE_ENABLED=true
RAG_QUERY_REWRITE_MODEL=qwen2.5:0.5b
RAG_QUERY_REWRITE_MAX_VARIANTS=3
```

Chay:

```bash
rag-chatbox
```

Can co:

```bash
ollama pull bge-m3
ollama pull qwen2.5:0.5b
```

## End-to-end eval

```bash
rag-eval --eval-file eval/questions.jsonl --mode rag --top-k 5
```

`rag-eval` do:
- retrieval hit rate
- reader exact match / F1
- rag exact match / F1
- citation hit rate

Neu muon chi benchmark reader:

```bash
rag-eval --mode reader
```

## Artifact contract

- Train sinh `outputs/checkpoints/best_model/`
- Export sinh `artifacts/readers/run_best/`
- RAG doc qua `RAG_READER_ARTIFACT_DIR`

## Khi nao can train lai?

- Khong can train lai chi vi doi pipeline chatbox
- Chi can export lai neu artifact cu khong hop contract moi hoac ban muon model reader moi hon
- Chi can train lai neu ban muon cap nhat weights QA

## Config

- Train: `config/defaults.yaml`
- Runtime RAG: `.env.example`
- Eval set: `eval/questions.jsonl`
