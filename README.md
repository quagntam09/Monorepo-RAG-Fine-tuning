# Monorepo RAG + Fine-tuning

Monorepo Python cho he thong RAG tren tai lieu PDF va fine-tuning reader model cho extractive question answering.

Du an hien gom 2 phan chinh:

- `training`: fine-tune DistilBERT QA, evaluate checkpoint va export reader artifact ONNX.
- `rag_chatbox`: nap PDF, build FAISS index, retrieve context, goi ONNX reader va dung Ollama LLM de tong hop cau tra loi co citation.

## 1. Yeu cau

- Python `>=3.10` khuyen nghi Python `3.12`.
- Linux/macOS hoac WSL tren Windows.
- Ollama dang chay local neu muon dung chatbox/RAG synthesis.
- Model Ollama mac dinh:
  - Embedding: `bge-m3`
  - LLM: `qwen2.5:0.5b`

## 2. Cai dat tren may local

Tao virtual environment va cai dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Tao file cau hinh moi truong:

```bash
cp .env.example .env
```

Chuan bi Ollama:

```bash
ollama pull bge-m3
ollama pull qwen2.5:0.5b
ollama serve
```

Dat tai lieu PDF vao thu muc:

```text
paper/
```

Mac dinh he thong doc cac file `**/*.pdf` trong `paper/`.

## 3. Cau hinh quan trong

Sua `.env` neu can:

```bash
RAG_DATA_DIR=./paper
RAG_FILE_GLOB=**/*.pdf
RAG_FAISS_INDEX_DIR=./.cache/faiss
RAG_EMBEDDING_MODEL=bge-m3
RAG_LLM_MODEL=qwen2.5:0.5b
RAG_TOP_K=5
RAG_FETCH_K=20
RAG_SCORE_THRESHOLD=0.35
RAG_READER_ARTIFACT_DIR=./artifacts/readers/run_best
RAG_READER_SERVICE_URL=http://localhost:8081
```

Artifact reader can co trong `artifacts/readers/run_best/`:

- `model.onnx` hoac `model_quantized.onnx`
- `model_metadata.json`
- `tokenizer.json`
- `tokenizer_config.json`

PDF, checkpoint, FAISS cache va model artifact lon khong nen commit vao git.

## 4. Chay RAG local

Build FAISS index offline:

```bash
rag-indexer --print-summary
```

Chay chatbox CLI:

```bash
rag-chatbox
```

Chay voi debug trace:

```bash
RAG_DEBUG_TRACE=true rag-chatbox --debug
```

## 5. Chay services local

Terminal 1, reader service:

```bash
rag-reader-service --host 0.0.0.0 --port 8081
```

Terminal 2, synthesis service:

```bash
rag-synthesis-service --host 0.0.0.0 --port 8080
```

Kiem tra health:

```bash
curl http://localhost:8081/healthz
curl http://localhost:8080/healthz
```

Goi API hoi dap:

```bash
curl -X POST http://localhost:8080/v1/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"RAG la gi?","chat_history":""}'
```

## 6. Fine-tuning reader

Cau hinh mac dinh nam o `config/defaults.yaml`.

Train:

```bash
rag-ft-train --config config/defaults.yaml
```

Evaluate checkpoint:

```bash
rag-ft-eval --config config/defaults.yaml --checkpoint-dir outputs/checkpoints/best_model
```

Export checkpoint sang ONNX artifact:

```bash
rag-ft-export \
  --config config/defaults.yaml \
  --checkpoint-dir outputs/checkpoints/best_model \
  --artifact-dir artifacts/readers/run_best
```

## 7. Evaluation

Eval reader tren `eval/questions.jsonl`:

```bash
rag-eval --mode reader --top-k 5 --limit 20 --output-json eval/reader_eval_latest.json
```

Eval RAG:

```bash
rag-eval --mode rag --top-k 5 --limit 10 --rag-answer-timeout-sec 60 \
  --output-json eval/rag_eval_latest_limited.json
```

Unit tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## 8. Docker deploy

Xem chi tiet trong `deploy/README.md`.

Lenh co ban:

```bash
docker compose -f deploy/docker-compose.yml --profile jobs run --rm indexer
docker compose -f deploy/docker-compose.yml up --build reader synthesis
```

## 9. Cau truc thu muc

```text
config/      Cau hinh training/export
deploy/      Dockerfile, docker-compose va deploy docs
eval/        Eval set va ket qua eval
scripts/     Script sync/fetch artifacts
src/training Fine-tuning, evaluation, ONNX export
src/rag_chatbox RAG ingestion, retrieval, pipeline, services
tests/       Unit tests
paper/       PDF local, bi ignore khoi git
artifacts/   Reader artifacts local, bi ignore khoi git
outputs/     Checkpoints local, bi ignore khoi git
.cache/      FAISS cache local, bi ignore khoi git
```

## 10. Artifact sync

Neu dung S3/object storage:

```bash
ARTIFACT_BUCKET=<bucket> scripts/fetch_artifacts.sh
ARTIFACT_BUCKET=<bucket> scripts/sync_artifacts.sh
```

Dung `fetch_artifacts.sh` truoc khi start service neu may moi chua co `artifacts/`, `.cache/` hoac `paper/`.
