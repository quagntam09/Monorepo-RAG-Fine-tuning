# Deploy

Thu muc nay chua Docker topology cho he thong RAG:

- `indexer`: job build FAISS index offline tu PDF trong `paper/`.
- `reader`: FastAPI service phuc vu ONNX reader tren port `8081`.
- `synthesis`: FastAPI service retrieve context, goi reader service va Ollama LLM tren port `8080`.

## 1. Chuan bi

Tu thu muc goc repo:

```bash
cp .env.deploy.example .env.deploy
```

Can co san cac thu muc local:

```text
paper/                       PDF dau vao
artifacts/readers/run_best/   ONNX reader artifact
.cache/faiss/                 FAISS index, co the build lai bang indexer
```

Neu artifact/cache duoc luu tren object storage:

```bash
ARTIFACT_BUCKET=<bucket> scripts/fetch_artifacts.sh
```

## 2. Cau hinh deploy

File `deploy/docker-compose.yml` dang dung `network_mode: "host"`.

Luu y:

- `reader` mac dinh lang nghe `8081`.
- `synthesis` mac dinh lang nghe `8080`.
- Compose override `RAG_READER_SERVICE_URL=http://localhost:8081` cho `synthesis`.
- May host can truy cap duoc Ollama tai `http://localhost:11434` hoac gia tri `RAG_OLLAMA_BASE_URL`/`OLLAMA_HOST`.

Bien moi truong quan trong:

```bash
RAG_DATA_DIR=./paper
RAG_FAISS_INDEX_DIR=./.cache/faiss
RAG_READER_ARTIFACT_DIR=./artifacts/readers/run_best
RAG_LLM_MODEL=qwen2.5:0.5b
RAG_EMBEDDING_MODEL=bge-m3
RAG_READER_SERVICE_URL=http://localhost:8081
```

## 3. Build index

Chay indexer job:

```bash
docker compose -f deploy/docker-compose.yml --profile jobs run --rm indexer
```

Job nay doc PDF tu `paper/`, tao chunk, build/load FAISS va ghi manifest trong `.cache/faiss/`.

## 4. Start services

```bash
docker compose -f deploy/docker-compose.yml up --build reader synthesis
```

Kiem tra:

```bash
curl http://localhost:8081/healthz
curl http://localhost:8080/healthz
```

Goi API:

```bash
curl -X POST http://localhost:8080/v1/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"RAG la gi?","chat_history":""}'
```

## 5. Artifact policy

- Khong commit PDF, checkpoint, ONNX model, FAISS index va file `.env` that.
- Luu file lon bang object storage hoac Git LFS.
- Repo chi nen commit code, cau hinh mau, manifest/metadata nho va eval set nho.

## 6. Troubleshooting ngan

- Loi khong thay PDF: kiem tra `paper/` va `RAG_FILE_GLOB`.
- Loi embedding/LLM: kiem tra Ollama dang chay va da pull `bge-m3`, `qwen2.5:0.5b`.
- Loi reader artifact: kiem tra `artifacts/readers/run_best/` co tokenizer, metadata va ONNX model.
- Synthesis khong goi duoc reader: kiem tra `curl http://localhost:8081/healthz` tren host.
- Ket qua retrieve cu: xoa `.cache/faiss/` roi chay lai indexer.
