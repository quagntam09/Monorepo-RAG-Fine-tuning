# Monorepo RAG + Fine-tuning

## Topology

- `indexer job`: build FAISS offline
- `reader service`: ONNX reader with micro-batching
- `llm synthesis service`: retrieval + reader + final generation

## Setup

```bash
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

## Run

- Train: `rag-ft-train --config config/defaults.yaml`
- Export: `rag-ft-export --config config/defaults.yaml --checkpoint-dir outputs/checkpoints/best_model --artifact-dir artifacts/readers/run_best`
- Index: `rag-indexer`
- Reader API: `rag-reader-service`
- Synthesis API: `rag-synthesis-service`
- Eval reader: `rag-eval --mode reader`
- Eval rag: `rag-eval --mode rag`

## Deploy

- See `deploy/README.md`
- Large PDFs/checkpoints/models should live in object storage or Git LFS
- Repo keeps only manifests and small metadata
