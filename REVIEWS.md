# Review du an Monorepo RAG + Fine-tuning

Ngay cap nhat: 2026-05-25

## 1. Tong quan

Du an la mot monorepo Python cho bai toan RAG tren tai lieu PDF ket hop reader model fine-tuned cho extractive question answering.

Muc tieu chinh:

1. Nap PDF tu `paper/`.
2. Tach tai lieu thanh chunk va tao FAISS index.
3. Retrieve cac chunk lien quan den cau hoi.
4. Dung DistilBERT ONNX reader de trich xuat ung vien cau tra loi.
5. Dung Ollama LLM de tong hop cau tra loi cuoi.
6. Gan citation theo source/page nam trong context da retrieve.

Package trong `pyproject.toml`: `rag-fine-tuning-monorepo`.

Python yeu cau: `>=3.10`.

## 2. Kien truc

Topology du an:

- `indexer job`: build FAISS index offline va ghi manifest.
- `reader service`: FastAPI service phuc vu ONNX reader, co micro-batching.
- `llm synthesis service`: retrieve context, goi reader service, goi LLM va tra answer.

Luong RAG:

```text
PDF -> loader -> splitter -> embeddings -> FAISS
Question -> retriever -> chunks -> ONNX reader -> LLM synthesis -> answer + Nguon
```

## 3. Cau truc thu muc

- `src/training/`: training, evaluation va ONNX export cho DistilBERT QA.
- `src/rag_chatbox/`: ingestion, retrieval, RAG pipeline, CLI, eval va services.
- `src/rag_chatbox/services/`: indexer job, reader service, synthesis service.
- `config/defaults.yaml`: cau hinh training/export mac dinh.
- `deploy/`: Dockerfile, docker-compose va huong dan deploy.
- `scripts/`: sync/fetch artifacts tu S3/object storage.
- `eval/`: eval set va ket qua eval.
- `tests/`: unit tests.
- `paper/`: PDF local, bi ignore khoi git.
- `artifacts/`: reader artifacts local, bi ignore khoi git.
- `outputs/`: training checkpoints local, bi ignore khoi git.
- `.cache/`: FAISS cache local, bi ignore khoi git.

## 4. Entry points

Du an khai bao cac command:

- `rag-ft-train`: train DistilBERT QA.
- `rag-ft-eval`: evaluate checkpoint.
- `rag-ft-export`: export checkpoint sang ONNX reader artifact.
- `rag-chatbox`: chay chatbox CLI.
- `rag-eval`: evaluate reader hoac RAG tren eval set.
- `rag-indexer`: build FAISS index.
- `rag-reader-service`: chay reader FastAPI service.
- `rag-synthesis-service`: chay synthesis FastAPI service.

## 5. Training pipeline

Module chinh:

- `config.py`: dataclass `TrainingConfig`, load/save YAML.
- `data_loader.py`: load dataset Hugging Face hoac file local.
- `dataset.py`: tokenize va align answer span.
- `modeling.py`: DistilBERT QA wrapper.
- `qa_head.py`: QA head va post-processing.
- `trainer.py`: training loop, checkpoint va best model.
- `evaluate.py`: evaluate checkpoint.
- `export.py`: export ONNX artifact va metadata.
- `metrics.py`: EM/F1 va SQuAD-style metrics.
- `vietnamese_utils.py`: ho tro xu ly tieng Viet.

Cau hinh mac dinh:

- Base model: `distilbert-base-multilingual-cased`.
- Dataset: `taidng/UIT-ViQuAD2.0`.
- Max length: `384`.
- Doc stride: `128`.
- Batch size: `8`.
- Epochs: `10`.
- Learning rate: `3.0e-5`.
- Mixed precision: `use_amp: true`.
- Best metric: `f1`.
- Checkpoint output: `outputs/checkpoints`.
- Artifact output: `artifacts/readers/run_best`.
- ONNX opset: `14`.

## 6. Reader artifact

Artifact local hien co theo cau truc:

- `artifacts/readers/run_best/model.onnx`
- `artifacts/readers/run_best/model_quantized.onnx`
- `artifacts/readers/run_best/model_metadata.json`
- `artifacts/readers/run_best/tokenizer.json`
- `artifacts/readers/run_best/tokenizer_config.json`

`reader_distilbert.py` dam nhan:

- Validate artifact contract.
- Load ONNX model bang ONNX Runtime.
- Chay answer extraction cho question/context pairs.
- Chon best non-empty answer theo `min_span_score`.

## 7. RAG ingestion va retrieval

Module chinh:

- `ingestion.py`: load PDF bang PyPDFLoader, clean text va split chunk.
- `retrieval.py`: build/load FAISS, manifest cache, threshold filter, fallback, query rewrite va TTL cache.
- `rag_pipeline.py`: build chatbot, goi reader + LLM, finalize answer va citation.
- `prompt_template.py`: prompt dung chung cho CLI/service.

Retrieval hien tai:

- Dung FAISS similarity search voi `fetch_k`.
- Chuan hoa relevance score.
- Tinh overlap score giua question va chunk.
- Rerank theo cong thuc `0.8 * relevance + 0.2 * overlap`.
- Filter bang `RAG_SCORE_THRESHOLD`.
- Neu khong co chunk qua threshold va fallback bat, lay top-k chunk tot nhat theo rerank.
- Co manifest de tranh rebuild FAISS khi documents/chunks/config khong doi.

## 8. Prompt va citation

Prompt yeu cau:

- Chi tra loi dua tren context.
- Neu context khong support thi noi khong biet.
- Uu tien reader candidate neu duoc context support.
- Tra loi cung ngon ngu voi cau hoi.
- Ket thuc bang section `Nguon:`.
- Chi cite source/page co trong retrieved context.

Logic citation:

- Tao danh sach allowed sources tu retrieved docs.
- Chap nhan citation match exact label hoac match theo basename source + page.
- Neu LLM khong cite dung format, fallback gan top allowed sources tu context.
- Neu khong co context, tra ve thong bao khong co trich dan hop le.

## 9. Services

Reader service: `src/rag_chatbox/services/reader_service.py`

- `GET /healthz`
- `POST /v1/reader/answers`
- Load ONNX reader tu `RAG_READER_ARTIFACT_DIR`.
- Micro-batching qua `READER_SERVICE_MAX_BATCH_SIZE` va `READER_SERVICE_BATCH_TIMEOUT_MS`.

Synthesis service: `src/rag_chatbox/services/synthesis_service.py`

- `GET /healthz`
- `POST /v1/chat/ask`
- Load documents va FAISS retriever luc startup.
- Goi reader service qua `RAG_READER_SERVICE_URL`.
- Goi Ollama LLM de tong hop cau tra loi.
- Tra ve `answer`, `retrieval_sources` va `debug`.

## 10. Docker va deploy

`deploy/docker-compose.yml` hien khai bao:

- `indexer`: profile `jobs`, mount `paper/`, `.cache/`, `outputs/`.
- `reader`: mount `artifacts/`, dung host network, port mac dinh `8081`.
- `synthesis`: mount `paper/`, `.cache/`, dung host network, port mac dinh `8080`.

Dockerfiles dung `python:3.12-slim`, install `requirements.txt` va package editable.

Lenh chinh:

```bash
docker compose -f deploy/docker-compose.yml --profile jobs run --rm indexer
docker compose -f deploy/docker-compose.yml up --build reader synthesis
```

## 11. Evaluation hien co

Eval set:

- `eval/questions.jsonl`

Reader eval gan nhat tu `eval/reader_eval_latest.json`:

```json
{
  "samples": 20,
  "requested_samples": 20,
  "mode": "reader",
  "top_k": 5,
  "truncated_by_runtime": false,
  "retrieval_hit_rate": 0.4,
  "reader_exact_match": 0.0,
  "reader_f1": 0.07915067658244529
}
```

Limited RAG eval tu `eval/rag_eval_latest_limited.json`:

```json
{
  "samples": 10,
  "requested_samples": 10,
  "mode": "rag",
  "top_k": 5,
  "truncated_by_runtime": false,
  "retrieval_hit_rate": 0.6,
  "reader_exact_match": 0.0,
  "reader_f1": 0.13580462220706124,
  "rag_exact_match": 0.0,
  "rag_f1": 0.12238194451194465,
  "citation_hit_rate": 0.5,
  "rag_timeout_count": 1
}
```

Full RAG eval tu `eval/rag_eval_latest.json`:

```json
{
  "samples": 20,
  "requested_samples": 20,
  "mode": "rag",
  "top_k": 5,
  "truncated_by_runtime": false,
  "retrieval_hit_rate": 0.45,
  "reader_exact_match": 0.0,
  "reader_f1": 0.11183835488007979,
  "rag_exact_match": 0.0,
  "rag_f1": 0.13711499075247408,
  "citation_hit_rate": 0.3,
  "rag_timeout_count": 0
}
```

Nhan xet dung theo metric hien co:

- Retrieval hit-rate con thap, tu `0.4` den `0.6` tuy eval mode/limit.
- Reader EM va RAG EM deu `0.0`, nghia la cau tra loi chua khop exact expected answer.
- F1 con thap, chat luong answer can tiep tuc cai thien.
- Citation hit-rate co ket qua nhung chua on dinh: `0.5` tren limited 10 mau va `0.3` tren full 20 mau.
- Limited RAG eval co 1 timeout, full RAG eval gan nhat khong timeout.

## 12. Tests va CI

Test files hien co:

- `tests/test_training_pipeline.py`
- `tests/test_export_artifact.py`
- `tests/test_rag_reader.py`
- `tests/test_retrieval.py`
- `tests/test_rag_pipeline.py`
- `tests/test_eval_eval_set.py`
- `tests/test_services_smoke.py`

CI:

- `.github/workflows/ci.yml`
  - Install dependencies.
  - Chay reader evaluation limit 5.
  - Validate reader artifact contract.
  - Chay unit tests bang `python -m unittest discover -s tests -p 'test_*.py'`.
- `.github/workflows/nightly.yml`
  - Chay RAG eval limit 10 theo lich 02:00 UTC hoac manual dispatch.

Lan cap nhat review nay chi sua tai lieu, chua chay lai test.

## 13. Trang thai git tai thoi diem cap nhat

`git status --short` truoc khi sua tai lieu cho thay co san cac thay doi chua commit:

- Modified: `deploy/docker-compose.yml`
- Modified: `eval/rag_eval_latest.json`
- Modified: `src/rag_chatbox/config.py`
- Modified: `src/rag_chatbox/prompt_template.py`
- Modified: `src/rag_chatbox/rag_pipeline.py`
- Modified: `src/rag_chatbox/retrieval.py`
- Modified: `src/rag_chatbox/services/synthesis_service.py`
- Added/untracked: `REVIEWS.md`
- Added/untracked: `eval/test_eval.json`

Can review ky truoc khi commit de khong tron thay doi code, deploy va eval output ngoai y muon.

## 14. Diem manh

- Tach kha ro training, retrieval, reader serving va synthesis serving.
- Co ONNX export va artifact contract cho reader.
- Retrieval co manifest/cache, threshold, fallback va debug trace.
- Citation co allowed-source filtering de giam cite sai context.
- Docker topology phu hop cach tach offline indexer, reader service va synthesis service.
- Co unit tests cho nhieu phan quan trong: training, export artifact, reader, retrieval, RAG pipeline va service import.

## 15. Rui ro va diem can cai thien

- Chat luong answer con thap theo EM/F1 hien co.
- Retrieval hit-rate chua cao, anh huong truc tiep den reader va LLM synthesis.
- RAG co the sinh cau tra loi dai hoac lan man neu context/reader candidate chua tot.
- Eval RAG phu thuoc Ollama local nen toc do va timeout co the khong on dinh giua cac may.
- `.gitignore` ignore `tests/`; neu them test moi co the bi bo sot khi commit.
- CI/eval yeu cau artifact va tai lieu phu hop ton tai trong workspace; may moi can fetch artifact/cache/PDF truoc.
- `.env` va `.env.deploy` la file local, khong nen commit.

## 16. Uu tien tiep theo

1. Chay lai unit tests sau khi on dinh working tree:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

2. Tune retrieval:

- Dieu chinh `RAG_SCORE_THRESHOLD`, `RAG_FETCH_K`, `RAG_TOP_K`.
- So sanh query rewrite bat/tat.
- Doc debug trace cua cac cau fail.

3. Tune reader:

- Thu `RAG_READER_MIN_SPAN_SCORE` cao hon de giam span yeu.
- Kiem tra reader answer tren chunk thuc te tu PDF.
- Danh gia lai checkpoint va artifact sau khi fine-tune.

4. Tune synthesis:

- Rut gon prompt neu answer bi dai.
- Ep answer bam sat reader/context hon.
- Them guardrail khi LLM tra loi khong co citation hop le.

5. Hoan thien deploy:

- Chuan hoa quy trinh fetch artifact/cache truoc khi start service.
- Them healthcheck Docker neu can chay dai han.
- Can nhac tach object storage bootstrap cho `paper/`, `.cache/`, `artifacts/`.
