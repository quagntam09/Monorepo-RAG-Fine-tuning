# Monorepo RAG + Fine-tuning Reader System

Chào mừng bạn đến với **Monorepo Python** tích hợp hệ thống **RAG (Retrieval-Augmented Generation)** nâng cao trên tài liệu PDF kết hợp với mô hình **Reader** được tinh chỉnh (Fine-tune) cục bộ để tối ưu bài toán trích xuất câu trả lời (Extractive Question Answering).

Hệ thống kết hợp sức mạnh tìm kiếm ngữ cảnh của **Retriever (FAISS)**, khả năng trích xuất chính xác của **Reader (DistilBERT ONNX)** và khả năng tổng hợp câu trả lời tự nhiên của **Generator (Ollama LLM)**.

> [!NOTE]
> Để hiểu sâu hơn về kiến trúc hệ thống, các tối ưu hóa toán học/thuật toán giúp tăng F1 Score và sơ đồ luồng dữ liệu chi tiết, vui lòng tham khảo [Tài liệu Hướng dẫn Kỹ thuật Toàn diện](file:///docs/PROJECT_GUIDE.md).

---

## 🚀 Các Tính Năng Nổi Bật

- **Kiến trúc Hybrid RAG + Reader**: Sử dụng mô hình Reader trích xuất span tốt nhất trước khi LLM sinh câu trả lời, tăng tối đa độ chính xác thực tế (Groundedness).
- **Token-Aware Ingestion**: Tách chunk tài liệu dựa trên Tokenizer của mô hình Reader để bảo đảm ngữ cảnh trọn vẹn và không bị tràn giới hạn xử lý.
- **Reranking & Overlap Scoring**: Kết hợp điểm số tương đồng Vector và độ trùng khớp từ khóa (Overlap) giúp tăng độ chính xác của tài liệu được truy xuất.
- **Micro-batching Reader Serving**: Phục vụ mô hình Reader được lượng hóa (Quantized INT8 ONNX) dưới dạng API siêu nhanh với cơ chế gom nhóm xử lý song song.
- **Loop-Breaker & Deduplication**: Thuật toán hậu xử lý triệt tiêu hoàn toàn hiện tượng lặp từ/vòng lặp vô hạn của các LLM siêu nhỏ (0.5B) chạy cục bộ.
- **Hệ Thống Đánh Giá Tự Động (Evaluation Set)**: Tích hợp sẵn bộ câu hỏi vàng 20 mẫu chuẩn SQuAD giúp đo lường tức thì chỉ số EM và F1 của hệ thống.

---

## 🛠️ 1. Yêu Cầu Hệ Thống

- **Hệ điều hành**: Linux (Ubuntu/Debian) hoặc Windows Subsystem for Linux (WSL2 - khuyến nghị).
- **Python**: Phiên bản `>=3.10` (Khuyến nghị `3.12`).
- **Ollama**: Đang hoạt động cục bộ trên máy host (cổng `11434`).
- **Mô hình trên Ollama**:
  - Embedding: `bge-m3`
  - LLM: `qwen2.5:0.5b`

---

## 📥 2. Cài Đặt trên Máy Local

Khởi tạo môi trường ảo Python và cài đặt các thư viện cần thiết:

```bash
# 1. Tạo và kích hoạt môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# 2. Cập nhật pip và cài đặt dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Thiết lập file biến môi trường cục bộ:
```bash
cp .env.example .env
```

Chuẩn bị Ollama:
```bash
ollama pull bge-m3
ollama pull qwen2.5:0.5b
```

Đặt các tài liệu PDF cần hỏi đáp vào thư mục:
```text
paper/
```
*(Mặc định hệ thống sẽ tự động quét toàn bộ các file `**/*.pdf` nằm trong thư mục `paper/`).*

---

## ⚙️ 3. Cấu Hình Quan Trọng (`.env`)

Bạn có thể chỉnh sửa tệp `.env` để tối ưu các tham số của RAG:

```bash
RAG_DATA_DIR=./paper
RAG_FILE_GLOB=**/*.pdf
RAG_FAISS_INDEX_DIR=./.cache/faiss
RAG_EMBEDDING_MODEL=bge-m3
RAG_LLM_MODEL=qwen2.5:0.5b
RAG_TOP_K=5
RAG_FETCH_K=20
RAG_SCORE_THRESHOLD=0.25 # Ngưỡng lọc độ tương đồng tối ưu
RAG_READER_ARTIFACT_DIR=./artifacts/readers/run_best
RAG_READER_SERVICE_URL=http://localhost:8081
```

> [!IMPORTANT]
> Hãy chắc chắn rằng bạn đã có các tệp mô hình Reader trong thư mục `artifacts/readers/run_best/` (bao gồm `model.onnx` hoặc `model_quantized.onnx`, `tokenizer.json` và cấu hình đi kèm) trước khi chạy hệ thống.

---

## 💻 4. Vận Hành Hệ Thống Cục Bộ

### A. Tạo FAISS Index Offline
Mỗi khi thêm tài liệu mới vào `paper/`, bạn cần chạy tác vụ build index vector database:
```bash
python3 -m rag_chatbox.services.indexer_job --print-summary
```

### B. Chạy Chatbot CLI Tương Tác
Trò chuyện và kiểm tra chất lượng RAG trực tiếp trên Terminal:
```bash
rag-chatbox
```

### C. Chạy Các Microservices (FastAPI)
- **Terminal 1 - Reader API Service** (Phục vụ suy luận ONNX):
  ```bash
  rag-reader-service --host 0.0.0.0 --port 8081
  ```
- **Terminal 2 - Synthesis API Service** (Điều phối RAG, truy xuất và gọi LLM):
  ```bash
  rag-synthesis-service --host 0.0.0.0 --port 8080
  ```

Kiểm tra trạng thái hoạt động:
```bash
curl http://localhost:8081/healthz
curl http://localhost:8080/healthz
```

Gửi yêu cầu hỏi đáp tới API RAG:
```bash
curl -X POST http://localhost:8080/v1/chat/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "RAG là gì?", "chat_history": ""}'
```

---

## 🏋️ 5. Huấn Luyện Tinh Chỉnh Mô Hình Reader

Toàn bộ cấu hình huấn luyện nằm tại tệp `config/defaults.yaml`.

### Huấn luyện (Train):
```bash
rag-ft-train --config config/defaults.yaml
```

### Đánh giá Checkpoint:
```bash
rag-ft-eval --config config/defaults.yaml --checkpoint-dir outputs/checkpoints/best_model
```

### Xuất mô hình sang ONNX lượng hóa:
```bash
rag-ft-export \
  --config config/defaults.yaml \
  --checkpoint-dir outputs/checkpoints/best_model \
  --artifact-dir artifacts/readers/run_best
```

---

## 📊 6. Hệ Thống Đánh Giá Tự Động (Evaluation)

### Đánh giá độc lập mô hình Reader:
```bash
rag-eval --mode reader --top-k 5 --limit 20 --output-json eval/reader_eval_latest.json
```

### Đánh giá toàn diện RAG (End-to-End):
```bash
rag-eval --mode rag --top-k 5 --limit 20 --rag-answer-timeout-sec 60 --output-json eval/rag_eval_latest.json
```

### Chạy hệ thống Unit Tests (Đảm bảo 100% các hàm lõi hoạt động đúng):
```bash
python -m unittest discover -s tests -p 'test_*.py'
```

---

## 🐳 7. Triển Khai Bằng Docker Compose

Xem tài liệu chi tiết tại [deploy/README.md](file:///deploy/README.md).

Quy trình khởi động nhanh gọn qua Docker:
```bash
# 1. Chuẩn bị biến môi trường deploy
cp .env.deploy.example .env.deploy

# 2. Build index offline trên máy host
python3 -m rag_chatbox.services.indexer_job --print-summary

# 3. Khởi chạy cụm Microservices (Reader + Synthesis)
docker compose -f deploy/docker-compose.yml up --build
```
*(Các dịch vụ Docker sử dụng chế độ mạng `host` để kết nối trực tiếp với dịch vụ Ollama đang lắng nghe trên máy Host).*

---

## 📁 8. Sơ Đồ Cấu Trúc Dự Án

```text
config/      # Cấu hình huấn luyện và export mô hình
deploy/      # Dockerfiles, Docker Compose và hướng dẫn deploy
docs/        # Tài liệu hướng dẫn kỹ thuật chi tiết của hệ thống
eval/        # Tập dữ liệu kiểm thử vàng và lịch sử kết quả đánh giá
scripts/     # Các script đồng bộ/tải nhanh artifacts từ Object Storage
src/         # Mã nguồn chính (training/ và rag_chatbox/)
tests/       # Hệ thống kiểm thử tự động (Unit Tests)
paper/       # PDF tài liệu đầu vào (Local - không commit git)
artifacts/   # ONNX Reader Model (Local - không commit git)
outputs/     # Checkpoint huấn luyện PyTorch (Local - không commit git)
.cache/      # Cơ sở dữ liệu FAISS cục bộ (Local - không commit git)
```

---

## 🔄 9. Đồng Bộ Hóa Artifacts

Khi triển khai trên máy chủ mới chưa có dữ liệu và mô hình huấn luyện, hãy sử dụng script đồng bộ từ S3/Object Storage (nếu được cấu hình):

```bash
# Tải dữ liệu, cache và mô hình về máy mới
export ARTIFACT_BUCKET=<tên-bucket-s3>
scripts/fetch_artifacts.sh

# Đồng bộ ngược kết quả huấn luyện cục bộ lên Cloud
scripts/sync_artifacts.sh
```

---

> [!CAUTION]
> **Quy định quan trọng dành cho các AI Assistant / Nhà phát triển**:
> Tuyệt đối **không thực hiện Git Commit** trực tiếp trên môi trường làm việc khi chưa có sự xác nhận của quản trị viên hệ thống. Mọi thay đổi mã nguồn phải được lưu giữ ở dạng working tree chưa commit.
