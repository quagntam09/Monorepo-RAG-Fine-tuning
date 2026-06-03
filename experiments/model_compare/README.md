# Model Compare: DistilRoBERTa + TinyBERT

Thư mục này triển khai pipeline fine-tuning độc lập để so sánh với baseline đang dùng trong dự án:
- Baseline: `distilbert-base-multilingual-cased` (flow chính hiện tại)
- Model bổ sung: `distilbert/distilroberta-base`
- Model bổ sung: `huawei-noah/TinyBERT_General_6L_768D`

Các script trong đây **không sửa** CLI chính `rag-ft-*`.

## 1. Train DistilRoBERTa

### Stage 1 (EN + VI)
```bash
python experiments/model_compare/train_autoqa.py \
  --config experiments/model_compare/configs/distilroberta_stage1.yaml
```

### Stage 2 (VI refine)
```bash
python experiments/model_compare/train_autoqa.py \
  --config experiments/model_compare/configs/distilroberta_stage2_vi.yaml
```

## 2. Train TinyBERT

### Stage 1 (EN + VI)
```bash
python experiments/model_compare/train_autoqa.py \
  --config experiments/model_compare/configs/tinybert_stage1.yaml
```

### Stage 2 (VI refine)
```bash
python experiments/model_compare/train_autoqa.py \
  --config experiments/model_compare/configs/tinybert_stage2_vi.yaml
```

## 3. Evaluate EN/VI

### DistilRoBERTa
```bash
python experiments/model_compare/eval_autoqa.py \
  --config experiments/model_compare/configs/distilroberta_eval_en.yaml \
  --checkpoint-dir outputs/model_compare/distilroberta/stage2_vi/best_model \
  --output-json outputs/model_compare/metrics/distilroberta_eval_en.json

python experiments/model_compare/eval_autoqa.py \
  --config experiments/model_compare/configs/distilroberta_eval_vi.yaml \
  --checkpoint-dir outputs/model_compare/distilroberta/stage2_vi/best_model \
  --output-json outputs/model_compare/metrics/distilroberta_eval_vi.json
```

### TinyBERT
```bash
python experiments/model_compare/eval_autoqa.py \
  --config experiments/model_compare/configs/tinybert_eval_en.yaml \
  --checkpoint-dir outputs/model_compare/tinybert/stage2_vi/best_model \
  --output-json outputs/model_compare/metrics/tinybert_eval_en.json

python experiments/model_compare/eval_autoqa.py \
  --config experiments/model_compare/configs/tinybert_eval_vi.yaml \
  --checkpoint-dir outputs/model_compare/tinybert/stage2_vi/best_model \
  --output-json outputs/model_compare/metrics/tinybert_eval_vi.json
```

## 4. Baseline DistilBERT để so sánh

Chạy lại đánh giá baseline hiện có:
```bash
rag-ft-eval --config config/eval_vi.yaml \
  --checkpoint-dir outputs/checkpoints_stage2_vi/best_model \
  > outputs/model_compare/metrics/distilbert_eval_vi.json
```

## 5. So sánh 3 model

```bash
python experiments/model_compare/compare_results.py \
  --result distilbert=outputs/model_compare/metrics/distilbert_eval_vi.json \
  --result distilroberta=outputs/model_compare/metrics/distilroberta_eval_vi.json \
  --result tinybert=outputs/model_compare/metrics/tinybert_eval_vi.json \
  --output-json outputs/model_compare/metrics/model_compare_vi.json
```

Script sẽ in bảng Markdown gồm `exact_match`, `f1`, `loss`, `span_exact_match`.
