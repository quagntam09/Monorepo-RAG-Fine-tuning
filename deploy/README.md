# Deploy Topology

- `indexer job`: builds FAISS index offline and writes `manifest.json`.
- `reader service`: serves ONNX reader with micro-batching.
- `llm synthesis service`: performs retrieval + reader call + final generation.

## Suggested runtime split

1. Offline indexing runs from CI or a scheduled job.
2. Reader service uses mounted artifact volume or object-storage sync.
3. Synthesis service mounts `paper/` and `.cache/` or reads them from object-storage sync.

## Artifact policy

- Keep large PDFs, checkpoints, and ONNX artifacts out of git.
- Store them in object storage or Git LFS.
- Commit only manifests and small metadata files.
