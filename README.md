# MITRE ATT&CK TTP Mapping & Lightweight RAG Evaluation

This repository contains a workflow for mapping security detection descriptions to MITRE ATT&CK Technique IDs ("TTP labeling") using embedding-based retrieval PLUS a local LLM (phi-4) reasoning / re-ranking step. The LLM is always invoked so its contribution can be evaluated end-to-end.

> Goal: Provide a concise, reproducible baseline you can extend (better models, richer metrics, vector DB, prompt strategies, etc.).

---
## Repository Contents
| File | Purpose |
|------|---------|
| `ttp_rag_labeling.ipynb` | Main notebook: data loading, embedding, retrieval, LLM refinement (always on), evaluation, inspection. |
| `MITRE_TPPs.xlsx` | Reference dataset of techniques (IDs, names, descriptions). |
| `MDE_SampleDetections.xlsx` | Detection descriptions and ground truth technique IDs in column `MiTRE_TTPs`. |

---
## High-Level Pipeline
1. Load MITRE technique reference (`MITRE_TPPs.xlsx`).
2. Concatenate ID + Name + Description into a single text string per technique.
3. Generate dense embeddings with `sentence-transformers/all-MiniLM-L6-v2` (fast, ~384-dim).
4. Build a cosine similarity index using `sklearn.neighbors.NearestNeighbors`.
5. Load detection dataset (`MDE_SampleDetections.xlsx`) and isolate: ID, Name (optional), Description, Ground Truth (`MiTRE_TTPs`).
6. For each detection description (RAG step):
    - Retrieve top-K candidate techniques (embedding similarity) to build a context block.
    - Provide detection description + context to the LLM with a strict system prompt.
    - LLM returns JSON: `{ "technique_ids": [...], "rationale": "..." }`.
7. Parse JSON, validate IDs against retrieved candidates, fallback to similarity if invalid/empty.
8. Using ground truth, compute evaluation metrics.
9. Inspect individual examples to sanity-check reasoning.

---
## Models & Components
| Component | Default | Notes |
|-----------|---------|-------|
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` | Small, fast, good baseline; can swap for larger model (`all-mpnet-base-v2`) for quality. |
| Index | `NearestNeighbors(metric='cosine')` | Simple in-memory; replace with FAISS/Chroma/Weaviate for scale or persistence. |
| LLM | `microsoft/phi-4` | Always used for reasoning / consolidation; retrieval provides candidate context. |
| Ground Truth Column | `MiTRE_TTPs` | Never used in prompts. |

---
## Configuration Knobs (in the Notebook)
| Variable | Purpose | Typical Values |
|----------|---------|----------------|
| `DIRECT_MODEL_NAME` | LLM model name | `microsoft/phi-4` |
| `EMBED_MODEL_NAME` | Embedding backbone | `all-MiniLM-L6-v2` |
| `k_retrieval` (function arg) | How many candidates retrieved | 5–15 |
| `top_n_final` (function arg) | Final top predicted IDs | 1–5 |
| `K` (evaluation cell) | Cutoff for Precision/Recall/F1 | 1–5 |

---
## Evaluation Metrics (Slim Set)
We intentionally reduced to three high‑signal metrics for early iteration:

1. F1@K (micro) – Harmonic mean of precision & recall on technique IDs within top K (single summary classification quality number).
2. Faithfulness – LLM‑judge score (0–1) indicating how well the rationale is grounded in the retrieved technique context (no hallucinated claims).
3. Relevance – LLM‑judge score (0–1) measuring how directly the rationale addresses the specific detection description (on‑topic, not generic).

Rationale for this choice:
- Precision@K & Recall@K are implicitly represented inside F1@K; keeping them separately adds noise early on.
- BLEU/ROUGE/BERTScore focus on surface or semantic similarity of wording; for ATT&CK mapping we care more about factual grounding (Faithfulness) and applicability (Relevance).
- Faithfulness + Relevance jointly answer: “Is it correct AND is it about the right thing?”

Gold Handling:
- Technique IDs extracted from `MiTRE_TTPs` via regex: `T[0-9]{4,5}(?:\.[0-9]{3})?` (sub‑techniques supported).
- Only IDs are used now for F1@K; the descriptive text remains available should you re‑enable lexical metrics later.

Planned (can re‑enable later): MRR, MAP, Coverage@K, ROUGE, BERTScore, Retrieval Recall@K, Context Utilization, Confusion matrix / clustering.

Bias Note: Faithfulness & Relevance currently use the same LLM that generated the rationale (self‑assessment). For stricter evaluation swap in a neutral judge model or sample manually.

---
## How To Run (Windows PowerShell Example)
1. Place Excel files in project root.
2. Open the notebook `ttp_rag_labeling.ipynb` in VS Code / Jupyter.
3. (Optional) Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Run Cell 2 (installs deps) – skip if already installed.
5. Run Cells 3 → 10 in order (or use Run All).
6. Inspect metrics & adjust parameters.

> If GPU memory is limited, switch to a smaller instruct model or quantized variant; LLM use is required for this evaluation design.

---
## Preventing Label Leakage
- Retrieval uses only the detection description text.
- `MiTRE_TTPs` is excluded from any context strings built for the model.
- Safety check ensures the ground truth column cannot be mis-assigned as description.

---
## Adding Persistence / Productionization (Ideas)
| Enhancement | Benefit | Quick Start |
|-------------|---------|-------------|
| Export embeddings to disk | Faster reload | Save `ttp_embeddings.npy` & serialized `col_map` |
| Vector DB (FAISS/Chroma) | Scale to thousands of techniques | Replace NearestNeighbors fit/query code |
| Prompt logging | Auditability | Append prompt/response pairs to a JSONL file |
| Cross-encoder re-rank | Better precision | Use `sentence-transformers` cross-encoder after initial retrieval |
| Multi-label thresholding | Control precision vs. recall | Add confidence threshold on similarity or LLM JSON output |
| Batch evaluation script | CI integration | Convert notebook cells into a Python script (`cli.py`) |

---
## RAG JSON Generation Logic
1. Retrieve top-K candidates (similarity order, larger pool than final output).
2. Build a compact context block: `index. TechniqueID | Name | truncated description`.
3. System prompt instructs the model to: use only provided techniques, avoid hallucination, output strict JSON.
4. User prompt includes the detection description and the candidate list.
5. Model output is parsed for JSON. Only technique IDs present in the retrieved set are kept; others are discarded; if parsing fails the prediction list is left empty (no similarity fallback to preserve evaluation purity).

**Benefits of JSON output:** Easier downstream auditing, deterministic parsing, simpler evaluation alignment.

---
## Reproducibility Tips
- Pin versions (optional) by exporting `pip freeze > requirements.txt` after a stable run.
- Set environment variable `PYTHONHASHSEED=0` for some hash-stable operations (not strictly required here).
- For consistent embeddings across runs: avoid model auto-updates by specifying exact model revision (e.g., `revision=` in `from_pretrained`).

---
## Error Handling & Troubleshooting
| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| CUDA OOM | Model too large | Disable LLM or use smaller model / reduce `max_new_tokens` |
| Slow retrieval | CPU-only environment | Use smaller embedding model or batch encode fewer at once |
| Empty predictions | LLM returned unexpected format | Check raw reply, adjust parser or fallback used |
| Column mapping error | Auto-detection failure | Manually set `det_map` in detection load cell |

---
---
## Roadmap Ideas
- Add MRR/MAP & calibration metrics
- Confusion heatmap (techniques vs. predicted)
- Multi-model ensemble (e.g., union of two embedding models)
- Guardrail to reject low-confidence matches
- Try structured JSON output from the LLM and validate schema
- Add a simple Streamlit UI for interactive mapping


---
## Quick FAQ
**Q: Can I run without the LLM?** Not in this configuration—the goal is to evaluate retrieval + LLM reasoning together.

**Q: How do I improve accuracy?** Strengthen embeddings (e.g., `all-mpnet-base-v2`), improve retrieval K, refine LLM prompt (clearer JSON schema, negative examples), or add a cross-encoder before passing candidates to the LLM.

**Q: How do I prevent hallucinated IDs?** Constrain output by post-filtering against the known Technique ID set from the reference file.

**Q: Can I map sub-techniques?** Yes—ensure sub-technique IDs are included in the reference file; embeddings treat them as additional entries.

---

