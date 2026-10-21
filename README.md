# Video RAG — Phase 1: Transcript-Based Video Question Answering

A modular, reproducible research baseline for question answering over educational and instructional videos using timestamp-preserving speech transcripts, dense vector retrieval with FAISS, and grounded LLM generation.

---

## 1. Research Objective

Video Question Answering (Video QA) requires bridging dense multimodal streams with semantic information retrieval. In this initial research phase (**Phase 1**), our objective is to build a rigorous **Transcript-Based Video RAG baseline** that:
1. Validates and extracts audio from raw video containers.
2. Performs speech-to-text with Whisper while strictly preserving speech timestamps.
3. Partitions transcripts into semantically coherent, timestamp-bounded sliding window chunks.
4. Generates dense embeddings using `BAAI/bge-small-en-v1.5` and indexes them in FAISS.
5. Performs top-$k$ vector similarity search and retrieves evidence chunks.
6. Feeds retrieved context into an LLM (Google Gemini API / grounded generator) to answer user queries with explicit video timestamp citations.
7. Measures quantitative retrieval (Recall@K, MRR), temporal grounding (IoU), and answer fidelity metrics.

This establishes the formal empirical baseline against which advanced temporal-aware and multimodal architectures will be compared in subsequent phases.

---

## 2. Problem Statement

Standard text-based RAG architectures operate on flat documents where temporal context is absent. When applied to video transcripts:
- Arbitrary text slicing destroys temporal boundaries, making it impossible to locate where in the video the answer was spoken.
- Long videos produce huge transcripts exceeding context windows and degrading attention focus.
- Hallucinations can point users to non-existent video segments.

Phase 1 addresses this by treating the video transcript as a sequence of timestamped intervals $[t_{\text{start}}, t_{\text{end}}]$, ensuring that all chunking, indexing, and retrieval operations maintain temporal provenance.

---

## 3. Architecture Overview

```
[Raw Video: sample.mp4]
         │
         ▼ (FFmpeg / imageio-ffmpeg)
[Extracted Audio: sample.wav (16kHz Mono)]
         │
         ▼ (Whisper / faster-whisper)
[Timestamped Transcript Segments]  -->  (data/transcripts/sample_transcript.json & .txt)
         │
         ▼ (TranscriptChunker: Windowed, Overlap-aware)
[Timestamped Chunks [start_time, end_time]]
         │
         ▼ (BGE Embeddings: BAAI/bge-small-en-v1.5)
[Dense Vector Matrix (384-dim, L2-normalized)]
         │
         ▼ (FAISS IndexFlatIP)
[Vector Store Index + Metadata JSON]
         │
  User Query ──► [Embed Query with BGE Prefix]
         │
         ▼ (FAISS Top-K Search)
[Top-K Evidence Chunks with Timestamps & Scores]
         │
         ▼ (Grounded Prompt Engineering)
[LLM: Gemini / Grounded Generator]
         │
         ▼
[Grounded Answer + Video Timestamp Citations [HH:MM:SS - HH:MM:SS]]
```

---

## 4. Technology Stack

* **Language**: Python 3.11+ / 3.13
* **Audio Extraction**: FFmpeg (via `VideoProcessor` with fallback to `imageio-ffmpeg`)
* **Speech-to-Text**: `faster-whisper` / `openai-whisper`
* **Embedding Model**: `BAAI/bge-small-en-v1.5` (via `sentence-transformers`)
* **Vector Store**: `faiss-cpu` (`IndexFlatIP` for exact cosine similarity)
* **LLM Generation**: Google Gemini API (`gemini-1.5-flash` / `gemini-2.0-flash`) via `google-genai` / `google-generativeai`
* **Data Processing & Viz**: `pandas`, `numpy`, `matplotlib`, `tabulate`

---

## 5. Project Directory Structure

```
Dmbi project/
├── data/
│   ├── videos/              # Raw video files (e.g., sample.mp4)
│   ├── audio/               # Extracted audio streams (e.g., sample.wav)
│   ├── transcripts/         # Timestamped JSON & formatted TXT transcripts
│   ├── vector_store/        # Persisted FAISS binary index and chunk metadata
│   └── evaluation/
│       ├── dev_questions.json         # Rapid development/debugging test cases
│       └── benchmark_questions.json   # 10-question evaluation benchmark
│
├── notebooks/
│   ├── 01_video_to_transcript.ipynb      # Phase 1A & 1B: Video to timestamped transcript
│   ├── 02_transcript_chunking.ipynb      # Phase 1C: Timestamp-preserving chunking
│   ├── 03_embeddings_and_indexing.ipynb  # Phase 1D & 1E: BGE embeddings & FAISS index
│   ├── 04_rag_question_answering.ipynb   # Phase 1F & 1G: End-to-end QA & interactive query loop
│   └── 05_baseline_evaluation.ipynb      # Phase 1 Evaluation: Quantitative metrics & plots
│
├── src/
│   ├── __init__.py
│   ├── video_processor.py   # Audio extraction & media validation
│   ├── transcription.py     # Whisper timestamped speech-to-text
│   ├── chunking.py          # Windowed timestamp-preserving chunker
│   ├── embeddings.py        # BGE dense vector embedder
│   ├── vector_store.py      # FAISS vector database wrapper
│   ├── retriever.py         # Top-k similarity retrieval
│   ├── llm.py               # Grounded prompt construction & Gemini client
│   ├── evaluation.py        # Metrics: Recall@K, MRR, Temporal IoU, Abstention
│   └── rag_pipeline.py      # Master VideoRAGPipeline orchestrator
│
├── experiments/
│   └── results/             # Benchmark JSON and CSV outputs
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## 6. Installation & Setup

### 1. Clone & Set Up Environment
```bash
# Create and activate virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and insert your Gemini API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

---

## 7. How the Pipeline Components Work

### Phase 1A: Video Processing (`src/video_processor.py`)
- Validates supported video extensions (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`).
- Uses FFmpeg to extract mono 16kHz PCM audio (`data/audio/<name>.wav`) suitable for Whisper acoustic processing without touching the original video file.

### Phase 1B: Speech-to-Text (`src/transcription.py`)
- Transcribes the audio with `faster-whisper` (or standard `whisper`).
- Preserves exact segment timestamps in structured JSON:
  ```json
  {
    "start": 12.4,
    "end": 18.7,
    "text": "Database normalization is the process of organizing data..."
  }
  ```
- Formats a human-readable text transcript with timestamps `[HH:MM:SS.s - HH:MM:SS.s]`.

### Phase 1C: Timestamp-Preserving Chunking (`src/chunking.py`)
- Groups consecutive speech segments into sliding windows targeting 300–500 tokens with configurable overlap (default 50 tokens).
- Calculates exact chunk boundaries:
  - `start_time = first_segment.start`
  - `end_time = last_segment.end`
- Outputs structured `TranscriptChunk` objects.

### Phase 1D: Dense Embeddings (`src/embeddings.py`)
- Utilizes `BAAI/bge-small-en-v1.5` yielding 384-dimensional dense vectors.
- Applies L2 normalization to enable exact cosine similarity using standard dot products.
- Prepends the required BGE retrieval prefix (`"Represent this sentence for searching relevant passages: "`) to query embeddings for asymmetric retrieval.

### Phase 1E: Vector Indexing & Retrieval (`src/vector_store.py` & `src/retriever.py`)
- Employs FAISS `IndexFlatIP` (Inner Product).
- Retrieves top-$k$ nearest chunks with similarity scores $\in [-1, 1]$.
- Formats evidence blocks with precise timestamp headers for LLM consumption.

### Phase 1F: Grounded LLM Generation (`src/llm.py`)
- Instructs LLM to answer strictly from retrieved context and refuse out-of-domain queries.
- Parses output into:
  - `Answer`: Factual summary.
  - `Evidence`: Cited timestamp ranges.
- Operates in offline grounded mode if no API key is provided.

---

## 8. Quantitative Evaluation Methodology (`src/evaluation.py`)

Evaluation questions are stored in `data/evaluation/benchmark_questions.json` with ground-truth temporal intervals $[t_{\text{start}}, t_{\text{end}}]$ and binary answerability flags.

### Metrics Computed:
1. **Retrieval**:
   - **Recall@K** ($K \in \{1, 3, 5\}$): Fraction of answerable queries where a ground-truth overlapping chunk was retrieved in the top-$K$.
   - **MRR (Mean Reciprocal Rank)**: $\frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$ of the first retrieved chunk overlapping the ground truth.
2. **Temporal Grounding**:
   - **Temporal IoU**: $\frac{\text{Intersection}(\text{Pred}, \text{GT})}{\text{Union}(\text{Pred}, \text{GT})}$ for the top retrieved chunk.
   - **Top-1 Overlap Hit Rate**: Percentage of queries where the top-1 retrieved chunk temporally overlaps the ground-truth interval.
3. **Answer Quality & Abstention**:
   - **Abstention Accuracy**: Correct refusal rate on unanswerable/out-of-domain questions.
4. **System Latency**:
   - Average retrieval latency, LLM generation latency, and end-to-end latency.

---

## 9. Current Limitations of Phase 1

1. **Audio Dependency**: Completely blind to non-verbal video content (visual slides, diagrams, on-screen demonstrations, equations without spoken descriptions).
2. **Speech-to-Text Errors**: Whisper acoustic hallucinations or domain-specific terminology errors directly propagate into the vector index.
3. **Fixed Window Boundaries**: Fixed-token chunking can occasionally split a single spoken explanation across chunk boundaries.
4. **Linear Temporal Scope**: Standard vector retrieval does not model long-range sequential video narrative or causal relationships across time.

---

## 10. Future Research Roadmap

```
┌────────────────────────────────────────────────────────┐
│  Phase 1: Transcript RAG Baseline (CURRENT)           │
│  - Whisper Speech-to-Text                              │
│  - Timestamp-preserving chunking                       │
│  - BGE Embeddings + FAISS FlatIP                       │
│  - Evidence timestamp grounding                        │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Phase 2: Temporal-Aware RAG (FUTURE)                  │
│  - Temporal graph modeling & hierarchical chunking     │
│  - Temporal distance decay & sequential re-ranking     │
│  - Interval-aware context expansion                    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Phase 3: Multimodal RAG (FUTURE)                      │
│  - Keyframe extraction & CLIP/SigLIP visual embeddings │
│  - OCR text indexing from video slides                 │
│  - Cross-modal fusion (Audio Transcript + Visuals)     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Phase 4: Comparative Evaluation & Ablation Study      │
│  - Comprehensive benchmark across Phase 1, 2, and 3    │
│  - End-to-end ablation on retrieval & answer quality   │
└────────────────────────────────────────────────────────┘
```
#
