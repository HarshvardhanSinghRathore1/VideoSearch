# VideoRAG Phase 1 — Comprehensive Codebase & Architecture Guide

This document explains the complete codebase structure, how every module works, where each file is located, and how data flows through the entire pipeline.

---

## 1. Project Directory Sitemap

```
Dmbi project/
│
├── src/                                  # BACKEND CORE PIPELINE MODULES
│   ├── __init__.py                       # Package exports for clean imports
│   ├── video_processor.py                # Step 1: Audio extraction & video validation
│   ├── transcription.py                  # Step 2: Whisper timestamped speech-to-text
│   ├── chunking.py                       # Step 3: Timestamp-preserving semantic chunking
│   ├── embeddings.py                     # Step 4: BAAI/bge-small-en-v1.5 dense vector embeddings
│   ├── vector_store.py                   # Step 5: FAISS vector database (IndexFlatIP)
│   ├── retriever.py                      # Step 6: Top-K similarity search & context formatting
│   ├── llm.py                            # Step 7: Grounded LLM answering (Gemini API + citations)
│   ├── evaluation.py                     # Step 8: Quantitative metrics (Recall@K, MRR, IoU)
│   ├── rag_pipeline.py                   # Master orchestrator tying all stages together
│   ├── api.py                            # FastAPI REST API layer for the web application
│   ├── create_notebooks.py               # Generator script for research notebooks
│   └── create_sample_video.py            # Generator for synthetic lecture test video & audio
│
├── frontend/                             # FRONTEND WEB APPLICATION (React + Vite)
│   ├── src/
│   │   ├── main.jsx                      # React entrypoint
│   │   ├── App.jsx                       # Root view router & global header
│   │   ├── services/
│   │   │   └── api.js                    # Centralized REST API client
│   │   ├── hooks/
│   │   │   └── useVideoRAG.js            # Custom React hook for video/player/chat/research state
│   │   ├── pages/
│   │   │   ├── Home.jsx                  # Screen 1: Upload & Quick Start
│   │   │   ├── Processing.jsx            # Screen 2: 7-stage live progress dashboard
│   │   │   └── VideoWorkspace.jsx        # Screen 3: Two-column Video Player + AI Chat
│   │   ├── components/
│   │   │   ├── VideoPlayer.jsx           # HTML5 video player + timestamp jump listener
│   │   │   ├── ChatPanel.jsx             # AI conversation stream & query input
│   │   │   ├── ChatMessage.jsx           # Grounded answer card & evidence chips
│   │   │   ├── EvidenceBadge.jsx         # Clickable timestamp chip triggering video seek
│   │   │   ├── VideoUploader.jsx         # Drag-and-drop file upload zone
│   │   │   ├── ProcessingStatus.jsx      # Pipeline stepper visualizer
│   │   │   ├── ResearchPanel.jsx         # Research Mode inspection drawer
│   │   │   ├── RetrievedChunk.jsx        # Ranked chunk card with similarity score
│   │   │   ├── LatencyMetrics.jsx        # Latency breakdown grid
│   │   │   └── ResearchMetadata.jsx      # System architecture modal
│   │   └── styles/
│   │       └── index.css                 # Dark research design system styles
│   ├── index.html                        # HTML root for Vite
│   ├── package.json                      # React & Vite npm dependencies
│   └── vite.config.js                    # Vite dev server config (proxying /api to port 8000)
│
├── notebooks/                            # JUPYTER RESEARCH EXPERIMENT NOTEBOOKS
│   ├── 01_video_to_transcript.ipynb      # Phase 1A & 1B walkthrough
│   ├── 02_transcript_chunking.ipynb      # Phase 1C chunking walkthrough
│   ├── 03_embeddings_and_indexing.ipynb  # Phase 1D & 1E vector indexing
│   ├── 04_rag_question_answering.ipynb   # Phase 1F & 1G interactive QA
│   └── 05_baseline_evaluation.ipynb      # Phase 1 quantitative benchmark runner
│
├── data/                                 # ARTIFACTS & STORAGE
│   ├── videos/                           # Raw video files (sample.mp4)
│   ├── audio/                            # Extracted 16kHz mono audio (sample.wav)
│   ├── transcripts/                      # JSON and TXT timestamped transcripts & chunks
│   ├── vector_store/                     # FAISS binary index (faiss.index) & metadata JSON
│   └── evaluation/
│       ├── benchmark_questions.json      # 10 formal evaluation questions with time bounds
│       └── dev_questions.json            # Development sanity check queries
│
├── experiments/
│   └── results/                          # BASELINE BENCHMARK OUTPUTS
│       ├── baseline_results.json         # JSON metrics and per-sample logs
│       ├── baseline_results.csv          # CSV tabular evaluation results
│       └── evaluation_plot.png           # Visualized Recall and Latency plots
│
├── run_server.py                         # FastAPI server launcher script
├── test_phase1_pipeline.py               # Automated end-to-end test suite
├── requirements.txt                      # Python dependencies list
├── .env                                  # Active environment variables (API keys, models)
├── .env.example                          # Environment template
└── README.md                             # Formal research documentation
```

---

## 2. Detailed Module-by-Module Explanation

### Backend Modules (`src/`)

#### 1. `src/video_processor.py`
* **Purpose**: Validates video files and extracts a clean audio stream using FFmpeg without modifying the original video container.
* **Key Class**: `VideoProcessor`
* **Key Functions**:
  * `validate_video_path(path)`: Ensures the file exists and has a supported format (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`).
  * `extract_audio(video_path, output_format="wav", sample_rate=16000, channels=1)`: Executes FFmpeg to output a 16kHz mono PCM WAV stream (`data/audio/<filename>.wav`).
  * `get_media_metadata(path)`: Returns file size, duration, format.

#### 2. `src/transcription.py`
* **Purpose**: Performs speech-to-text while strictly preserving start and end timestamps for every spoken sentence.
* **Key Class**: `Transcriber`, `TranscriptSegment`
* **Key Functions**:
  * `transcribe(audio_path)`: Uses `faster-whisper` (or standard `whisper`) to output `[TranscriptSegment(start=0.0, end=45.0, text="...")]`.
  * `save_transcripts(segments, video_name)`: Exports both structured `data/transcripts/<name>_transcript.json` and human-readable `.txt` with timestamps `[00:00:00 - 00:00:45]`.
  * `format_timestamp(seconds)`: Converts seconds into clean `HH:MM:SS.ss` format.

#### 3. `src/chunking.py`
* **Purpose**: Partitions consecutive speech segments into sliding window chunks while keeping exact temporal boundaries.
* **Key Class**: `TranscriptChunker`, `TranscriptChunk`
* **Key Functions**:
  * `chunk_segments(segments, target_chunk_size=120, chunk_overlap=25)`: Groups segments into chunks where `start_time = first_segment.start` and `end_time = last_segment.end`.
  * `save_chunks(chunks, path)` & `load_chunks(path)`: Serializes chunks into JSON.

#### 4. `src/embeddings.py`
* **Purpose**: Computes dense vector representations for transcript chunks and search queries.
* **Key Class**: `BGEEmbedder`
* **Key Functions**:
  * `embed_chunks(chunks)`: Encodes chunk text into 384-dimensional dense vectors using `BAAI/bge-small-en-v1.5` with L2 normalization.
  * `embed_query(query)`: Encodes query text with the BGE retrieval prefix (`"Represent this sentence for searching relevant passages: "`).

#### 5. `src/vector_store.py`
* **Purpose**: FAISS vector database storing dense embeddings with metadata persistence.
* **Key Class**: `FAISSVectorStore`
* **Key Functions**:
  * `add_chunks(chunks, embeddings)`: Inserts normalized vectors into FAISS `IndexFlatIP` (Inner Product = exact Cosine Similarity).
  * `search(query_vec, top_k=5)`: Finds top-K most similar chunks and returns cosine scores.
  * `save(index_path, metadata_path)` & `load(...)`: Persists `faiss.index` binary alongside `chunks_metadata.json`.

#### 6. `src/retriever.py`
* **Purpose**: Connects query embedding to vector search and formats structured context blocks for the LLM.
* **Key Class**: `TranscriptRetriever`
* **Key Functions**:
  * `retrieve(query, top_k=5)`: Returns ranked list of chunks with similarity scores, start/end timestamps, and formatted strings.
  * `format_context_for_llm(chunks)`: Formats context as numbered evidence blocks `[Evidence 1] (Time: 00:00:00 - 00:00:45) Text...`.

#### 7. `src/llm.py`
* **Purpose**: Prompts the LLM with strict grounding instructions to prevent hallucination and cite timestamps.
* **Key Class**: `GroundedLLMGenerator`
* **Key Functions**:
  * `generate_answer(query, retrieved_chunks, context_str)`: Calls Gemini API (`gemini-3.6-flash` with model fallbacks) or offline grounded engine.
  * Ensures output has distinct:
    * `Answer: ...`
    * `Evidence: [00:00:00 - 00:00:45]`
    * Refusal if out-of-domain: *"The provided video transcript does not contain information to answer this question."*

#### 8. `src/evaluation.py`
* **Purpose**: Computes quantitative research metrics on benchmark datasets.
* **Key Class**: `RAGEvaluator`, `EvaluationSampleResult`
* **Metrics**:
  * Retrieval: Recall@1, Recall@3, Recall@5, MRR
  * Temporal Grounding: Temporal IoU, Top-1 Temporal Overlap Hit Rate
  * Answer Quality: Abstention Accuracy on unanswerables
  * Latency: Retrieval Latency, Generation Latency, Total Latency.

#### 9. `src/rag_pipeline.py`
* **Purpose**: Master orchestrator class providing high-level methods:
  * `process_video_to_transcript(video_path)`
  * `index_transcript(segments, video_name)`
  * `query(question, top_k=5)` (Baseline B: RAG)
  * `query_baseline_a_full_transcript(question)` (Baseline A: Full transcript comparison)
  * `run_evaluation(dataset_path)`

#### 10. `src/api.py`
* **Purpose**: FastAPI REST API server exposing endpoints to the React frontend:
  * `POST /api/videos/upload`: Multipart file upload.
  * `POST /api/videos/{id}/process`: Background multi-stage processing.
  * `GET /api/videos/{id}/status`: Live progress status polling.
  * `POST /api/videos/{id}/query`: Grounded RAG query endpoint.
  * `GET /api/videos/{id}/video`: Streaming video with HTTP 206 Range support for video seeking.
  * `GET /api/system/info`: Metadata about models and parameters.

---

### Frontend Modules (`frontend/`)

#### 1. State Management & API Client
* **`frontend/src/services/api.js`**: Centralized HTTP client sending requests to `/api/...`.
* **`frontend/src/hooks/useVideoRAG.js`**: React custom hook holding all UI state, video player `useRef()`, chat history, processing stage polling, and `seekVideo(startTimeSeconds)`.

#### 2. Pages
* **`frontend/src/pages/Home.jsx`** (Screen 1): Video upload area, drag-and-drop, format validation, and pre-indexed sample lecture quick-start button.
* **`frontend/src/pages/Processing.jsx`** (Screen 2): Real-time 7-stage pipeline visualizer (Video Uploaded $\rightarrow$ Audio $\rightarrow$ Transcribe $\rightarrow$ Chunk $\rightarrow$ Embed $\rightarrow$ Index $\rightarrow$ Ready).
* **`frontend/src/pages/VideoWorkspace.jsx`** (Screen 3): Two-column layout with HTML5 video player on the left and grounded AI chat on the right.

#### 3. Components
* **`VideoPlayer.jsx`**: Native HTML5 `<video>` player with time indicators and suggested query chips.
* **`ChatPanel.jsx` & `ChatMessage.jsx`**: Conversation stream, loading states, grounded answer cards, and source tags.
* **`EvidenceBadge.jsx`**: Clickable timestamp chip (e.g. `[00:01:25 - 00:02:10]`). When clicked, it sets `videoRef.current.currentTime = startTime` and starts playback.
* **`ResearchPanel.jsx`**: Slide-out Research Mode drawer displaying latency breakdowns, top-K ranked chunks with cosine similarity scores, user query, LLM context, and raw outputs.
* **`ResearchMetadata.jsx`**: Modal showing Phase 1 pipeline architecture.

---

## 3. How Data Flows End-to-End

```
1. USER UPLOADS VIDEO (sample.mp4)
   └─► React sends POST /api/videos/upload
   └─► React initiates POST /api/videos/{id}/process
   
2. BACKEND MULTI-STAGE PIPELINE (src/api.py + src/rag_pipeline.py)
   ├─► Stage 1: Validate file format (.mp4)
   ├─► Stage 2: Extract audio stream -> data/audio/sample.wav (16kHz Mono)
   ├─► Stage 3: Whisper Transcribe -> data/transcripts/sample_transcript.json
   ├─► Stage 4: Windowed Chunking -> data/transcripts/sample_chunks.json
   ├─► Stage 5: BGE Embeddings -> 384-dim normalized vector matrix
   ├─► Stage 6: Ingest into FAISS IndexFlatIP -> data/vector_store/faiss.index
   └─► Stage 7: Pipeline Ready!

3. USER ASKS QUESTION: "What is the primary goal of database normalization?"
   └─► React sends POST /api/videos/{id}/query
   └─► Retriever embeds query with BGE prefix
   └─► FAISS searches top-5 nearest chunks (Cosine similarity score = 0.8710)
   └─► LLM (Gemini 3.6 Flash) generates grounded answer citing [00:00:00 - 00:00:45]
   └─► Backend returns JSON with answer, evidence timestamps, ranked chunks, latencies

4. INTERACTIVE TIMESTAMP SEEKING
   └─► User clicks evidence badge [00:00:00 - 00:00:45] in Chat
   └─► React executes: videoRef.current.currentTime = 0.0; videoRef.current.play();
   └─► HTML5 video player jumps directly to that point in the lecture.

5. RESEARCH MODE INSPECTION
   └─► User clicks "Research Mode" in the header
   └─► Side drawer reveals retrieval latency (0.015s), cosine similarity (0.8710),
       exact chunk text, and full context sent to the LLM.
```

---

## 4. How to Run the System

### Step 1: Start Backend
```bash
python run_server.py --port 8000
```

### Step 2: Start Frontend
```bash
cd frontend
npm run dev
```

### Step 3: Open Browser
Open **`http://localhost:5173`** to use the full interface.
