"""
Notebook Generator Script
Constructs the 5 research notebooks with clean markdown explanations and reproducible code cells.
"""

import json
from pathlib import Path

def make_code_cell(source: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def make_md_cell(source: str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")]
    }

def build_notebook(cells, output_path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Generated notebook: {output_path}")

def generate_all():
    # -------------------------------------------------------------
    # Notebook 01: Video to Transcript
    # -------------------------------------------------------------
    nb1_cells = [
        make_md_cell("""# Video RAG — Phase 1: Notebook 01
## Video Audio Extraction and Timestamped Speech-to-Text Transcription

This notebook demonstrates the first two stages of the research pipeline:
1. **Phase 1A**: Validating the video file and extracting mono 16kHz audio using FFmpeg without modifying the original video.
2. **Phase 1B**: Transcribing the speech into timestamp-preserved segments using Whisper (`[{"start": float, "end": float, "text": str}]`).
"""),
        make_code_cell("""import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.append(str(Path.cwd().parent))

from src.video_processor import VideoProcessor
from src.transcription import Transcriber, format_timestamp
"""),
        make_md_cell("### Step 1: Video File Validation & Audio Extraction"),
        make_code_cell("""video_path = "../data/videos/sample.mp4"

processor = VideoProcessor(output_audio_dir="../data/audio")
metadata = processor.get_media_metadata(video_path)
print("Video Metadata:")
for k, v in metadata.items():
    print(f"  {k}: {v}")

# Extract audio stream (16kHz mono WAV suitable for Whisper)
audio_path = processor.extract_audio(video_path, output_format="wav", overwrite=True)
print(f"\\nExtracted Audio Path: {audio_path}")
"""),
        make_md_cell("### Step 2: Timestamped Speech-to-Text Transcription with Whisper"),
        make_code_cell("""# Initialize Transcriber (supports faster-whisper and openai-whisper)
transcriber = Transcriber(model_size="base", output_dir="../data/transcripts")

# Transcribe the audio
segments = transcriber.transcribe(audio_path, language="en")

print(f"Total Speech Segments Extracted: {len(segments)}")
"""),
        make_md_cell("### Step 3: Inspect Timestamped Segments Table"),
        make_code_cell("""# Display segments in a clean pandas DataFrame
records = [
    {
        "Index": i + 1,
        "Start Time (s)": seg.start,
        "End Time (s)": seg.end,
        "Formatted Range": f"[{format_timestamp(seg.start)} - {format_timestamp(seg.end)}]",
        "Duration (s)": round(seg.duration, 2),
        "Text": seg.text
    }
    for i, seg in enumerate(segments)
]
df_segments = pd.DataFrame(records)
df_segments.head(10)
"""),
        make_md_cell("### Step 4: Save Structured JSON & Human-Readable TXT Transcripts"),
        make_code_cell("""saved_files = transcriber.save_transcripts(
    segments=segments,
    video_name="sample.mp4",
    metadata={"video_size_mb": metadata["size_mb"]}
)

print(f"Structured JSON: {saved_files['json']}")
print(f"Human-Readable TXT: {saved_files['txt']}")
"""),
    ]
    build_notebook(nb1_cells, "notebooks/01_video_to_transcript.ipynb")

    # -------------------------------------------------------------
    # Notebook 02: Transcript Chunking
    # -------------------------------------------------------------
    nb2_cells = [
        make_md_cell("""# Video RAG — Phase 1: Notebook 02
## Timestamp-Preserving Transcript Chunking

In this notebook, we transform the raw sequence of speech segments into **semantically coherent, timestamp-bounded chunks**.

### Key Rules:
- Chunks must strictly preserve start and end timestamps (`start_time`, `end_time`).
- Chunk size is configurable (target 300–500 tokens/words).
- Overlap is configurable to maintain context across boundaries.
"""),
        make_code_cell("""import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path.cwd().parent))

from src.transcription import Transcriber, format_timestamp
from src.chunking import TranscriptChunker
"""),
        make_md_cell("### Step 1: Load Timestamped Transcript JSON"),
        make_code_cell("""transcript_json_path = "../data/transcripts/sample_transcript.json"

segments = Transcriber.load_transcript_json(transcript_json_path)
print(f"Loaded {len(segments)} segments from {transcript_json_path}")
"""),
        make_md_cell("### Step 2: Configure and Execute Sliding-Window Chunker"),
        make_code_cell("""# Initialize chunker with target token size of ~350 and overlap of ~50
chunker = TranscriptChunker(
    target_chunk_size=350,
    chunk_overlap=50,
    min_chunk_size=40
)

chunks = chunker.chunk_segments(segments, source_video="sample.mp4")
print(f"Generated {len(chunks)} timestamped chunks.")
"""),
        make_md_cell("### Step 3: Inspect Generated Chunks and Temporal Boundaries"),
        make_code_cell("""chunk_records = [
    {
        "Chunk ID": c.chunk_id,
        "Time Range": c.formatted_time_range(),
        "Start (s)": c.start_time,
        "End (s)": c.end_time,
        "Duration (s)": round(c.duration, 2),
        "Approx Tokens": c.token_count_approx,
        "Snippet": c.text[:90] + "..."
    }
    for c in chunks
]
df_chunks = pd.DataFrame(chunk_records)
df_chunks
"""),
        make_md_cell("### Step 4: Persist Chunks to Disk"),
        make_code_cell("""chunks_output_path = "../data/transcripts/sample_chunks.json"
chunker.save_chunks(chunks, chunks_output_path)
print(f"Saved chunks to {chunks_output_path}")
"""),
    ]
    build_notebook(nb2_cells, "notebooks/02_transcript_chunking.ipynb")

    # -------------------------------------------------------------
    # Notebook 03: Embeddings & Indexing
    # -------------------------------------------------------------
    nb3_cells = [
        make_md_cell("""# Video RAG — Phase 1: Notebook 03
## Dense Embeddings & FAISS Vector Indexing

In this notebook, we:
1. Generate dense vector embeddings for each chunk using **`BAAI/bge-small-en-v1.5`**.
2. Apply **L2 Normalization** on embeddings.
3. Build a **FAISS `IndexFlatIP`** vector index (Inner Product = exact Cosine Similarity on normalized vectors).
4. Persist the vector index and associated metadata for downstream retrieval.
"""),
        make_code_cell("""import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path.cwd().parent))

from src.chunking import TranscriptChunker
from src.embeddings import BGEEmbedder
from src.vector_store import FAISSVectorStore
"""),
        make_md_cell("### Step 1: Load Chunks"),
        make_code_cell("""chunks = TranscriptChunker.load_chunks("../data/transcripts/sample_chunks.json")
print(f"Loaded {len(chunks)} chunks for indexing.")
"""),
        make_md_cell("### Step 2: Initialize BGE Embedding Model"),
        make_code_cell("""embedder = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5", normalize_embeddings=True)
print(f"Embedding Model: {embedder.model_name}")
print(f"Vector Dimension: {embedder.dimension}")
"""),
        make_md_cell("### Step 3: Compute Embeddings for Chunks"),
        make_code_cell("""chunk_embeddings = embedder.embed_chunks(chunks, show_progress_bar=True)
print(f"Chunk Embeddings Matrix Shape: {chunk_embeddings.shape}")
print(f"L2 Norm Verification (first vector): {np.linalg.norm(chunk_embeddings[0]):.4f}")
"""),
        make_md_cell("### Step 4: Ingest into FAISS Vector Store"),
        make_code_cell("""vector_store = FAISSVectorStore(dimension=embedder.dimension)
vector_store.add_chunks(chunks, chunk_embeddings)
print(f"Total Vectors Indexed: {vector_store.total_vectors}")
"""),
        make_md_cell("### Step 5: Test Similarity Search on Sample Query"),
        make_code_cell("""test_query = "What is database normalization?"
query_vec = embedder.embed_query(test_query)
results = vector_store.search(query_vec, top_k=3)

print(f"Query: '{test_query}'\\n")
for rank, res in enumerate(results, 1):
    print(f"Rank {rank} (Cosine Similarity Score: {res['score']:.4f}):")
    print(f"  Chunk ID: {res['chunk_id']} | Time: [{res['start_time']}s - {res['end_time']}s]")
    print(f"  Text: {res['text'][:120]}...\\n")
"""),
        make_md_cell("### Step 6: Save FAISS Index and Metadata"),
        make_code_cell("""vector_store.save(
    index_path="../data/vector_store/faiss.index",
    metadata_path="../data/vector_store/chunks_metadata.json"
)
print("Saved FAISS index and metadata successfully.")
"""),
    ]
    build_notebook(nb3_cells, "notebooks/03_embeddings_and_indexing.ipynb")

    # -------------------------------------------------------------
    # Notebook 04: RAG Question Answering
    # -------------------------------------------------------------
    nb4_cells = [
        make_md_cell("""# Video RAG — Phase 1: Notebook 04
## Timestamp-Grounded Question Answering Pipeline

This notebook demonstrates the complete end-to-end question answering pipeline:
1. User asks a question about the video.
2. The query is embedded and matched against the FAISS index.
3. Top-k evidence chunks with timestamps are retrieved.
4. The LLM (Gemini API or grounded baseline) answers using **strictly** the retrieved context and cites timestamps.
5. Out-of-scope / unanswerable questions are cleanly refused.
"""),
        make_code_cell("""import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

from src.rag_pipeline import VideoRAGPipeline
"""),
        make_md_cell("### Step 1: Initialize the Video RAG Pipeline"),
        make_code_cell("""# Initialize pipeline with pre-built FAISS index
pipeline = VideoRAGPipeline()
pipeline.load_index("../data/vector_store")
pipeline.current_video = "sample.mp4"
print("Video RAG Pipeline ready.")
"""),
        make_md_cell("### Step 2: Test Grounded Question Answering"),
        make_code_cell("""def ask_and_display(question: str, top_k: int = 3):
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    result = pipeline.query(question, top_k=top_k)
    
    print(f"\\nANSWER:\\n{result['answer']}\\n")
    print(f"EVIDENCE CITATION: {result['evidence_raw']}")
    print(f"LATENCY: Retrieval={result['retrieval_latency_sec']}s | Generation={result['generation_latency_sec']}s | Total={result['total_latency_sec']}s")
    
    print("\\nRETRIEVED CONTEXT CHUNKS:")
    for i, c in enumerate(result['retrieved_chunks'], 1):
        print(f"  [{i}] {c['chunk_id']} | Score: {c['score']:.4f} | Time: {c['formatted_time']}")
        print(f"      {c['text'][:100]}...")
"""),
        make_code_cell("""# Test Case 1: In-domain Definition Question
ask_and_display("What is the primary goal of database normalization?")
"""),
        make_code_cell("""# Test Case 2: Specific Concept Question
ask_and_display("What is required for a table to satisfy First Normal Form?")
"""),
        make_code_cell("""# Test Case 3: Negative Test Case (Out of Domain - Expected Refusal)
ask_and_display("What is the capital of France?")
"""),
        make_md_cell("### Step 3: Comparative Analysis — Baseline A vs Baseline B"),
        make_code_cell("""# Compare Baseline B (RAG Top-K Retrieval) with Baseline A (Full Transcript Context)
q = "How does Second Normal Form eliminate partial dependencies?"

print("--- Baseline B: Transcript RAG (Top-3) ---")
res_b = pipeline.query(q, top_k=3)
print(f"Answer: {res_b['answer']}")
print(f"Latency: {res_b['total_latency_sec']}s")
"""),
        make_md_cell("### Step 4: Optional Interactive Loop (Uncomment to use interactively)"),
        make_code_cell("""# Uncomment below to run an interactive terminal query loop:
# while True:
#     user_q = input("\\nAsk a question about the video (or 'exit'): ")
#     if user_q.strip().lower() in ('exit', 'quit', 'q'):
#         break
#     ask_and_display(user_q)
"""),
    ]
    build_notebook(nb4_cells, "notebooks/04_rag_question_answering.ipynb")

    # -------------------------------------------------------------
    # Notebook 05: Baseline Evaluation
    # -------------------------------------------------------------
    nb5_cells = [
        make_md_cell("""# Video RAG — Phase 1: Notebook 05
## Quantitative Baseline Evaluation Benchmark

This notebook runs the formal evaluation benchmark on the annotated dataset:
- **Retrieval Metrics**: Recall@1, Recall@3, Recall@5, MRR
- **Temporal Grounding Metrics**: Temporal IoU, Top-1 Overlap Hit Rate
- **Answer Quality Metrics**: Abstention Accuracy for unanswerable questions
- **System Metrics**: Average Retrieval Latency, Generation Latency, Total Latency

Results are exported to `experiments/results/baseline_results.json` and `.csv`.
"""),
        make_code_cell("""import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(str(Path.cwd().parent))

from src.rag_pipeline import VideoRAGPipeline
from src.evaluation import RAGEvaluator
"""),
        make_md_cell("### Step 1: Initialize Pipeline & Load Dataset"),
        make_code_cell("""pipeline = VideoRAGPipeline()
pipeline.load_index("../data/vector_store")
pipeline.current_video = "sample.mp4"

dataset_path = "../data/evaluation/benchmark_questions.json"
evaluator = RAGEvaluator(output_dir="../experiments/results")
questions = evaluator.load_evaluation_dataset(dataset_path)

print(f"Loaded {len(questions)} evaluation questions from {dataset_path}")
"""),
        make_md_cell("### Step 2: Run Full Benchmark"),
        make_code_cell("""# Execute evaluation across all benchmark questions with top_k=5
results_summary = pipeline.run_evaluation(
    eval_dataset_path=dataset_path,
    experiment_name="baseline_results",
    top_k=5
)
"""),
        make_md_cell("### Step 3: Evaluation Summary Metrics"),
        make_code_cell("""print("=" * 60)
print("        VIDEO RAG PHASE 1 — EVALUATION RESULTS")
print("=" * 60)
print(f"Total Questions Evaluated:    {results_summary['counts']['total_queries']}")
print(f"  - Answerable Queries:       {results_summary['counts']['answerable_queries']}")
print(f"  - Unanswerable Queries:     {results_summary['counts']['unanswerable_queries']}")
print("-" * 60)
print("RETRIEVAL METRICS:")
for k, v in results_summary['retrieval_metrics'].items():
    print(f"  {k:20s}: {v}")

print("-" * 60)
print("TEMPORAL GROUNDING METRICS:")
for k, v in results_summary['temporal_grounding_metrics'].items():
    print(f"  {k:20s}: {v}")

print("-" * 60)
print("ANSWER & ABSTENTION METRICS:")
for k, v in results_summary['answer_metrics'].items():
    print(f"  {k:20s}: {v}")

print("-" * 60)
print("LATENCY (Seconds):")
for k, v in results_summary['latency_metrics_sec'].items():
    print(f"  {k:20s}: {v}s")
print("=" * 60)
"""),
        make_md_cell("### Step 4: Detailed Sample Results Table"),
        make_code_cell("""df_results = pd.read_csv("../experiments/results/baseline_results.csv")
display_cols = [
    "sample_id", "question", "answerable", "retrieval_hit_at_1", 
    "top1_temporal_iou", "abstention_correct", "total_latency_sec"
]
df_results[display_cols]
"""),
        make_md_cell("### Step 5: Metric Visualizations"),
        make_code_cell("""# Plot Retrieval & Temporal Performance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Retrieval Recall & MRR
ret_metrics = results_summary['retrieval_metrics']
axes[0].bar(list(ret_metrics.keys()), list(ret_metrics.values()), color="#2b5c8f", edgecolor="black")
axes[0].set_ylim(0, 1.1)
axes[0].set_title("Retrieval Performance (Recall@K & MRR)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Score")
axes[0].grid(axis="y", linestyle="--", alpha=0.7)
for i, (k, v) in enumerate(ret_metrics.items()):
    if v is not None:
        axes[0].text(i, v + 0.03, f"{v:.2f}", ha="center", fontweight="bold")

# Plot 2: Latency Breakdown
lat_metrics = results_summary['latency_metrics_sec']
axes[1].bar(["Retrieval", "Generation", "Total"], list(lat_metrics.values()), color="#3f9b0b", edgecolor="black")
axes[1].set_title("Average Latency Breakdown (Seconds)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Seconds")
axes[1].grid(axis="y", linestyle="--", alpha=0.7)
for i, (k, v) in enumerate(lat_metrics.items()):
    axes[1].text(i, v + 0.01, f"{v:.3f}s", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("../experiments/results/evaluation_plot.png", dpi=150)
plt.show()
"""),
    ]
    build_notebook(nb5_cells, "notebooks/05_baseline_evaluation.ipynb")

if __name__ == "__main__":
    generate_all()
