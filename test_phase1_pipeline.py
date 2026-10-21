"""
Phase 1 End-to-End Verification Test Script
Executes all stages of the Video RAG Phase 1 pipeline:
1. Video audio extraction
2. Timestamped transcription verification
3. Timestamp-preserving chunking
4. BGE dense embedding generation
5. FAISS vector store save & load
6. Top-K query retrieval with similarity scores
7. Grounded LLM answering (5 normal + 2 unanswerable questions)
8. Formal evaluation benchmark execution with metrics export
"""

import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.video_processor import VideoProcessor
from src.transcription import Transcriber, TranscriptSegment, format_timestamp
from src.chunking import TranscriptChunker
from src.embeddings import BGEEmbedder
from src.vector_store import FAISSVectorStore
from src.retriever import TranscriptRetriever
from src.llm import GroundedLLMGenerator
from src.evaluation import RAGEvaluator
from src.rag_pipeline import VideoRAGPipeline


def run_tests():
    print("=" * 80)
    print("STARTING VIDEO RAG PHASE 1 VERIFICATION TESTS")
    print("=" * 80)

    # 1. Test VideoProcessor
    print("\n[TEST 1/8] Testing VideoProcessor...")
    processor = VideoProcessor(output_audio_dir="data/audio")
    v_path = Path("data/videos/sample.mp4")
    meta = processor.get_media_metadata(v_path)
    print(f"  Video Metadata: {meta}")
    extracted_audio = processor.extract_audio(v_path, overwrite=True)
    assert extracted_audio.exists(), f"Audio extraction failed: {extracted_audio}"
    print(f"  Audio successfully extracted to: {extracted_audio}")

    # 2. Test Transcriber
    print("\n[TEST 2/8] Testing Transcriber & Timestamp Preservation...")
    transcriber = Transcriber(output_dir="data/transcripts")
    # Load transcript segments
    json_path = Path("data/transcripts/sample_transcript.json")
    if json_path.exists():
        segments = Transcriber.load_transcript_json(json_path)
    else:
        segments = transcriber.transcribe(extracted_audio)
    assert len(segments) > 0, "No segments found in transcript!"
    for i, seg in enumerate(segments[:3], 1):
        print(f"  Segment {i}: [{format_timestamp(seg.start)} - {format_timestamp(seg.end)}] {seg.text[:60]}...")
    transcriber.save_transcripts(segments, "sample.mp4")

    # 3. Test Chunking
    print("\n[TEST 3/8] Testing Timestamp-Preserving Chunking...")
    chunker = TranscriptChunker(target_chunk_size=75, chunk_overlap=15)
    chunks = chunker.chunk_segments(segments, source_video="sample.mp4")
    assert len(chunks) > 0, "No chunks generated!"
    for c in chunks:
        print(f"  Chunk {c.chunk_id}: [{c.start_time}s - {c.end_time}s] ({c.token_count_approx} tokens) -> {c.text[:70]}...")
        assert c.start_time < c.end_time, f"Invalid chunk timestamps: {c.start_time} >= {c.end_time}"
    chunker.save_chunks(chunks, "data/transcripts/sample_chunks.json")

    # 4. Test Embeddings
    print("\n[TEST 4/8] Testing BGE Embeddings (BAAI/bge-small-en-v1.5)...")
    embedder = BGEEmbedder(model_name="BAAI/bge-small-en-v1.5", normalize_embeddings=True)
    chunk_embeddings = embedder.embed_chunks(chunks)
    print(f"  Embedding Dimension: {embedder.dimension}")
    print(f"  Chunk Embeddings Matrix: shape={chunk_embeddings.shape}, dtype={chunk_embeddings.dtype}")
    assert chunk_embeddings.shape == (len(chunks), 384), f"Unexpected embedding shape: {chunk_embeddings.shape}"

    # 5. Test FAISS Vector Store Save & Load
    print("\n[TEST 5/8] Testing FAISS Vector Store Indexing, Persistence & Reloading...")
    vstore = FAISSVectorStore(dimension=embedder.dimension)
    vstore.add_chunks(chunks, chunk_embeddings)
    assert vstore.total_vectors == len(chunks), "Vector store count mismatch!"
    
    idx_dir = Path("data/vector_store")
    vstore.save(idx_dir / "faiss.index", idx_dir / "chunks_metadata.json")
    
    reloaded_vstore = FAISSVectorStore(dimension=embedder.dimension)
    reloaded_vstore.load(idx_dir / "faiss.index", idx_dir / "chunks_metadata.json")
    assert reloaded_vstore.total_vectors == len(chunks), "Reloaded vector store count mismatch!"
    print(f"  Vector store successfully saved and reloaded with {reloaded_vstore.total_vectors} vectors.")

    # 6. Test Retriever
    print("\n[TEST 6/8] Testing TranscriptRetriever...")
    retriever = TranscriptRetriever(embedder=embedder, vector_store=reloaded_vstore, default_top_k=3)
    query_text = "What is the primary goal of database normalization?"
    results = retriever.retrieve(query_text, top_k=3)
    assert len(results) > 0, "Retriever returned 0 chunks!"
    print(f"  Query: '{query_text}'")
    for r in results:
        print(f"    - [{r['chunk_id']}] Score: {r['score']:.4f} | Time: {r['formatted_time']} | {r['text'][:60]}...")

    # 7. Test Pipeline on 5 Normal + 2 Unanswerable Questions
    print("\n[TEST 7/8] Testing End-to-End Grounded QA on 7 Test Queries...")
    pipeline = VideoRAGPipeline(
        embedder=embedder,
        vector_store=reloaded_vstore,
    )
    pipeline.segments = segments
    pipeline.chunks = chunks
    pipeline.current_video = "sample.mp4"

    test_queries = [
        # 5 In-domain answerable questions
        ("What is the primary goal of database normalization?", True),
        ("What is required for a table to satisfy First Normal Form (1NF)?", True),
        ("How does Second Normal Form eliminate partial dependencies?", True),
        ("What is transitive dependency and how is it resolved in 3NF?", True),
        ("What is the purpose of the K-Means clustering algorithm?", True),
        # 2 Out-of-domain unanswerable questions
        ("What is the capital city of France?", False),
        ("How does quantum key distribution work?", False),
    ]

    for q_text, is_answerable in test_queries:
        print("-" * 75)
        print(f"Q: {q_text} [Expected Answerable: {is_answerable}]")
        ans_res = pipeline.query(q_text, top_k=3)
        print(f"Answer: {ans_res['answer']}")
        print(f"Evidence: {ans_res['evidence_raw']}")
        print(f"Latency: {ans_res['total_latency_sec']}s (Ret: {ans_res['retrieval_latency_sec']}s, Gen: {ans_res['generation_latency_sec']}s)")

    # 8. Test Formal Evaluation Benchmark
    print("\n[TEST 8/8] Running Benchmark Evaluation on data/evaluation/benchmark_questions.json...")
    benchmark_path = Path("data/evaluation/benchmark_questions.json")
    summary = pipeline.run_evaluation(
        eval_dataset_path=benchmark_path,
        experiment_name="baseline_results",
        top_k=5
    )

    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION SUMMARY METRICS")
    print("=" * 80)
    print(f"Total Evaluated: {summary['counts']['total_queries']} questions")
    print(f"Retrieval Recall@1:           {summary['retrieval_metrics']['Recall@1']}")
    print(f"Retrieval Recall@3:           {summary['retrieval_metrics']['Recall@3']}")
    print(f"Retrieval Recall@5:           {summary['retrieval_metrics']['Recall@5']}")
    print(f"Retrieval MRR:                {summary['retrieval_metrics']['MRR']}")
    print(f"Mean Top-1 Temporal IoU:      {summary['temporal_grounding_metrics']['Mean_Top1_IoU']}")
    print(f"Top-1 Temporal Hit Rate:      {summary['temporal_grounding_metrics']['Top1_Temporal_Hit_Rate']}")
    print(f"Abstention Accuracy:          {summary['answer_metrics']['Abstention_Accuracy']}")
    print(f"Average Total Latency:        {summary['latency_metrics_sec']['Avg_Total_Latency']}s")
    print("=" * 80)

    # Check generated output files
    assert Path("experiments/results/baseline_results.json").exists(), "Missing baseline_results.json"
    assert Path("experiments/results/baseline_results.csv").exists(), "Missing baseline_results.csv"
    print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_tests()
