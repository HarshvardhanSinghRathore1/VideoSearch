"""
FastAPI REST API Layer for VideoRAG Phase 1
Acts as a clean, thin API interface over the existing Phase 1 VideoRAGPipeline.
Supports video upload, real-time stage-by-stage status tracking, grounded RAG queries,
and HTTP Range video streaming for native HTML5 video player seeking.
"""

import os
import re
import uuid
import time
import shutil
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.video_processor import VideoProcessor, SUPPORTED_FORMATS
from src.transcription import Transcriber, format_timestamp
from src.chunking import TranscriptChunker
from src.embeddings import BGEEmbedder, DEFAULT_MODEL_NAME
from src.vector_store import FAISSVectorStore
from src.retriever import TranscriptRetriever
from src.llm import GroundedLLMGenerator
from src.rag_pipeline import VideoRAGPipeline
from src.question_generator import QuestionGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("VideoRAG_API")

app = FastAPI(
    title="VideoRAG Phase 1 API",
    description="Transcript-Grounded Video Question Answering REST API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
DATA_DIR = Path("data")
VIDEOS_DIR = DATA_DIR / "videos"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
VECTOR_DIR = DATA_DIR / "vector_store"
FRONTEND_DIR = Path("frontend")

for d in [VIDEOS_DIR, AUDIO_DIR, TRANSCRIPTS_DIR, VECTOR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Shared embedding model, LLM generator, and question generator
global_embedder: Optional[BGEEmbedder] = None
global_llm_generator: Optional[GroundedLLMGenerator] = None
global_question_generator: Optional[QuestionGenerator] = None
pipelines: Dict[str, VideoRAGPipeline] = {}
processing_jobs: Dict[str, Dict[str, Any]] = {}
video_metadata_registry: Dict[str, Dict[str, Any]] = {}


def get_embedder() -> BGEEmbedder:
    """Singleton getter for BGE embedding model."""
    global global_embedder
    if global_embedder is None:
        global_embedder = BGEEmbedder(model_name=DEFAULT_MODEL_NAME, normalize_embeddings=True)
    return global_embedder


def get_llm_generator() -> GroundedLLMGenerator:
    """Singleton getter for GroundedLLMGenerator."""
    global global_llm_generator
    if global_llm_generator is None:
        global_llm_generator = GroundedLLMGenerator()
    return global_llm_generator


def get_question_generator() -> QuestionGenerator:
    """Singleton getter for QuestionGenerator."""
    global global_question_generator
    if global_question_generator is None:
        global_question_generator = QuestionGenerator(llm_generator=get_llm_generator())
    return global_question_generator


def initialize_existing_videos():
    """Initializes sample.mp4 and any existing videos in data/videos on startup."""
    for video_file in VIDEOS_DIR.glob("*"):
        if video_file.suffix.lower() in SUPPORTED_FORMATS:
            stem = video_file.stem
            video_id = "sample" if video_file.name == "sample.mp4" else stem
            is_sample = (video_file.name == "sample.mp4")
            
            video_metadata_registry[video_id] = {
                "id": video_id,
                "filename": video_file.name,
                "path": str(video_file.resolve()),
                "size_mb": round(video_file.stat().st_size / (1024 * 1024), 2),
                "status": "ready",
                "is_sample": is_sample
            }
            try:
                # Check for dedicated or global vector store
                idx_file = VECTOR_DIR / f"{stem}_faiss.index"
                meta_file = VECTOR_DIR / f"{stem}_metadata.json"
                if not (idx_file.exists() and meta_file.exists()):
                    if is_sample:
                        idx_file = VECTOR_DIR / "faiss.index"
                        meta_file = VECTOR_DIR / "chunks_metadata.json"

                pipe = VideoRAGPipeline(embedder=get_embedder(), llm_generator=get_llm_generator())
                if idx_file.exists() and meta_file.exists():
                    pipe.vector_store.load(idx_file, meta_file)
                    pipe.current_video = video_file.name
                    
                    t_json = TRANSCRIPTS_DIR / f"{stem}_transcript.json"
                    if t_json.exists():
                        pipe.segments = Transcriber.load_transcript_json(t_json)
                    pipelines[video_id] = pipe
                    logger.info(f"Initialized pre-indexed pipeline for video '{video_id}' ({video_file.name}).")
                else:
                    video_metadata_registry[video_id]["status"] = "uploaded"
            except Exception as e:
                logger.warning(f"Could not pre-load vector store for {video_file.name}: {e}")


@app.on_event("startup")
async def startup_event():
    initialize_existing_videos()


# --- Pydantic Request Models ---
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


# --- Video Streaming with HTTP Range Support ---
def send_bytes_range_requests(
    file_path: Path,
    range_header: Optional[str] = None
) -> StreamingResponse:
    """Streams video file supporting HTTP 206 Partial Content for native player seeking."""
    file_size = file_path.stat().st_size
    content_type = "video/mp4"
    if file_path.suffix.lower() == ".webm":
        content_type = "video/webm"
    elif file_path.suffix.lower() == ".mov":
        content_type = "video/quicktime"
    elif file_path.suffix.lower() == ".mkv":
        content_type = "video/x-matroska"

    if range_header:
        byte1, byte2 = 0, None
        match = re.search(r"bytes=(\d+)-(\d*)", range_header)
        if match:
            g1, g2 = match.groups()
            byte1 = int(g1)
            if g2:
                byte2 = int(g2)

        if byte2 is None or byte2 >= file_size:
            byte2 = file_size - 1

        length = byte2 - byte1 + 1

        def iterfile():
            with open(file_path, "rb") as f:
                f.seek(byte1)
                chunk_size = 1024 * 512
                remaining = length
                while remaining > 0:
                    read_size = min(chunk_size, remaining)
                    data = f.read(read_size)
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {byte1}-{byte2}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
            "Content-Type": content_type,
        }
        return StreamingResponse(iterfile(), status_code=206, headers=headers)
    else:
        def iterfile_full():
            with open(file_path, "rb") as f:
                while chunk := f.read(1024 * 512):
                    yield chunk

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": content_type,
        }
        return StreamingResponse(iterfile_full(), status_code=200, headers=headers)


# --- Background Pipeline Worker ---
def run_video_pipeline_background(video_id: str, video_path: Path):
    """Executes the actual multi-stage processing tracking real progress."""
    job = processing_jobs[video_id]
    try:
        embedder = get_embedder()
        pipeline = VideoRAGPipeline(embedder=embedder, llm_generator=get_llm_generator())
        pipeline.current_video = video_path.name

        # Stage 1: Video Uploaded
        job["stages"]["upload"]["status"] = "completed"
        job["progress"] = 15

        # Stage 2: Audio Extraction
        job["current_stage"] = "audio"
        job["stages"]["audio"]["status"] = "processing"
        job["message"] = "Extracting mono audio stream using FFmpeg..."
        audio_path = pipeline.video_processor.extract_audio(video_path, overwrite=True)
        job["stages"]["audio"]["status"] = "completed"
        job["progress"] = 30

        # Stage 3: Speech Transcription
        job["current_stage"] = "transcribe"
        job["stages"]["transcribe"]["status"] = "processing"
        job["message"] = "Performing speech-to-text with timestamp preservation..."
        
        # Check if pre-existing ground-truth transcript exists for sample
        sample_json = TRANSCRIPTS_DIR / f"{video_path.stem}_transcript.json"
        if sample_json.exists() and video_id == "sample":
            segments = Transcriber.load_transcript_json(sample_json)
        else:
            segments = pipeline.transcriber.transcribe(audio_path)
            pipeline.transcriber.save_transcripts(segments, video_path.name)
        
        pipeline.segments = segments
        job["stages"]["transcribe"]["status"] = "completed"
        job["segment_count"] = len(segments)
        job["progress"] = 55

        # Stage 4: Transcript Chunking
        job["current_stage"] = "chunk"
        job["stages"]["chunk"]["status"] = "processing"
        job["message"] = "Chunking transcript with sliding-window temporal boundaries..."
        chunks = pipeline.chunker.chunk_segments(segments, source_video=video_path.name)
        pipeline.chunks = chunks
        job["stages"]["chunk"]["status"] = "completed"
        job["chunk_count"] = len(chunks)
        job["progress"] = 70

        # Stage 5: Embedding Generation
        job["current_stage"] = "embed"
        job["stages"]["embed"]["status"] = "processing"
        job["message"] = f"Generating dense vectors via {DEFAULT_MODEL_NAME}..."
        embeddings = embedder.embed_chunks(chunks)
        job["stages"]["embed"]["status"] = "completed"
        job["progress"] = 85

        # Stage 6: FAISS Indexing
        job["current_stage"] = "index"
        job["stages"]["index"]["status"] = "processing"
        job["message"] = "Constructing FAISS IndexFlatIP vector database..."
        pipeline.vector_store.add_chunks(chunks, embeddings)
        job["stages"]["index"]["status"] = "completed"
        job["progress"] = 95

        # Stage 7: Ready
        job["current_stage"] = "ready"
        job["stages"]["ready"]["status"] = "completed"
        job["status"] = "completed"
        job["progress"] = 100
        job["message"] = "Video processing completed. Ready for questions."

        # Register pipeline
        pipelines[video_id] = pipeline
        video_metadata_registry[video_id]["status"] = "ready"
        logger.info(f"Video {video_id} successfully processed and registered.")

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {e}", exc_info=True)
        current = job.get("current_stage", "upload")
        if current in job["stages"]:
            job["stages"][current]["status"] = "failed"
        job["status"] = "failed"
        job["error"] = str(e)
        job["message"] = f"Processing failed at stage '{current}': {str(e)}"


# --- REST Endpoints ---

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "VideoRAG API"}


@app.get("/api/system/info")
async def get_system_info():
    """Returns metadata about the research pipeline."""
    return {
        "system": "VideoRAG",
        "phase": "Phase 1 — Transcript RAG Baseline",
        "embedding_model": DEFAULT_MODEL_NAME,
        "embedding_dimension": 384,
        "vector_store": "FAISS IndexFlatIP (Cosine Similarity)",
        "speech_to_text": "Whisper / faster-whisper",
        "llm_provider": "Google Gemini (or offline grounded engine)",
        "supported_formats": list(SUPPORTED_FORMATS),
    }


@app.get("/api/videos/list")
async def list_videos():
    """Lists available videos in the system."""
    video_list = []
    for v_id, meta in video_metadata_registry.items():
        v_path = Path(meta["path"])
        if v_path.exists():
            video_list.append({
                "id": v_id,
                "filename": meta["filename"],
                "size_mb": meta["size_mb"],
                "status": meta.get("status", "ready" if v_id in pipelines else "pending"),
                "is_sample": meta.get("is_sample", False),
            })
    return {"videos": video_list}


@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    """Uploads a video and registers it for processing."""
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    video_id = str(uuid.uuid4())[:8]
    clean_filename = f"{video_id}_{Path(file.filename).name}"
    target_path = VIDEOS_DIR / clean_filename

    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    file_size_mb = round(target_path.stat().st_size / (1024 * 1024), 2)
    video_metadata_registry[video_id] = {
        "id": video_id,
        "filename": file.filename,
        "path": str(target_path.resolve()),
        "size_mb": file_size_mb,
        "status": "uploaded",
        "is_sample": False
    }

    return {
        "id": video_id,
        "filename": file.filename,
        "size_mb": file_size_mb,
        "status": "uploaded",
        "message": "Video uploaded successfully. Ready to process."
    }


@app.post("/api/videos/{video_id}/process")
async def process_video(video_id: str, background_tasks: BackgroundTasks):
    """Triggers the 7-stage processing pipeline for a video."""
    if video_id not in video_metadata_registry:
        raise HTTPException(status_code=404, detail="Video ID not found.")

    meta = video_metadata_registry[video_id]
    v_path = Path(meta["path"])
    if not v_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing on disk.")

    # Initialize stage tracking
    processing_jobs[video_id] = {
        "video_id": video_id,
        "status": "processing",
        "current_stage": "upload",
        "progress": 10,
        "message": "Initializing pipeline...",
        "stages": {
            "upload": {"label": "Video uploaded", "status": "completed"},
            "audio": {"label": "Audio extraction", "status": "pending"},
            "transcribe": {"label": "Speech transcription", "status": "pending"},
            "chunk": {"label": "Transcript chunking", "status": "pending"},
            "embed": {"label": "Embedding generation", "status": "pending"},
            "index": {"label": "FAISS indexing", "status": "pending"},
            "ready": {"label": "Ready for questions", "status": "pending"},
        }
    }

    # Run processing asynchronously
    thread = threading.Thread(target=run_video_pipeline_background, args=(video_id, v_path))
    thread.daemon = True
    thread.start()

    return {
        "video_id": video_id,
        "status": "processing",
        "message": "Processing started."
    }


@app.get("/api/videos/{video_id}/status")
async def get_processing_status(video_id: str):
    """Returns stage-by-stage status of video processing."""
    if video_id in processing_jobs:
        return processing_jobs[video_id]
    
    # If already ready from startup
    if video_id in pipelines:
        meta = video_metadata_registry.get(video_id, {})
        return {
            "video_id": video_id,
            "status": "completed",
            "progress": 100,
            "message": "Video is ready for questions.",
            "stages": {
                "upload": {"label": "Video uploaded", "status": "completed"},
                "audio": {"label": "Audio extraction", "status": "completed"},
                "transcribe": {"label": "Speech transcription", "status": "completed"},
                "chunk": {"label": "Transcript chunking", "status": "completed"},
                "embed": {"label": "Embedding generation", "status": "completed"},
                "index": {"label": "FAISS indexing", "status": "completed"},
                "ready": {"label": "Ready for questions", "status": "completed"},
            }
        }

    if video_id in video_metadata_registry:
        return {
            "video_id": video_id,
            "status": "pending",
            "progress": 0,
            "message": "Video uploaded, awaiting processing.",
            "stages": {
                "upload": {"label": "Video uploaded", "status": "completed"},
                "audio": {"label": "Audio extraction", "status": "pending"},
                "transcribe": {"label": "Speech transcription", "status": "pending"},
                "chunk": {"label": "Transcript chunking", "status": "pending"},
                "embed": {"label": "Embedding generation", "status": "pending"},
                "index": {"label": "FAISS indexing", "status": "pending"},
                "ready": {"label": "Ready for questions", "status": "pending"},
            }
        }

    raise HTTPException(status_code=404, detail="Video ID not found.")


@app.get("/api/videos/{video_id}/video")
async def stream_video(video_id: str, request: Request):
    """Streams video file supporting HTML5 seeking and scrubbing."""
    if video_id not in video_metadata_registry:
        raise HTTPException(status_code=404, detail="Video ID not found.")

    v_path = Path(video_metadata_registry[video_id]["path"])
    if not v_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing.")

    range_header = request.headers.get("Range")
    return send_bytes_range_requests(v_path, range_header)


@app.get("/api/videos/{video_id}/transcript")
async def get_transcript(video_id: str):
    """Returns the full timestamped transcript segments."""
    if video_id not in pipelines:
        raise HTTPException(status_code=400, detail="Video has not been processed yet.")

    pipe = pipelines[video_id]
    return {
        "video_id": video_id,
        "video_name": pipe.current_video,
        "segment_count": len(pipe.segments),
        "segments": [seg.to_dict() for seg in pipe.segments]
    }


@app.get("/api/videos/{video_id}/suggested-questions")
@app.post("/api/videos/{video_id}/suggested-questions")
async def get_suggested_questions(video_id: str, force: bool = False):
    """
    Returns 4 to 6 dynamically generated questions grounded in the video's transcript.
    Uses disk caching to return instantly on subsequent visits.
    """
    if video_id not in video_metadata_registry and video_id not in pipelines:
        raise HTTPException(status_code=404, detail="Video ID not found.")

    qg = get_question_generator()
    pipe = pipelines.get(video_id)
    segments = pipe.segments if pipe and pipe.segments else None

    try:
        cached_before = qg.get_cached_questions(video_id) is not None if not force else False
        questions = qg.generate_questions(video_id, segments_or_chunks=segments, force_regenerate=force)
        return {
            "video_id": video_id,
            "questions": questions,
            "count": len(questions),
            "cached": cached_before,
            "source": "transcript",
            "model": "Gemini (Grounded)"
        }
    except Exception as e:
        logger.error(f"Error generating suggested questions for {video_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate suggested questions: {str(e)}")


@app.post("/api/videos/{video_id}/query")
async def query_video(video_id: str, req: QueryRequest):
    """Executes a grounded RAG query on the video transcript."""
    if video_id not in pipelines:
        raise HTTPException(status_code=400, detail="Video is not processed or ready for queries.")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    pipeline = pipelines[video_id]
    result = pipeline.query(question=question, top_k=req.top_k)

    # Check for refusal / abstention
    refusal_keywords = [
        "not contain information",
        "not found in the video",
        "does not mention",
        "cannot be determined",
        "not provided",
    ]
    is_abstained = any(k in result["answer"].lower() for k in refusal_keywords)

    # Format structured evidence items
    evidence_items = []
    if not is_abstained and result.get("evidence_timestamps"):
        for ev in result["evidence_timestamps"]:
            evidence_items.append({
                "chunk_id": ev["chunk_id"],
                "start": ev["start"],
                "end": ev["end"],
                "formatted": ev["formatted"],
            })

    # Format ranked chunks for Research Mode
    ranked_chunks = []
    for rank, chunk in enumerate(result["retrieved_chunks"], 1):
        ranked_chunks.append({
            "rank": rank,
            "chunk_id": chunk["chunk_id"],
            "score": chunk["score"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "formatted_time": chunk["formatted_time"],
            "text": chunk["text"],
            "source_video": chunk.get("source_video", pipeline.current_video or ""),
        })

    return {
        "question": question,
        "answer": result["answer"],
        "evidence": evidence_items,
        "evidence_raw": result.get("evidence_raw", "None" if is_abstained else ""),
        "retrieved_chunks": ranked_chunks,
        "abstained": is_abstained,
        "source_video": pipeline.current_video or "video",
        "context_str": result["context_str"],
        "latencies": {
            "retrieval_sec": result["retrieval_latency_sec"],
            "generation_sec": result["generation_latency_sec"],
            "total_sec": result["total_latency_sec"],
        },
        "model": result.get("model", ""),
        "top_k": req.top_k,
    }


# Mount Static Frontend Files
dist_dir = FRONTEND_DIR / "dist"
if dist_dir.exists():
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")
elif FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
