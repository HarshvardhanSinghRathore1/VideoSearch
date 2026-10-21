import React from 'react';

/**
 * System Architecture and Research Scope Modal.
 */
export function ResearchMetadata({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>VideoRAG Phase 1 Architecture</h3>
          <button className="btn-icon" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">
          <div className="arch-flow">
            <div className="arch-step">1. Video Container (MP4/MKV/MOV/AVI/WEBM)</div>
            <div className="arch-arrow">↓</div>
            <div className="arch-step">2. Audio Extraction via FFmpeg (16kHz Mono WAV)</div>
            <div className="arch-arrow">↓</div>
            <div className="arch-step">3. Speech Transcription via Whisper (Timestamped JSON & TXT)</div>
            <div className="arch-arrow">↓</div>
            <div className="arch-step">4. Windowed Chunking with Contiguous Temporal Spans</div>
            <div className="arch-arrow">↓</div>
            <div className="arch-step">5. Dense Embedding with BAAI/bge-small-en-v1.5 (384-dim)</div>
            <div className="arch-arrow">↓</div>
            <div className="arch-step">6. FAISS IndexFlatIP Vector Similarity Search</div>
            <div className="arch-arrow">↓</div>
            <div className="arch-step">7. Grounded LLM Prompting with Timestamp Citations</div>
          </div>
          <div className="arch-footer-note">
            <strong>Research Scope Note:</strong> Phase 1 implements strictly transcript-based RAG. Visual RAG, OCR, frame analysis, and multimodal embeddings belong to future research phases.
          </div>
        </div>
      </div>
    </div>
  );
}
