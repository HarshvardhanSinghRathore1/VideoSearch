import React from 'react';

const STAGES_CONFIG = [
  { id: 'upload', number: 1, title: 'Video Uploaded', desc: 'Container validated and registered' },
  { id: 'audio', number: 2, title: 'Audio Extraction', desc: 'FFmpeg 16kHz mono PCM stream' },
  { id: 'transcribe', number: 3, title: 'Speech Transcription', desc: 'Whisper speech-to-text with timestamps' },
  { id: 'chunk', number: 4, title: 'Transcript Chunking', desc: 'Sliding window with temporal bounds' },
  { id: 'embed', number: 5, title: 'Embedding Generation', desc: 'BAAI/bge-small-en-v1.5 (384-dim)' },
  { id: 'index', number: 6, title: 'FAISS Indexing', desc: 'IndexFlatIP vector similarity database' },
  { id: 'ready', number: 7, title: 'Ready', desc: 'Video ready for grounded QA' },
];

/**
 * Screen 2 Processing Dashboard Stepper Component.
 */
export function ProcessingStatus({ processingStatus, onEnterWorkspace }) {
  const { status, progress, message, stages, error } = processingStatus;

  return (
    <div className="processing-container card">
      <div className="processing-header">
        {status !== 'completed' && status !== 'failed' && (
          <div className="spinner-large"></div>
        )}
        {status === 'completed' && (
          <div style={{ fontSize: '2rem', color: 'var(--accent-emerald)' }}>✓</div>
        )}
        {status === 'failed' && (
          <div style={{ fontSize: '2rem', color: 'var(--accent-rose)' }}>✕</div>
        )}
        <div>
          <h2>Processing Video Pipeline</h2>
          <p className="status-subtitle">
            {message || 'Executing transcript extraction, chunking, and vector indexing...'}
          </p>
        </div>
      </div>

      <div className="progress-bar-container">
        <div className="progress-bar-fill" style={{ width: `${progress}%` }}></div>
      </div>

      <div className="pipeline-stepper">
        {STAGES_CONFIG.map((s) => {
          const stageData = stages?.[s.id] || { status: 'pending' };
          const stageStatus = stageData.status || 'pending';

          return (
            <div key={s.id} className={`step-item ${stageStatus}`}>
              <div className="step-icon">
                {stageStatus === 'completed' ? '✓' : s.number}
              </div>
              <div className="step-content">
                <div className="step-title">{s.title}</div>
                <div className="step-desc">{s.desc}</div>
              </div>
              <div className={`step-badge badge-${stageStatus}`}>
                {stageStatus.charAt(0).toUpperCase() + stageStatus.slice(1)}
              </div>
            </div>
          );
        })}
      </div>

      {error && <div className="alert-box error">{error}</div>}

      {status === 'completed' && (
        <div className="processing-footer">
          <button className="btn btn-primary" onClick={onEnterWorkspace} type="button">
            Enter Video QA Workspace →
          </button>
        </div>
      )}
    </div>
  );
}
