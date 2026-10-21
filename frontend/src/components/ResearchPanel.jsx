import React from 'react';
import { LatencyMetrics } from './LatencyMetrics';
import { RetrievedChunk } from './RetrievedChunk';

/**
 * Collapsible Research Mode Drawer / Inspection Panel.
 * Separates User Query, Retrieved Evidence, LLM Context, Generated Answer, and Question Generation metadata.
 */
export function ResearchPanel({ isOpen, onClose, queryResult, onSeek, suggestedQuestions = [] }) {
  const chunks = queryResult?.retrieved_chunks || [];
  const latencies = queryResult?.latencies;
  const topK = queryResult?.top_k || 5;

  return (
    <aside className={`research-drawer ${isOpen ? 'open' : ''}`}>
      <div className="drawer-header">
        <div className="drawer-title">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="20" x2="18" y2="10"></line>
            <line x1="12" y1="20" x2="12" y2="4"></line>
            <line x1="6" y1="20" x2="6" y2="14"></line>
          </svg>
          <h3>Research Mode — Retrieval & Latency Inspector</h3>
        </div>
        <button className="btn-icon" onClick={onClose} type="button">✕</button>
      </div>

      <div className="drawer-body">
        {/* Latency Breakdown */}
        <LatencyMetrics latencies={latencies} topK={topK} />

        {/* Configuration Summary */}
        <div className="config-summary-card">
          <div className="config-row">
            <span>Embedding Model:</span>
            <strong>BAAI/bge-small-en-v1.5 (384-dim)</strong>
          </div>
          <div className="config-row">
            <span>Vector Store:</span>
            <strong>FAISS IndexFlatIP (Cosine Sim)</strong>
          </div>
          <div className="config-row">
            <span>Speech Engine:</span>
            <strong>Whisper Timestamped Segments</strong>
          </div>
          <div className="config-row">
            <span>LLM Backend:</span>
            <strong>{queryResult?.model || 'Gemini Grounded'}</strong>
          </div>
        </div>

        {/* Dynamic Question Generation Metadata */}
        <div className="research-section">
          <h4>Dynamic Question Generation</h4>
          <div className="config-summary-card">
            <div className="config-row">
              <span>Source:</span>
              <strong>Transcript</strong>
            </div>
            <div className="config-row">
              <span>Generation Model:</span>
              <strong>Gemini</strong>
            </div>
            <div className="config-row">
              <span>Number of Suggestions:</span>
              <strong>{suggestedQuestions.length}</strong>
            </div>
          </div>
        </div>

        {/* 1. User Query */}
        <div className="research-section">
          <h4>1. User Query</h4>
          <div className="code-box">{queryResult?.question || 'No active query'}</div>
        </div>

        {/* 2. Top-K Retrieved Chunks */}
        <div className="research-section">
          <h4>2. Top-K Retrieved Transcript Chunks ({chunks.length})</h4>
          {chunks.length > 0 ? (
            <div className="chunks-list">
              {chunks.map((chunk) => (
                <RetrievedChunk key={chunk.chunk_id || chunk.rank} chunk={chunk} onSeek={onSeek} />
              ))}
            </div>
          ) : (
            <p className="placeholder-text" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Ask a question to inspect retrieved FAISS chunks.
            </p>
          )}
        </div>

        {/* 3. Context Sent to LLM */}
        <div className="research-section">
          <h4>3. Context Sent to LLM</h4>
          <pre className="code-box pre-wrap">
            {queryResult?.context_str || 'No context generated'}
          </pre>
        </div>

        {/* 4. Generated Output */}
        <div className="research-section">
          <h4>4. Generated Grounded Output</h4>
          <pre className="code-box pre-wrap">
            {queryResult?.answer || 'No response generated'}
          </pre>
        </div>
      </div>
    </aside>
  );
}
