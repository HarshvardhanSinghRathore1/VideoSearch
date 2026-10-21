import React from 'react';

/**
 * Individual ranked retrieved chunk card for Research Mode inspection.
 */
export function RetrievedChunk({ chunk, onSeek }) {
  const { rank, chunk_id, score, start_time, end_time, formatted_time, text, source_video } = chunk;

  return (
    <div className="chunk-card">
      <div className="chunk-card-header">
        <span className="chunk-rank">Rank #{rank} ({chunk_id})</span>
        <span className="chunk-score-pill">Cosine Sim: {Number(score).toFixed(4)}</span>
      </div>
      <div className="chunk-text">{text}</div>
      <div className="chunk-footer">
        <button
          className="timestamp-chip"
          onClick={() => onSeek && onSeek(start_time)}
          title="Jump video to this chunk"
          type="button"
        >
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
          {formatted_time || `[${start_time}s - ${end_time}s]`}
        </button>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{source_video}</span>
      </div>
    </div>
  );
}
