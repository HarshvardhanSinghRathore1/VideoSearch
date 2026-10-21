import React from 'react';
import { EvidenceBadge } from './EvidenceBadge';

/**
 * Individual Chat Message renderer supporting User and Grounded AI Assistant messages.
 */
export function ChatMessage({ message, onSeek }) {
  if (message.role === 'user') {
    return (
      <div className="chat-msg user">
        <div className="msg-bubble-user">{message.content}</div>
      </div>
    );
  }

  // Assistant Message
  if (message.loading) {
    return (
      <div className="chat-msg assistant">
        <div className="msg-card-assistant">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            <span className="spinner-large" style={{ width: '16px', height: '16px', borderWidth: '2px' }}></span>
            Searching transcript & generating grounded answer...
          </div>
        </div>
      </div>
    );
  }

  if (message.error) {
    return (
      <div className="chat-msg assistant">
        <div className="msg-card-assistant" style={{ borderColor: 'var(--accent-rose)' }}>
          <div style={{ color: '#FCA5A5', fontSize: '0.85rem' }}>Error: {message.error}</div>
        </div>
      </div>
    );
  }

  const { data } = message;
  const isAbstained = data?.abstained;
  const answerText = isAbstained
    ? "I couldn't find information in this video that answers this question."
    : data?.answer || "No response generated.";

  const evidenceItems = (!isAbstained && data?.evidence) ? data.evidence : [];

  return (
    <div className="chat-msg assistant">
      <div className="msg-card-assistant">
        <div className="msg-answer-text">{answerText}</div>

        {evidenceItems.length > 0 && (
          <div className="msg-evidence-box">
            <span className="evidence-label">Evidence Citation:</span>
            {evidenceItems.map((ev, i) => (
              <EvidenceBadge
                key={ev.chunk_id || i}
                start={ev.start}
                end={ev.end}
                formatted={ev.formatted}
                onSeek={onSeek}
              />
            ))}
          </div>
        )}

        <div className="msg-meta-footer">
          <span>Source: <strong>{data?.source_video || 'video'}</strong></span>
          <span>Latency: <strong>{data?.latencies?.total_sec || 0}s</strong></span>
        </div>
      </div>
    </div>
  );
}
