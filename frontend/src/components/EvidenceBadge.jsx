import React from 'react';

/**
 * Clickable Evidence Timestamp Chip.
 * Clicking directly seeks the HTML5 video player and initiates playback.
 */
export function EvidenceBadge({ start, end, formatted, onSeek }) {
  const formatSeconds = (secs) => {
    if (isNaN(secs) || secs === null) return '00:00:00';
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = Math.floor(secs % 60);
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

  const displayText = formatted || `[${formatSeconds(start)} – ${formatSeconds(end)}]`;

  return (
    <button
      className="timestamp-chip"
      onClick={() => onSeek && onSeek(start)}
      title={`Seek video to ${displayText}`}
      type="button"
    >
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <polygon points="5 3 19 12 5 21 5 3"></polygon>
      </svg>
      <span>{displayText}</span>
    </button>
  );
}
