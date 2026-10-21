import React, { useState } from 'react';
import { getVideoUrl } from '../services/api';

/**
 * Native HTML5 Video Player with timestamp seeking and dynamic transcript-grounded suggested query chips.
 */
export function VideoPlayer({
  videoRef,
  videoId,
  videoTitle,
  onSelectSuggestedQuery,
  suggestedQuestions = [],
  isGeneratingQuestions = false,
  suggestedQuestionsError = null,
  onRetrySuggestedQuestions = null,
}) {
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const formatSeconds = (secs) => {
    if (isNaN(secs) || secs === null) return '00:00:00';
    const h = Math.floor(secs / 3600);
    const m = Math.floor((secs % 3600) / 60);
    const s = Math.floor(secs % 60);
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const videoSrc = videoId ? getVideoUrl(videoId) : '';

  return (
    <div className="workspace-left card">
      <div className="video-container">
        <video
          ref={videoRef}
          src={videoSrc}
          controls
          preload="metadata"
          playsInline
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
        >
          Your browser does not support HTML5 video playback.
        </video>
      </div>

      <div className="video-details-bar">
        <div className="video-title-box">
          <span className="video-indicator-dot"></span>
          <span className="video-title-text">{videoTitle || 'Video Player'}</span>
        </div>
        <div className="video-time-box">
          <span>{formatSeconds(currentTime)}</span> / <span>{formatSeconds(duration)}</span>
        </div>
      </div>

      <div className="sample-questions-box">
        <div className="box-label-header">
          <span className="box-label">Suggested Questions</span>
          {suggestedQuestions.length > 0 && (
            <span className="badge-grounded">Transcript Grounded</span>
          )}
        </div>

        {isGeneratingQuestions ? (
          <div className="suggested-loading-state">
            <span className="spinner-inline">⟳</span>
            <span>Generating questions from transcript...</span>
          </div>
        ) : suggestedQuestionsError ? (
          <div className="suggested-error-state">
            <p className="error-text">Unable to generate suggestions. You can ask a question about the video manually.</p>
            {onRetrySuggestedQuestions && (
              <button
                className="btn-retry-suggested"
                onClick={() => onRetrySuggestedQuestions(videoId, true)}
                type="button"
              >
                Retry
              </button>
            )}
          </div>
        ) : suggestedQuestions && suggestedQuestions.length > 0 ? (
          <div className="chips-container">
            {suggestedQuestions.map((question, idx) => (
              <button
                key={idx}
                className="query-chip dynamic-query-chip"
                onClick={() => onSelectSuggestedQuery(question)}
                type="button"
                title="Click to ask this question"
              >
                <span className="chip-bullet">•</span> {question}
              </button>
            ))}
          </div>
        ) : (
          <div className="suggested-empty-state">
            <span>No suggestions available for this video yet. Type your question on the right.</span>
          </div>
        )}
      </div>
    </div>
  );
}
