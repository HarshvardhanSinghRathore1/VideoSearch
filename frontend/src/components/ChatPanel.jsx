import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from './ChatMessage';

/**
 * Chat Panel Component for questioning the video transcript.
 */
export function ChatPanel({
  chatHistory,
  isQuerying,
  onAskQuestion,
  onClearChat,
  onSeek,
  initialQuestion = '',
}) {
  const [inputText, setInputText] = useState(initialQuestion);
  const streamEndRef = useRef(null);

  // Sync initialQuestion if passed from suggested chips
  useEffect(() => {
    if (initialQuestion) {
      setInputText(initialQuestion);
    }
  }, [initialQuestion]);

  // Auto-scroll on new message
  useEffect(() => {
    streamEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, isQuerying]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!inputText.trim() || isQuerying) return;
    onAskQuestion(inputText);
    setInputText('');
  };

  return (
    <div className="workspace-right card">
      <div className="chat-header">
        <div className="chat-header-title">
          <h3>Transcript-Grounded AI</h3>
          <span className="badge badge-indigo">Gemini Grounded</span>
        </div>
        <button
          className="btn-icon"
          onClick={onClearChat}
          title="Clear conversation history"
          type="button"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
          </svg>
        </button>
      </div>

      {/* Message Stream */}
      <div className="chat-stream">
        {chatHistory.length === 0 ? (
          <div className="chat-empty-state">
            <div className="empty-icon">💬</div>
            <h4>Ask anything about the video</h4>
            <p>Type a question below or pick a suggested topic to search timestamped speech evidence.</p>
          </div>
        ) : (
          chatHistory.map((msg) => (
            <ChatMessage key={msg.id} message={msg} onSeek={onSeek} />
          ))
        )}
        <div ref={streamEndRef} />
      </div>

      {/* Input Form */}
      <div className="chat-input-wrapper">
        <form className="chat-form" onSubmit={handleSubmit}>
          <input
            type="text"
            className="chat-input"
            placeholder="Ask a question about the video transcript..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            disabled={isQuerying}
          />
          <button
            type="submit"
            className="btn btn-primary btn-send"
            disabled={isQuerying || !inputText.trim()}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="22" y1="2" x2="11" y2="13"></line>
              <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
}
