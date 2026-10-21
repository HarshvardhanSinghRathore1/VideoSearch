import React, { useState } from 'react';
import { VideoPlayer } from '../components/VideoPlayer';
import { ChatPanel } from '../components/ChatPanel';

/**
 * Screen 3: Two-Column Video QA Workspace Page.
 */
export function VideoWorkspace({
  videoRef,
  videoId,
  videoName,
  chatHistory,
  isQuerying,
  suggestedQuestions = [],
  isGeneratingQuestions = false,
  suggestedQuestionsError = null,
  onRetrySuggestedQuestions = null,
  onAskQuestion,
  onClearChat,
  onSeek,
}) {
  const [activeSuggestedQuery, setActiveSuggestedQuery] = useState('');

  const handleSelectSuggested = (query) => {
    setActiveSuggestedQuery(query);
    onAskQuestion(query);
  };

  return (
    <section className="view-screen active workspace-screen">
      <div className="workspace-layout">
        {/* Left Column: Native HTML5 Video Player with Dynamic Questions */}
        <VideoPlayer
          videoRef={videoRef}
          videoId={videoId}
          videoTitle={videoName}
          suggestedQuestions={suggestedQuestions}
          isGeneratingQuestions={isGeneratingQuestions}
          suggestedQuestionsError={suggestedQuestionsError}
          onRetrySuggestedQuestions={onRetrySuggestedQuestions}
          onSelectSuggestedQuery={handleSelectSuggested}
        />

        {/* Right Column: Grounded AI Chat */}
        <ChatPanel
          chatHistory={chatHistory}
          isQuerying={isQuerying}
          onAskQuestion={onAskQuestion}
          onClearChat={onClearChat}
          onSeek={onSeek}
          initialQuestion={activeSuggestedQuery}
        />
      </div>
    </section>
  );
}
