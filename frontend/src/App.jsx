import React, { useState } from 'react';
import { useVideoRAG } from './hooks/useVideoRAG';
import { Home } from './pages/Home';
import { Processing } from './pages/Processing';
import { VideoWorkspace } from './pages/VideoWorkspace';
import { ResearchPanel } from './components/ResearchPanel';
import { ResearchMetadata } from './components/ResearchMetadata';

export function App() {
  const {
    currentScreen,
    currentVideoId,
    currentVideoName,
    processingStatus,
    chatHistory,
    isQuerying,
    latestQueryResult,
    researchMode,
    suggestedQuestions,
    isGeneratingQuestions,
    suggestedQuestionsError,
    fetchSuggestedQuestions,
    videoRef,
    seekVideo,
    loadSampleVideo,
    uploadAndProcess,
    enterWorkspace,
    askQuestion,
    clearChat,
    toggleResearchMode,
    navigateTo,
  } = useVideoRAG();

  const [isArchModalOpen, setIsArchModalOpen] = useState(false);

  return (
    <>
      {/* Global Application Header */}
      <header className="app-header">
        <div className="header-brand" onClick={() => navigateTo('home')}>
          <div className="logo-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
              <polygon points="23 7 16 12 23 17 23 7"></polygon>
              <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
            </svg>
          </div>
          <div>
            <h1 className="app-title">VideoRAG</h1>
            <p className="app-subtitle">Ask questions about any video using transcript-grounded AI.</p>
          </div>
        </div>

        <div className="header-actions">
          <div className="phase-badge" title="Phase 1: Speech Transcript Baseline">
            Phase 1: Transcript RAG
          </div>

          <button
            className={`btn-toggle-research ${researchMode ? 'active' : ''}`}
            onClick={toggleResearchMode}
            title="Toggle detailed research and retrieval metrics"
            type="button"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="20" x2="18" y2="10"></line>
              <line x1="12" y1="20" x2="12" y2="4"></line>
              <line x1="6" y1="20" x2="6" y2="14"></line>
            </svg>
            <span>Research Mode</span>
            <span className="toggle-indicator">{researchMode ? 'ON' : 'OFF'}</span>
          </button>

          <button
            className="btn-icon"
            onClick={() => setIsArchModalOpen(true)}
            title="View System Architecture"
            type="button"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="16" x2="12" y2="12"></line>
              <line x1="12" y1="8" x2="12.01" y2="8"></line>
            </svg>
          </button>
        </div>
      </header>

      {/* Main Screen View */}
      <main className="app-container">
        {currentScreen === 'home' && (
          <Home
            onUploadAndProcess={uploadAndProcess}
            onLoadSample={loadSampleVideo}
          />
        )}

        {currentScreen === 'processing' && (
          <Processing
            processingStatus={processingStatus}
            onEnterWorkspace={enterWorkspace}
          />
        )}

        {currentScreen === 'workspace' && (
          <VideoWorkspace
            videoRef={videoRef}
            videoId={currentVideoId}
            videoName={currentVideoName}
            chatHistory={chatHistory}
            isQuerying={isQuerying}
            suggestedQuestions={suggestedQuestions}
            isGeneratingQuestions={isGeneratingQuestions}
            suggestedQuestionsError={suggestedQuestionsError}
            onRetrySuggestedQuestions={fetchSuggestedQuestions}
            onAskQuestion={askQuestion}
            onClearChat={clearChat}
            onSeek={seekVideo}
          />
        )}
      </main>

      {/* Research Mode Inspection Drawer */}
      <ResearchPanel
        isOpen={researchMode}
        onClose={toggleResearchMode}
        queryResult={latestQueryResult}
        suggestedQuestions={suggestedQuestions}
        onSeek={seekVideo}
      />

      {/* System Architecture Modal */}
      <ResearchMetadata
        isOpen={isArchModalOpen}
        onClose={() => setIsArchModalOpen(false)}
      />
    </>
  );
}
export default App;
