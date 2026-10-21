/**
 * Custom React Hook managing VideoRAG state, video player references,
 * processing lifecycle, chat history, and research debugging data.
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import * as api from '../services/api';

export function useVideoRAG() {
  const [currentScreen, setCurrentScreen] = useState('home'); // 'home' | 'processing' | 'workspace'
  const [currentVideoId, setCurrentVideoId] = useState(null);
  const [currentVideoName, setCurrentVideoName] = useState(null);

  // Processing State
  const [processingStatus, setProcessingStatus] = useState({
    status: 'idle', // 'idle' | 'processing' | 'completed' | 'failed'
    progress: 0,
    message: '',
    stages: {},
    error: null,
  });

  // Chat State
  const [chatHistory, setChatHistory] = useState([]);
  const [isQuerying, setIsQuerying] = useState(false);
  const [latestQueryResult, setLatestQueryResult] = useState(null);

  // Research Mode
  const [researchMode, setResearchMode] = useState(false);
  const [systemInfo, setSystemInfo] = useState(null);

  // Dynamic Suggested Questions State
  const [suggestedQuestions, setSuggestedQuestions] = useState([]);
  const [isGeneratingQuestions, setIsGeneratingQuestions] = useState(false);
  const [suggestedQuestionsError, setSuggestedQuestionsError] = useState(null);

  // HTML5 Video Player Ref
  const videoRef = useRef(null);

  // Fetch system metadata on load
  useEffect(() => {
    api.getSystemInfo()
      .then(info => setSystemInfo(info))
      .catch(err => console.warn('System info fetch warning:', err));
  }, []);

  /**
   * Fetches dynamic suggested questions for the currently active video.
   */
  const fetchSuggestedQuestions = useCallback(async (videoId, force = false) => {
    if (!videoId) return;
    setIsGeneratingQuestions(true);
    setSuggestedQuestionsError(null);
    try {
      const data = await api.getSuggestedQuestions(videoId, force);
      if (data && Array.isArray(data.questions) && data.questions.length > 0) {
        setSuggestedQuestions(data.questions);
      } else {
        setSuggestedQuestions([]);
      }
    } catch (err) {
      console.error('Failed to fetch suggested questions:', err);
      setSuggestedQuestionsError(err.message || 'Unable to generate suggestions.');
      setSuggestedQuestions([]);
    } finally {
      setIsGeneratingQuestions(false);
    }
  }, []);

  // Automatically fetch suggested questions whenever active video changes
  useEffect(() => {
    if (currentVideoId && currentScreen === 'workspace') {
      fetchSuggestedQuestions(currentVideoId);
    }
  }, [currentVideoId, currentScreen, fetchSuggestedQuestions]);

  /**
   * Seeks the HTML5 video player to the target start timestamp and starts playback.
   */
  const seekVideo = useCallback((startTimeSeconds) => {
    if (videoRef.current) {
      const time = parseFloat(startTimeSeconds);
      if (!isNaN(time)) {
        videoRef.current.currentTime = time;
        videoRef.current.play().catch(e => console.log('Autoplay policy notice:', e));
      }
    }
  }, []);

  /**
   * Quick-loads the pre-indexed sample lecture.
   */
  const loadSampleVideo = useCallback(() => {
    setCurrentVideoId('sample');
    setCurrentVideoName('sample.mp4');
    setChatHistory([]);
    setLatestQueryResult(null);
    setCurrentScreen('workspace');
  }, []);

  /**
   * Uploads and initiates processing of a user-selected video file.
   */
  const uploadAndProcess = useCallback(async (file) => {
    try {
      setCurrentScreen('processing');
      setProcessingStatus({
        status: 'processing',
        progress: 10,
        message: 'Uploading video container to backend...',
        stages: {
          upload: { label: 'Video uploaded', status: 'processing' },
          audio: { label: 'Audio extraction', status: 'pending' },
          transcribe: { label: 'Speech transcription', status: 'pending' },
          chunk: { label: 'Transcript chunking', status: 'pending' },
          embed: { label: 'Embedding generation', status: 'pending' },
          index: { label: 'FAISS indexing', status: 'pending' },
          ready: { label: 'Ready for questions', status: 'pending' },
        },
        error: null,
      });

      // 1. Upload video
      const uploadRes = await api.uploadVideo(file);
      const videoId = uploadRes.id;
      setCurrentVideoId(videoId);
      setCurrentVideoName(uploadRes.filename);

      // 2. Trigger processing
      await api.processVideo(videoId);

      // 3. Poll status
      const pollInterval = setInterval(async () => {
        try {
          const statusData = await api.getVideoStatus(videoId);
          setProcessingStatus({
            status: statusData.status,
            progress: statusData.progress || 0,
            message: statusData.message || '',
            stages: statusData.stages || {},
            error: statusData.error || null,
          });

          if (statusData.status === 'completed') {
            clearInterval(pollInterval);
          } else if (statusData.status === 'failed') {
            clearInterval(pollInterval);
          }
        } catch (pollErr) {
          console.error('Status poll error:', pollErr);
        }
      }, 1000);

    } catch (err) {
      setProcessingStatus(prev => ({
        ...prev,
        status: 'failed',
        error: err.message,
        message: `Processing failed: ${err.message}`,
      }));
    }
  }, []);

  /**
   * Transitions from completed processing dashboard to the Video QA Workspace.
   */
  const enterWorkspace = useCallback(() => {
    setChatHistory([]);
    setLatestQueryResult(null);
    setCurrentScreen('workspace');
  }, []);

  /**
   * Submits a question query to the RAG backend.
   */
  const askQuestion = useCallback(async (questionText) => {
    if (!questionText || !questionText.trim() || !currentVideoId) return;

    const query = questionText.trim();
    const userMsg = { id: `user-${Date.now()}`, role: 'user', content: query };
    const placeholderId = `assistant-${Date.now()}`;

    setChatHistory(prev => [
      ...prev,
      userMsg,
      { id: placeholderId, role: 'assistant', loading: true }
    ]);
    setIsQuerying(true);

    try {
      const result = await api.queryVideo(currentVideoId, query, 5);
      setLatestQueryResult(result);

      setChatHistory(prev => prev.map(msg => {
        if (msg.id === placeholderId) {
          return {
            id: placeholderId,
            role: 'assistant',
            loading: false,
            data: result,
          };
        }
        return msg;
      }));

    } catch (err) {
      setChatHistory(prev => prev.map(msg => {
        if (msg.id === placeholderId) {
          return {
            id: placeholderId,
            role: 'assistant',
            loading: false,
            error: err.message,
          };
        }
        return msg;
      }));
    } finally {
      setIsQuerying(false);
    }
  }, [currentVideoId]);

  const clearChat = useCallback(() => {
    setChatHistory([]);
  }, []);

  const toggleResearchMode = useCallback(() => {
    setResearchMode(prev => !prev);
  }, []);

  const navigateTo = useCallback((screen) => {
    setCurrentScreen(screen);
  }, []);

  return {
    currentScreen,
    currentVideoId,
    currentVideoName,
    processingStatus,
    chatHistory,
    isQuerying,
    latestQueryResult,
    researchMode,
    systemInfo,
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
  };
}
