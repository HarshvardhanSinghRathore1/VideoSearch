/**
 * Centralized API Service for VideoRAG Frontend
 * Communicates with the FastAPI backend REST endpoints.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || "";

/**
 * Uploads a video file to the backend.
 * @param {File} file 
 * @returns {Promise<{id: string, filename: string, size_mb: number, status: string}>}
 */
export async function uploadVideo(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/api/videos/upload`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Upload failed with status ${res.status}`);
  }

  return res.json();
}

/**
 * Initiates the 7-stage processing pipeline for a given video ID.
 * @param {string} videoId 
 * @returns {Promise<{video_id: string, status: string, message: string}>}
 */
export async function processVideo(videoId) {
  const res = await fetch(`${API_BASE}/api/videos/${encodeURIComponent(videoId)}/process`, {
    method: "POST",
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Processing trigger failed with status ${res.status}`);
  }

  return res.json();
}

/**
 * Retrieves the real-time stage-by-stage processing status.
 * @param {string} videoId 
 * @returns {Promise<{video_id: string, status: string, progress: number, message: string, stages: Object}>}
 */
export async function getVideoStatus(videoId) {
  const res = await fetch(`${API_BASE}/api/videos/${encodeURIComponent(videoId)}/status`);

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Status fetch failed with status ${res.status}`);
  }

  return res.json();
}

/**
 * Executes a transcript-grounded question-answering query.
 * @param {string} videoId 
 * @param {string} question 
 * @param {number} topK 
 * @returns {Promise<Object>}
 */
export async function queryVideo(videoId, question, topK = 5) {
  const res = await fetch(`${API_BASE}/api/videos/${encodeURIComponent(videoId)}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Query failed with status ${res.status}`);
  }

  return res.json();
}

/**
 * Returns the URL for streaming video with HTTP Range support for HTML5 <video>.
 * @param {string} videoId 
 * @returns {string}
 */
export function getVideoUrl(videoId) {
  return `${API_BASE}/api/videos/${encodeURIComponent(videoId)}/video`;
}

/**
 * Fetches the full timestamped transcript segments for a processed video.
 * @param {string} videoId 
 * @returns {Promise<Object>}
 */
export async function getTranscript(videoId) {
  const res = await fetch(`${API_BASE}/api/videos/${encodeURIComponent(videoId)}/transcript`);

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch transcript`);
  }

  return res.json();
}

/**
 * Lists all registered and processed videos.
 * @returns {Promise<{videos: Array}>}
 */
export async function listVideos() {
  const res = await fetch(`${API_BASE}/api/videos/list`);

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to list videos`);
  }

  return res.json();
}

/**
 * Retrieves research architecture and model info.
 * @returns {Promise<Object>}
 */
export async function getSystemInfo() {
  const res = await fetch(`${API_BASE}/api/system/info`);

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch system info`);
  }

  return res.json();
}

/**
 * Fetches dynamic, transcript-grounded suggested questions for a video.
 * @param {string} videoId 
 * @param {boolean} force 
 * @returns {Promise<{video_id: string, questions: string[], count: number, cached: boolean, source: string, model: string}>}
 */
export async function getSuggestedQuestions(videoId, force = false) {
  const res = await fetch(
    `${API_BASE}/api/videos/${encodeURIComponent(videoId)}/suggested-questions${force ? '?force=true' : ''}`
  );

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch suggested questions (${res.status})`);
  }

  return res.json();
}
