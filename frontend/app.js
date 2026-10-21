/**
 * VideoRAG — Frontend Application Controller
 * Handles video upload, real-time stage tracking, HTML5 video seeking on evidence clicks,
 * grounded chat interaction, and Research Mode metrics inspection.
 */

// Application State
const state = {
  currentVideoId: null,
  currentVideoName: null,
  selectedFile: null,
  researchMode: false,
  statusPollInterval: null,
  latestQueryResult: null,
};

// DOM Elements
const elements = {
  // Screens
  screenUpload: document.getElementById("screen-upload"),
  screenProcessing: document.getElementById("screen-processing"),
  screenWorkspace: document.getElementById("screen-workspace"),

  // Header & Controls
  btnResearchToggle: document.getElementById("btn-research-toggle"),
  researchIndicator: document.getElementById("research-indicator"),
  researchDrawer: document.getElementById("research-drawer"),
  btnCloseDrawer: document.getElementById("btn-close-drawer"),
  btnSystemInfo: document.getElementById("btn-system-info"),
  modalSystemInfo: document.getElementById("modal-system-info"),
  btnCloseModal: document.getElementById("btn-close-modal"),

  // Upload Screen
  dropZone: document.getElementById("drop-zone"),
  fileInput: document.getElementById("file-input"),
  btnBrowse: document.getElementById("btn-browse"),
  selectedFileCard: document.getElementById("selected-file-card"),
  fileName: document.getElementById("file-name"),
  fileSize: document.getElementById("file-size"),
  btnClearFile: document.getElementById("btn-clear-file"),
  btnProcessVideo: document.getElementById("btn-process-video"),
  btnLoadSample: document.getElementById("btn-load-sample"),
  uploadError: document.getElementById("upload-error"),

  // Processing Screen
  processingStatusMsg: document.getElementById("processing-status-msg"),
  progressBarFill: document.getElementById("progress-bar-fill"),
  processingFooter: document.getElementById("processing-footer"),
  btnEnterWorkspace: document.getElementById("btn-enter-workspace"),
  processingError: document.getElementById("processing-error"),

  // Workspace Screen
  videoPlayer: document.getElementById("video-player"),
  workspaceVideoTitle: document.getElementById("workspace-video-title"),
  currentTimeDisplay: document.getElementById("current-time-display"),
  durationDisplay: document.getElementById("duration-display"),
  chatStream: document.getElementById("chat-stream"),
  chatEmptyState: document.getElementById("chat-empty-state"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  btnSendChat: document.getElementById("btn-send-chat"),
  btnClearChat: document.getElementById("btn-clear-chat"),
  sampleQueryChips: document.getElementById("sample-query-chips"),

  // Research Drawer Elements
  resRetrievalLat: document.getElementById("res-retrieval-lat"),
  resGenLat: document.getElementById("res-gen-lat"),
  resTotalLat: document.getElementById("res-total-lat"),
  resTopK: document.getElementById("res-top-k"),
  resLlmModel: document.getElementById("res-llm-model"),
  resUserQuery: document.getElementById("res-user-query"),
  resChunksList: document.getElementById("res-chunks-list"),
  resLlmContext: document.getElementById("res-llm-context"),
  resGeneratedAns: document.getElementById("res-generated-ans"),
};

// --- Initialization ---
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  fetchSystemInfo();
});

// --- View Router ---
function showScreen(screenId) {
  [elements.screenUpload, elements.screenProcessing, elements.screenWorkspace].forEach(screen => {
    screen.classList.remove("active");
  });
  const target = document.getElementById(screenId);
  if (target) {
    target.classList.add("active");
  }
}

// --- Format Utilities ---
function formatSeconds(secs) {
  if (isNaN(secs) || secs === null) return "00:00:00";
  const hours = Math.floor(secs / 3600);
  const minutes = Math.floor((secs % 3600) / 60);
  const seconds = Math.floor(secs % 60);
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

// --- Event Listeners Setup ---
function setupEventListeners() {
  // Research Toggle
  elements.btnResearchToggle.addEventListener("click", toggleResearchMode);
  elements.btnCloseDrawer.addEventListener("click", () => setResearchMode(false));

  // System Info Modal
  elements.btnSystemInfo.addEventListener("click", () => elements.modalSystemInfo.style.display = "flex");
  elements.btnCloseModal.addEventListener("click", () => elements.modalSystemInfo.style.display = "none");
  elements.modalSystemInfo.addEventListener("click", (e) => {
    if (e.target === elements.modalSystemInfo) elements.modalSystemInfo.style.display = "none";
  });

  // File Upload Interactions
  elements.btnBrowse.addEventListener("click", (e) => {
    e.stopPropagation();
    elements.fileInput.click();
  });
  elements.dropZone.addEventListener("click", () => elements.fileInput.click());
  elements.fileInput.addEventListener("change", handleFileSelected);

  // Drag & Drop
  ["dragenter", "dragover"].forEach(name => {
    elements.dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      elements.dropZone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach(name => {
    elements.dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      elements.dropZone.classList.remove("dragover");
    });
  });
  elements.dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelected({ target: { files } });
    }
  });

  elements.btnClearFile.addEventListener("click", clearSelectedFile);
  elements.btnProcessVideo.addEventListener("click", startVideoProcessing);

  // Sample Video Button
  elements.btnLoadSample.addEventListener("click", loadSampleVideoWorkspace);

  // Processing Navigation
  elements.btnEnterWorkspace.addEventListener("click", () => {
    loadVideoIntoWorkspace(state.currentVideoId, state.currentVideoName);
  });

  // Video Player Events
  elements.videoPlayer.addEventListener("timeupdate", () => {
    elements.currentTimeDisplay.textContent = formatSeconds(elements.videoPlayer.currentTime);
  });
  elements.videoPlayer.addEventListener("loadedmetadata", () => {
    elements.durationDisplay.textContent = formatSeconds(elements.videoPlayer.duration);
  });

  // Chat Submission
  elements.chatForm.addEventListener("submit", handleChatSubmit);
  elements.btnClearChat.addEventListener("click", clearChatStream);

  // Suggested Chips
  elements.sampleQueryChips.addEventListener("click", (e) => {
    const chip = e.target.closest(".query-chip");
    if (chip) {
      const q = chip.getAttribute("data-q");
      elements.chatInput.value = q;
      handleChatSubmit(new Event("submit"));
    }
  });
}

// --- Research Mode Controller ---
function toggleResearchMode() {
  setResearchMode(!state.researchMode);
}

function setResearchMode(active) {
  state.researchMode = active;
  if (active) {
    elements.researchDrawer.classList.add("open");
    elements.btnResearchToggle.classList.add("active");
    elements.researchIndicator.textContent = "ON";
    if (state.latestQueryResult) {
      renderResearchDrawer(state.latestQueryResult);
    }
  } else {
    elements.researchDrawer.classList.remove("open");
    elements.btnResearchToggle.classList.remove("active");
    elements.researchIndicator.textContent = "OFF";
  }
}

// --- Fetch System Info ---
async function fetchSystemInfo() {
  try {
    const res = await fetch("/api/system/info");
    const data = await res.json();
    if (data.llm_provider) {
      elements.resLlmModel.textContent = data.llm_provider;
    }
  } catch (err) {
    console.warn("System info fetch warning:", err);
  }
}

// --- File Selection Handlers ---
function handleFileSelected(e) {
  const file = e.target.files[0];
  if (!file) return;

  const validExts = [".mp4", ".mkv", ".mov", ".avi", ".webm"];
  const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
  
  if (!validExts.includes(ext)) {
    showUploadError(`Unsupported video format "${ext}". Supported: MP4, MKV, MOV, AVI, WEBM.`);
    return;
  }

  state.selectedFile = file;
  elements.fileName.textContent = file.name;
  elements.fileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
  
  elements.selectedFileCard.style.display = "flex";
  elements.dropZone.style.display = "none";
  elements.btnProcessVideo.disabled = false;
  elements.uploadError.style.display = "none";
}

function clearSelectedFile() {
  state.selectedFile = null;
  elements.fileInput.value = "";
  elements.selectedFileCard.style.display = "none";
  elements.dropZone.style.display = "block";
  elements.btnProcessVideo.disabled = true;
  elements.uploadError.style.display = "none";
}

function showUploadError(msg) {
  elements.uploadError.textContent = msg;
  elements.uploadError.style.display = "block";
}

// --- Video Processing Controller ---
async function startVideoProcessing() {
  if (!state.selectedFile) return;

  showScreen("screen-processing");
  resetPipelineSteps();
  elements.processingStatusMsg.textContent = "Uploading video to server...";
  elements.progressBarFill.style.width = "10%";
  elements.processingError.style.display = "none";
  elements.processingFooter.style.display = "none";

  const formData = new FormData();
  formData.append("file", state.selectedFile);

  try {
    const uploadRes = await fetch("/api/videos/upload", {
      method: "POST",
      body: formData
    });

    if (!uploadRes.ok) {
      const err = await uploadRes.json();
      throw new Error(err.detail || "Upload failed");
    }

    const uploadData = await uploadRes.json();
    state.currentVideoId = uploadData.id;
    state.currentVideoName = uploadData.filename;

    // Trigger processing
    const procRes = await fetch(`/api/videos/${state.currentVideoId}/process`, { method: "POST" });
    if (!procRes.ok) {
      const err = await procRes.json();
      throw new Error(err.detail || "Processing trigger failed");
    }

    // Begin status polling
    startStatusPolling(state.currentVideoId);

  } catch (err) {
    showProcessingError(`Processing failed: ${err.message}`);
  }
}

function resetPipelineSteps() {
  const steps = ["upload", "audio", "transcribe", "chunk", "embed", "index", "ready"];
  steps.forEach(id => {
    const el = document.getElementById(`step-${id}`);
    if (el) {
      el.className = "step-item";
      const badge = el.querySelector(".step-badge");
      if (badge) {
        badge.className = "step-badge badge-pending";
        badge.textContent = "Pending";
      }
    }
  });
}

function startStatusPolling(videoId) {
  if (state.statusPollInterval) clearInterval(state.statusPollInterval);

  state.statusPollInterval = setInterval(async () => {
    try {
      const res = await fetch(`/api/videos/${videoId}/status`);
      if (!res.ok) throw new Error("Failed to fetch processing status");
      
      const statusData = await res.json();
      updatePipelineUI(statusData);

      if (statusData.status === "completed") {
        clearInterval(state.statusPollInterval);
        elements.processingStatusMsg.textContent = "✓ Video is fully indexed and ready for questions.";
        elements.progressBarFill.style.width = "100%";
        elements.processingFooter.style.display = "block";
      } else if (statusData.status === "failed") {
        clearInterval(state.statusPollInterval);
        showProcessingError(statusData.error || "An error occurred during video processing.");
      }
    } catch (err) {
      console.error("Polling error:", err);
    }
  }, 1000);
}

function updatePipelineUI(statusData) {
  if (statusData.message) {
    elements.processingStatusMsg.textContent = statusData.message;
  }
  if (statusData.progress) {
    elements.progressBarFill.style.width = `${statusData.progress}%`;
  }

  if (statusData.stages) {
    for (const [stageKey, stageInfo] of Object.entries(statusData.stages)) {
      const el = document.getElementById(`step-${stageKey}`);
      if (!el) continue;

      el.className = `step-item ${stageInfo.status}`;
      const badge = el.querySelector(".step-badge");
      if (badge) {
        badge.className = `step-badge badge-${stageInfo.status}`;
        badge.textContent = stageInfo.status.charAt(0).toUpperCase() + stageInfo.status.slice(1);
      }
    }
  }
}

function showProcessingError(msg) {
  elements.processingError.textContent = msg;
  elements.processingError.style.display = "block";
}

// --- Quick Load Sample Video Workspace ---
function loadSampleVideoWorkspace() {
  state.currentVideoId = "sample";
  state.currentVideoName = "sample.mp4";
  loadVideoIntoWorkspace("sample", "sample.mp4");
}

function loadVideoIntoWorkspace(videoId, videoName) {
  showScreen("screen-workspace");
  elements.workspaceVideoTitle.textContent = videoName || "Video Player";
  
  // Set streaming endpoint for native HTML5 player
  elements.videoPlayer.src = `/api/videos/${videoId}/video`;
  elements.videoPlayer.load();
  elements.chatInput.focus();
}

// --- Video Seeking by Timestamp Click ---
window.seekVideoTo = function(seconds, chipEl) {
  if (!elements.videoPlayer) return;

  const targetSec = parseFloat(seconds);
  if (isNaN(targetSec)) return;

  // 1. Set currentTime
  elements.videoPlayer.currentTime = targetSec;

  // 2. Play video
  elements.videoPlayer.play().catch(e => console.log("Autoplay note:", e));

  // 3. Visual pulse indicator on clicked chip
  document.querySelectorAll(".timestamp-chip").forEach(c => c.classList.remove("active-seek"));
  if (chipEl) {
    chipEl.classList.add("active-seek");
  }
};

// --- Chat Controller ---
async function handleChatSubmit(e) {
  e.preventDefault();
  const query = elements.chatInput.value.trim();
  if (!query) return;

  if (elements.chatEmptyState) {
    elements.chatEmptyState.style.display = "none";
  }

  // 1. Render User Message
  renderUserMessage(query);
  elements.chatInput.value = "";
  elements.btnSendChat.disabled = true;

  // 2. Render Thinking Placeholder
  const placeholderId = `ai-msg-${Date.now()}`;
  renderThinkingMessage(placeholderId);

  try {
    const res = await fetch(`/api/videos/${state.currentVideoId}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: query, top_k: 5 })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Query failed");
    }

    const result = await res.json();
    state.latestQueryResult = result;

    // 3. Replace Thinking Message with Grounded Answer Card
    renderAssistantMessage(placeholderId, result);

    // 4. Update Research Mode Drawer if active
    if (state.researchMode) {
      renderResearchDrawer(result);
    }

  } catch (err) {
    renderErrorMessage(placeholderId, `Error: ${err.message}`);
  } finally {
    elements.btnSendChat.disabled = false;
  }
}

function renderUserMessage(text) {
  const msgDiv = document.createElement("div");
  msgDiv.className = "chat-msg user";
  msgDiv.innerHTML = `
    <div class="msg-bubble-user">${escapeHtml(text)}</div>
  `;
  elements.chatStream.appendChild(msgDiv);
  elements.chatStream.scrollTop = elements.chatStream.scrollHeight;
}

function renderThinkingMessage(id) {
  const msgDiv = document.createElement("div");
  msgDiv.className = "chat-msg assistant";
  msgDiv.id = id;
  msgDiv.innerHTML = `
    <div class="msg-card-assistant">
      <div style="display: flex; align-items: center; gap: 8px; color: var(--text-secondary); font-size: 0.85rem;">
        <span class="spinner-large" style="width: 16px; height: 16px; border-width: 2px;"></span>
        Searching transcript & generating grounded answer...
      </div>
    </div>
  `;
  elements.chatStream.appendChild(msgDiv);
  elements.chatStream.scrollTop = elements.chatStream.scrollHeight;
}

function renderAssistantMessage(id, data) {
  const el = document.getElementById(id);
  if (!el) return;

  const isAbstained = data.abstained;
  let evidenceHtml = "";

  if (!isAbstained && data.evidence && data.evidence.length > 0) {
    const chips = data.evidence.map(ev => {
      const startSec = ev.start;
      const fmtTime = ev.formatted || `[${formatSeconds(startSec)} - ${formatSeconds(ev.end)}]`;
      return `
        <button class="timestamp-chip" onclick="seekVideoTo(${startSec}, this)" title="Click to seek video">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
          ${fmtTime}
        </button>
      `;
    }).join("");

    evidenceHtml = `
      <div class="msg-evidence-box">
        <span class="evidence-label">Evidence Citation:</span>
        ${chips}
      </div>
    `;
  }

  const answerText = isAbstained
    ? "I couldn't find information in this video that answers this question."
    : data.answer;

  el.innerHTML = `
    <div class="msg-card-assistant">
      <div class="msg-answer-text">${escapeHtml(answerText)}</div>
      ${evidenceHtml}
      <div class="msg-meta-footer">
        <span>Source: <strong>${escapeHtml(data.source_video)}</strong></span>
        <span>Latency: <strong>${data.latencies?.total_sec || 0}s</strong></span>
      </div>
    </div>
  `;
  elements.chatStream.scrollTop = elements.chatStream.scrollHeight;
}

function renderErrorMessage(id, errorText) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = `
    <div class="msg-card-assistant" style="border-color: var(--accent-rose);">
      <div style="color: #FCA5A5; font-size: 0.85rem;">${escapeHtml(errorText)}</div>
    </div>
  `;
}

function clearChatStream() {
  elements.chatStream.innerHTML = "";
  if (elements.chatEmptyState) {
    elements.chatStream.appendChild(elements.chatEmptyState);
    elements.chatEmptyState.style.display = "block";
  }
}

// --- Research Mode Drawer Renderer ---
function renderResearchDrawer(data) {
  elements.resRetrievalLat.textContent = `${data.latencies?.retrieval_sec || 0}s`;
  elements.resGenLat.textContent = `${data.latencies?.generation_sec || 0}s`;
  elements.resTotalLat.textContent = `${data.latencies?.total_sec || 0}s`;
  elements.resTopK.textContent = data.top_k || 5;

  elements.resUserQuery.textContent = data.question || "—";
  elements.resLlmContext.textContent = data.context_str || "—";
  elements.resGeneratedAns.textContent = data.answer || "—";

  // Render Retrieved Chunks List
  if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
    elements.resChunksList.innerHTML = data.retrieved_chunks.map(c => {
      const scorePct = Math.round((c.score || 0) * 100);
      return `
        <div class="chunk-card">
          <div class="chunk-card-header">
            <span class="chunk-rank">Rank #${c.rank} (${c.chunk_id})</span>
            <span class="chunk-score-pill">Cosine Sim: ${(c.score || 0).toFixed(4)}</span>
          </div>
          <div class="chunk-text">${escapeHtml(c.text)}</div>
          <div class="chunk-footer">
            <button class="timestamp-chip" onclick="seekVideoTo(${c.start_time}, this)">
              ▶ ${c.formatted_time || `[${c.start_time}s - ${c.end_time}s]`}
            </button>
            <span style="font-size: 0.7rem; color: var(--text-muted);">${escapeHtml(c.source_video)}</span>
          </div>
        </div>
      `;
    }).join("");
  } else {
    elements.resChunksList.innerHTML = `<p class="placeholder-text">No chunks retrieved.</p>`;
  }
}

// Helper to sanitize HTML
function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
