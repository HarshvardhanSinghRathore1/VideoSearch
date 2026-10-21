import React, { useState, useRef } from 'react';

const SUPPORTED_EXTS = ['.mp4', '.mkv', '.mov', '.avi', '.webm'];

/**
 * Screen 1 Video Uploader Component.
 */
export function VideoUploader({ onUploadAndProcess, onLoadSample }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const validateAndSelect = (file) => {
    if (!file) return;
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (!SUPPORTED_EXTS.includes(ext)) {
      setErrorMsg(`Unsupported format "${ext}". Supported: MP4, MKV, MOV, AVI, WEBM.`);
      setSelectedFile(null);
      return;
    }
    setErrorMsg(null);
    setSelectedFile(file);
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    validateAndSelect(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    validateAndSelect(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleClear = () => {
    setSelectedFile(null);
    setErrorMsg(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleProcess = () => {
    if (selectedFile) {
      onUploadAndProcess(selectedFile);
    }
  };

  return (
    <div className="upload-grid">
      {/* Upload Zone Card */}
      <div className="card upload-card">
        {!selectedFile ? (
          <div
            className={`drop-zone ${isDragOver ? 'dragover' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="drop-icon">
              <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
            </div>
            <h3>Drag and drop your video file here</h3>
            <p className="drop-help">or click to browse from your computer</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp4,.mkv,.mov,.avi,.webm"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
            <button
              type="button"
              className="btn btn-secondary"
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              Browse Files
            </button>
            <div className="format-tags">
              <span>MP4</span>
              <span>MKV</span>
              <span>MOV</span>
              <span>AVI</span>
              <span>WEBM</span>
            </div>
          </div>
        ) : (
          <div className="selected-file-box">
            <div className="file-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="23 7 16 12 23 17 23 7"></polygon>
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
              </svg>
            </div>
            <div className="file-info">
              <div className="file-name">{selectedFile.name}</div>
              <div className="file-meta">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</div>
            </div>
            <button className="btn-icon" onClick={handleClear} title="Clear file" type="button">✕</button>
          </div>
        )}

        {errorMsg && <div className="alert-box error">{errorMsg}</div>}

        <div className="upload-actions">
          <button
            className="btn btn-primary"
            disabled={!selectedFile}
            onClick={handleProcess}
            type="button"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="5 3 19 12 5 21 5 3"></polygon>
            </svg>
            Process Video
          </button>
        </div>
      </div>

      {/* Pre-loaded Sample Option */}
      <div className="card sample-card">
        <div className="sample-header">
          <h3>Quick Start with Sample Video</h3>
          <span className="badge badge-teal">Pre-indexed</span>
        </div>
        <p className="sample-desc">
          Test the pipeline immediately with the pre-configured lecture on <strong>Database Normalization & Data Mining</strong> (1NF, 2NF, 3NF, BCNF, Preprocessing, K-Means Clustering).
        </p>
        <div className="sample-meta">
          <div><strong>File:</strong> sample.mp4</div>
          <div><strong>Duration:</strong> 05:45 (345s)</div>
          <div><strong>Topics:</strong> 8 structured sections</div>
        </div>
        <button className="btn btn-accent full-width" onClick={onLoadSample} type="button">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
            <line x1="16" y1="13" x2="8" y2="13"></line>
            <line x1="16" y1="17" x2="8" y2="17"></line>
          </svg>
          Open Sample Lecture Workspace
        </button>
      </div>
    </div>
  );
}
