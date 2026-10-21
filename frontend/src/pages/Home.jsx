import React from 'react';
import { VideoUploader } from '../components/VideoUploader';

/**
 * Screen 1: Home / Upload Page.
 */
export function Home({ onUploadAndProcess, onLoadSample }) {
  return (
    <section className="view-screen active">
      <div className="upload-hero">
        <div className="hero-badge">Research Prototype Baseline</div>
        <h2>Video-to-Knowledge Pipeline</h2>
        <p className="hero-desc">
          Upload educational or instructional videos to transcribe speech, generate dense vector embeddings, and interactively query with exact timestamp citations.
        </p>
      </div>

      <VideoUploader onUploadAndProcess={onUploadAndProcess} onLoadSample={onLoadSample} />
    </section>
  );
}
