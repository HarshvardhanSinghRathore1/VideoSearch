import React from 'react';

/**
 * Displays quantitative latency breakdown and retrieval settings.
 */
export function LatencyMetrics({ latencies, topK = 5 }) {
  const retrievalSec = latencies?.retrieval_sec ?? '—';
  const genSec = latencies?.generation_sec ?? '—';
  const totalSec = latencies?.total_sec ?? '—';

  return (
    <div className="metrics-grid">
      <div className="metric-card">
        <div className="metric-label">Retrieval Latency</div>
        <div className="metric-value">{typeof retrievalSec === 'number' ? `${retrievalSec}s` : retrievalSec}</div>
      </div>
      <div className="metric-card">
        <div className="metric-label">Generation Latency</div>
        <div className="metric-value">{typeof genSec === 'number' ? `${genSec}s` : genSec}</div>
      </div>
      <div className="metric-card">
        <div className="metric-label">Total Latency</div>
        <div className="metric-value">{typeof totalSec === 'number' ? `${totalSec}s` : totalSec}</div>
      </div>
      <div className="metric-card">
        <div className="metric-label">Top-K Chunks</div>
        <div className="metric-value">{topK}</div>
      </div>
    </div>
  );
}
