import React from 'react';
import { ProcessingStatus } from '../components/ProcessingStatus';

/**
 * Screen 2: Processing Dashboard Page.
 */
export function Processing({ processingStatus, onEnterWorkspace }) {
  return (
    <section className="view-screen active">
      <ProcessingStatus
        processingStatus={processingStatus}
        onEnterWorkspace={onEnterWorkspace}
      />
    </section>
  );
}
