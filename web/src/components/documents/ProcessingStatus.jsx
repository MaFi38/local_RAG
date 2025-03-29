import { useState, useEffect } from 'react';
import { useApp } from '../../hooks/useApp';

export default function ProcessingStatus() {
  const { documents } = useApp();
  const [processingDocs, setProcessingDocs] = useState([]);

  useEffect(() => {
    // In a real app, this would connect to your backend to get processing status
    const processingStatus = documents.filter(doc => doc.status === 'processing');
    setProcessingDocs(processingStatus);
  }, [documents]);

  if (processingDocs.length === 0) return null;

  return (
    <div className="p-4 bg-yellow-50 border-b">
      <div className="flex items-center">
        <svg
          className="animate-spin -ml-1 mr-3 h-5 w-5 text-yellow-700"
          fill="none"
          viewBox="0 0 24 24"
        >
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth="4"
          />
          <path
            className="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
          />
        </svg>
        <span className="text-sm text-yellow-700">
          Processing {processingDocs.length} document(s)...
        </span>
      </div>
    </div>
  );
} 