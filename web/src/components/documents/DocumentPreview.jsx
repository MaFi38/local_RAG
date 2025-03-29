import { useState } from 'react';
import { Document, Page } from 'react-pdf';
import { format } from 'date-fns';

export default function DocumentPreview({ document }) {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);

  if (!document) return null;

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
  };

  return (
    <div className="p-4">
      <div className="mb-4">
        <h3 className="text-lg font-medium text-gray-900">{document.name}</h3>
        <p className="text-sm text-gray-500">
          Uploaded on {format(new Date(document.timestamp), 'MMM d, yyyy')}
        </p>
      </div>

      {document.type === 'pdf' ? (
        <div className="border rounded-lg p-4">
          <Document
            file={document.url}
            onLoadSuccess={onDocumentLoadSuccess}
          >
            <Page pageNumber={pageNumber} />
          </Document>
          
          <div className="mt-4 flex justify-between items-center">
            <button
              onClick={() => setPageNumber(prev => Math.max(prev - 1, 1))}
              disabled={pageNumber <= 1}
              className="px-3 py-1 bg-primary text-white rounded-md disabled:opacity-50"
            >
              Previous
            </button>
            
            <span className="text-sm text-gray-500">
              Page {pageNumber} of {numPages}
            </span>
            
            <button
              onClick={() => setPageNumber(prev => Math.min(prev + 1, numPages))}
              disabled={pageNumber >= numPages}
              className="px-3 py-1 bg-primary text-white rounded-md disabled:opacity-50"
            >
              Next
            </button>
          </div>
        </div>
      ) : (
        <div className="border rounded-lg p-4">
          <pre className="whitespace-pre-wrap text-sm">
            {document.content}
          </pre>
        </div>
      )}
    </div>
  );
} 