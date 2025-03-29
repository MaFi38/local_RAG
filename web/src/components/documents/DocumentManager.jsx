import { useState, useEffect } from 'react';
import { uploadDocument, listDocuments, deleteDocument } from '../../services/document.service';
import { TrashIcon } from '@heroicons/react/24/outline';

export default function DocumentManager() {
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const docs = await listDocuments('user-1'); // We'll get user ID from auth later
      setDocuments(docs);
    } catch (err) {
      setError('Failed to load documents');
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setError('');
    setSuccess('');

    try {
      await uploadDocument(file, 'user-1');
      await loadDocuments(); // Refresh the list
      setSuccess('Document uploaded successfully!');
      e.target.value = ''; // Reset file input
    } catch (err) {
      setError(err.message || 'Failed to upload document');
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (documentId) => {
    try {
      await deleteDocument(documentId, 'user-1');
      setDocuments(docs => docs.filter(doc => doc.id !== documentId));
    } catch (err) {
      setError('Failed to delete document');
    }
  };

  return (
    <div className="w-64 border-l bg-white flex flex-col h-full">
      <div className="p-4 border-b">
        <h3 className="text-lg font-medium text-gray-900">Documents</h3>
        
        {error && (
          <div className="text-red-500 text-sm mt-2">{error}</div>
        )}
        {success && (
          <div className="text-green-500 text-sm mt-2">{success}</div>
        )}

        <div className="mt-4">
          <input
            type="file"
            accept=".pdf,.doc,.docx,.txt"
            onChange={handleFileUpload}
            disabled={uploading}
            className="block w-full text-sm text-gray-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-md file:border-0
              file:text-sm file:font-semibold
              file:bg-primary file:text-white
              hover:file:bg-primary-dark
              disabled:opacity-50"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {documents.length === 0 ? (
          <div className="text-center text-gray-500">
            No documents uploaded yet
          </div>
        ) : (
          <div className="space-y-2">
            {documents.map((doc) => (
              <div key={doc.id} className="document-item">
                <span className="flex-1 truncate">{doc.name}</span>
                <button
                  onClick={() => handleDelete(doc.id)}
                  className="p-1 hover:bg-gray-100 rounded"
                >
                  <TrashIcon className="h-4 w-4 text-gray-500" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
} 