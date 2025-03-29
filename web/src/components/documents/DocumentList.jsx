import { useState, useEffect } from 'react';
import { useAuth } from '../auth/AuthContext';
import { documentService } from '../../services/api';
import { TrashIcon } from '@heroicons/react/24/outline';

export default function DocumentList(reload) {
  // const { user } = useAuth();
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  
  const user = localStorage.getItem('user');
  const userp = JSON.parse(user);
  const userId = userp.user_id;

  useEffect(() => {
    loadDocuments();
  }, [userId, reload]);

  const loadDocuments = async () => {
    try {
      const docs = await documentService.listDocuments(userId);
      console.log(docs);
      setDocuments(docs);
    } catch (err) {
      setError('Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (documentId) => {
    console.log(documentId);
    try {
      
      const respone = await documentService.deleteDocument(documentId, userId);
      console.log(respone);
      setDocuments(docs => docs.filter(doc => doc.id !== documentId));
    } catch (err) {
      setError('Failed to delete document');
    }
  };

  if (loading) {
    return <div className="p-4 text-center text-white bg-gray-800">Loading documents...</div>;
  }

  if (error) {
    return <div className="p-4 text-red-500 bg-gray-800">{error}</div>;
  }


  return (
    <div className="flex-1 overflow-y-auto bg-[#1C1E21]">
      <h3 className="text-lg font-medium text-white p-4">Documents</h3>
      
      {documents.length === 0 ? (
        <div className="text-center text-gray-400 p-4">
          No documents uploaded yet
        </div>
      ) : (
        <div className="space-y-2">
          {documents.map((doc) => (
            <div key={doc.id} className="document-item bg-gray-800 p-2 rounded">
              <span className="flex-1 truncate text-white">{doc.filename}</span>
              <button
                onClick={() => handleDelete(doc.id)}
                className="p-1 hover:bg-gray-700 rounded"
              >
                <TrashIcon className="h-4 w-4 text-gray-300" />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
} 