import { useState, useEffect } from 'react';
import { useApp } from '../../hooks/useApp';
import { LoadingSpinner } from '../common/LoadingStates';

export default function DocumentSearch() {
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const { documents, searchDocuments } = useApp();
  const [filteredDocs, setFilteredDocs] = useState(documents || []);

  useEffect(() => {
    const search = async () => {
      if (searchQuery.trim()) {
        setIsSearching(true);
        try {
          const results = await searchDocuments(searchQuery);
          setFilteredDocs(results || []);
        } catch (error) {
          console.error('Error searching documents:', error);
          setFilteredDocs([]);
        } finally {
          setIsSearching(false);
        }
      } else {
        setFilteredDocs(documents || []);
      }
    };

    const debounce = setTimeout(search, 300);
    return () => clearTimeout(debounce);
  }, [searchQuery, documents]);

  if (!Array.isArray(filteredDocs)) {
    return <div className="text-gray-300">Loading...</div>;
  }

  return (
    <div className="flex flex-col" style={{ backgroundColor: '#1C1E21' }}>
      <div className="relative">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search documents..."
          className="p-2 rounded bg-gray-800 text-white placeholder-gray-400"
        />
        {isSearching ? (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">
            <LoadingSpinner size="small" />
          </div>
        ) : (
          <svg
            className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        )}
      </div>

      <div className="space-y-2 mt-2">
        {filteredDocs.map((doc) => (
          <div
            key={doc.id}
            className="p-3 bg-gray-800 rounded-lg border border-gray-700 hover:border-blue-500 transition-colors text-white"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1 min-w-0">
                <h3 className="text-sm font-medium text-gray-200 truncate">
                  {doc.name}
                </h3>
                <p className="text-xs text-gray-400">
                  {new Date(doc.uploadedAt).toLocaleDateString()}
                </p>
              </div>
              <button
                onClick={() => doc.onDelete?.(doc.id)}
                className="ml-2 text-gray-400 hover:text-red-500 transition-colors"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                  />
                </svg>
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
} 