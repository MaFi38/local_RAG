import { useState } from 'react';
import { useApp } from '../../hooks/useApp';

export default function ConversationContext() {
  const { currentSession, sessions, setSessions } = useApp();
  const [showContext, setShowContext] = useState(false);
  
  const currentMessages = currentSession ? sessions[currentSession]?.messages || [] : [];
  
  const clearContext = () => {
    if (currentSession) {
      setSessions(prev => ({
        ...prev,
        [currentSession]: {
          ...prev[currentSession],
          messages: []
        }
      }));
    }
  };

  return (
    <div className="border-t">
      <div className="p-4">
        <button
          onClick={() => setShowContext(!showContext)}
          className="flex items-center justify-between w-full"
        >
          <span className="text-sm font-medium text-gray-700">
            Conversation Context
          </span>
          <svg
            className={`w-5 h-5 transform ${showContext ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </button>

        {showContext && (
          <div className="mt-2">
            <div className="text-sm text-gray-500 mb-2">
              Messages in current context: {currentMessages.length}
            </div>
            <button
              onClick={clearContext}
              className="text-sm text-red-600 hover:text-red-800"
            >
              Clear Context
            </button>
          </div>
        )}
      </div>
    </div>
  );
} 