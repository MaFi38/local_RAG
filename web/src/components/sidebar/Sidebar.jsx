import { useState } from 'react';
import { useAuth } from '../auth/AuthContext';
import SessionList from './SessionList';
import ModelSelector from './ModelSelector';
import { chatService } from '../../services/api';
import { createNewSession } from '../../utils/sessionManager';

export default function Sidebar() {
  const { user, sessions, setSessions, currentSession, setCurrentSession } = useAuth();
  const [isCreatingSession, setIsCreatingSession] = useState(false);

  const handleCreateSession = async () => {
    if (isCreatingSession) return;
    setIsCreatingSession(true);

    try {
      const newSession = createNewSession();
      setSessions(prev => ({
        ...prev,
        [newSession.id]: newSession
      }));
      setCurrentSession(newSession.id);
    } finally {
      setIsCreatingSession(false);
    }
  };

  const handleDeleteSession = async (sessionId) => {
    try {
      await chatService.deleteSession(sessionId, user.id);
      setSessions(prev => {
        const newSessions = { ...prev };
        delete newSessions[sessionId];
        return newSessions;
      });

      if (currentSession === sessionId) {
        setCurrentSession(null);
      }
    } catch (error) {
      console.error('Error deleting session:', error);
    }
  };

  return (
    <div className="w-64 bg-gray-50 border-r h-full flex flex-col">
      <div className="p-4">
        <button
          onClick={handleCreateSession}
          disabled={isCreatingSession}
          className="w-full py-2 px-4 bg-primary text-white rounded-md hover:bg-primary-dark disabled:opacity-50"
        >
          New Chat
        </button>
      </div>

      <SessionList
        sessions={sessions}
        currentSession={currentSession}
        onSelectSession={setCurrentSession}
        onDeleteSession={handleDeleteSession}
      />

      <div className="mt-auto p-4 border-t">
        <ModelSelector
          currentSession={currentSession}
          sessions={sessions}
          setSessions={setSessions}
        />
      </div>
    </div>
  );
} 