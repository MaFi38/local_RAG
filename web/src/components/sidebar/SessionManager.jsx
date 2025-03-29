import { useState } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { format } from 'date-fns';
import { TrashIcon } from '@heroicons/react/24/outline';

export default function SessionManager() {
  const [sessions, setSessions] = useState({});
  const [currentSession, setCurrentSession] = useState(null);
  const [selectedModel, setSelectedModel] = useState('gpt-4');

  const isSessionEmpty = (sessionId) => {
    const session = sessions[sessionId];
    if (!session) return true;
    
    // Check if messages array exists and has content
    const hasMessages = session.messages && 
                       session.messages.length > 0 && 
                       session.messages.some(msg => msg.content && msg.content.trim() !== '');

    
    
    console.log('Session check:', sessionId, {
      session,
      hasMessages,
      messages: session.messages
    });
    
    return !hasMessages;
  };

  const handleSessionChange = (sessionId) => {
    // Check all sessions and remove empty ones except the one we're switching to
    setSessions(prev => {
      const newSessions = { ...prev };
      Object.keys(newSessions).forEach(sid => {
        console.log('Checking session:', sid,newSessions[sid]);
        if (sid !== sessionId && isSessionEmpty(sid)) {
          deleteSession(newSessions[sid]);
        }
      });
      return newSessions;
    });
    
    setCurrentSession(sessionId);
  };

  const createNewSession = () => {
    const newSession = {
      id: uuidv4(),
      timestamp: new Date().toISOString(),
      model: selectedModel,
      messages: []
    };

    setSessions(prev => ({
      ...prev,
      [newSession.id]: newSession
    }));
    handleSessionChange(newSession.id);
  };

  const deleteSession = (sessionId) => {
    setSessions(prev => {
      const newSessions = { ...prev };
      delete newSessions[sessionId];
      return newSessions;
    });

    if (currentSession === sessionId) {
      setCurrentSession(null);
    }
  };

  return (
    <div className="w-64 bg-white border-r flex flex-col h-full">
      <div className="p-4">
        <button
          onClick={createNewSession}
          className="w-full py-2 px-4 bg-primary text-white rounded-md hover:bg-primary-dark"
        >
          New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto">
        {Object.entries(sessions)
          .sort(([, a], [, b]) => new Date(b.timestamp) - new Date(a.timestamp))
          .map(([sessionId, session]) => (
            <div
              key={sessionId}
              className={`flex justify-between items-center p-3 cursor-pointer hover:bg-gray-100 ${
                sessionId === currentSession ? 'bg-gray-100' : ''
              }`}
              onClick={() => handleSessionChange(sessionId)}
            >
              <div className="flex-1 min-w-0">
                <p className="text-sm truncate">
                  {session.messages[0]?.content || 'New Chat'}
                </p>
                <p className="text-xs text-gray-500">
                  {format(new Date(session.timestamp), 'MMM d, yyyy')}
                </p>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteSession(sessionId);
                }}
                className="p-1 hover:bg-gray-200 rounded"
              >
                <TrashIcon className="h-4 w-4 text-gray-500" />
              </button>
            </div>
          ))}
      </div>

      <div className="p-4 border-t">
        <label className="block text-sm font-medium text-gray-700">
          Model
        </label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary focus:ring-primary sm:text-sm"
        >
          <option value="gpt-4">GPT-4</option>
          <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
        </select>
      </div>
    </div>
  );
} 