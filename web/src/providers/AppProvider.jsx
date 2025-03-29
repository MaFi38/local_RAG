import { useState, useEffect } from 'react';
import { AppContext } from '../context/AppContext';

export function AppProvider({ children }) {
  const [sessions, setSessions] = useState({});
  const [currentSession, setCurrentSession] = useState(null);
  const [preferences, setPreferences] = useState({
    theme: 'light',
    fontSize: 'medium',
    autoSave: true
  });

  useEffect(() => {
    const storedSessions = localStorage.getItem('sessions');
    const storedPreferences = localStorage.getItem('preferences');

    if (storedSessions) setSessions(JSON.parse(storedSessions));
    if (storedPreferences) setPreferences(JSON.parse(storedPreferences));
  }, []);

  const createNewSession = () => {
    const sessionId = `session_${Date.now()}`;
    const newSession = {
      id: sessionId,
      title: 'New Chat',
      createdAt: new Date().toISOString(),
      messages: [],
      settings: {
        model: 'gpt-4',
        temperature: 0.7,
        systemPrompt: 'You are a helpful AI assistant.'
      }
    };

    setSessions(prev => ({
      ...prev,
      [sessionId]: newSession
    }));

    return sessionId;
  };

  const value = {
    sessions,
    setSessions,
    currentSession,
    setCurrentSession,
    preferences,
    createNewSession
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
} 