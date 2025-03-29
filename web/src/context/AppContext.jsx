import { createContext, useContext, useState, useEffect } from 'react';

// Create a separate file for the context
export const AppContext = createContext(null);

// Create the hook as a named function declaration
function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}

// Create the provider as a named function declaration
function AppProvider({ children }) {
  const [sessions, setSessions] = useState({});
  const [currentSession, setCurrentSession] = useState(null);
  const [preferences, setPreferences] = useState({
    theme: 'light',
    fontSize: 'medium',
    autoSave: true
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

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
      [sessionId]: newSession,
      ...prev,
    }));

    return sessionId;
  };

  const fetchChatHistory = async (userId) => {
    if (!userId) {
      setError('User ID is required to fetch chat history');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Clear localStorage first
      localStorage.removeItem('sessions');
      
      const response = await fetch(`http://localhost:5173/chat-history?user_id=${userId}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch chat history: ${response.status}`);
      }

      const data = await response.json();
      
      // Transform the API response to match our session format
      const transformedSessions = {};
      
      Object.entries(data).forEach(([sessionId, sessionData]) => {
        // Create an array of interleaved messages (user query followed by assistant response)
        const messages = [];
        
        // Process each query-response pair and add them as sequential messages
        sessionData.queries.forEach(query => {
          // Add user message
          messages.push({
            id: `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            role: 'user',
            content: query.query,
            timestamp: query.timestamp,
            status: 'complete'
          });
          
          // Add assistant message with document data
          messages.push({
            id: `assistant_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            role: 'assistant',
            content: query.response,
            timestamp: query.timestamp,
            status: 'complete',
            documents: query.documents || {}, // Include document highlighting data
            userId: userId // Include user ID for document access
          });
        });
        
        // Sort messages by timestamp if needed
        messages.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        
        transformedSessions[sessionId] = {
          id: sessionId,
          title: sessionData.name || `Chat from ${new Date(sessionData.timestamp).toLocaleDateString()}`,
          createdAt: sessionData.timestamp,
          messages: messages,
          settings: {
            model: sessionData.model || 'gpt-4o-mini',
            temperature: 0.7,
            systemPrompt: 'You are a helpful AI assistant.'
          }
        };
      });
      
      // Sort sessions by createdAt date (newest first)
      const sortedSessions = Object.fromEntries(
        Object.entries(transformedSessions)
          .sort(([, a], [, b]) => new Date(b.createdAt) - new Date(a.createdAt))
      );
      
      setSessions(sortedSessions);
      
      // Save to localStorage
      localStorage.setItem('sessions', JSON.stringify(sortedSessions));
      
      // Set the most recent session as current if no current session
      if (!currentSession && Object.keys(sortedSessions).length > 0) {
        setCurrentSession(Object.keys(sortedSessions)[0]);
      }
      
    } catch (err) {
      setError(err.message);
      console.error('Error fetching chat history:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const updateSessionName = async (sessionId, newName) => {
    if (!sessionId) {
      setError('Session ID is required to update chat name');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Check if session exists and has no messages
      if (sessions[sessionId]?.messages?.length === 0 || sessions[sessionId]?.messages?.length === undefined) {
        // Update only frontend state for empty sessions
        setSessions(prev => {
          const updatedSessions = { ...prev };
          if (updatedSessions[sessionId]) {
            updatedSessions[sessionId] = {
              ...updatedSessions[sessionId],
              title: newName
            };
          }
          return updatedSessions;
        });

        // Update localStorage
        const updatedSessions = { ...sessions };
        if (updatedSessions[sessionId]) {
          updatedSessions[sessionId] = {
            ...updatedSessions[sessionId],
            title: newName
          };
        }
        localStorage.setItem('sessions', JSON.stringify(updatedSessions));
        setIsLoading(false);
        return;
      }

      const user = localStorage.getItem('user');
      const { user_id } = JSON.parse(user);

      const response = await fetch('http://localhost:5173/chat-name', {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id,
          session_id: sessionId,
          name: newName
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to update chat name: ${response.status}`);
      }

      // Update local state
      setSessions(prev => {
        const updatedSessions = { ...prev };
        if (updatedSessions[sessionId]) {
          updatedSessions[sessionId] = {
            ...updatedSessions[sessionId],
            title: newName
          };
        }
        return updatedSessions;
      });

      // Update localStorage
      const updatedSessions = { ...sessions };
      if (updatedSessions[sessionId]) {
        updatedSessions[sessionId] = {
          ...updatedSessions[sessionId],
          title: newName
        };
      }
      localStorage.setItem('sessions', JSON.stringify(updatedSessions));

    } catch (err) {
      setError(err.message);
      console.error('Error updating chat name:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const deleteSession = async (sessionId) => {
    if (!sessionId) {
      setError('Session ID is required to delete chat');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // Check if session exists and has no messages
      if (sessions[sessionId]?.messages?.length === 0 || sessions[sessionId]?.messages?.length === undefined) {
        // Update only frontend state for empty sessions
        setSessions(prev => {
          const newSessions = { ...prev };
          delete newSessions[sessionId];
          return newSessions;
        });

        // Update localStorage
        const updatedSessions = { ...sessions };
        delete updatedSessions[sessionId];
        localStorage.setItem('sessions', JSON.stringify(updatedSessions));

        // If the deleted session was the current session, set a new current session
        if (currentSession === sessionId) {
          const remainingSessions = Object.keys(updatedSessions);
          if (remainingSessions.length > 0) {
            setCurrentSession(remainingSessions[0]);
          } else {
            setCurrentSession(null);
          }
        }
        setIsLoading(false);
        return;
      }

      const user = localStorage.getItem('user');
      const { user_id } = JSON.parse(user);

      const response = await fetch(
        `http://localhost:5173/delete-chat-history?user_id=${user_id}&session_id=${sessionId}`,
        {
          method: 'DELETE'
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to delete chat: ${response.status}`);
      }

      // Update local state by removing the session
      setSessions(prev => {
        const newSessions = { ...prev };
        delete newSessions[sessionId];
        return newSessions;
      });

      // Update localStorage
      const updatedSessions = { ...sessions };
      delete updatedSessions[sessionId];
      localStorage.setItem('sessions', JSON.stringify(updatedSessions));

      // If the deleted session was the current session, set a new current session
      if (currentSession === sessionId) {
        const remainingSessions = Object.keys(updatedSessions);
        if (remainingSessions.length > 0) {
          setCurrentSession(remainingSessions[0]);
        } else {
          setCurrentSession(null);
        }
      }

    } catch (err) {
      setError(err.message);
      console.error('Error deleting chat:', err);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const value = {
    sessions,
    setSessions,
    currentSession,
    setCurrentSession,
    preferences,
    createNewSession,
    fetchChatHistory,
    updateSessionName,
    deleteSession,
    isLoading,
    error
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
}

// Export both as named exports
export { useApp, AppProvider }; 