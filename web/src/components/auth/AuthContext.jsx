import { createContext, useState, useContext, useEffect } from 'react';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [authenticated, setAuthenticated] = useState(false);
  const [sessions, setSessions] = useState({});
  const [currentSession, setCurrentSession] = useState(null);

  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser && storedUser !== 'undefined') {
      const userData = JSON.parse(storedUser);
      if (userData) {
        setUser(userData);
        setAuthenticated(true);
      }
    }else{
      localStorage.removeItem('user');
    }
  }, []);

  const login = (userData) => {
    setUser(userData);
    setAuthenticated(true);
    localStorage.setItem('user', JSON.stringify(userData));
  };
  
  const logout = () => {
    setUser(null);
    setAuthenticated(false);
    setSessions({});
    setCurrentSession(null);
    localStorage.removeItem('user');
  };

  const isSessionEmpty = (sessionId) => {
    const session = sessions[sessionId];
    if (!session) return true;
    
    // Check if messages array exists and has content
    const hasMessages = session.messages && 
                       session.messages.length > 0 && 
                       session.messages.some(msg => msg.content && msg.content.trim() !== '');
    
    return !hasMessages;
  };

  const handleSessionChange = (sessionId) => {
    console.log('handleSessionChange', sessionId);
    // Check all sessions and remove empty ones except the one we're switching to
    setSessions(prev => {
      const newSessions = { ...prev };
      Object.keys(newSessions).forEach(sid => {
        console.log('Checking session:', sid, newSessions[sid]);
        if (sid !== sessionId && isSessionEmpty(sid)) {
          delete newSessions[sid];
        }
      });
      return newSessions;
    });
    
    setCurrentSession(sessionId);
  };

  return (
    <AuthContext.Provider value={{
      user,
      authenticated,
      sessions,
      setSessions,
      currentSession,
      setCurrentSession: handleSessionChange,
      login,
      logout,
      isSessionEmpty
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext); 