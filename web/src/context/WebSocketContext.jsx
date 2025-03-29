import { createContext, useContext, useEffect, useState } from 'react';
import { io } from 'socket.io-client';

const WebSocketContext = createContext(null);

const socket = io(import.meta.env.VITE_WS_URL || 'http://localhost:8000', {
  autoConnect: false
});

export function WebSocketProvider({ children }) {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    function onConnect() {
      setIsConnected(true);
    }

    function onDisconnect() {
      setIsConnected(false);
    }

    socket.on('connect', onConnect);
    socket.on('disconnect', onDisconnect);

    // Connect to WebSocket server
    socket.connect();

    return () => {
      socket.off('connect', onConnect);
      socket.off('disconnect', onDisconnect);
      socket.disconnect();
    };
  }, []);

  const subscribe = (event, callback) => {
    socket.on(event, callback);
    return () => socket.off(event, callback);
  };

  const emit = (event, data) => {
    socket.emit(event, data);
  };

  return (
    <WebSocketContext.Provider value={{ isConnected, subscribe, emit, socket }}>
      {children}
    </WebSocketContext.Provider>
  );
}

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
}; 