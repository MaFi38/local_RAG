import { useEffect, useState } from 'react';
import { useWebSocket } from '../../context/WebSocketContext';

export default function TypingIndicator({ sessionId }) {
  const [typingUsers, setTypingUsers] = useState(new Set());
  const { subscribe } = useWebSocket();

  useEffect(() => {
    const handleTypingStart = (data) => {
      if (data.sessionId === sessionId) {
        setTypingUsers(prev => new Set([...prev, data.userId]));
      }
    };

    const handleTypingStop = (data) => {
      if (data.sessionId === sessionId) {
        setTypingUsers(prev => {
          const newSet = new Set(prev);
          newSet.delete(data.userId);
          return newSet;
        });
      }
    };

    const unsubscribeStart = subscribe('typing:start', handleTypingStart);
    const unsubscribeStop = subscribe('typing:stop', handleTypingStop);

    return () => {
      unsubscribeStart();
      unsubscribeStop();
    };
  }, [sessionId, subscribe]);

  if (typingUsers.size === 0) return null;

  return (
    <div className="px-4 py-2 text-sm text-gray-500 italic">
      <div className="flex items-center space-x-2">
        <div className="flex space-x-1">
          <span className="animate-bounce">•</span>
          <span className="animate-bounce" style={{ animationDelay: '0.2s' }}>•</span>
          <span className="animate-bounce" style={{ animationDelay: "0.4s" }}>•</span>
        </div>
        <span>
          {typingUsers.size === 1
            ? 'Someone is typing'
            : `${typingUsers.size} people are typing`}
        </span>
      </div>
    </div>
  );
} 