import { useState, useEffect } from 'react';
import { useApp } from '../../hooks/useApp';

export default function TokenCounter() {
  const { currentSession, sessions } = useApp();
  const [tokenCount, setTokenCount] = useState({
    total: 0,
    current: 0
  });

  useEffect(() => {
    if (currentSession && sessions[currentSession]) {
      // Simple estimation: ~4 chars per token
      const messages = sessions[currentSession].messages;
      const currentTokens = Math.ceil(
        messages.reduce((acc, msg) => acc + msg.content.length / 4, 0)
      );
      
      setTokenCount(prev => ({
        ...prev,
        current: currentTokens
      }));
    }
  }, [currentSession, sessions]);

  return (
    <div className="px-4 py-2 text-xs text-gray-400 bg-[#1C1E21]">
      <div className="flex justify-between">
        <span>Session Tokens: {tokenCount.current}</span>
        <span>Total Tokens: {tokenCount.total}</span>
      </div>
      <div className="mt-1 w-full bg-gray-800 rounded-full h-1.5">
        <div
          className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
          style={{ width: `${Math.min((tokenCount.current / 4000) * 100, 100)}%` }}
        />
      </div>
    </div>
  );
} 