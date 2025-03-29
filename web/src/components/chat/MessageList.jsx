import { memo } from 'react';

function MessageList({ messages, messagesEndRef }) {
  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((message, index) => (
        <div
          key={index}
          className={`chat-message ${
            message.role === 'user' ? 'user-message' : 'assistant-message'
          }`}
        >
          <div className="text-sm text-gray-500">
            {message.role === 'user' ? 'You' : 'Assistant'}
          </div>
          <div className="mt-1">{message.content}</div>
        </div>
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
}

export default memo(MessageList); 