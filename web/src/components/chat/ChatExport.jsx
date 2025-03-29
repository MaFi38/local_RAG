import { useState } from 'react';
import { useApp } from '../../hooks/useApp';
import { format } from 'date-fns';

export default function ChatExport() {
  const { currentSession, sessions } = useApp();
  const [exportFormat, setExportFormat] = useState('txt');

  const exportChat = () => {
    if (!currentSession) return;

    const session = sessions[currentSession];
    const messages = session.messages;
    
    let content = '';
    if (exportFormat === 'txt') {
      content = messages.map(msg => 
        `${msg.role.toUpperCase()} [${format(new Date(msg.timestamp), 'yyyy-MM-dd HH:mm:ss')}]:\n${msg.content}\n\n`
      ).join('');
    } else if (exportFormat === 'json') {
      content = JSON.stringify(messages, null, 2);
    }

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${format(new Date(), 'yyyy-MM-dd-HH-mm')}.${exportFormat}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="p-4 border-t border-gray-700 bg-[#1C1E21]">
      <div className="flex items-center justify-between">
        <select
          value={exportFormat}
          onChange={(e) => setExportFormat(e.target.value)}
          className="mr-2 rounded-md bg-gray-800 border-gray-700 text-gray-200 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
        >
          <option value="txt">Text Format</option>
          <option value="json">JSON Format</option>
        </select>
        <button
          onClick={exportChat}
          disabled={!currentSession}
          className="px-3 py-1 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:opacity-50 transition-colors"
        >
          Export Chat
        </button>
      </div>
    </div>
  );
} 