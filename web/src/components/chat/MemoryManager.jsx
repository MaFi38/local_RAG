import { useState } from 'react';
import { useApp } from '../../hooks/useApp';

export default function MemoryManager() {
  const [memoryType, setMemoryType] = useState('chat');
  const { updateMemorySettings } = useApp();

  const handleMemoryChange = (type) => {
    setMemoryType(type);
    updateMemorySettings({ type });
  };

  return (
    <div className="p-4 border-t">
      <h3 className="text-sm font-medium text-gray-700 mb-2">Memory Type</h3>
      <div className="space-y-2">
        <label className="flex items-center">
          <input
            type="radio"
            value="chat"
            checked={memoryType === 'chat'}
            onChange={() => handleMemoryChange('chat')}
            className="h-4 w-4 text-primary border-gray-300 focus:ring-primary"
          />
          <span className="ml-2 text-sm text-gray-700">Chat Memory</span>
        </label>
        
        <label className="flex items-center">
          <input
            type="radio"
            value="vector"
            checked={memoryType === 'vector'}
            onChange={() => handleMemoryChange('vector')}
            className="h-4 w-4 text-primary border-gray-300 focus:ring-primary"
          />
          <span className="ml-2 text-sm text-gray-700">Vector Memory</span>
        </label>

        <label className="flex items-center">
          <input
            type="radio"
            value="none"
            checked={memoryType === 'none'}
            onChange={() => handleMemoryChange('none')}
            className="h-4 w-4 text-primary border-gray-300 focus:ring-primary"
          />
          <span className="ml-2 text-sm text-gray-700">No Memory</span>
        </label>
      </div>
    </div>
  );
} 