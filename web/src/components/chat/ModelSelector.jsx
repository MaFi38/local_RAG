import { useState } from 'react';
import { useApp } from '../../hooks/useApp';

export default function ModelSelector({ sessionId }) {
  const { sessions, setSessions } = useApp();
  const currentModel = sessions[sessionId]?.settings?.model || 'gpt-4o-mini';

  const models = [
    { id: 'gpt-4', name: 'GPT-4', description: 'Most capable model, best for complex tasks' },
    { id: 'gpt-3.5-turbo', name: 'GPT-3.5 Turbo', description: 'Fast and efficient for most tasks' },
    { id: 'gpt-4o', name: 'GPT-4o Turbo', description: 'Latest GPT-4o model with enhanced capabilities' },
    { id: 'gpt-4o-mini', name: 'GPT-4o Mini', description: 'Latest GPT-4o Mini model with enhanced capabilities' },
  ];
  

  const handleModelChange = (modelId) => {
    setSessions(prev => ({
      ...prev,
      [sessionId]: {
        ...prev[sessionId],
        settings: {
          ...prev[sessionId].settings,
          model: modelId
        }
      }
    }));
  };

  return (
    <div className="px-4 py-2 border-b border-gray-700">
      <select
        value={currentModel}
        onChange={(e) => handleModelChange(e.target.value)}
        className="w-full p-2 bg-gray-800 border border-gray-700 text-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
      >
        {models.map(model => (
          <option 
            key={model.id} 
            value={model.id}
            className="bg-gray-800 text-gray-200"
          >
            {model.name}
          </option>
        ))}
      </select>
      <div className="mt-1 text-xs text-gray-400">
        {models.find(m => m.id === currentModel)?.description}
      </div>
    </div>
  );
} 