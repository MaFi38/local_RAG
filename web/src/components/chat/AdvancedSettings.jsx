import { useState } from 'react';
import { useApp } from '../../hooks/useApp';

export default function AdvancedSettings() {
  const { currentSession, sessions, setSessions } = useApp();
  const [showSettings, setShowSettings] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [systemPrompt, setSystemPrompt] = useState(
    "You are a helpful AI assistant. Answer questions clearly and accurately."
  );

  const handleSettingsChange = () => {
    if (currentSession) {
      setSessions(prev => ({
        ...prev,
        [currentSession]: {
          ...prev[currentSession],
          settings: {
            temperature,
            systemPrompt
          }
        }
      }));
    }
  };

  return (
    <div className="border-t">
      <button
        onClick={() => setShowSettings(!showSettings)}
        className="flex items-center justify-between w-full p-4"
      >
        <span className="text-sm font-medium text-gray-700">Advanced Settings</span>
        <svg
          className={`w-5 h-5 transform ${showSettings ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {showSettings && (
        <div className="p-4 pt-0 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Temperature: {temperature}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full"
            />
            <p className="text-xs text-gray-500 mt-1">
              Lower values make responses more focused, higher values make them more creative.
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              System Prompt
            </label>
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-primary focus:ring-primary sm:text-sm"
              rows="3"
            />
          </div>

          <button
            onClick={handleSettingsChange}
            className="w-full px-4 py-2 bg-primary text-white rounded-md hover:bg-primary-dark"
          >
            Apply Settings
          </button>
        </div>
      )}
    </div>
  );
} 