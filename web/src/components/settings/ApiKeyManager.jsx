import { useState } from 'react';
import { useApp } from '../../hooks/useApp';

export default function ApiKeyManager() {
  const { updateApiKey } = useApp();
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    updateApiKey(apiKey);
    setShowKey(false);
  };

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium text-gray-300">API Key</h3>
      <form onSubmit={handleSubmit}>
        <div className="relative">
          <input
            type={showKey ? 'text' : 'password'}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Enter your OpenAI API key"
            className="w-full pr-20 px-3 py-2 bg-gray-800 border border-gray-700 text-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder-gray-500"
          />
          <button
            type="button"
            onClick={() => setShowKey(!showKey)}
            className="absolute right-2 top-1/2 -translate-y-1/2 px-2 py-1 text-sm text-gray-400 hover:text-gray-300"
          >
            {showKey ? 'Hide' : 'Show'}
          </button>
        </div>
        <button
          type="submit"
          className="mt-2 w-full px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
        >
          Save API Key
        </button>
      </form>
      <p className="text-xs text-gray-400">
        Your API key is stored securely and never shared with third parties.
      </p>
    </div>
  );
} 