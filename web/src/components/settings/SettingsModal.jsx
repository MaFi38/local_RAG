import { useState } from 'react';
import ApiKeyManager from './ApiKeyManager';
import { useApp } from '../../hooks/useApp';

export default function SettingsModal({ isOpen, onClose }) {
  const { updatePreferences, preferences } = useApp();
  const [theme, setTheme] = useState(preferences?.theme || 'light');

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-[#1C1E21] rounded-lg shadow-xl w-full max-w-md text-gray-200">
        <div className="flex justify-between items-center p-4 border-b border-gray-700">
          <h2 className="text-lg font-medium">Settings</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-300"
          >
            <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-4 space-y-4">
          <ApiKeyManager />
          
          <div className="border-t border-gray-700 pt-4">
            <h3 className="text-sm font-medium text-gray-300 mb-2">Theme</h3>
            <select
              value={theme}
              onChange={(e) => {
                setTheme(e.target.value);
                updatePreferences({ theme: e.target.value });
              }}
              className="w-full px-3 py-2 bg-gray-800 border border-gray-700 text-gray-200 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="light">Light</option>
              <option value="dark">Dark</option>
              <option value="system">System</option>
            </select>
          </div>

          <div className="border-t border-gray-700 pt-4">
            <h3 className="text-sm font-medium text-gray-300 mb-2">About</h3>
            <p className="text-sm text-gray-400">
              Version: 1.0.0
              <br />
              Made with ❤️ by Your Team
            </p>
          </div>
        </div>

        <div className="p-4 border-t border-gray-700 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
} 