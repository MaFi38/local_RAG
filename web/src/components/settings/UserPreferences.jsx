import { useState, useEffect } from 'react';
import { useApp } from '../../hooks/useApp';

export default function UserPreferences() {
  const { updatePreferences, preferences } = useApp();
  const [theme, setTheme] = useState(preferences?.theme || 'light');
  const [fontSize, setFontSize] = useState(preferences?.fontSize || 'medium');
  const [autoSave, setAutoSave] = useState(preferences?.autoSave || true);
  const [notifications, setNotifications] = useState(preferences?.notifications || {
    chat: true,
    documents: true,
    system: true
  });

  useEffect(() => {
    // Apply theme
    document.documentElement.classList.remove('light', 'dark');
    document.documentElement.classList.add(theme);
  }, [theme]);

  const handleSave = () => {
    const newPreferences = {
      theme,
      fontSize,
      autoSave,
      notifications
    };
    updatePreferences(newPreferences);
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow">
      <h2 className="text-lg font-medium text-gray-900 mb-4">Preferences</h2>

      <div className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-700">Theme</label>
          <select
            value={theme}
            onChange={(e) => setTheme(e.target.value)}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary focus:ring-primary sm:text-sm"
          >
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700">Font Size</label>
          <select
            value={fontSize}
            onChange={(e) => setFontSize(e.target.value)}
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-primary focus:ring-primary sm:text-sm"
          >
            <option value="small">Small</option>
            <option value="medium">Medium</option>
            <option value="large">Large</option>
          </select>
        </div>

        <div>
          <div className="flex items-center">
            <input
              type="checkbox"
              checked={autoSave}
              onChange={(e) => setAutoSave(e.target.checked)}
              className="h-4 w-4 text-primary focus:ring-primary border-gray-300 rounded"
            />
            <label className="ml-2 block text-sm text-gray-900">
              Auto-save conversations
            </label>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Notifications
          </label>
          <div className="space-y-2">
            {Object.entries(notifications).map(([key, value]) => (
              <div key={key} className="flex items-center">
                <input
                  type="checkbox"
                  checked={value}
                  onChange={(e) => setNotifications(prev => ({
                    ...prev,
                    [key]: e.target.checked
                  }))}
                  className="h-4 w-4 text-primary focus:ring-primary border-gray-300 rounded"
                />
                <label className="ml-2 block text-sm text-gray-900">
                  {key.charAt(0).toUpperCase() + key.slice(1)} notifications
                </label>
              </div>
            ))}
          </div>
        </div>

        <button
          onClick={handleSave}
          className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
        >
          Save Preferences
        </button>
      </div>
    </div>
  );
} 