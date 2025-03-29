export default function ModelSelector({ currentSession, sessions, setSessions }) {
  const handleModelChange = (model) => {
    if (!currentSession) return;

    setSessions(prev => ({
      ...prev,
      [currentSession]: {
        ...prev[currentSession],
        model
      }
    }));
  };

  const currentModel = currentSession ? sessions[currentSession]?.model : 'gpt-4';

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700">
        Model
      </label>
      <select
        value={currentModel}
        onChange={(e) => handleModelChange(e.target.value)}
        disabled={!currentSession}
        className="block w-full rounded-md border-gray-300 shadow-sm focus:border-primary focus:ring-primary sm:text-sm"
      >
        <option value="gpt-4">GPT-4</option>
        <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
      </select>
    </div>
  );
} 