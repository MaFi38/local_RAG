import { format } from 'date-fns';
import { TrashIcon } from '@heroicons/react/24/outline';

export default function SessionList({ sessions, currentSession, onSelectSession, onDeleteSession }) {
  return (
    <div className="flex-1 overflow-y-auto">
      {Object.entries(sessions)
        .sort(([, a], [, b]) => new Date(b.timestamp) - new Date(a.timestamp))
        .map(([sessionId, session]) => (
          <div
            key={sessionId}
            className={`flex justify-between items-center p-3 cursor-pointer hover:bg-gray-100 ${
              sessionId === currentSession ? 'bg-gray-100' : ''
            }`}
            // onClick={() =>{
            //   console.log('SessionList', sessionId);
            //    onSelectSession(sessionId)
            //   }}
          >
            <div className="flex-1 min-w-0">
              <p className="text-sm truncate">
                {session.messages?.[0]?.content || 'New Chat'}
              </p>
              <p className="text-xs text-gray-500">
                {format(new Date(session.timestamp), 'MMM d, yyyy')}
              </p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation();
                onDeleteSession(sessionId);
              }}
              className="p-1 hover:bg-gray-200 rounded"
            >
              <TrashIcon className="h-4 w-4 text-gray-500" />
            </button>
          </div>
        ))}
    </div>
  );
} 