import { useApp } from '../../hooks/useApp';

export function LoadingSpinner({ size = 'medium' }) {
  const sizeClasses = {
    small: 'w-4 h-4',
    medium: 'w-8 h-8',
    large: 'w-12 h-12'
  };

  return (
    <div className="flex justify-center items-center">
      <div
        className={`${sizeClasses[size]} animate-spin rounded-full border-4 border-gray-200 border-t-primary`}
      />
    </div>
  );
}

export function LoadingOverlay({ message = 'Loading...' }) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white p-6 rounded-lg shadow-xl">
        <LoadingSpinner size="large" />
        <p className="mt-4 text-center text-gray-700">{message}</p>
      </div>
    </div>
  );
}

export function LoadingButton({ loading, children, ...props }) {
  return (
    <button
      disabled={loading}
      className="relative px-4 py-2 bg-primary text-white rounded-md hover:bg-primary-dark disabled:opacity-50"
      {...props}
    >
      {loading ? (
        <div className="flex items-center">
          <LoadingSpinner size="small" />
          <span className="ml-2">Loading...</span>
        </div>
      ) : (
        children
      )}
    </button>
  );
} 