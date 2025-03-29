import React from 'react';

interface ErrorFallbackProps {
  message: string;
  details?: string;
}

export function ErrorFallback({ message, details }: ErrorFallbackProps) {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      padding: '20px',
      textAlign: 'center',
      color: '#666'
    }}>
      <div style={{
        fontSize: '64px',
        marginBottom: '20px'
      }}>
        📄
      </div>
      <h2 style={{
        margin: '0 0 10px 0',
        color: '#333'
      }}>
        {message}
      </h2>
      {details && (
        <p style={{
          margin: '0',
          fontSize: '14px'
        }}>
          {details}
        </p>
      )}
    </div>
  );
} 