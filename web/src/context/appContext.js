import { createContext, useState } from 'react';

export const AppContext = createContext();

export function AppProvider({ children }) {
  // ... existing code ...

  const uploadDocument = async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await fetch('/api/documents/upload', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        throw new Error('Upload failed');
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error uploading document:', error);
      throw error;
    }
  };

  return (
    <AppContext.Provider value={{
      // ... other values
      uploadDocument
    }}>
      {children}
    </AppContext.Provider>
  );
} 