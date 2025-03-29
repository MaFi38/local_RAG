import { useState, useEffect } from 'react';
import { LoadingSpinner } from '../common/LoadingStates';
import { uploadDocument } from '../../services/document.service';

export default function DocumentUpload({setReload, reload}) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState(0);

  const loadingMessages = [
    "Uploading your document...",
    "Just a moment...",
    "Almost there...",
    "Processing your file...",
    "Getting everything ready...",
    "Hang on tight...",
    "Making progress...",
    "Just a few more seconds..."
  ];

  useEffect(() => {
    let interval;
    if (isUploading) {
      interval = setInterval(() => {
        setLoadingMessage((prev) => (prev + 1) % loadingMessages.length);
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [isUploading]);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    await handleFiles(files);
  };

  const handleFileInput = async (e) => {
    const files = Array.from(e.target.files);
    await handleFiles(files);
  };

  const handleFiles = async (files) => {
    setIsUploading(true);
    try {
      for (const file of files) {
        const user = localStorage.getItem('user');
        const userp = JSON.parse(user);
        const userId = userp.user_id;
        await uploadDocument(file, userId);
      }
      setReload(!reload);
    } catch (error) {
      console.error('Error uploading files:', error);
      alert('Failed to upload file: ' + error.message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="p-4 border-b border-gray-700 bg-[#1C1E21]">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-4 text-center ${
          isDragging ? 'border-blue-500 bg-blue-500/5' : 'border-gray-700'
        }`}
      >
        {isUploading ? (
          <div className="flex items-center justify-center space-x-2 text-gray-200">
            <LoadingSpinner size="small" />
            <span>{loadingMessages[loadingMessage]}</span>
          </div>
        ) : (
          <>
            <label className="cursor-pointer">
              <input
                type="file"
                multiple
                onChange={handleFileInput}
                className="hidden"
                accept=".pdf,.doc,.docx,.txt"
              />
              <div className="space-y-2">
                <svg className="mx-auto h-12 w-12 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <div className="text-sm text-gray-300">
                  Drop files here or <span className="text-blue-500 hover:text-blue-400">browse</span>
                </div>
                <div className="text-xs text-gray-500">
                  Supports PDF, DOC, DOCX, TXT
                </div>
              </div>
            </label>
          </>
        )}
      </div>
    </div>
  );
} 