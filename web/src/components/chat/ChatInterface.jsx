import { useState, useEffect, useRef } from "react";
import { useApp } from "../../hooks/useApp";
import { LoadingSpinner } from "../common/LoadingStates";
import { uploadDocument } from "../../services/document.service";
import { getChatResponse } from "../../services/chat.service";

export default function ChatInterface({ sessionId, isDocumentOpen, isSidebarOpen }) {
  const { sessions, setSessions } = useApp();
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const [showUploadPopover, setShowUploadPopover] = useState(false);
  const fileInputRef = useRef(null);
  const imageInputRef = useRef(null);
  const [isUploading, setIsUploading] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [showPdfViewer, setShowPdfViewer] = useState(false);
  const [selectedMessage, setSelectedMessage] = useState(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [sessions[sessionId]?.messages]);

  useEffect(() => {
    if (sessionId && sessions[sessionId]?.messages?.length > 0) {
      scrollToBottom();
    }
  }, [sessionId, sessions]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading) return;

    const newMessage = {
      role: "user",
      content: inputMessage,
      timestamp: new Date().toISOString(),
    };

    setIsLoading(true);
    setInputMessage("");
    setStreamingText(""); // Reset streaming text

    try {
      // Add user message to the session
      setSessions((prev) => ({
        ...prev,
        [sessionId]: {
          ...prev[sessionId],
          messages: [...(prev[sessionId]?.messages || []), newMessage],
        },
      }));
      console.log(sessions);
      console.log(sessionId);

      // Get response from the chat service
      const user = localStorage.getItem("user");
      const userp = JSON.parse(user);
      const userId = userp.user_id;
      const model = sessions[sessionId]?.settings?.model || 'gpt-4o-mini';
      // or get this from props/state if you have multiple models
      console.log(inputMessage, sessionId, model, userId);
      const response = await getChatResponse(
        inputMessage,
        sessionId,
        model,
        userId
      );
      // const response = {
      //   role: "system",
      // content: inputMessage,
      // timestamp: new Date().toISOString(),
      // };

      console.log(response);
      // Start streaming the response
      let streamText = "";
      const chars = response.answer.split("");
      setIsStreaming(true);

      const streamInterval = setInterval(() => {
        if (chars.length > 0) {
          const nextChar = chars.shift();
          streamText += nextChar;
          setStreamingText(streamText);
        } else {
          clearInterval(streamInterval);
          setIsStreaming(false);

          // Add assistant's response to the session with document data
          setSessions((prev) => ({
            ...prev,
            [sessionId]: {
              ...prev[sessionId],
              messages: [
                ...(prev[sessionId]?.messages || []),
                {
                  role: "assistant",
                  content: response.answer,
                  timestamp: new Date().toISOString(),
                  documents: response.documents || {},  // Save document data
                  userId: response.user_id // Save user ID
                },
              ],
            },
          }));
          setIsLoading(false);
        }
      }, 50);
    } catch (error) {
      console.error("Error sending message:", error);
      // Add error message to the chat
      setSessions((prev) => ({
        ...prev,
        [sessionId]: {
          ...prev[sessionId],
          messages: [
            ...(prev[sessionId]?.messages || []),
            {
              role: "assistant",
              content:
                "Sorry, there was an error processing your request. Please try again.",
              timestamp: new Date().toISOString(),
              error: true,
            },
          ],
        },
      }));
      setIsLoading(false);
      setIsStreaming(false);
    }
  };

  const handleFiles = async (files) => {
    setIsUploading(true);
    try {
      for (const file of files) {
        const user = localStorage.getItem("user");
        const userp = JSON.parse(user);
        const userId = userp.user_id;
        await uploadDocument(file, userId);
      }
    } catch (error) {
      console.error("Error uploading files:", error);
      alert("Failed to upload file: " + error.message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileChange = async (e, type) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    if (type === "image") {
      const imageFiles = files.filter((file) => file.type.startsWith("image/"));
      if (imageFiles.length !== files.length) {
        alert("Please select only image files");
        e.target.value = "";
        return;
      }
      await handleFiles(imageFiles);
    } else {
      const allowedTypes = [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
      ];
      const validFiles = files.filter((file) =>
        allowedTypes.includes(file.type)
      );
      if (validFiles.length !== files.length) {
        alert("Please select only PDF, DOC, DOCX, or TXT files");
        e.target.value = "";
        return;
      }
      await handleFiles(validFiles);
    }
    e.target.value = "";
  };

  const handleFileUpload = (type) => {
    if (type === "image") {
      imageInputRef.current?.click();
    } else {
      fileInputRef.current?.click();
    }
    setShowUploadPopover(false);
  };

  return (
    <div className="flex flex-col h-full bg-[#0D1117]">
      {showPdfViewer ? (
        <div 
          className="flex-1 relative"
          style={{
            marginLeft: isSidebarOpen ? '256px' : '0',
            marginRight: isDocumentOpen ? '300px' : '0',
            transition: 'all 300ms',
            padding: 0
          }}
        >
          <button
            onClick={() => {
              setShowPdfViewer(false);
              setSelectedMessage(null);
            }}
            className="absolute top-4 right-4 z-10 p-2 bg-gray-800 rounded-full text-gray-200 hover:bg-gray-700"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
          <iframe
            src={`http://localhost:3003/react-pdf-highlighter/react-pdf-highlighter?userId=${selectedMessage?.userId || ''}&documents=${encodeURIComponent(JSON.stringify(selectedMessage?.documents || {}))}`}
            className="w-full h-full border-none"
            title="PDF Viewer"
            style={{ display: 'block' }}
          />
        </div>
      ) : (
        <div 
          className="flex-1 overflow-y-auto p-4 space-y-4 pb-24"
          style={{
            marginLeft: isSidebarOpen ? '256px' : '0',
            marginRight: isDocumentOpen ? '300px' : '0',
            transition: 'all 300ms'
          }}
        >
          {sessions[sessionId]?.messages?.length > 0 ? (
            sessions[sessionId]?.messages?.map((message, index) => (
              <div
                key={`message-${index}`}
                className={`p-4 rounded-lg ${
                  message.role === "user"
                    ? "bg-primary text-white ml-12"
                    : message.role === "system"
                    ? "bg-gray-800 text-gray-200"
                    : "bg-gray-800 text-gray-200 mr-12"
                }`}
                style={{
                  marginRight: message.role === "system" ? (isDocumentOpen ? '300px' : '48px') : undefined,
                  marginLeft: message.role === "system" ? (isSidebarOpen ? '256px' : '0') : undefined
                }}
              >
                <div className="flex justify-between items-start">
                  <div>{message.content}</div>
                  {(message.role === "assistant" || message.role === "system") && (
                    <button
                      onClick={() => {
                        setSelectedMessage(message);
                        setShowPdfViewer(true);
                      }}
                      className="ml-4 px-2 py-1 text-xs bg-gray-700 hover:bg-gray-600 rounded-md flex items-center gap-1"
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-4 w-4"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M12 6v6m0 0v6m0-6h6m-6 0H6"
                        />
                      </svg>
                      Show Source
                    </button>
                  )}
                </div>
              </div>
            ))
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-center text-gray-400 p-8">
              <div className="text-4xl mb-4">👋</div>
              <h2 className="text-xl font-semibold mb-2">Welcome to your chat session</h2>
              <p className="max-w-md">Type a message below to start a conversation with the AI assistant.</p>
            </div>
          )}
          {isStreaming && (
            <div className="p-4 rounded-lg bg-gray-800 text-gray-200 mr-12">
              {streamingText}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      )}

      <div
        className={`border-t border-gray-800 p-4 fixed bottom-0 bg-[#0D1117] transition-all duration-300`}
        style={{
          left: isSidebarOpen ? '256px' : '0',
          right: isDocumentOpen ? '300px' : '0',
        }}
      >
        <form onSubmit={handleSendMessage} className="max-w-5xl mx-auto">
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => handleFileChange(e, "file")}
            accept=".pdf,.doc,.docx,.txt"
            className="hidden"
          />
          <input
            type="file"
            ref={imageInputRef}
            onChange={(e) => handleFileChange(e, "image")}
            accept="image/*"
            className="hidden"
          />

          <div className="flex space-x-4">
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowUploadPopover(!showUploadPopover)}
                className="px-3 py-2 border border-gray-600 rounded-md hover:bg-gray-800 text-gray-400"
              >
                +
              </button>

              {showUploadPopover && (
                <div className="absolute bottom-12 left-0 bg-[#1C1E21] text-gray-200 rounded-lg shadow-lg p-4 w-64 border border-gray-700">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-lg font-medium">Attach</span>
                    <button
                      onClick={() => setShowUploadPopover(false)}
                      className="text-gray-400 hover:text-gray-200"
                    >
                      ×
                    </button>
                  </div>
                  <button
                    onClick={() => handleFileUpload("image")}
                    className="flex items-center space-x-3 w-full p-2 hover:bg-gray-900 rounded"
                  >
                    <span>📷</span>
                    <div className="text-left">
                      <div>Upload Image</div>
                      <div className="text-sm text-gray-400">
                        Choose from local files on your device
                      </div>
                    </div>
                  </button>
                  <button
                    onClick={() => handleFileUpload("file")}
                    className="flex items-center space-x-3 w-full p-2 hover:bg-gray-900 rounded mt-2"
                  >
                    <span>📎</span>
                    <div className="text-left">
                      <div>Upload File</div>
                      <div className="text-sm text-gray-400">
                        Choose from local files on your device
                      </div>
                    </div>
                  </button>
                </div>
              )}
            </div>

            <div className="flex-1 relative">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Query:"
                disabled={isLoading}
                className="w-full p-2 px-4 pr-12 bg-[#1C1E21] border border-blue-500 text-gray-200 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                type="submit"
                disabled={isLoading || !inputMessage.trim()}
                className="absolute right-3 top-1/2 -translate-y-1/2"
              >
                {isLoading ? (
                  <LoadingSpinner />
                ) : (
                  <svg
                    className="w-5 h-5 text-gray-400"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    stroke="none"
                  >
                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
