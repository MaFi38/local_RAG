import { useEffect, useState } from 'react';
import { useApp } from '../../hooks/useApp';
import { LuArrowLeftFromLine, LuArrowRightFromLine } from "react-icons/lu";
import ChatInterface from '../chat/ChatInterface';
import ChatExport from '../chat/ChatExport';
import ModelSelector from '../chat/ModelSelector';
import Sidebar from './Sidebar';
import Header from './Header';
import DocumentContext from '../documents/DocumentContext';

export default function MainLayout() {
  const { currentSession, setCurrentSession, createNewSession } = useApp();
  const [isDocumentOpen, setIsDocumentOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [selectedFile, setSelectedFile] = useState(null);

  const user = localStorage.getItem("user");
  const userp = JSON.parse(user);
  const userId = userp.user_id;

  useEffect(() => {
    if (!currentSession) {
      let id = createNewSession(userId)
      setCurrentSession(id)
    }
  }, []);

  useEffect(() => {
    return () => {
      if (selectedFile) {
        URL.revokeObjectURL(selectedFile.path);
      }
    };
  }, [selectedFile]);

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-[#1C1E21] text-gray-200">
      <Header setIsSidebarOpen={setIsSidebarOpen} setIsDocumentOpen={setIsDocumentOpen} isSidebarOpen={isSidebarOpen} />
      
      <div className="flex-1 flex overflow-hidden relative">
        <aside 
          className={`fixed top-[60px] left-0 h-full z-10 transition-transform duration-300 ${
            isSidebarOpen ? 'translate-x-0' : '-translate-x-64'
          }`}
        >
          <Sidebar isOpen={isSidebarOpen} />
        </aside>

        <main className="flex-1 overflow-hidden flex flex-col bg-[#1C1E21] relative">
          {currentSession ? (
            <>
              <div className="flex-1 overflow-hidden">
                <ChatInterface 
                  sessionId={currentSession} 
                  isDocumentOpen={isDocumentOpen} 
                  isSidebarOpen={isSidebarOpen}
                />
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-full text-gray-400">
              Welcome to POC Copilot
            </div>
          )}
        </main>

        <aside 
          className={`fixed right-0 top-0 h-full w-[300px] border-l border-gray-700 bg-[#1C1E21] transition-transform duration-300 ease-in-out ${
            isDocumentOpen ? 'translate-x-0' : 'translate-x-full'
          }`}
        >
          <DocumentContext setIsDocumentOpen={setIsDocumentOpen} />
        </aside>
      </div>
    </div>
  );
} 