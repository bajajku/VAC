'use client';

import React, { useState, useEffect, forwardRef, useImperativeHandle } from 'react';
import { Plus, MessageSquare, Clock, ChevronDown, Trash2, Edit2, Check, X } from 'lucide-react';
import { sessionService, ChatSession } from '../services/sessionService';

interface SessionManagerProps {
  currentSessionId: string | null;
  onSessionChange: (sessionId: string) => void;
  onNewSession: () => void;
}

export interface SessionManagerRef {
  refreshSessions: () => Promise<void>;
}

const SessionManager = forwardRef<SessionManagerRef, SessionManagerProps>(({
  currentSessionId,
  onSessionChange,
  onNewSession
}, ref) => {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [loading, setLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [editingSession, setEditingSession] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    setLoading(true);
    try {
      const result = await sessionService.listSessions();
      if (result.success && result.data) {
        setSessions(result.data);
      } else {
        console.error('Failed to load sessions:', result.error);
      }
    } catch (error) {
      console.error('Error loading sessions:', error);
    } finally {
      setLoading(false);
    }
  };

  // Expose refresh method via ref
  useImperativeHandle(ref, () => ({
    refreshSessions: loadSessions
  }));

  const handleNewSession = async () => {
    try {
      const result = await sessionService.createNewSession();
      if (result.success && result.data) {
        // Add new session to the list
        setSessions(prev => [result.data!, ...prev]);
        // Switch to new session
        onSessionChange(result.data.session_id);
        sessionService.saveSessionToLocalStorage(result.data.session_id);
        setIsOpen(false);
        onNewSession();
      } else {
        console.error('Failed to create new session:', result.error);
      }
    } catch (error) {
      console.error('Error creating new session:', error);
    }
  };

  const handleSessionSelect = (sessionId: string) => {
    onSessionChange(sessionId);
    sessionService.saveSessionToLocalStorage(sessionId);
    setIsOpen(false);
  };

  const handleDeleteSession = async (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    
    if (confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) {
      try {
        const result = await sessionService.deleteSession(sessionId);
        if (result.success) {
          setSessions(prev => prev.filter(s => s.session_id !== sessionId));
          
          // If we deleted the current session, create a new one
          if (sessionId === currentSessionId) {
            handleNewSession();
          }
        } else {
          console.error('Failed to delete session:', result.error);
        }
      } catch (error) {
        console.error('Error deleting session:', error);
      }
    }
  };

  const handleEditSession = (sessionId: string, currentTitle: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingSession(sessionId);
    setEditTitle(currentTitle || '');
  };

  const handleSaveEdit = async (sessionId: string) => {
    try {
      const result = await sessionService.updateSession(sessionId, { title: editTitle });
      if (result.success && result.data) {
        setSessions(prev => 
          prev.map(s => s.session_id === sessionId ? result.data! : s)
        );
      } else {
        console.error('Failed to update session:', result.error);
      }
    } catch (error) {
      console.error('Error updating session:', error);
    } finally {
      setEditingSession(null);
      setEditTitle('');
    }
  };

  const handleCancelEdit = () => {
    setEditingSession(null);
    setEditTitle('');
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 1) {
      return 'Today';
    } else if (diffDays === 2) {
      return 'Yesterday';
    } else if (diffDays <= 7) {
      return `${diffDays - 1} days ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  const getCurrentSession = () => {
    return sessions.find(s => s.session_id === currentSessionId);
  };

  const currentSession = getCurrentSession();

  return (
    <div className="relative">
      {/* Session Selector Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center space-x-2 px-4 py-2 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors min-w-[200px]"
      >
        <MessageSquare className="w-4 h-4 text-slate-600" />
        <span className="text-sm text-slate-700 truncate flex-1 text-left">
          {currentSession?.title || 'Select Conversation'}
        </span>
        <ChevronDown className={`w-4 h-4 text-slate-500 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute top-full left-0 mt-2 w-80 bg-white border border-slate-200 rounded-lg shadow-lg z-50 max-h-96 overflow-y-auto">
          {/* New Session Button */}
          <button
            onClick={handleNewSession}
            className="w-full flex items-center space-x-3 px-4 py-3 hover:bg-blue-50 transition-colors border-b border-slate-100"
          >
            <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
              <Plus className="w-4 h-4 text-blue-600" />
            </div>
            <span className="text-sm font-medium text-slate-900">Start New Conversation</span>
          </button>

          {/* Sessions List */}
          {loading ? (
            <div className="p-4 text-center text-slate-500">
              <div className="animate-spin w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full mx-auto"></div>
              <span className="text-sm mt-2 block">Loading conversations...</span>
            </div>
          ) : sessions.length === 0 ? (
            <div className="p-4 text-center text-slate-500">
              <MessageSquare className="w-8 h-8 mx-auto mb-2 text-slate-400" />
              <span className="text-sm">No conversations yet</span>
            </div>
          ) : (
            <div className="py-2">
              {sessions.map((session) => (
                <div
                  key={session.session_id}
                  className={`flex items-center justify-between px-4 py-3 hover:bg-slate-50 cursor-pointer transition-colors ${
                    session.session_id === currentSessionId ? 'bg-blue-50 border-r-2 border-blue-500' : ''
                  }`}
                  onClick={() => handleSessionSelect(session.session_id)}
                >
                  <div className="flex items-center space-x-3 flex-1 min-w-0">
                    <div className="w-8 h-8 bg-slate-100 rounded-lg flex items-center justify-center">
                      <MessageSquare className="w-4 h-4 text-slate-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      {editingSession === session.session_id ? (
                        <div className="flex items-center space-x-2">
                          <input
                            type="text"
                            value={editTitle}
                            onChange={(e) => setEditTitle(e.target.value)}
                            className="flex-1 px-2 py-1 text-sm border border-slate-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                            onClick={(e) => e.stopPropagation()}
                            autoFocus
                          />
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleSaveEdit(session.session_id);
                            }}
                            className="p-1 text-green-600 hover:bg-green-100 rounded"
                          >
                            <Check className="w-3 h-3" />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleCancelEdit();
                            }}
                            className="p-1 text-red-600 hover:bg-red-100 rounded"
                          >
                            <X className="w-3 h-3" />
                          </button>
                        </div>
                      ) : (
                        <>
                          <div className="text-sm font-medium text-slate-900 truncate">
                            {session.title || `Chat ${session.session_id.slice(-8)}`}
                          </div>
                          <div className="flex items-center space-x-2 text-xs text-slate-500">
                            <Clock className="w-3 h-3" />
                            <span>{formatDate(session.updated_at)}</span>
                            <span>•</span>
                            <span>{session.message_count} messages</span>
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                  
                  {editingSession !== session.session_id && (
                    <div className="flex items-center space-x-1">
                      <button
                        onClick={(e) => handleEditSession(session.session_id, session.title || '', e)}
                        className="p-1 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded"
                        title="Edit title"
                      >
                        <Edit2 className="w-3 h-3" />
                      </button>
                      <button
                        onClick={(e) => handleDeleteSession(session.session_id, e)}
                        className="p-1 text-slate-400 hover:text-red-600 hover:bg-red-100 rounded"
                        title="Delete conversation"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
});

SessionManager.displayName = 'SessionManager';

export default SessionManager; 