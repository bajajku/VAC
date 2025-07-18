"use client"
import React, { useState, useRef, useEffect } from 'react';
import { Send, Shield, User, ArrowLeft, AlertCircle, CheckCircle2, ExternalLink, LogOut } from 'lucide-react';
import Link from 'next/link';
// Removed unnecessary dotenv import
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import FeedbackRating from '../../components/FeedbackRating';
import { feedbackService } from '../../services/feedbackService';
import SessionManager, { SessionManagerRef } from '../../components/SessionManager';
import { sessionService } from '../../services/sessionService';
import { logoutService } from '../../services/logoutService';
import Cookies from 'js-cookie';
import { useRouter } from 'next/navigation';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const INITIAL_GREETING = "Hello, I'm your confidential support assistant. I'm here to listen and provide supportive guidance in a safe, judgment-free space. How can I help you today?";

type QuestionAnswer = {
  question: string;
  answer: string;
};

type FeedbackData = {
  questionAnswer: QuestionAnswer;
  responseId: string;
  overallRating?: number;
  accuracy?: number;
  helpfulness?: number;
  clarity?: number;
  vote: 'like' | 'dislike' | null;
  comment: string;
  expertNotes: string;
  timestamp: Date;
  sessionId?: string;
};

type Message = {
  id: number;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  feedbackSubmitted?: boolean;
  sources?: string[]; // Array of source URLs
};

const ChatPage = () => {
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello, I'm your confidential support assistant. I'm here to listen and provide supportive guidance in a safe, judgment-free space. How can I help you today?",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [lastQuestion, setLastQuestion] = useState('');
  const [lastAnswer, setLastAnswer] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const sessionManagerRef = useRef<SessionManagerRef>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoadingSession, setIsLoadingSession] = useState(false);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (!Cookies.get('token')) {
      router.push('/auth/login');
      return;
    }
    scrollToBottom();
  }, [router, messages]);

  useEffect(() => {
    const savedSessionId = localStorage.getItem('chatSessionId');
    if (savedSessionId) {
      setSessionId(savedSessionId);
      loadSessionHistory(savedSessionId);
    } else {
      // If no saved session, create a new one
      createInitialSession();
    }
  }, []);

  const createInitialSession = async () => {
    try {
      const result = await sessionService.createNewSession();
      if (result.success && result.data) {
        setSessionId(result.data.session_id);
        sessionService.saveSessionToLocalStorage(result.data.session_id);
        // Keep the initial bot message
        setMessages([{
          id: 1,
          text: INITIAL_GREETING,
          sender: 'bot',
          timestamp: new Date()
        }]);
      }
    } catch (error) {
      console.error('Error creating initial session:', error);
    }
  };

  const loadSessionHistory = async (sessionId: string) => {
    setIsLoadingSession(true);
    try {
      const result = await sessionService.getSessionMessages(sessionId);
      if (result.success && result.data) {
        // Convert backend messages to frontend format
        const convertedMessages = result.data.map((msg, index) => 
          sessionService.convertBackendMessage(msg, index + 1)
        );
        
        // Add initial bot message if no messages exist
        if (convertedMessages.length === 0) {
          setMessages([{
            id: 1,
            text: INITIAL_GREETING,
            sender: 'bot',
            timestamp: new Date()
          }]);
        } else {
          setMessages(convertedMessages);
        }
      } else {
        console.error('Failed to load session history:', result.error);
      }
    } catch (error) {
      console.error('Error loading session history:', error);
    } finally {
      setIsLoadingSession(false);
    }
  };

  const handleSessionChange = (newSessionId: string) => {
    setSessionId(newSessionId);
    loadSessionHistory(newSessionId);
  };

  const handleNewSession = () => {
    // Reset messages to initial state
    setMessages([{
      id: 1,
      text: "Hello, I'm your confidential support assistant. I'm here to listen and provide supportive guidance in a safe, judgment-free space. How can I help you today?",
      sender: 'bot',
      timestamp: new Date()
    }]);
    setInputText('');
    setLastQuestion('');
    setLastAnswer('');
  };

  const handleSendMessage = async (e?: React.KeyboardEvent<HTMLInputElement> | React.MouseEvent<HTMLButtonElement>) => {
    e?.preventDefault();
    if (!inputText.trim()) return;

    const userMessage: Message = {
      id: Date.now(),
      text: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);

    try {
      // Try to refresh token if needed
      const token = Cookies.get('token');
      if (!token) {
        router.push('/auth/login');
        return;
      }

      // Try to decode the token to check expiration
      const tokenParts = token.split('.');
      if (tokenParts.length === 3) {
        const payload = JSON.parse(atob(tokenParts[1]));
        const expiresAt = payload.exp * 1000; // Convert to milliseconds
        const now = Date.now();
        const fiveMinutes = 5 * 60 * 1000;

        // If token expires in less than 5 minutes, refresh it
        if (expiresAt - now < fiveMinutes) {
          const refreshToken = Cookies.get('refresh_token');
          if (refreshToken) {
            const refreshResponse = await fetch(`${BASE_URL}/auth/refresh`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
                'ngrok-skip-browser-warning': 'true'
              },
              body: JSON.stringify({ refresh_token: refreshToken })
            });

            if (refreshResponse.ok) {
              const data = await refreshResponse.json();
              Cookies.set('token', data.access_token, { expires: 7, sameSite: 'Lax' });
              Cookies.set('refresh_token', data.refresh_token, { expires: 7, sameSite: 'Lax' });
            } else {
              router.push('/auth/login');
              return;
            }
          }
        }
      }

      const response = await fetch(`${BASE_URL}/stream_async`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${Cookies.get('token')}`,
          'ngrok-skip-browser-warning': 'true'
        },
        body: JSON.stringify({ 
          question: inputText,
          session_id: sessionId
        })
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder('utf-8');
      let fullText = '';
                const collectedSources: string[] = [];
      let botMessageId: number | null = null;
      
      if (reader) {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
      
          const chunk = decoder.decode(value);
          console.log('Raw chunk:', chunk);
      
          // Remove "data: " prefix
          let cleanChunk = chunk.replace(/^data: /gm, '').replace(/\n\n/g, '');
          
          // Extract and remove session_id BEFORE processing other content
          if (cleanChunk.includes('[SESSION_ID]')) {
            const sessionMatch = cleanChunk.match(/\[SESSION_ID\](.*?)\[\/SESSION_ID\]/);
            if (sessionMatch && !sessionId) {
              setSessionId(sessionMatch[1]);
              localStorage.setItem('chatSessionId', sessionMatch[1]);
            }
            // Remove session ID from the chunk so it doesn't appear in chat
            cleanChunk = cleanChunk.replace(/\[SESSION_ID\].*?\[\/SESSION_ID\]/g, '');
          }

          // Extract sources from the chunk and collect them (don't display yet)
          if (cleanChunk.includes('[SOURCE]')) {
            const sourceMatches = cleanChunk.match(/\[SOURCE\](.*?)\[\/SOURCE\]/g);
            if (sourceMatches) {
              sourceMatches.forEach(match => {
                const sourceContent = match.replace(/\[SOURCE\]|\[\/SOURCE\]/g, '').trim();
                // Better duplicate checking - normalize URLs
                const normalizedSource = sourceContent.toLowerCase();
                const isDuplicate = collectedSources.some(existing => 
                  existing.toLowerCase() === normalizedSource
                );
                if (sourceContent && !isDuplicate) {
                  collectedSources.push(sourceContent);
                }
              });
            }
            // Remove source markers from the chunk so they don't appear in chat
            cleanChunk = cleanChunk.replace(/\[SOURCE\].*?\[\/SOURCE\]/g, '');
          }
          
          // Only add to fullText if there's actual content after removing markers
          if (cleanChunk.trim()) {
            fullText += cleanChunk;
          }

          // Update UI progressively with text content only (no sources yet)
          if (fullText.trim()) {
            setMessages(prev => {
              const last = prev[prev.length - 1];
              if (last && last.sender === 'bot' && last.id === botMessageId) {
                // Update existing bot message
                return [...prev.slice(0, -1), { 
                  ...last, 
                  text: fullText
                }];
              } else {
                // Create new bot message
                const newBotMessage: Message = {
                  id: Date.now() + 1,
                  text: fullText,
                  sender: 'bot',
                  timestamp: new Date()
                };
                botMessageId = newBotMessage.id;
                return [...prev, newBotMessage];
              }
            });
          }
        }

        // After streaming is complete, add sources to the final message
        if (collectedSources.length > 0 && botMessageId) {
          setMessages(prev => 
            prev.map(msg => 
              msg.id === botMessageId 
                ? { ...msg, sources: collectedSources }
                : msg
            )
          );
        }
      }
      
      setLastQuestion(inputText);
      setLastAnswer(fullText);
      
      // Trigger session list refresh to update message counts
      if (sessionManagerRef.current) {
        sessionManagerRef.current.refreshSessions();
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        text: "I apologize, but I'm having trouble connecting right now. Please try again in a moment. If you need immediate support, please reach out to a healthcare professional or the Veterans Crisis Line.",
        sender: 'bot',
        timestamp: new Date()
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleFeedbackSubmit = async (feedback: FeedbackData) => {
    try {
      const result = await feedbackService.submitFeedback(feedback);
      if (result.success) {
        // Mark the message as having feedback submitted
        setMessages(prev => 
          prev.map(msg => 
            msg.id.toString() === feedback.responseId 
              ? { ...msg, feedbackSubmitted: true }
              : msg
          )
        );
        console.log('Feedback submitted successfully:', result.id);
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
    }
  };

  const handleLogout = async () => {
    try {
      const result = await logoutService.logout();
      if (result.success) {
        // Clear local storage
        localStorage.removeItem('chatSessionId');
        // Redirect to login page
        router.push('/auth/login');
      } else {
        console.error('Logout failed:', result.error);
      }
    } catch (error) {
      console.error('Error during logout:', error);
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit', 
      hour12: true 
    });
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <header className="bg-white/90 backdrop-blur-md border-b border-slate-200/50 px-4 sm:px-6 py-4 shadow-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Link 
              href="/" 
              className="w-8 h-8 bg-slate-100 hover:bg-slate-200 rounded-lg flex items-center justify-center transition-colors"
              aria-label="Return to home"
            >
              <ArrowLeft className="w-4 h-4 text-slate-600" />
            </Link>
            <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-indigo-700 rounded-xl flex items-center justify-center">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-semibold text-slate-900">Confidential Support Session</h1>
              <p className="text-sm text-slate-600">Secure • Private • Supportive</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <SessionManager
              ref={sessionManagerRef}
              currentSessionId={sessionId}
              onSessionChange={handleSessionChange}
              onNewSession={handleNewSession}
            />
            <button
              onClick={handleLogout}
              className="flex items-center space-x-2 px-3 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-colors"
              aria-label="Logout"
            >
              <LogOut className="w-4 h-4" />
              <span className="text-sm font-medium hidden sm:inline">Logout</span>
            </button>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-sm text-slate-600 hidden sm:inline">Secure Connection</span>
            </div>
          </div>
        </div>
      </header>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6 space-y-6">
        <div className="max-w-4xl mx-auto">
          {/* Loading Session Indicator */}
          {isLoadingSession && (
            <div className="flex items-center justify-center py-8">
              <div className="flex items-center space-x-3">
                <div className="animate-spin w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full"></div>
                <span className="text-slate-600">Loading conversation...</span>
              </div>
            </div>
          )}
          {!isLoadingSession && messages.map((message) => (
            <div
              key={message.id}
              className={`flex items-start space-x-4 animate-in slide-in-from-bottom-2 duration-300 ${
                message.sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''
              }`}
            >
              {/* Avatar */}
              <div className={`w-10 h-10 rounded-2xl flex items-center justify-center flex-shrink-0 shadow-sm ${
                message.sender === 'user' 
                  ? 'bg-gradient-to-r from-blue-500 to-blue-600' 
                  : 'bg-gradient-to-r from-indigo-500 to-purple-600'
              }`}>
                {message.sender === 'user' ? (
                  <User className="w-5 h-5 text-white" />
                ) : (
                  <Shield className="w-5 h-5 text-white" />
                )}
              </div>

              {/* Message Content */}
              <div className={`flex-1 max-w-2xl ${
                message.sender === 'user' ? 'items-end' : 'items-start'
              }`}>
                <div className={`rounded-2xl px-4 py-3 shadow-sm ${
                  message.sender === 'user'
                    ? 'bg-blue-600 text-white ml-auto'
                    : 'bg-white border border-slate-200 text-slate-800'
                }`}>
                  {message.sender === 'bot' ? (
                    <div className="prose prose-sm max-w-none">
                      <ReactMarkdown 
                        remarkPlugins={[remarkGfm]}
                        components={{
                          // Custom components for better styling
                          p: ({ children }) => <p className="mb-3 last:mb-0 leading-relaxed text-sm text-slate-800">{children}</p>,
                          ul: ({ children }) => <ul className="list-disc pl-4 mb-3 space-y-1 text-slate-800">{children}</ul>,
                          ol: ({ children }) => <ol className="list-decimal pl-4 mb-3 space-y-1 text-slate-800">{children}</ol>,
                          li: ({ children }) => <li className="text-sm leading-relaxed text-slate-800">{children}</li>,
                          h1: ({ children }) => <h1 className="text-lg font-semibold mb-2 text-slate-800">{children}</h1>,
                          h2: ({ children }) => <h2 className="text-base font-medium mb-2 text-slate-700">{children}</h2>,
                          h3: ({ children }) => <h3 className="text-sm font-medium mb-1 text-slate-600">{children}</h3>,
                          strong: ({ children }) => <strong className="font-semibold text-slate-800">{children}</strong>,
                          em: ({ children }) => <em className="italic text-slate-700">{children}</em>,
                          blockquote: ({ children }) => (
                            <blockquote className="border-l-4 border-blue-200 pl-3 py-1 my-2 bg-blue-50 rounded-r text-slate-700">
                              {children}
                            </blockquote>
                          ),
                          code: ({ children }) => (
                            <code className="bg-slate-100 px-1 py-0.5 rounded text-xs font-mono text-slate-800">{children}</code>
                          )
                        }}
                                              >
                          {message.text}
                        </ReactMarkdown>
                    </div>
                  ) : (
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.text}</p>
                  )}
                </div>

                {/* Sources Section for Bot Messages */}
                {message.sender === 'bot' && message.sources && message.sources.length > 0 && (
                  <div className="mt-3 p-3 bg-gradient-to-r from-slate-50 to-blue-50 border border-slate-200 rounded-lg shadow-sm">
                    <div className="flex items-center mb-2">
                      <ExternalLink className="w-3 h-3 text-slate-500 mr-2" />
                      <span className="text-xs font-medium text-slate-600">Sources:</span>
                    </div>
                    <div className="space-y-1">
                      {message.sources.map((source, idx) => {
                        // Extract title from URL or use a default
                        const getSourceTitle = (url: string) => {
                          try {
                            const urlObj = new URL(url);
                            const pathParts = urlObj.pathname.split('/').filter(Boolean);
                            const lastPart = pathParts[pathParts.length - 1];
                            if (lastPart && lastPart !== 'index.html') {
                              return lastPart.replace(/[-_]/g, ' ').replace('.html', '').replace(/\b\w/g, l => l.toUpperCase());
                            }
                            return urlObj.hostname;
                          } catch {
                            return `Source ${idx + 1}`;
                          }
                        };

                        return (
                          <div key={idx} className="group">
                            <a 
                              href={source}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-xs text-blue-600 hover:text-blue-800 hover:underline block transition-colors duration-150 group-hover:bg-white/60 p-1 rounded"
                              title={source}
                            >
                              <div className="flex items-center space-x-1">
                                <span className="font-medium">{getSourceTitle(source)}</span>
                                <ExternalLink className="w-2.5 h-2.5 opacity-60" />
                              </div>
                              <div className="text-slate-500 truncate mt-0.5 text-xs">
                                {source}
                              </div>
                            </a>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Sources Loading Indicator for Bot Messages while typing */}
                {message.sender === 'bot' && isTyping && messages[messages.length - 1]?.id === message.id && (
                  <div className="mt-3 p-2 bg-slate-50 border border-slate-200 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-slate-400 rounded-full animate-pulse"></div>
                      <span className="text-xs text-slate-500">Loading sources...</span>
                    </div>
                  </div>
                )}
                
                <div className={`flex items-center mt-2 space-x-2 ${
                  message.sender === 'user' ? 'justify-end' : 'justify-start'
                }`}>
                  <span className="text-xs text-slate-500">
                    {formatTime(message.timestamp)}
                  </span>
                  {message.sender === 'bot' && message.feedbackSubmitted && (
                    <div className="flex items-center space-x-1">
                      <CheckCircle2 className="w-3 h-3 text-green-500" />
                      <span className="text-xs text-green-600">Feedback received</span>
                    </div>
                  )}
                </div>
                
                {/* Feedback Component for Bot Messages */}
                {message.sender === 'bot' && !message.feedbackSubmitted && message.id !== 1 && (
                  <div className="mt-3">
                    <FeedbackRating
                      responseId={message.id.toString()}
                      sessionId={sessionId || undefined}
                      onSubmit={(data) => {
                        handleFeedbackSubmit({
                          ...data,
                          questionAnswer: {
                            question: lastQuestion,
                            answer: lastAnswer
                          }
                        });
                      }}
                      compact={true}
                    />
                  </div>
                )}
              </div>
            </div>
          ))}

          {/* Typing Indicator */}
          {!isLoadingSession && isTyping && (
            <div className="flex items-start space-x-4 animate-in slide-in-from-bottom-2 duration-300">
              <div className="w-10 h-10 rounded-2xl bg-gradient-to-r from-indigo-500 to-purple-600 flex items-center justify-center shadow-sm">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <div className="bg-white border border-slate-200 rounded-2xl px-4 py-3 shadow-sm">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
              <span className="text-xs text-slate-500 self-end">Thinking...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Privacy Notice */}
      <div className="px-4 sm:px-6 py-2 bg-blue-50/50 border-t border-blue-200/30">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center space-x-2 text-xs text-blue-700">
            <AlertCircle className="w-3 h-3" />
            <span>This conversation is confidential. For crisis situations, contact emergency services or the Veterans Crisis Line: 1-800-273-8255</span>
          </div>
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-white/90 backdrop-blur-md border-t border-slate-200/50 px-4 sm:px-6 py-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center space-x-3">
            <div className="flex-1 relative">
              <input
                ref={inputRef}
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage(e)}
                placeholder="Share what's on your mind... (Press Enter to send)"
                className="w-full bg-slate-50 border border-slate-200 rounded-2xl px-4 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 text-slate-800 placeholder-slate-500 text-sm"
                disabled={isTyping}
                aria-label="Type your message"
              />
            </div>
            <button
              onClick={handleSendMessage}
              disabled={!inputText.trim() || isTyping}
              className="w-11 h-11 bg-gradient-to-r from-blue-600 to-indigo-700 text-white rounded-2xl flex items-center justify-center hover:from-blue-700 hover:to-indigo-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105 active:scale-95"
              aria-label="Send message"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatPage;