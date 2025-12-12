"use client"
import React, { useState, useRef, useEffect } from 'react';
import { Send, Shield, User, ArrowLeft, CheckCircle2, ExternalLink, LogOut, Wind } from 'lucide-react';
import CrisisResources from '../../components/CrisisResources';
import GroundingTools from '../../components/GroundingTools';
import ThemeToggle from '../../components/ThemeToggle';
import VoiceInput from '../../components/VoiceInput';
import TTS from '../../components/TTS';
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
const INITIAL_GREETING = "Welcome. I'm here to support you in a safe, confidential space. There's no judgment here—only understanding. Take your time, and share whatever feels right. How can I help you today?";

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
  
  // Detailed feedback categories (1-5 star ratings)
  retrieval_relevance?: number;
  hallucination?: number;
  noise_robustness?: number;
  negative_rejection?: number;
  information_integration?: number;
  counterfactual_robustness?: number;
  privacy_breach?: number;
  malicious_use?: number;
  security_breach?: number;
  out_of_domain?: number;
  completeness?: number;
  brand_damage?: number;
  empathy?: number;
  sensitivity?: number;
  
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
  tts?: boolean;
};

const ChatPage = () => {
  const router = useRouter();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello, I'm your confidential support assistant. I'm here to listen and provide supportive guidance in a safe, judgment-free space. How can I help you today?",
      sender: 'bot',
      timestamp: new Date(),
      tts: true
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
  const [showGroundingTools, setShowGroundingTools] = useState(false);

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
            timestamp: new Date(),
            tts: true
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
      timestamp: new Date(),
      tts: true
    }]);
    setInputText('');
    setLastQuestion('');
    setLastAnswer('');
  };

  const handleSendMessage = (e?: React.KeyboardEvent<HTMLInputElement> | React.MouseEvent<HTMLButtonElement>) => {
    e?.preventDefault();
    if (!inputText.trim() || isTyping) return;
  
    const userMessage: Message = {
      id: Date.now(),
      text: inputText,
      sender: 'user',
      timestamp: new Date(),
      tts: false
    };
  
    setMessages(prev => [...prev, userMessage]);
    setLastQuestion(inputText);
    setInputText('');
    setIsTyping(true);
  };

  const handleVoiceTranscriptChange = (transcript: string) => {
    // Only update if there's a transcript and it's different from current input
    if (transcript && transcript !== inputText) {
      setInputText(transcript);
    }
  };

  useEffect(() => {
    if (!isTyping || !lastQuestion) return;
  
    const processStream = async () => {
      try {
        const token = Cookies.get('token');
        if (!token) {
          router.push('/auth/login');
          return;
        }

        // Lightweight token check - only validate if token is obviously expired
        try {
          const tokenParts = token.split('.');
          if (tokenParts.length === 3) {
            const payload = JSON.parse(atob(tokenParts[1]));
            const expiresAt = payload.exp * 1000;
            const now = Date.now();
            
            // Only refresh if token expires in less than 1 minute (reduced from 5 minutes)
            if (expiresAt <= now + 60000) {
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
        } catch (tokenError) {
          console.warn('Token validation error:', tokenError);
          // Continue with existing token if parsing fails
        }

        const response = await fetch(`${BASE_URL}/stream_async_optimized`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
            'ngrok-skip-browser-warning': 'true'
          },
          body: JSON.stringify({
            question: lastQuestion,
            session_id: sessionId
          })
        });
  
        if (!response.body) {
          throw new Error("Response body is null");
        }
  
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let fullText = '';
        const collectedSources: string[] = [];
        let botMessageId: number | null = null;
        
        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            setIsTyping(false);
            setLastAnswer(fullText);
            if (sessionManagerRef.current) {
              sessionManagerRef.current.refreshSessions();
            }
            // Final update - ensure everything is in sync
            console.log('🔍 Final update - Collected sources:', collectedSources);
            console.log('🔍 Final update - Bot message ID:', botMessageId);
            if (botMessageId) {
              console.log('🎯 Final sync of message with sources!');
              setMessages(prev =>
                prev.map(msg =>
                  msg.id === botMessageId
                    ? { ...msg, text: fullText, sources: [...collectedSources] }
                    : msg
                )
              );
            }
            break;
          }
  
          
          const chunk = decoder.decode(value, { stream: true });
          
          // Fast path: Skip processing if no special markers
          const hasMarkers = chunk.includes('[SESSION_ID]') || chunk.includes('[SOURCE]');
          let cleanChunk = chunk.replace(/^data: /gm, '').replace(/\n\n/g, '');
          
          if (hasMarkers) {
            // Extract and remove session_id BEFORE processing other content
            if (cleanChunk.includes('[SESSION_ID]')) {
              const sessionMatch = cleanChunk.match(/\[SESSION_ID\](.*?)\[\/SESSION_ID\]/);
              if (sessionMatch && !sessionId) {
                const newSessionId = sessionMatch[1];
                setSessionId(newSessionId);
                localStorage.setItem('chatSessionId', newSessionId);
              }
              cleanChunk = cleanChunk.replace(/\[SESSION_ID\].*?\[\/SESSION_ID\]/g, '');
            }

            // Extract sources from the chunk and collect them (don't display yet)
            if (cleanChunk.includes('[SOURCE]')) {
              console.log('🔍 Found [SOURCE] in chunk:', cleanChunk);
              const sourceMatches = cleanChunk.match(/\[SOURCE\](.*?)\[\/SOURCE\]/g);
              console.log('🔍 Source matches:', sourceMatches);
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
                    console.log('✅ Added source:', sourceContent);
                    console.log('📝 Total collected sources:', collectedSources.length);
                  }
                });
              }
              // Remove source markers from the chunk so they don't appear in chat
              cleanChunk = cleanChunk.replace(/\[SOURCE\].*?\[\/SOURCE\]/g, '');
            }
          }
      
          // Add content to fullText
          if (cleanChunk.trim()) {
            fullText += cleanChunk;
          }

          // Ensure we have a bot message created as soon as we get any content
          if (!botMessageId && (fullText || collectedSources.length > 0)) {
            const newBotMessage: Message = {
              id: Date.now() + 1,
              text: fullText,
              sender: 'bot',
              timestamp: new Date(),
              sources: [...collectedSources],
              tts: true
            };
            botMessageId = newBotMessage.id;
            setMessages(prev => [...prev, newBotMessage]);
          } else if (botMessageId) {
            // Update existing bot message with current text and sources
            setMessages(prev =>
              prev.map(msg =>
                msg.id === botMessageId
                  ? { ...msg, text: fullText, sources: [...collectedSources] }
                  : msg
              )
            );
          }
        }
  
      } catch (error) {
        console.error('Error sending message:', error);
        setMessages(prev => [...prev, {
          id: Date.now() + 1,
          text: "I'm having a moment of difficulty connecting, but your conversation is safe. Please take a breath and try again when you're ready. If you need immediate support, the Veterans Crisis Line is available 24/7 at 1-833-456-4566.",
          sender: 'bot',
          timestamp: new Date(),
          tts: true
        }]);
        setIsTyping(false);
      }
    };
  
    processStream();
  
  }, [isTyping, lastQuestion, sessionId, router]);

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
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800 safe-area-inset-top safe-area-inset-bottom transition-colors duration-300">
      {/* Header */}
      <header className="bg-white/90 dark:bg-slate-900/95 backdrop-blur-md border-b border-slate-200/50 dark:border-slate-700/50 px-3 sm:px-4 md:px-6 py-3 sm:py-4 shadow-sm mobile-header-landscape transition-colors">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2 sm:space-x-3 flex-1 min-w-0">
            <Link
              href="/"
              className="w-8 h-8 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-lg flex items-center justify-center transition-colors touch-target"
              aria-label="Return to home"
            >
              <ArrowLeft className="w-4 h-4 text-slate-600 dark:text-slate-400" />
            </Link>
            <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-r from-blue-600 to-indigo-700 rounded-lg sm:rounded-xl flex items-center justify-center">
              <Shield className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
            </div>
            <div className="min-w-0 flex-1">
              <h1 className="text-sm sm:text-lg font-semibold text-slate-900 dark:text-slate-100 truncate">Confidential Support Session</h1>
              <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 hidden sm:block">Secure • Private • Supportive</p>
            </div>
          </div>
          <div className="flex items-center space-x-2 sm:space-x-3">
            <div className="hidden md:block">
              <SessionManager
                ref={sessionManagerRef}
                currentSessionId={sessionId}
                onSessionChange={handleSessionChange}
                onNewSession={handleNewSession}
              />
            </div>
            <ThemeToggle />
            <button
              onClick={handleLogout}
              className="flex items-center space-x-1 sm:space-x-2 px-2 sm:px-3 py-2 bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-lg hover:bg-red-100 dark:hover:bg-red-900/50 transition-colors touch-target"
              aria-label="Logout"
            >
              <LogOut className="w-4 h-4" />
              <span className="text-sm font-medium hidden sm:inline">Logout</span>
            </button>
            <div className="hidden sm:flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-slate-600 dark:text-slate-400 hidden lg:inline">Secure</span>
            </div>
          </div>
        </div>
        
        {/* Mobile Session Manager */}
        <div className="md:hidden mt-3">
          <SessionManager
            ref={sessionManagerRef}
            currentSessionId={sessionId}
            onSessionChange={handleSessionChange}
            onNewSession={handleNewSession}
          />
        </div>
      </header>

      {/* Crisis Resources Banner */}
      <CrisisResources variant="banner" />

      {/* Grounding Tools Modal */}
      {showGroundingTools && (
        <GroundingTools
          variant="modal"
          onClose={() => setShowGroundingTools(false)}
        />
      )}

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto px-3 sm:px-4 md:px-6 py-4 sm:py-6 space-y-4 sm:space-y-6 mobile-scroll">
        <div className="max-w-4xl mx-auto">
          {/* Loading Session Indicator */}
          {isLoadingSession && (
            <div className="flex items-center justify-center py-8">
              <div className="flex flex-col items-center space-y-3">
                <div className="animate-spin w-6 h-6 border-2 border-teal-500 border-t-transparent rounded-full"></div>
                <span className="text-slate-600 text-sm">Preparing your safe space...</span>
              </div>
            </div>
          )}
          {!isLoadingSession && messages.map((message) => (
            <div
              key={message.id}
              className={`flex items-start space-x-2 sm:space-x-4 animate-in slide-in-from-bottom-2 duration-300 ${
                message.sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''
              }`}
            >
              {/* Avatar */}
              <div className={`w-8 h-8 sm:w-10 sm:h-10 rounded-xl sm:rounded-2xl flex items-center justify-center flex-shrink-0 shadow-sm ${
                message.sender === 'user' 
                  ? 'bg-gradient-to-r from-blue-500 to-blue-600' 
                  : 'bg-gradient-to-r from-indigo-500 to-purple-600'
              }`}>
                {message.sender === 'user' ? (
                  <User className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
                ) : (
                  <Shield className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
                )}
              </div>

              {/* Message Content */}
              <div className={`flex-1 max-w-[85%] sm:max-w-2xl ${
                message.sender === 'user' ? 'items-end' : 'items-start'
              }`}>
                <div className={`rounded-xl sm:rounded-2xl px-3 sm:px-4 py-2 sm:py-3 shadow-sm ${
                  message.sender === 'user'
                    ? 'bg-blue-600 text-white ml-auto'
                    : 'bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-800 dark:text-slate-200'
                }`}>
                  {message.sender === 'bot' ? (
                    <div className="prose prose-sm max-w-none dark:prose-invert">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          // Custom components for better styling
                          p: ({ children }) => <p className="mb-2 sm:mb-3 last:mb-0 leading-relaxed text-xs sm:text-sm text-slate-800 dark:text-slate-200">{children}</p>,
                          ul: ({ children }) => <ul className="list-disc pl-3 sm:pl-4 mb-2 sm:mb-3 space-y-1 text-slate-800 dark:text-slate-200">{children}</ul>,
                          ol: ({ children }) => <ol className="list-decimal pl-3 sm:pl-4 mb-2 sm:mb-3 space-y-1 text-slate-800 dark:text-slate-200">{children}</ol>,
                          li: ({ children }) => <li className="text-xs sm:text-sm leading-relaxed text-slate-800 dark:text-slate-200">{children}</li>,
                          h1: ({ children }) => <h1 className="text-sm sm:text-lg font-semibold mb-2 text-slate-800 dark:text-slate-100">{children}</h1>,
                          h2: ({ children }) => <h2 className="text-sm sm:text-base font-medium mb-2 text-slate-700 dark:text-slate-200">{children}</h2>,
                          h3: ({ children }) => <h3 className="text-xs sm:text-sm font-medium mb-1 text-slate-600 dark:text-slate-300">{children}</h3>,
                          strong: ({ children }) => <strong className="font-semibold text-slate-800 dark:text-slate-100">{children}</strong>,
                          em: ({ children }) => <em className="italic text-slate-700 dark:text-slate-300">{children}</em>,
                          blockquote: ({ children }) => (
                            <blockquote className="border-l-4 border-blue-200 dark:border-blue-700 pl-2 sm:pl-3 py-1 my-2 bg-blue-50 dark:bg-blue-900/30 rounded-r text-slate-700 dark:text-slate-300">
                              {children}
                            </blockquote>
                          ),
                          code: ({ children }) => (
                            <code className="bg-slate-100 dark:bg-slate-700 px-1 py-0.5 rounded text-xs font-mono text-slate-800 dark:text-slate-200">{children}</code>
                          )
                        }}
                      >
                        {message.text}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <p className="text-xs sm:text-sm leading-relaxed whitespace-pre-wrap">{message.text}</p>
                  )}
                </div>

                {/* Sources Section for Bot Messages */}
                {message.sender === 'bot' && message.sources && message.sources.length > 0 && (
                  <div className="mt-2 sm:mt-3 p-2 sm:p-3 bg-gradient-to-r from-slate-50 to-blue-50 dark:from-slate-800 dark:to-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg shadow-sm">
                    <div className="flex items-center mb-1 sm:mb-2">
                      <ExternalLink className="w-3 h-3 text-slate-500 dark:text-slate-400 mr-1 sm:mr-2" />
                      <span className="text-xs font-medium text-slate-600 dark:text-slate-300">Sources:</span>
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
                              className="text-xs text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 hover:underline block transition-colors duration-150 group-hover:bg-white/60 dark:group-hover:bg-slate-600/60 p-1 rounded touch-target"
                              title={source}
                            >
                              <div className="flex items-center space-x-1">
                                <span className="font-medium text-xs">{getSourceTitle(source)}</span>
                                <ExternalLink className="w-2.5 h-2.5 opacity-60" />
                              </div>
                              <div className="text-slate-500 dark:text-slate-400 truncate mt-0.5 text-xs">
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
                  <div className="mt-2 sm:mt-3 p-2 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-600 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-slate-400 dark:bg-slate-500 rounded-full animate-pulse"></div>
                      <span className="text-xs text-slate-500 dark:text-slate-400">Loading sources...</span>
                    </div>
                  </div>
                )}
                
                <div className={`flex items-center mt-1 sm:mt-2 space-x-2 ${
                  message.sender === 'user' ? 'justify-end' : 'justify-start'
                }`}>
                  <span className="text-xs text-slate-500">
                    {formatTime(message.timestamp)}
                  </span>
                  
                  {/* TTS Button for Bot Messages */}
                  {message.sender === 'bot' && message.tts && (
                    <TTS 
                      text={message.text} 
                      compact={true}
                      className="ml-1"
                    />
                  )}
                  
                  {message.sender === 'bot' && message.feedbackSubmitted && (
                    <div className="flex items-center space-x-1">
                      <CheckCircle2 className="w-3 h-3 text-green-500" />
                      <span className="text-xs text-green-600 hidden sm:inline">Feedback received</span>
                    </div>
                  )}
                </div>
                
                {/* Feedback Component for Bot Messages */}
                {message.sender === 'bot' && !message.feedbackSubmitted && message.id !== 1 && (
                  <div className="mt-2 sm:mt-3">
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
            <div className="flex items-start space-x-2 sm:space-x-4 animate-in slide-in-from-bottom-2 duration-300">
              <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-xl sm:rounded-2xl bg-gradient-to-r from-indigo-500 to-purple-600 flex items-center justify-center shadow-sm">
                <Shield className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
              </div>
              <div className="bg-white border border-slate-200 rounded-xl sm:rounded-2xl px-3 sm:px-4 py-2 sm:py-3 shadow-sm">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
              <span className="text-xs text-slate-500 self-end hidden sm:inline">Taking a moment to understand...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Wellness Bar */}
      <div className="px-3 sm:px-4 md:px-6 py-2 bg-gradient-to-r from-teal-50/80 to-emerald-50/80 dark:from-teal-900/30 dark:to-emerald-900/30 border-t border-teal-200/30 dark:border-teal-800/30 transition-colors">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-1 sm:space-x-2 text-xs text-teal-700 dark:text-teal-400">
            <Shield className="w-3 h-3 flex-shrink-0" />
            <span className="leading-tight">Your conversation is confidential and secure.</span>
          </div>
          <button
            onClick={() => setShowGroundingTools(true)}
            className="flex items-center space-x-1.5 px-3 py-1.5 bg-teal-100 dark:bg-teal-800/50 hover:bg-teal-200 dark:hover:bg-teal-800 text-teal-700 dark:text-teal-300 rounded-full text-xs font-medium transition-colors"
            aria-label="Open grounding tools"
          >
            <Wind className="w-3 h-3" />
            <span className="hidden sm:inline">Need a moment?</span>
            <span className="sm:hidden">Pause</span>
          </button>
        </div>
      </div>

      {/* Input Area */}
      <div className="bg-white/90 dark:bg-slate-900/95 backdrop-blur-md border-t border-slate-200/50 dark:border-slate-700/50 px-3 sm:px-4 md:px-6 py-3 sm:py-4 mobile-input-landscape safe-area-inset-bottom transition-colors">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center space-x-2 sm:space-x-3">
            <div className="flex-1 relative">
              <input
                ref={inputRef}
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendMessage(e)}
                placeholder="Take your time... share when you're ready"
                className="w-full bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl sm:rounded-2xl px-3 sm:px-4 py-2.5 sm:py-3 pr-10 sm:pr-12 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 focus:border-transparent transition-all duration-200 text-slate-800 dark:text-slate-200 placeholder-slate-500 dark:placeholder-slate-400 text-sm sm:text-base"
                disabled={isTyping}
                aria-label="Type your message"
              />
            </div>

            {/* Voice Input Component */}
            <VoiceInput
              onTranscriptChange={handleVoiceTranscriptChange}
              disabled={isTyping}
            />

            <button
              onClick={handleSendMessage}
              disabled={!inputText.trim() || isTyping}
              className="w-10 h-10 sm:w-11 sm:h-11 bg-gradient-to-r from-blue-600 to-indigo-700 text-white rounded-xl sm:rounded-2xl flex items-center justify-center hover:from-blue-700 hover:to-indigo-800 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105 active:scale-95 touch-target"
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