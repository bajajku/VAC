"use client"
import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sparkles } from 'lucide-react';
import dotenv from 'dotenv';

dotenv.config();
const BASE_URL = process.env.NEXT_PUBLIC_API_URL;
const ChatPage = () => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! I'm your AI assistant. How can I help you today?",
      sender: 'bot',
      timestamp: new Date()
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const savedSessionId = localStorage.getItem('chatSessionId');
    if (savedSessionId) {
      setSessionId(savedSessionId);
    }
  }, []);

  const handleSendMessage = async (e?: React.KeyboardEvent<HTMLInputElement> | React.MouseEvent<HTMLButtonElement>) => {
    e?.preventDefault();
    if (!inputText.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);

    const response = await fetch(`${BASE_URL}/stream_async`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ 
        question: inputText,
        session_id: sessionId
      })
    });

    const reader = response.body?.getReader();
    const decoder = new TextDecoder('utf-8');
    let fullText = '';
    
    if (reader) {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
    
        const chunk = decoder.decode(value);
        console.log('Raw chunk:', chunk);  // Debug: you'll see "data: xxx\n\n"
    
        // Remove "data: " prefix
        let cleanChunk = chunk.replace(/^data: /gm, '').replace(/\n\n/g, '');
        
        // Extract and remove session_id BEFORE adding to fullText
        if (cleanChunk.includes('[SESSION_ID]')) {
          const sessionMatch = cleanChunk.match(/\[SESSION_ID\](.*?)\[\/SESSION_ID\]/);
          if (sessionMatch && !sessionId) {
            setSessionId(sessionMatch[1]);
            localStorage.setItem('chatSessionId', sessionMatch[1]);
          }
          // Remove session ID from the chunk so it doesn't appear in chat
          cleanChunk = cleanChunk.replace(/\[SESSION_ID\].*?\[\/SESSION_ID\]/g, '');
        }
        
        // Only add to fullText if there's actual content after removing session ID
        if (cleanChunk.trim()) {
          fullText += cleanChunk;
        
          // Update UI progressively only when there's actual content
          setMessages(prev => {
            const last = prev[prev.length - 1];
            if (last && last.sender === 'bot') {
              return [...prev.slice(0, -1), { ...last, text: fullText }];
            } else {
              return [...prev, {
                id: Date.now() + 1,
                text: fullText,
                sender: 'bot',
                timestamp: new Date()
              }];
            }
          });
        }
      }
    }  
    setIsTyping(false);
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-md border-b border-slate-200/50 px-3 sm:px-6 py-3 sm:py-4 shadow-sm">
        <div className="flex items-center space-x-2 sm:space-x-3">
          <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
            <Sparkles className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
          </div>
          <div>
            <h1 className="text-base sm:text-lg font-semibold text-slate-800">AI Assistant</h1>
            <p className="text-xs sm:text-sm text-slate-500">Always here to help</p>
          </div>
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto px-3 sm:px-4 py-4 sm:py-6 space-y-3 sm:space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex items-start space-x-2 sm:space-x-3 animate-in slide-in-from-bottom-2 duration-300 ${
              message.sender === 'user' ? 'flex-row-reverse space-x-reverse' : ''
            }`}
          >
            {/* Avatar */}
            <div className={`w-6 h-6 sm:w-8 sm:h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
              message.sender === 'user' 
                ? 'bg-gradient-to-r from-green-400 to-blue-500' 
                : 'bg-gradient-to-r from-purple-400 to-pink-400'
            }`}>
              {message.sender === 'user' ? (
                <User className="w-3 h-3 sm:w-4 sm:h-4 text-white" />
              ) : (
                <Bot className="w-3 h-3 sm:w-4 sm:h-4 text-white" />
              )}
            </div>

            {/* Message Bubble */}
            <div className={`max-w-[280px] sm:max-w-xs md:max-w-sm lg:max-w-md xl:max-w-lg ${
              message.sender === 'user' ? 'ml-auto' : 'mr-auto'
            }`}>
              <div className={`rounded-2xl px-3 sm:px-4 py-2 sm:py-3 shadow-sm ${
                message.sender === 'user'
                  ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white'
                  : 'bg-white border border-slate-200 text-slate-800'
              }`}>
                <p className="text-sm leading-relaxed">{message.text}</p>
              </div>
              <p className={`text-xs text-slate-500 mt-1 ${
                message.sender === 'user' ? 'text-right' : 'text-left'
              }`}>
                {formatTime(message.timestamp)}
              </p>
            </div>
          </div>
        ))}

        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex items-start space-x-2 sm:space-x-3 animate-in slide-in-from-bottom-2 duration-300">
            <div className="w-6 h-6 sm:w-8 sm:h-8 rounded-full bg-gradient-to-r from-purple-400 to-pink-400 flex items-center justify-center">
              <Bot className="w-3 h-3 sm:w-4 sm:h-4 text-white" />
            </div>
            <div className="bg-white border border-slate-200 rounded-2xl px-3 sm:px-4 py-2 sm:py-3 shadow-sm">
              <div className="flex space-x-1">
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="bg-white/80 backdrop-blur-md border-t border-slate-200/50 px-3 sm:px-4 py-3 sm:py-4">
        <div className="flex items-center justify-center">
          <div className="flex items-center space-x-2 sm:space-x-3 w-full max-w-4xl">
            <div className="flex-1 relative">
              <input
                ref={inputRef}
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage(e)}
                placeholder="Type your message..."
                className="w-full bg-slate-100 border border-slate-200 rounded-full px-3 sm:px-4 py-2.5 sm:py-3 pr-10 sm:pr-12 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 text-slate-800 placeholder-slate-500 text-sm sm:text-base"
                disabled={isTyping}
              />
            </div>
            <button
              onClick={handleSendMessage}
              disabled={!inputText.trim() || isTyping}
              className="w-9 h-9 sm:w-10 sm:h-10 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-full flex items-center justify-center hover:from-blue-600 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105 active:scale-95"
            >
              <Send className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatPage;