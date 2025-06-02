'use client';

import React from 'react';

type ChatMessageProps = {
  message: string;
  sender: 'user' | 'bot';
};

const ChatMessage: React.FC<ChatMessageProps> = ({ message, sender }) => {
  return (
    <div
      style={{
        marginBottom: '0.75rem',
        textAlign: sender === 'user' ? 'right' : 'left',
      }}
    >
      <div
        style={{
          display: 'inline-block',
          padding: '0.5rem 1rem',
          borderRadius: '1rem',
          backgroundColor: sender === 'user' ? '#4f46e5' : '#e5e7eb',
          color: sender === 'user' ? 'white' : 'black',
          maxWidth: '80%',
        }}
      >
        {message}
      </div>
    </div>
  );
};

export default ChatMessage;
