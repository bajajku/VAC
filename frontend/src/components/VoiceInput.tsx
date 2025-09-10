'use client'
import React, { useEffect } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { Mic, MicOff, Square } from 'lucide-react';

interface VoiceInputProps {
  onTranscriptChange: (transcript: string) => void;
  disabled?: boolean;
}

const VoiceInput: React.FC<VoiceInputProps> = ({ onTranscriptChange, disabled = false }) => {
  const {
    transcript,
    listening,
    resetTranscript,
    browserSupportsSpeechRecognition
  } = useSpeechRecognition();

  // Update parent component when transcript changes
  useEffect(() => {
    onTranscriptChange(transcript);
  }, [transcript, onTranscriptChange]);

  const startListening = () => {
    if (!disabled && browserSupportsSpeechRecognition) {
      resetTranscript();
      SpeechRecognition.startListening({ continuous: true, interimResults: true });
    }
  };

  const stopListening = () => {
    if (browserSupportsSpeechRecognition) {
      SpeechRecognition.stopListening();
    }
  };

  if (!browserSupportsSpeechRecognition) {
    return null; // Don't show anything if not supported
  }

  return (
    <div className="flex items-center space-x-2">
      {/* Voice Status Indicator */}
      {listening && (
        <div className="flex items-center space-x-1 text-blue-600 text-xs">
          <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
          <span>Listening...</span>
        </div>
      )}
      
      {/* Voice Control Buttons */}
      <div className="flex items-center space-x-1">
        {!listening ? (
          <button
            onClick={startListening}
            disabled={disabled}
            className="w-8 h-8 bg-green-500 hover:bg-green-600 text-white rounded-lg flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105 active:scale-95"
            aria-label="Start voice input"
          >
            <Mic className="w-4 h-4" />
          </button>
        ) : (
          <button
            onClick={stopListening}
            className="w-8 h-8 bg-red-500 hover:bg-red-600 text-white rounded-lg flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 transition-all duration-200 hover:scale-105 active:scale-95"
            aria-label="Stop voice input"
          >
            <Square className="w-4 h-4" />
          </button>
        )}
        
        {transcript && (
          <button
            onClick={resetTranscript}
            disabled={disabled}
            className="w-8 h-8 bg-slate-500 hover:bg-slate-600 text-white rounded-lg flex items-center justify-center focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105 active:scale-95"
            aria-label="Clear voice input"
          >
            <MicOff className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
};

export default VoiceInput;