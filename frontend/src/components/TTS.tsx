'use client'
import { useState, useEffect } from "react";
import { Volume2, Play, Pause } from "lucide-react";

interface TTSProps {
  text: string;
  className?: string;
  compact?: boolean;
}

export default function TTS({ text, className = "", compact = false }: TTSProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isSupported, setIsSupported] = useState(false);

  useEffect(() => {
    setIsSupported("speechSynthesis" in window);
  }, []);

  const speak = () => {
    if (!isSupported) {
      alert("Sorry, your browser does not support text-to-speech.");
      return;
    }

    // Stop any current speech
    speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.rate = 0.9; // Slightly slower for better comprehension
    utterance.pitch = 1;
    utterance.volume = 0.8;

    utterance.onstart = () => {
      setIsPlaying(true);
    };

    utterance.onend = () => {
      setIsPlaying(false);
    };

    utterance.onerror = () => {
      setIsPlaying(false);
    };

    speechSynthesis.speak(utterance);
  };

  const stop = () => {
    speechSynthesis.cancel();
    setIsPlaying(false);
  };

  if (!isSupported) {
    return null;
  }

  if (compact) {
    return (
      <button
        onClick={isPlaying ? stop : speak}
        className={`w-6 h-6 rounded-full flex items-center justify-center transition-all duration-200 hover:scale-105 ${
          isPlaying 
            ? 'bg-red-500 hover:bg-red-600 text-white' 
            : 'bg-blue-500 hover:bg-blue-600 text-white'
        } ${className}`}
        aria-label={isPlaying ? "Stop speech" : "Play speech"}
      >
        {isPlaying ? (
          <Pause className="w-3 h-3" />
        ) : (
          <Volume2 className="w-3 h-3" />
        )}
      </button>
    );
  }

  return (
    <div className={`p-4 space-y-2 ${className}`}>
      <div className="flex items-center space-x-2">
        <button
          onClick={isPlaying ? stop : speak}
          className={`px-4 py-2 rounded-lg flex items-center space-x-2 transition-all duration-200 ${
            isPlaying
              ? 'bg-red-500 hover:bg-red-600 text-white'
              : 'bg-blue-500 hover:bg-blue-600 text-white'
          }`}
        >
          {isPlaying ? (
            <>
              <Pause className="w-4 h-4" />
              <span>Stop</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              <span>Speak</span>
            </>
          )}
        </button>
        {isPlaying && (
          <div className="flex items-center space-x-1 text-blue-600 text-sm">
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
            <span>Speaking...</span>
          </div>
        )}
      </div>
    </div>
  );
}
