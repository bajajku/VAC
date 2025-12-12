"use client"
import React, { useState, useEffect, useCallback } from 'react';
import { Wind, Pause, Play, X, Eye, Hand, Ear, Heart, Coffee } from 'lucide-react';

interface GroundingToolsProps {
  onClose?: () => void;
  variant?: 'modal' | 'inline' | 'floating';
}

type GroundingTechnique = 'breathing' | '54321' | null;

const GroundingTools: React.FC<GroundingToolsProps> = ({
  onClose,
  variant = 'modal'
}) => {
  const [activeTechnique, setActiveTechnique] = useState<GroundingTechnique>(null);
  const [breathPhase, setBreathPhase] = useState<'inhale' | 'hold' | 'exhale' | 'rest'>('inhale');
  const [breathCount, setBreathCount] = useState(0);
  const [isBreathing, setIsBreathing] = useState(false);
  const [timer, setTimer] = useState(4);
  const [groundingStep, setGroundingStep] = useState(0);

  const breathingCycle = useCallback(() => {
    const phases: Array<{ phase: 'inhale' | 'hold' | 'exhale' | 'rest'; duration: number }> = [
      { phase: 'inhale', duration: 4 },
      { phase: 'hold', duration: 4 },
      { phase: 'exhale', duration: 4 },
      { phase: 'rest', duration: 2 }
    ];

    let currentPhaseIndex = 0;
    let countdown = phases[0].duration;

    const interval = setInterval(() => {
      countdown--;
      setTimer(countdown);

      if (countdown <= 0) {
        currentPhaseIndex = (currentPhaseIndex + 1) % phases.length;
        const nextPhase = phases[currentPhaseIndex];
        setBreathPhase(nextPhase.phase);
        countdown = nextPhase.duration;
        setTimer(countdown);

        if (currentPhaseIndex === 0) {
          setBreathCount(prev => prev + 1);
        }
      }
    }, 1000);

    return interval;
  }, []);

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;

    if (isBreathing && activeTechnique === 'breathing') {
      interval = breathingCycle();
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isBreathing, activeTechnique, breathingCycle]);

  const startBreathing = () => {
    setActiveTechnique('breathing');
    setIsBreathing(true);
    setBreathPhase('inhale');
    setBreathCount(0);
    setTimer(4);
  };

  const stopBreathing = () => {
    setIsBreathing(false);
  };

  const groundingSteps = [
    { icon: Eye, sense: "SEE", prompt: "Name 5 things you can see around you", color: "blue" },
    { icon: Hand, sense: "TOUCH", prompt: "Name 4 things you can physically feel", color: "green" },
    { icon: Ear, sense: "HEAR", prompt: "Name 3 things you can hear right now", color: "purple" },
    { icon: Coffee, sense: "SMELL", prompt: "Name 2 things you can smell", color: "orange" },
    { icon: Heart, sense: "TASTE", prompt: "Name 1 thing you can taste", color: "red" }
  ];

  const getBreathingMessage = () => {
    switch (breathPhase) {
      case 'inhale': return 'Breathe in slowly...';
      case 'hold': return 'Hold gently...';
      case 'exhale': return 'Breathe out slowly...';
      case 'rest': return 'Rest...';
    }
  };

  const getBreathingCircleSize = () => {
    switch (breathPhase) {
      case 'inhale': return 'scale-110';
      case 'hold': return 'scale-110';
      case 'exhale': return 'scale-90';
      case 'rest': return 'scale-90';
    }
  };

  const content = (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-teal-100 p-2 rounded-full">
            <Wind className="w-5 h-5 text-teal-600" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Take a Moment</h2>
            <p className="text-sm text-gray-600">Grounding techniques to help you feel centered</p>
          </div>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
            aria-label="Close grounding tools"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        )}
      </div>

      {/* Technique Selection */}
      {!activeTechnique && (
        <div className="grid gap-4 md:grid-cols-2">
          {/* Breathing Exercise Card */}
          <button
            onClick={startBreathing}
            className="p-6 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl border border-blue-200 hover:border-blue-300 hover:shadow-md transition-all text-left group"
          >
            <div className="bg-blue-100 w-12 h-12 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <Wind className="w-6 h-6 text-blue-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">4-4-4 Breathing</h3>
            <p className="text-sm text-gray-600">
              A calming breathing exercise to help regulate your nervous system
            </p>
          </button>

          {/* 5-4-3-2-1 Technique Card */}
          <button
            onClick={() => {
              setActiveTechnique('54321');
              setGroundingStep(0);
            }}
            className="p-6 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border border-purple-200 hover:border-purple-300 hover:shadow-md transition-all text-left group"
          >
            <div className="bg-purple-100 w-12 h-12 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <Eye className="w-6 h-6 text-purple-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">5-4-3-2-1 Grounding</h3>
            <p className="text-sm text-gray-600">
              Use your senses to anchor yourself in the present moment
            </p>
          </button>
        </div>
      )}

      {/* Breathing Exercise */}
      {activeTechnique === 'breathing' && (
        <div className="flex flex-col items-center py-8">
          {/* Breathing Circle */}
          <div className={`w-48 h-48 rounded-full bg-gradient-to-br from-blue-400 to-cyan-400 flex items-center justify-center transition-transform duration-1000 ease-in-out ${getBreathingCircleSize()}`}>
            <div className="w-40 h-40 rounded-full bg-white flex flex-col items-center justify-center">
              <span className="text-4xl font-light text-blue-600">{timer}</span>
              <span className="text-sm text-blue-500 mt-1">{breathPhase}</span>
            </div>
          </div>

          {/* Instructions */}
          <p className="text-lg text-gray-700 mt-6 text-center font-medium">
            {getBreathingMessage()}
          </p>

          {/* Breath Counter */}
          <p className="text-sm text-gray-500 mt-2">
            Completed breaths: {breathCount}
          </p>

          {/* Controls */}
          <div className="flex gap-4 mt-6">
            <button
              onClick={() => isBreathing ? stopBreathing() : startBreathing()}
              className={`flex items-center gap-2 px-6 py-3 rounded-full font-medium transition-colors ${
                isBreathing
                  ? 'bg-gray-200 hover:bg-gray-300 text-gray-700'
                  : 'bg-blue-500 hover:bg-blue-600 text-white'
              }`}
            >
              {isBreathing ? (
                <><Pause className="w-5 h-5" /> Pause</>
              ) : (
                <><Play className="w-5 h-5" /> Resume</>
              )}
            </button>
            <button
              onClick={() => {
                stopBreathing();
                setActiveTechnique(null);
              }}
              className="px-6 py-3 rounded-full font-medium bg-gray-100 hover:bg-gray-200 text-gray-700 transition-colors"
            >
              Done
            </button>
          </div>

          {/* Tips */}
          <div className="mt-8 p-4 bg-blue-50 rounded-lg max-w-md">
            <p className="text-sm text-blue-800 text-center">
              <strong>Tip:</strong> Place one hand on your chest and one on your belly.
              Try to breathe so that only your belly hand moves.
            </p>
          </div>
        </div>
      )}

      {/* 5-4-3-2-1 Grounding */}
      {activeTechnique === '54321' && (
        <div className="space-y-6">
          {/* Progress Indicator */}
          <div className="flex justify-center gap-2">
            {groundingSteps.map((_, index) => (
              <div
                key={index}
                className={`w-3 h-3 rounded-full transition-colors ${
                  index < groundingStep
                    ? 'bg-green-500'
                    : index === groundingStep
                    ? 'bg-purple-500'
                    : 'bg-gray-200'
                }`}
              />
            ))}
          </div>

          {/* Current Step */}
          {groundingStep < 5 ? (
            <div className="text-center py-8">
              <div className={`mx-auto w-20 h-20 rounded-full flex items-center justify-center mb-6 bg-${groundingSteps[groundingStep].color}-100`}>
                {React.createElement(groundingSteps[groundingStep].icon, {
                  className: `w-10 h-10 text-${groundingSteps[groundingStep].color}-600`
                })}
              </div>
              <div className="text-6xl font-bold text-purple-600 mb-4">
                {5 - groundingStep}
              </div>
              <p className="text-xl text-gray-800 font-medium mb-2">
                {groundingSteps[groundingStep].sense}
              </p>
              <p className="text-gray-600 mb-8">
                {groundingSteps[groundingStep].prompt}
              </p>
              <p className="text-sm text-gray-500 mb-6">
                Take your time. There&apos;s no rush.
              </p>

              {/* Navigation */}
              <div className="flex justify-center gap-4">
                {groundingStep > 0 && (
                  <button
                    onClick={() => setGroundingStep(prev => prev - 1)}
                    className="px-6 py-3 rounded-full font-medium bg-gray-100 hover:bg-gray-200 text-gray-700 transition-colors"
                  >
                    Previous
                  </button>
                )}
                <button
                  onClick={() => setGroundingStep(prev => prev + 1)}
                  className="px-8 py-3 rounded-full font-medium bg-purple-500 hover:bg-purple-600 text-white transition-colors"
                >
                  {groundingStep === 4 ? 'Complete' : 'Next'}
                </button>
              </div>
            </div>
          ) : (
            /* Completion */
            <div className="text-center py-8">
              <div className="mx-auto w-20 h-20 rounded-full bg-green-100 flex items-center justify-center mb-6">
                <Heart className="w-10 h-10 text-green-600" />
              </div>
              <h3 className="text-2xl font-semibold text-gray-900 mb-4">
                Well done
              </h3>
              <p className="text-gray-600 mb-8 max-w-md mx-auto">
                You&apos;ve completed the grounding exercise. Take a moment to notice how you feel.
                It&apos;s okay if you need to do this again.
              </p>
              <div className="flex justify-center gap-4">
                <button
                  onClick={() => setGroundingStep(0)}
                  className="px-6 py-3 rounded-full font-medium bg-purple-100 hover:bg-purple-200 text-purple-700 transition-colors"
                >
                  Do Again
                </button>
                <button
                  onClick={() => setActiveTechnique(null)}
                  className="px-6 py-3 rounded-full font-medium bg-green-500 hover:bg-green-600 text-white transition-colors"
                >
                  I&apos;m Ready to Continue
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Back to Selection */}
      {activeTechnique && !isBreathing && activeTechnique === 'breathing' && (
        <button
          onClick={() => setActiveTechnique(null)}
          className="w-full py-3 text-gray-600 hover:text-gray-800 text-sm font-medium transition-colors"
        >
          ← Try a different technique
        </button>
      )}
    </div>
  );

  if (variant === 'floating') {
    return (
      <div className="fixed bottom-24 right-4 w-96 max-w-[calc(100vw-2rem)] bg-white rounded-2xl shadow-2xl border border-gray-200 p-6 z-50">
        {content}
      </div>
    );
  }

  if (variant === 'inline') {
    return (
      <div className="bg-white rounded-xl border border-gray-200 p-6">
        {content}
      </div>
    );
  }

  // Modal variant
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full max-h-[90vh] overflow-y-auto p-6">
        {content}
      </div>
    </div>
  );
};

export default GroundingTools;
