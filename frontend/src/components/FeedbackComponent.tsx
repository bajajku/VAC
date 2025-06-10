'use client';

import React, { useState } from 'react';
import { feedbackService, FeedbackCreate } from '../services/feedbackService';

interface FeedbackComponentProps {
  sessionId: string;
  question: string;
  answer: string;
  onFeedbackSubmitted?: (success: boolean) => void;
  className?: string;
}

const FeedbackComponent: React.FC<FeedbackComponentProps> = ({
  sessionId,
  question,
  answer,
  onFeedbackSubmitted,
  className = ''
}) => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [feedback, setFeedback] = useState<{
    type: 'positive' | 'negative' | 'suggestion' | null;
    rating?: number;
    comment?: string;
  }>({ type: null });
  const [showDetailedForm, setShowDetailedForm] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleQuickFeedback = async (type: 'positive' | 'negative') => {
    setIsSubmitting(true);
    
    try {
      const result = await feedbackService.createFeedback({
        session_id: sessionId,
        question,
        answer,
        feedback_type: type,
      });

      if (result.success) {
        setSubmitted(true);
        onFeedbackSubmitted?.(true);
      } else {
        console.error('Failed to submit feedback:', result.error);
        onFeedbackSubmitted?.(false);
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      onFeedbackSubmitted?.(false);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDetailedFeedback = async () => {
    if (!feedback.type) return;
    
    setIsSubmitting(true);
    
    try {
      const result = await feedbackService.createFeedback({
        session_id: sessionId,
        question,
        answer,
        feedback_type: feedback.type,
        feedback_text: feedback.comment,
        rating: feedback.rating,
      });

      if (result.success) {
        setSubmitted(true);
        onFeedbackSubmitted?.(true);
      } else {
        console.error('Failed to submit feedback:', result.error);
        onFeedbackSubmitted?.(false);
      }
    } catch (error) {
      console.error('Error submitting feedback:', error);
      onFeedbackSubmitted?.(false);
    } finally {
      setIsSubmitting(false);
    }
  };

  if (submitted) {
    return (
      <div className={`flex items-center text-sm text-green-600 ${className}`}>
        <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
        Thank you for your feedback!
      </div>
    );
  }

  if (showDetailedForm) {
    return (
      <div className={`bg-gray-50 p-4 rounded-lg ${className}`}>
        <h4 className="text-sm font-medium text-gray-900 mb-3">Provide Detailed Feedback</h4>
        
        {/* Feedback Type */}
        <div className="mb-3">
          <label className="block text-xs font-medium text-gray-700 mb-1">Type</label>
          <div className="flex space-x-2">
            {(['positive', 'negative', 'suggestion'] as const).map((type) => (
              <button
                key={type}
                onClick={() => setFeedback({ ...feedback, type })}
                className={`px-3 py-1 text-xs rounded-full capitalize ${
                  feedback.type === type
                    ? 'bg-blue-100 text-blue-800 border border-blue-300'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {type}
              </button>
            ))}
          </div>
        </div>

        {/* Rating */}
        <div className="mb-3">
          <label className="block text-xs font-medium text-gray-700 mb-1">Rating (optional)</label>
          <div className="flex space-x-1">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                onClick={() => setFeedback({ ...feedback, rating: star })}
                className={`text-lg ${
                  feedback.rating && star <= feedback.rating
                    ? 'text-yellow-400'
                    : 'text-gray-300 hover:text-yellow-300'
                }`}
              >
                ★
              </button>
            ))}
          </div>
        </div>

        {/* Comment */}
        <div className="mb-3">
          <label className="block text-xs font-medium text-gray-700 mb-1">Comment (optional)</label>
          <textarea
            value={feedback.comment || ''}
            onChange={(e) => setFeedback({ ...feedback, comment: e.target.value })}
            placeholder="Share your thoughts about this response..."
            className="w-full px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            rows={3}
          />
        </div>

        {/* Actions */}
        <div className="flex space-x-2">
          <button
            onClick={handleDetailedFeedback}
            disabled={!feedback.type || isSubmitting}
            className="px-4 py-2 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
          </button>
          <button
            onClick={() => setShowDetailedForm(false)}
            className="px-4 py-2 bg-gray-200 text-gray-700 text-sm rounded-md hover:bg-gray-300"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <span className="text-xs text-gray-600">Was this helpful?</span>
      
      {/* Quick feedback buttons */}
      <button
        onClick={() => handleQuickFeedback('positive')}
        disabled={isSubmitting}
        className="flex items-center space-x-1 px-2 py-1 text-xs text-green-700 bg-green-50 hover:bg-green-100 rounded-md disabled:opacity-50"
      >
        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
          <path d="M2 10.5a1.5 1.5 0 113 0v6a1.5 1.5 0 01-3 0v-6zM6 10.333v5.43a2 2 0 001.106 1.79l.05.025A4 4 0 008.943 18h5.416a2 2 0 001.962-1.608l1.2-6A2 2 0 0015.56 8H12V4a2 2 0 00-2-2 1 1 0 00-1 1v.667a4 4 0 01-.8 2.4L6.8 7.933a4 4 0 00-.8 2.4z" />
        </svg>
        <span>Yes</span>
      </button>
      
      <button
        onClick={() => handleQuickFeedback('negative')}
        disabled={isSubmitting}
        className="flex items-center space-x-1 px-2 py-1 text-xs text-red-700 bg-red-50 hover:bg-red-100 rounded-md disabled:opacity-50"
      >
        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20" style={{ transform: 'rotate(180deg)' }}>
          <path d="M2 10.5a1.5 1.5 0 113 0v6a1.5 1.5 0 01-3 0v-6zM6 10.333v5.43a2 2 0 001.106 1.79l.05.025A4 4 0 008.943 18h5.416a2 2 0 001.962-1.608l1.2-6A2 2 0 0015.56 8H12V4a2 2 0 00-2-2 1 1 0 00-1 1v.667a4 4 0 01-.8 2.4L6.8 7.933a4 4 0 00-.8 2.4z" />
        </svg>
        <span>No</span>
      </button>
      
      <button
        onClick={() => setShowDetailedForm(true)}
        className="text-xs text-blue-600 hover:text-blue-800 underline"
      >
        More feedback
      </button>
    </div>
  );
};

export default FeedbackComponent; 