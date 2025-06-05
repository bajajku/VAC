import React, { useState } from 'react';
import { Star, ThumbsUp, ThumbsDown, MessageSquare, Send } from 'lucide-react';

type QuestionAnswer = {
  question: string;
  answer: string;
};

type FeedbackData = {
  questionAnswer: QuestionAnswer;
  responseId: string;
  overallRating: number; // 1-5 stars
  accuracy: number; // 1-5 stars
  helpfulness: number; // 1-5 stars
  clarity: number; // 1-5 stars
  vote: 'like' | 'dislike' | null;
  comment: string;
  expertNotes: string;
  timestamp: Date;
  sessionId?: string;
};

type FeedbackRatingProps = {
  responseId: string;
  sessionId?: string;
  onSubmit: (data: FeedbackData) => void;
  compact?: boolean; // For different display modes
};

const StarRating: React.FC<{
  rating: number;
  onRating: (rating: number) => void;
  label: string;
  size?: 'sm' | 'md' | 'lg';
}> = ({ rating, onRating, label, size = 'sm' }) => {
  const [hoverRating, setHoverRating] = useState(0);
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6'
  };

  return (
    <div className="flex flex-col space-y-1">
      <label className="text-xs text-slate-600 font-medium">{label}</label>
      <div className="flex space-x-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            type="button"
            className={`${sizeClasses[size]} transition-colors duration-150 focus:outline-none`}
            onMouseEnter={() => setHoverRating(star)}
            onMouseLeave={() => setHoverRating(0)}
            onClick={() => onRating(star)}
          >
            <Star
              className={`w-full h-full ${
                star <= (hoverRating || rating)
                  ? 'text-yellow-400 fill-yellow-400'
                  : 'text-slate-300'
              }`}
            />
          </button>
        ))}
      </div>
    </div>
  );
};

const FeedbackRating: React.FC<FeedbackRatingProps> = ({ 
  responseId, 
  sessionId, 
  onSubmit, 
  compact = false 
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [feedback, setFeedback] = useState<Partial<FeedbackData>>({
    responseId,
    sessionId,
    overallRating: 0,
    accuracy: 0,
    helpfulness: 0,
    clarity: 0,
    vote: null,
    comment: '',
    expertNotes: '',
  });

  const handleSubmit = () => {
    const completeFeedback: FeedbackData = {
      questionAnswer: { question: '', answer: '' }, // Will be overridden by parent component
      responseId,
      sessionId,
      overallRating: feedback.overallRating || 0,
      accuracy: feedback.accuracy || 0,
      helpfulness: feedback.helpfulness || 0,
      clarity: feedback.clarity || 0,
      vote: feedback.vote || null,
      comment: feedback.comment || '',
      expertNotes: feedback.expertNotes || '',
      timestamp: new Date(),
    };

    onSubmit(completeFeedback);
    setSubmitted(true);
  };

  const updateFeedback = (field: keyof FeedbackData, value: any) => {
    setFeedback(prev => ({ ...prev, [field]: value }));
  };

  const hasAnyRating = feedback.overallRating! > 0 || feedback.vote !== null || 
                      feedback.comment !== '' || feedback.expertNotes !== '';

  if (submitted) {
    return (
      <div className="bg-green-50 border border-green-200 rounded-lg p-3 mt-2">
        <div className="flex items-center space-x-2">
          <div className="w-5 h-5 bg-green-500 rounded-full flex items-center justify-center">
            <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
          </div>
          <p className="text-sm text-green-700 font-medium">Thank you for your feedback!</p>
        </div>
      </div>
    );
  }

  if (compact && !isExpanded) {
    return (
      <div className="flex items-center space-x-2 mt-2 py-1">
        <button
          onClick={() => updateFeedback('vote', feedback.vote === 'like' ? null : 'like')}
          className={`p-1.5 rounded-full transition-colors duration-150 ${
            feedback.vote === 'like' 
              ? 'bg-green-100 text-green-600' 
              : 'text-slate-400 hover:text-green-500 hover:bg-green-50'
          }`}
        >
          <ThumbsUp className="w-4 h-4" />
        </button>
        <button
          onClick={() => updateFeedback('vote', feedback.vote === 'dislike' ? null : 'dislike')}
          className={`p-1.5 rounded-full transition-colors duration-150 ${
            feedback.vote === 'dislike' 
              ? 'bg-red-100 text-red-600' 
              : 'text-slate-400 hover:text-red-500 hover:bg-red-50'
          }`}
        >
          <ThumbsDown className="w-4 h-4" />
        </button>
        <button
          onClick={() => setIsExpanded(true)}
          className="p-1.5 rounded-full text-slate-400 hover:text-blue-500 hover:bg-blue-50 transition-colors duration-150"
        >
          <MessageSquare className="w-4 h-4" />
        </button>
        {hasAnyRating && (
          <button
            onClick={handleSubmit}
            className="px-3 py-1 bg-blue-500 text-white text-xs rounded-full hover:bg-blue-600 transition-colors duration-150"
          >
            Submit
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 mt-3 space-y-4">
      {/* Quick Vote */}
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-slate-700">Rate this response</h4>
        {compact && (
          <button
            onClick={() => setIsExpanded(false)}
            className="text-slate-400 hover:text-slate-600 text-xs"
          >
            Collapse
          </button>
        )}
      </div>

      <div className="flex items-center space-x-4">
        <button
          onClick={() => updateFeedback('vote', feedback.vote === 'like' ? null : 'like')}
          className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-all duration-150 ${
            feedback.vote === 'like' 
              ? 'bg-green-100 text-green-700 border border-green-300' 
              : 'bg-white border border-slate-200 hover:bg-green-50 hover:border-green-200'
          }`}
        >
          <ThumbsUp className="w-4 h-4" />
          <span className="text-sm">Helpful</span>
        </button>
        <button
          onClick={() => updateFeedback('vote', feedback.vote === 'dislike' ? null : 'dislike')}
          className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-all duration-150 ${
            feedback.vote === 'dislike' 
              ? 'bg-red-100 text-red-700 border border-red-300' 
              : 'bg-white border border-slate-200 hover:bg-red-50 hover:border-red-200'
          }`}
        >
          <ThumbsDown className="w-4 h-4" />
          <span className="text-sm">Not helpful</span>
        </button>
      </div>

      {/* Detailed Ratings */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 pt-2 border-t border-slate-200">
        <StarRating
          rating={feedback.overallRating!}
          onRating={(rating) => updateFeedback('overallRating', rating)}
          label="Overall"
        />
        <StarRating
          rating={feedback.accuracy!}
          onRating={(rating) => updateFeedback('accuracy', rating)}
          label="Accuracy"
        />
        <StarRating
          rating={feedback.helpfulness!}
          onRating={(rating) => updateFeedback('helpfulness', rating)}
          label="Helpfulness"
        />
        <StarRating
          rating={feedback.clarity!}
          onRating={(rating) => updateFeedback('clarity', rating)}
          label="Clarity"
        />
      </div>

      {/* Comment Section */}
      <div className="space-y-3">
        <div>
          <label className="block text-xs font-medium text-slate-600 mb-2">
            General Feedback
          </label>
          <textarea
            value={feedback.comment}
            onChange={(e) => updateFeedback('comment', e.target.value)}
            placeholder="What did you think about this response? Any suggestions?"
            className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            rows={2}
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-600 mb-2">
            Expert Notes (Optional)
          </label>
          <textarea
            value={feedback.expertNotes}
            onChange={(e) => updateFeedback('expertNotes', e.target.value)}
            placeholder="Detailed notes for training improvement, ideal response structure, missing information, etc."
            className="w-full px-3 py-2 border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            rows={3}
          />
        </div>
      </div>

      {/* Submit Button */}
      <div className="flex justify-end pt-2 border-t border-slate-200">
        <button
          onClick={handleSubmit}
          disabled={!hasAnyRating}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-500 text-white text-sm rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-150"
        >
          <Send className="w-4 h-4" />
          <span>Submit Feedback</span>
        </button>
      </div>
    </div>
  );
};

export default FeedbackRating; 