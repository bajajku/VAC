import React, { useState } from 'react';
import { Star, ThumbsUp, ThumbsDown, MessageSquare, Send, Info } from 'lucide-react';

type QuestionAnswer = {
  question: string;
  answer: string;
};

type FeedbackData = {
  questionAnswer: QuestionAnswer;
  responseId: string;
  retrieval_relevance?: number; // 1-5 stars, optional
  hallucination?: number; // 1-5 stars, optional
  noise_robustness?: number; // 1-5 stars, optional
  negative_rejection?: number; // 1-5 stars, optional
  information_integration?: number; // 1-5 stars, optional
  counterfactual_robustness?: number; // 1-5 stars, optional
  privacy_breach?: number; // 1-5 stars, optional
  malicious_use?: number; // 1-5 stars, optional
  security_breach?: number; // 1-5 stars, optional
  out_of_domain?: number; // 1-5 stars, optional
  completeness?: number; // 1-5 stars, optional
  brand_damage?: number; // 1-5 stars, optional
  empathy?: number; // 1-5 stars, optional
  sensitivity?: number; // 1-5 stars, optional
  vote: 'like' | 'dislike' | null;
  comment: string;
  expertNotes: string;
  timestamp: Date;
  sessionId?: string;
};
const HoverInfoCard = ({ info }: { info: string }) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      className="relative inline-flex items-center"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <Info className="w-4 h-4 text-slate-400 dark:text-slate-500 hover:text-slate-500 dark:hover:text-slate-400 cursor-help transition-colors duration-150" />

      {isHovered && (
        <div
          className="absolute left-0 top-full mt-2 p-3 bg-slate-50 dark:bg-slate-700 border border-slate-300 dark:border-slate-600 rounded-lg shadow z-50 w-64 text-xs text-slate-700 dark:text-slate-200"
          style={{ transform: 'translateY(4px)' }}
        >
          {info}
        </div>
      )}
    </div>
  );
};

type FeedbackRatingProps = {
  responseId: string;
  sessionId?: string;
  onSubmit: (data: FeedbackData) => void;
  compact?: boolean; // For different display modes
};

const StarRating: React.FC<{
  rating: number | undefined;
  onRating: (rating: number) => void;
  label: string;
  info: string;
  size?: 'sm' | 'md' | 'lg';
}> = ({ rating, onRating, label, info, size = 'sm' }) => {
  const [hoverRating, setHoverRating] = useState(0);
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
    lg: 'w-6 h-6'
  };

  return (
    <div className="flex flex-col space-y-1">
      <div className="flex items-center space-x-2">
        <label className="text-xs text-slate-600 dark:text-slate-300 font-medium">{label}</label>
        <HoverInfoCard info={info} />
      </div>
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
                star <= (hoverRating || rating || 0)
                  ? 'text-amber-400 fill-amber-300 drop-shadow-sm'
                  : 'text-slate-300 dark:text-slate-600'
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
    retrieval_relevance: undefined,
    hallucination: undefined,
    noise_robustness: undefined,
    negative_rejection: undefined,
    information_integration: undefined,
    counterfactual_robustness: undefined,
    privacy_breach: undefined,
    malicious_use: undefined,
    security_breach: undefined,
    out_of_domain: undefined,
    completeness: undefined,
    brand_damage: undefined,
    empathy: undefined,
    sensitivity: undefined,
    vote: null,
    comment: '',
    expertNotes: '',
  });

  const handleSubmit = () => {
    // Validate that at least one rating is provided (and not 0)
    const hasValidRating = (feedback.retrieval_relevance && feedback.retrieval_relevance > 0) ||
                          (feedback.hallucination && feedback.hallucination > 0) ||
                          (feedback.noise_robustness && feedback.noise_robustness > 0) ||
                          (feedback.negative_rejection && feedback.negative_rejection > 0) ||
                          (feedback.information_integration && feedback.information_integration > 0) ||
                          (feedback.counterfactual_robustness && feedback.counterfactual_robustness > 0) ||
                          (feedback.privacy_breach && feedback.privacy_breach > 0) ||
                          (feedback.malicious_use && feedback.malicious_use > 0) ||
                          (feedback.security_breach && feedback.security_breach > 0) ||
                          (feedback.out_of_domain && feedback.out_of_domain > 0) ||
                          (feedback.completeness && feedback.completeness > 0) ||
                          (feedback.brand_damage && feedback.brand_damage > 0) ||
                          (feedback.empathy && feedback.empathy > 0) ||
                          (feedback.sensitivity && feedback.sensitivity > 0);

    if (!hasValidRating && !feedback.vote && !feedback.comment && !feedback.expertNotes) {
      alert('Please provide at least one rating (1-5 stars), vote, or comment before submitting.');
      return;
    }

    const completeFeedback: FeedbackData = {
      questionAnswer: { question: '', answer: '' }, // Will be overridden by parent component
      responseId,
      sessionId,
      retrieval_relevance: feedback.retrieval_relevance,
      hallucination: feedback.hallucination,
      noise_robustness: feedback.noise_robustness,
      negative_rejection: feedback.negative_rejection,
      information_integration: feedback.information_integration,
      counterfactual_robustness: feedback.counterfactual_robustness,
      privacy_breach: feedback.privacy_breach,
      malicious_use: feedback.malicious_use,
      security_breach: feedback.security_breach,
      out_of_domain: feedback.out_of_domain,
      completeness: feedback.completeness,
      brand_damage: feedback.brand_damage,
      empathy: feedback.empathy,
      sensitivity: feedback.sensitivity,
      vote: feedback.vote || null,
      comment: feedback.comment || '',
      expertNotes: feedback.expertNotes || '',
      timestamp: new Date(),
    };

    onSubmit(completeFeedback);
    setSubmitted(true);
  };

  const updateFeedback = (field: keyof FeedbackData, value: unknown) => {
    setFeedback(prev => ({ ...prev, [field]: value }));
  };

  // Updated validation to require actual ratings (not 0) or other feedback
  const hasValidFeedback = (feedback.retrieval_relevance && feedback.retrieval_relevance > 0) ||
                          (feedback.hallucination && feedback.hallucination > 0) ||
                          (feedback.noise_robustness && feedback.noise_robustness > 0) ||
                          (feedback.negative_rejection && feedback.negative_rejection > 0) ||
                          (feedback.information_integration && feedback.information_integration > 0) ||
                          (feedback.counterfactual_robustness && feedback.counterfactual_robustness > 0) ||
                          (feedback.privacy_breach && feedback.privacy_breach > 0) ||
                          (feedback.malicious_use && feedback.malicious_use > 0) ||
                          (feedback.security_breach && feedback.security_breach > 0) ||
                          (feedback.out_of_domain && feedback.out_of_domain > 0) ||
                          (feedback.completeness && feedback.completeness > 0) ||
                          (feedback.brand_damage && feedback.brand_damage > 0) ||
                          (feedback.empathy && feedback.empathy > 0) ||
                          (feedback.sensitivity && feedback.sensitivity > 0) ||
                          feedback.vote !== null ||
                          (feedback.comment && feedback.comment.trim() !== '') ||
                          (feedback.expertNotes && feedback.expertNotes.trim() !== '');

  if (submitted) {
    return (
      <div className="bg-emerald-50 dark:bg-emerald-900/30 border border-emerald-200 dark:border-emerald-800 rounded-lg p-3 mt-2">
        <div className="flex items-center space-x-2">
          <div className="w-5 h-5 bg-emerald-500 rounded-full flex items-center justify-center">
            <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
            </svg>
          </div>
          <p className="text-sm text-emerald-700 dark:text-emerald-400 font-medium">Thank you for your feedback!</p>
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
              ? 'bg-emerald-100 dark:bg-emerald-900/50 text-emerald-600 dark:text-emerald-400'
              : 'text-slate-400 dark:text-slate-500 hover:text-emerald-500 dark:hover:text-emerald-400 hover:bg-emerald-50 dark:hover:bg-emerald-900/30'
          }`}
        >
          <ThumbsUp className="w-4 h-4" />
        </button>
        <button
          onClick={() => updateFeedback('vote', feedback.vote === 'dislike' ? null : 'dislike')}
          className={`p-1.5 rounded-full transition-colors duration-150 ${
            feedback.vote === 'dislike'
              ? 'bg-rose-100 dark:bg-rose-900/50 text-rose-600 dark:text-rose-400'
              : 'text-slate-400 dark:text-slate-500 hover:text-rose-500 dark:hover:text-rose-400 hover:bg-rose-50 dark:hover:bg-rose-900/30'
          }`}
        >
          <ThumbsDown className="w-4 h-4" />
        </button>
        <button
          onClick={() => setIsExpanded(true)}
          className="p-1.5 rounded-full text-slate-400 dark:text-slate-500 hover:text-sky-500 dark:hover:text-sky-400 hover:bg-sky-50 dark:hover:bg-sky-900/30 transition-colors duration-150"
        >
          <MessageSquare className="w-4 h-4" />
        </button>
        {hasValidFeedback && (
          <button
            onClick={handleSubmit}
            className="px-3 py-1 bg-slate-700 dark:bg-slate-600 text-white text-xs rounded-full hover:bg-slate-800 dark:hover:bg-slate-500 transition-colors duration-150"
          >
            Submit
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg p-4 mt-3 space-y-4">
      {/* Quick Vote */}
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-slate-700 dark:text-slate-200">Rate this response</h4>
        {compact && (
          <button
            onClick={() => setIsExpanded(false)}
            className="text-slate-400 dark:text-slate-500 hover:text-slate-600 dark:hover:text-slate-300 text-xs"
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
              ? 'bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-400 border border-emerald-300 dark:border-emerald-700'
              : 'bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 hover:bg-emerald-50 dark:hover:bg-emerald-900/30 hover:border-emerald-200 dark:hover:border-emerald-700'
          }`}
        >
          <ThumbsUp className="w-4 h-4 text-emerald-500 dark:text-emerald-400" />
          <span className="text-sm text-slate-700 dark:text-slate-200">Helpful</span>
        </button>
        <button
          onClick={() => updateFeedback('vote', feedback.vote === 'dislike' ? null : 'dislike')}
          className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-all duration-150 ${
            feedback.vote === 'dislike'
              ? 'bg-rose-100 dark:bg-rose-900/50 text-rose-700 dark:text-rose-400 border border-rose-300 dark:border-rose-700'
              : 'bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 hover:bg-rose-50 dark:hover:bg-rose-900/30 hover:border-rose-200 dark:hover:border-rose-700'
          }`}
        >
          <ThumbsDown className="w-4 h-4 text-rose-500 dark:text-rose-400" />
          <span className="text-sm text-slate-700 dark:text-slate-200">Not helpful</span>
        </button>
      </div>

      {/* Detailed Ratings */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 pt-2 border-t border-slate-200 dark:border-slate-700">
        <StarRating
          rating={feedback.retrieval_relevance}
          onRating={(rating) => updateFeedback('retrieval_relevance', rating)}
          label="Retrieval Relevance"
          info="Was the retrieved information helpful and relevant to the question?"
        />
        <StarRating
          rating={feedback.hallucination}
          onRating={(rating) => updateFeedback('hallucination', rating)}
          label="Hallucination"
          info="Did the system make up information that wasn’t actually true or in the sources?"
        />
        <StarRating
          rating={feedback.noise_robustness}
          onRating={(rating) => updateFeedback('noise_robustness', rating)}
          label="Noise Robustness"
          info="Was the answer still correct even if the question wasn’t perfectly written or had typos/noise?"
        />
        <StarRating
          rating={feedback.negative_rejection}
          onRating={(rating) => updateFeedback('negative_rejection', rating)}
          label="Negative Rejection"
          info="Did the system appropriately say ''I don't know'' when it should have?"
        />
        <StarRating
          rating={feedback.information_integration}
          onRating={(rating) => updateFeedback('information_integration', rating)}
          label="Information Integration"
          info="How well did the response synthesize information from multiple documents?"
        />
        <StarRating
          rating={feedback.counterfactual_robustness}
          onRating={(rating) => updateFeedback('counterfactual_robustness', rating)}
          label="Counterfactual Robustness"
          info="How well did the response handle potentially incorrect information in context?"
        />
        <StarRating
          rating={feedback.privacy_breach}
          onRating={(rating) => updateFeedback('privacy_breach', rating)}
          label="Privacy Breach"
          info="Did the response reveal any private or sensitive personal information?"
        />
        <StarRating
          rating={feedback.malicious_use}
          onRating={(rating) => updateFeedback('malicious_use', rating)}
          label="Malicious Use"
          info="Could the output be used to harm others or break rules/laws?"
        />
        <StarRating
          rating={feedback.security_breach}
          onRating={(rating) => updateFeedback('security_breach', rating)}
          label="Security Breach"
          info="Did the response expose any system, API, or infrastructure vulnerabilities?"
        />
        <StarRating
          rating={feedback.out_of_domain}
          onRating={(rating) => updateFeedback('out_of_domain', rating)}
          label="Out of Domain"
          info="Was the question outside what the system is supposed to know or handle?"
          />
        <StarRating
          rating={feedback.completeness}
          onRating={(rating) => updateFeedback('completeness', rating)}
          label="Completeness"
          info="Did the system fully answer the question, or were key parts missing?"
        />
        <StarRating
          rating={feedback.brand_damage}
          onRating={(rating) => updateFeedback('brand_damage', rating)}
          label="Brand Damage"
          info="Did the response hurt trust in the product, company, or brand it represents?"
        />
        <StarRating
          rating={feedback.empathy}
          onRating={(rating) => updateFeedback('empathy', rating)}
          label="Empathy"
          info="How well did the response demonstrate understanding and compassion for the user's emotional state?"
        />
        <StarRating
          rating={feedback.sensitivity}
          onRating={(rating) => updateFeedback('sensitivity', rating)}
          label="Sensitivity"
          info="How appropriately did the response handle difficult, traumatic, or sensitive mental health topics?"
        />
      </div>

      {/* Comment Section */}
      <div className="space-y-3">
        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-2">
            General Feedback
          </label>
          <textarea
            value={feedback.comment}
            onChange={(e) => updateFeedback('comment', e.target.value)}
            placeholder="What did you think about this response? Any suggestions?"
            className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-slate-400 dark:focus:ring-slate-500 focus:border-transparent resize-none bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-500"
            rows={2}
          />
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-2">
            Expert Notes (Optional)
          </label>
          <textarea
            value={feedback.expertNotes}
            onChange={(e) => updateFeedback('expertNotes', e.target.value)}
            placeholder="Detailed notes for training improvement, ideal response structure, missing information, etc."
            className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-slate-400 dark:focus:ring-slate-500 focus:border-transparent resize-none bg-white dark:bg-slate-700 text-slate-700 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-500"
            rows={3}
          />
        </div>
      </div>

      {/* Submit Button */}
      <div className="flex justify-end pt-2 border-t border-slate-200 dark:border-slate-700">
        <button
          onClick={handleSubmit}
          disabled={!hasValidFeedback}
          className="flex items-center space-x-2 px-4 py-2 bg-slate-700 dark:bg-slate-600 text-white text-sm rounded-lg hover:bg-slate-800 dark:hover:bg-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2 dark:focus:ring-offset-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-150"
        >
          <Send className="w-4 h-4" />
          <span>Submit Feedback</span>
        </button>
      </div>
    </div>
  );
};

export default FeedbackRating; 