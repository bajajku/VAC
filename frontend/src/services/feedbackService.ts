import Cookies from 'js-cookie';
import { refreshTokenIfNeeded } from '../utils/refreshToken';

// Types for the new MongoDB-based feedback system
type FeedbackCreate = {
  session_id: string;
  question: string;
  answer: string;
  feedback_type: 'positive' | 'negative' | 'suggestion';
  feedback_text?: string;
  rating?: number; // 1-5 scale
  user_id?: string;
  
  // Detailed feedback categories (1-5 star ratings)
  retrieval_relevance?: number;
  hallucination?: number;
  noise_robustness?: number;
  negative_rejection?: number;
  privacy_breach?: number;
  malicious_use?: number;
  security_breach?: number;
  out_of_domain?: number;
  completeness?: number;
  brand_damage?: number;
  
  // Additional feedback fields
  vote?: string;
  comment?: string;
  expert_notes?: string;
};

type FeedbackResponse = {
  id: string;
  session_id: string;
  question: string;
  answer: string;
  feedback_type: 'positive' | 'negative' | 'suggestion';
  feedback_text?: string;
  rating?: number;
  user_id?: string;
  
  // Detailed feedback categories
  retrieval_relevance?: number;
  hallucination?: number;
  noise_robustness?: number;
  negative_rejection?: number;
  privacy_breach?: number;
  malicious_use?: number;
  security_breach?: number;
  out_of_domain?: number;
  completeness?: number;
  brand_damage?: number;
  
  vote?: string;
  comment?: string;
  expert_notes?: string;
  
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};

type FeedbackStats = {
  total_feedback: number;
  positive_count: number;
  negative_count: number;
  suggestion_count: number;
  average_rating?: number;
  recent_feedback: FeedbackResponse[];
};

// Legacy types for backward compatibility
type QuestionAnswer = {
  question: string;
  answer: string;
}

type LegacyFeedbackData = {
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
  privacy_breach?: number;
  malicious_use?: number;
  security_breach?: number;
  out_of_domain?: number;
  completeness?: number;
  brand_damage?: number;
  
  vote: 'like' | 'dislike' | null;
  comment: string;
  expertNotes: string;
  sessionId?: string;
};

type FeedbackStorageItem = LegacyFeedbackData & {
  id: string;
};

class FeedbackService {
  private baseUrl: string;
  private localStorageKey = 'chatbot_feedback';

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  }

<<<<<<< HEAD
  private async getHeaders(): Promise<HeadersInit> {
    // Try to refresh token if needed
    const isValid = await refreshTokenIfNeeded();
    if (!isValid) {
      // Token refresh failed, redirect to login
      window.location.href = '/auth/login';
      throw new Error('Authentication failed');
    }

    const token = Cookies.get('token');
    return {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
=======
  private getHeaders(): HeadersInit {
    return {
      'Content-Type': 'application/json',
>>>>>>> 85f5f79 (feat: Add ngrok header to API requests across authentication and feedback services)
      'ngrok-skip-browser-warning': 'true'
    };
  }

  // New MongoDB-based feedback methods
  async createFeedback(feedback: FeedbackCreate): Promise<{ success: boolean; data?: FeedbackResponse; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/feedback`, {
        method: 'POST',
<<<<<<< HEAD
        headers: await this.getHeaders(),
=======
        headers: this.getHeaders(),
>>>>>>> 85f5f79 (feat: Add ngrok header to API requests across authentication and feedback services)
        body: JSON.stringify(feedback),
      });

      if (response.ok) {
        const data = await response.json();
        return { success: true, data };
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Error creating feedback:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async getFeedbackById(feedbackId: string): Promise<{ success: boolean; data?: FeedbackResponse; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/feedback/${feedbackId}`, {
        headers: { 'ngrok-skip-browser-warning': 'true' }
      });

      if (response.ok) {
        const data = await response.json();
        return { success: true, data };
      } else if (response.status === 404) {
        return { success: false, error: 'Feedback not found' };
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Error fetching feedback:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async getSessionFeedback(sessionId: string): Promise<{ success: boolean; data?: FeedbackResponse[]; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/feedback/session/${sessionId}`, {
        headers: { 'ngrok-skip-browser-warning': 'true' }
      });

      if (response.ok) {
        const data = await response.json();
        return { success: true, data };
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Error fetching session feedback:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async getFeedbackStats(): Promise<{ success: boolean; data?: FeedbackStats; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/feedback-stats`, {
        headers: { 'ngrok-skip-browser-warning': 'true' }
      });

      if (response.ok) {
        const data = await response.json();
        return { success: true, data };
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Error fetching feedback stats:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  // Utility method to submit simple feedback with just rating and type
  async submitSimpleFeedback(
    sessionId: string,
    question: string,
    answer: string,
    isPositive: boolean,
    rating?: number,
    comment?: string
  ): Promise<{ success: boolean; data?: FeedbackResponse; error?: string }> {
    return this.createFeedback({
      session_id: sessionId,
      question,
      answer,
      feedback_type: isPositive ? 'positive' : 'negative',
      feedback_text: comment,
      rating,
    });
  }

  // Legacy methods for backward compatibility
  private storeLocally(feedback: LegacyFeedbackData): string {
    const id = `feedback_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const feedbackWithId: FeedbackStorageItem = { ...feedback, id };
    
    const existingFeedback = this.getLocalFeedback();
    const updatedFeedback = [...existingFeedback, feedbackWithId];
    
    localStorage.setItem(this.localStorageKey, JSON.stringify(updatedFeedback));
    return id;
  }

  private getLocalFeedback(): FeedbackStorageItem[] {
    try {
      const stored = localStorage.getItem(this.localStorageKey);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Error parsing local feedback:', error);
      return [];
    }
  }

  // Legacy method - converts old format to new format and submits
  async submitFeedback(feedback: LegacyFeedbackData): Promise<{ success: boolean; id?: string; error?: string }> {
    try {
      // Convert legacy format to new format
      const newFeedback: FeedbackCreate = {
        session_id: feedback.sessionId || `session_${Date.now()}`,
        question: feedback.questionAnswer.question,
        answer: feedback.questionAnswer.answer,
        feedback_type: feedback.vote === 'like' ? 'positive' : feedback.vote === 'dislike' ? 'negative' : 'suggestion',
        feedback_text: feedback.comment || feedback.expertNotes,
        // Only include rating if it's a valid rating (1-5), not 0 or undefined
        rating: feedback.overallRating && feedback.overallRating > 0 ? feedback.overallRating : undefined,
        
        // Map detailed categories from legacy feedback
        retrieval_relevance: feedback.retrieval_relevance && feedback.retrieval_relevance > 0 ? feedback.retrieval_relevance : undefined,
        hallucination: feedback.hallucination && feedback.hallucination > 0 ? feedback.hallucination : undefined,
        noise_robustness: feedback.noise_robustness && feedback.noise_robustness > 0 ? feedback.noise_robustness : undefined,
        negative_rejection: feedback.negative_rejection && feedback.negative_rejection > 0 ? feedback.negative_rejection : undefined,
        privacy_breach: feedback.privacy_breach && feedback.privacy_breach > 0 ? feedback.privacy_breach : undefined,
        malicious_use: feedback.malicious_use && feedback.malicious_use > 0 ? feedback.malicious_use : undefined,
        security_breach: feedback.security_breach && feedback.security_breach > 0 ? feedback.security_breach : undefined,
        out_of_domain: feedback.out_of_domain && feedback.out_of_domain > 0 ? feedback.out_of_domain : undefined,
        completeness: feedback.completeness && feedback.completeness > 0 ? feedback.completeness : undefined,
        brand_damage: feedback.brand_damage && feedback.brand_damage > 0 ? feedback.brand_damage : undefined,
        
        // Additional fields
        vote: feedback.vote || undefined,
        comment: feedback.comment || undefined,
        expert_notes: feedback.expertNotes || undefined,
      };

      const result = await this.createFeedback(newFeedback);
      
      if (result.success && result.data) {
        return { success: true, id: result.data.id };
      } else {
        throw new Error(result.error || 'Unknown error');
      }
    } catch (error) {
      console.warn('API submission failed, storing locally:', error);
      const id = this.storeLocally(feedback);
      return { success: true, id };
    }
  }

  async exportFeedback(): Promise<FeedbackStorageItem[]> {
    try {
      // Try to get from new API first
      const statsResult = await this.getFeedbackStats();
      if (statsResult.success && statsResult.data) {
        // Convert new format to legacy format for export compatibility
        return statsResult.data.recent_feedback.map(feedback => ({
          id: feedback.id,
          questionAnswer: {
            question: feedback.question,
            answer: feedback.answer
          },
          responseId: feedback.id,
          overallRating: feedback.rating && feedback.rating > 0 ? feedback.rating : undefined,
          
          // Map detailed categories from new format back to legacy format
          retrieval_relevance: feedback.retrieval_relevance,
          hallucination: feedback.hallucination,
          noise_robustness: feedback.noise_robustness,
          negative_rejection: feedback.negative_rejection,
          privacy_breach: feedback.privacy_breach,
          malicious_use: feedback.malicious_use,
          security_breach: feedback.security_breach,
          out_of_domain: feedback.out_of_domain,
          completeness: feedback.completeness,
          brand_damage: feedback.brand_damage,
          
          accuracy: undefined, // Not available in new format
          helpfulness: undefined, // Not available in new format
          clarity: undefined, // Not available in new format
                     vote: (feedback.vote === 'like' || feedback.vote === 'dislike') ? feedback.vote : 
                 (feedback.feedback_type === 'positive' ? 'like' as const : 
                 feedback.feedback_type === 'negative' ? 'dislike' as const : null),
          comment: feedback.comment || feedback.feedback_text || '',
          expertNotes: feedback.expert_notes || '',
          timestamp: new Date(feedback.created_at),
          sessionId: feedback.session_id
        }));
      }
      
      return this.getLocalFeedback();
    } catch (error) {
      console.error('Error exporting feedback:', error);
      return this.getLocalFeedback();
    }
  }

  clearLocalFeedback(): void {
    localStorage.removeItem(this.localStorageKey);
  }
}

export const feedbackService = new FeedbackService();
export type { 
  FeedbackCreate, 
  FeedbackResponse, 
  FeedbackStats,
  LegacyFeedbackData as FeedbackData, 
  FeedbackStorageItem 
}; 