// Types for the new MongoDB-based feedback system
type FeedbackCreate = {
  session_id: string;
  question: string;
  answer: string;
  feedback_type: 'positive' | 'negative' | 'suggestion';
  feedback_text?: string;
  rating?: number; // 1-5 scale
  user_id?: string;
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
  metadata: Record<string, any>;
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
  vote: 'like' | 'dislike' | null;
  comment: string;
  expertNotes: string;
  timestamp: Date;
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

  // New MongoDB-based feedback methods
  async createFeedback(feedback: FeedbackCreate): Promise<{ success: boolean; data?: FeedbackResponse; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
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
      const response = await fetch(`${this.baseUrl}/feedback/${feedbackId}`);

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
      const response = await fetch(`${this.baseUrl}/feedback/session/${sessionId}`);

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
      const response = await fetch(`${this.baseUrl}/feedback-stats`);

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
          accuracy: undefined, // Not available in new format
          helpfulness: undefined, // Not available in new format
          clarity: undefined, // Not available in new format
          vote: feedback.feedback_type === 'positive' ? 'like' as const : 
                feedback.feedback_type === 'negative' ? 'dislike' as const : null,
          comment: feedback.feedback_text || '',
          expertNotes: '',
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