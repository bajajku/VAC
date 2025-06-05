
type QuestionAnswer = {
  question: string;
  answer: string;
}

type FeedbackData = {
  questionAnswer: QuestionAnswer;
  responseId: string;
  overallRating: number;
  accuracy: number;
  helpfulness: number;
  clarity: number;
  vote: 'like' | 'dislike' | null;
  comment: string;
  expertNotes: string;
  timestamp: Date;
  sessionId?: string;
};

type FeedbackStorageItem = FeedbackData & {
  id: string;
};

class FeedbackService {
  private baseUrl: string;
  private localStorageKey = 'chatbot_feedback';

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || '';
  }

  // Store feedback locally (fallback when API is not available)
  private storeLocally(feedback: FeedbackData): string {
    const id = `feedback_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const feedbackWithId: FeedbackStorageItem = { ...feedback, id };
    
    const existingFeedback = this.getLocalFeedback();
    const updatedFeedback = [...existingFeedback, feedbackWithId];
    
    localStorage.setItem(this.localStorageKey, JSON.stringify(updatedFeedback));
    return id;
  }

  // Get locally stored feedback
  private getLocalFeedback(): FeedbackStorageItem[] {
    try {
      const stored = localStorage.getItem(this.localStorageKey);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Error parsing local feedback:', error);
      return [];
    }
  }

  // Submit feedback - tries API first, falls back to local storage
  async submitFeedback(feedback: FeedbackData): Promise<{ success: boolean; id?: string; error?: string }> {
    try {
      // Try to submit to API if endpoint exists
      if (this.baseUrl) {
        const response = await fetch(`${this.baseUrl}/feedback`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(feedback),
        });

        if (response.ok) {
          const result = await response.json();
          return { success: true, id: result.id };
        } else {
          throw new Error(`API Error: ${response.status}`);
        }
      }
      
      // Fallback to local storage
      throw new Error('No API endpoint configured');
    } catch (error) {
      console.warn('API submission failed, storing locally:', error);
      const id = this.storeLocally(feedback);
      return { success: true, id };
    }
  }

  // Get feedback statistics for analytics
  async getFeedbackStats(): Promise<{
    totalFeedback: number;
    averageRatings: {
      overall: number;
      accuracy: number;
      helpfulness: number;
      clarity: number;
    };
    voteDistribution: {
      likes: number;
      dislikes: number;
      neutral: number;
    };
  }> {
    try {
      // Try to get from API first
      if (this.baseUrl) {
        const response = await fetch(`${this.baseUrl}/feedback/stats`);
        if (response.ok) {
          return await response.json();
        }
      }
      
      // Fallback to local analysis
      const localFeedback = this.getLocalFeedback();
      return this.calculateLocalStats(localFeedback);
    } catch (error) {
      console.error('Error fetching feedback stats:', error);
      // Fallback to local analysis
      const localFeedback = this.getLocalFeedback();
      return this.calculateLocalStats(localFeedback);
      
    }
  }

  private calculateLocalStats(feedback: FeedbackStorageItem[]) {
    if (feedback.length === 0) {
      return {
        totalFeedback: 0,
        averageRatings: { overall: 0, accuracy: 0, helpfulness: 0, clarity: 0 },
        voteDistribution: { likes: 0, dislikes: 0, neutral: 0 }
      };
    }

    const ratingsSum = feedback.reduce((acc, item) => ({
      overall: acc.overall + item.overallRating,
      accuracy: acc.accuracy + item.accuracy,
      helpfulness: acc.helpfulness + item.helpfulness,
      clarity: acc.clarity + item.clarity,
    }), { overall: 0, accuracy: 0, helpfulness: 0, clarity: 0 });

    const votes = feedback.reduce((acc, item) => ({
      likes: acc.likes + (item.vote === 'like' ? 1 : 0),
      dislikes: acc.dislikes + (item.vote === 'dislike' ? 1 : 0),
      neutral: acc.neutral + (item.vote === null ? 1 : 0),
    }), { likes: 0, dislikes: 0, neutral: 0 });

    return {
      totalFeedback: feedback.length,
      averageRatings: {
        overall: ratingsSum.overall / feedback.length,
        accuracy: ratingsSum.accuracy / feedback.length,
        helpfulness: ratingsSum.helpfulness / feedback.length,
        clarity: ratingsSum.clarity / feedback.length,
      },
      voteDistribution: votes
    };
  }

  // Export feedback data (for analysis or backup)
  async exportFeedback(): Promise<FeedbackStorageItem[]> {
    try {
      if (this.baseUrl) {
        const response = await fetch(`${this.baseUrl}/feedback/export`);
        if (response.ok) {
          return await response.json();
        }
      }
      
      return this.getLocalFeedback();
    } catch (error) {
      console.error('Error exporting feedback:', error);
      return this.getLocalFeedback();
    }
  }

  // Clear local feedback (useful for testing or privacy)
  clearLocalFeedback(): void {
    localStorage.removeItem(this.localStorageKey);
  }
}

export const feedbackService = new FeedbackService();
export type { FeedbackData, FeedbackStorageItem }; 