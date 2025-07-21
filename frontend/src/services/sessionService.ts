import { refreshTokenIfNeeded } from '../utils/refreshToken';
// Types for chat session management
import Cookies from 'js-cookie';
export type ChatMessage = {
  id: string;
  content: string;
  sender: 'user' | 'assistant';
  timestamp: string;
  sources?: string[];
  metadata?: Record<string, unknown>;
};

export type ChatSession = {
  id: string;
  session_id: string;
  title?: string;
  user_id?: string;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
  message_count: number;
  metadata?: Record<string, unknown>;
};

export type NewSessionRequest = {
  user_id?: string;
  title?: string;
};

export type SessionUpdateRequest = {
  title?: string;
  metadata?: Record<string, unknown>;
};

// Frontend message type
export type FrontendMessage = {
  id: number;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  feedbackSubmitted?: boolean;
  sources?: string[];
};

class SessionService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  }
  private async getAuthHeaders(): Promise<HeadersInit> {
    // Try to refresh token if needed
    const isValid = await refreshTokenIfNeeded();
    if (!isValid) {
      // Token refresh failed, redirect to login
      window.location.href = '/auth/login';
      throw new Error('Authentication failed');
    }

    const token = Cookies.get('token');
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
      'ngrok-skip-browser-warning': 'true'
    };
  }

  async createNewSession(request: NewSessionRequest = {}): Promise<{ success: boolean; data?: ChatSession; error?: string }> {
    try {

      const response = await fetch(`${this.baseUrl}/sessions/new`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify(request),
      });

      if (response.ok) {
        const data = await response.json();
        return { success: true, data };
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Error creating new session:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async listSessions(userId?: string, limit: number = 20): Promise<{ success: boolean; data?: ChatSession[]; error?: string }> {
    try {
      const params = new URLSearchParams();
      if (userId) params.append('user_id', userId);
      if (limit) params.append('limit', limit.toString());

      const response = await fetch(`${this.baseUrl}/sessions?${params}`, {
        headers: await this.getAuthHeaders()
      });

      if (response.ok) {
        const data = await response.json();
        return { success: true, data };
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Error listing sessions:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async getSession(sessionId: string): Promise<{ success: boolean; data?: ChatSession; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/sessions/${sessionId}`, {
        headers: await this.getAuthHeaders()
      });

      if (response.ok) {
        const data = await response.json();
        return { success: true, data };
      } else if (response.status === 404) {
        return { success: false, error: 'Session not found' };
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Error fetching session:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async getSessionMessages(sessionId: string): Promise<{ success: boolean; data?: ChatMessage[]; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/sessions/${sessionId}/messages`, {
        headers: await this.getAuthHeaders()
      });

      if (response.ok) {
        const data = await response.json();
        return { success: true, data };
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Error fetching session messages:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async updateSession(sessionId: string, updateData: SessionUpdateRequest): Promise<{ success: boolean; data?: ChatSession; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/sessions/${sessionId}`, {
        method: 'PUT',
        headers: await this.getAuthHeaders(),
        body: JSON.stringify(updateData),
      });

      if (response.ok) {
        const data = await response.json();
        return { success: true, data };
      } else if (response.status === 404) {
        return { success: false, error: 'Session not found' };
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Error updating session:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async deleteSession(sessionId: string): Promise<{ success: boolean; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/sessions/${sessionId}`, {
        method: 'DELETE',
        headers: await this.getAuthHeaders()
      });

      if (response.ok) {
        return { success: true };
      } else if (response.status === 404) {
        return { success: false, error: 'Session not found' };
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (error) {
      console.error('Error deleting session:', error);
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  // Helper methods for localStorage compatibility
  saveSessionToLocalStorage(sessionId: string): void {
    localStorage.setItem('chatSessionId', sessionId);
  }

  getSessionFromLocalStorage(): string | null {
    return localStorage.getItem('chatSessionId');
  }

  clearSessionFromLocalStorage(): void {
    localStorage.removeItem('chatSessionId');
  }

  // Convert backend message format to frontend format
  convertBackendMessage(backendMessage: ChatMessage, messageId: number): FrontendMessage {
    return {
      id: messageId,
      text: backendMessage.content,
      sender: backendMessage.sender === 'assistant' ? 'bot' : backendMessage.sender,
      timestamp: new Date(backendMessage.timestamp),
      feedbackSubmitted: false,
      sources: backendMessage.sources || []
    };
  }

  // Convert frontend message format to backend format
  convertFrontendMessage(frontendMessage: FrontendMessage): ChatMessage {
    return {
      id: frontendMessage.id.toString(),
      content: frontendMessage.text,
      sender: frontendMessage.sender === 'bot' ? 'assistant' : frontendMessage.sender,
      timestamp: frontendMessage.timestamp.toISOString(),
      metadata: {}
    };
  }
}

export const sessionService = new SessionService();
export default sessionService; 