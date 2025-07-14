import Cookies from 'js-cookie';
import { refreshTokenIfNeeded } from '../utils/refreshToken';
class LogoutService {
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

  async logout(): Promise<{ success: boolean; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/auth/logout`, {
        method: 'POST',
        headers: await this.getAuthHeaders(),
      });

      if (response.ok) {
        // Remove the token cookies
        Cookies.remove('token');
        Cookies.remove('refresh_token');
        return { success: true };
      } else {
        return { success: false, error: 'Failed to logout' };
      }
    } catch (error) {
      return { success: false, error: 'Failed to logout ' + error };
    }
  }
}

export const logoutService = new LogoutService();