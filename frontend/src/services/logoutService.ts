import Cookies from 'js-cookie';
<<<<<<< HEAD
import { refreshTokenIfNeeded } from '../utils/refreshToken';
=======



>>>>>>> c833bc7 (feat: Implement chat session management in the API and frontend (#8))
class LogoutService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  }
<<<<<<< HEAD
  
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
=======

  private getAuthHeaders(): HeadersInit {
    const token = Cookies.get('token');
    return {
      'Authorization': `Bearer ${token}`,
<<<<<<< HEAD
      'Content-Type': 'application/json'
>>>>>>> c833bc7 (feat: Implement chat session management in the API and frontend (#8))
=======
      'Content-Type': 'application/json',
      'ngrok-skip-browser-warning': 'true'
>>>>>>> 85f5f79 (feat: Add ngrok header to API requests across authentication and feedback services)
    };
  }

  async logout(): Promise<{ success: boolean; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/auth/logout`, {
        method: 'POST',
<<<<<<< HEAD
        headers: await this.getAuthHeaders(),
      });

      if (response.ok) {
        // Remove the token cookies
        Cookies.remove('token');
        Cookies.remove('refresh_token');
=======
        headers: this.getAuthHeaders(),
      });

      if (response.ok) {
        // Remove the token cookie
        Cookies.remove('token');
>>>>>>> c833bc7 (feat: Implement chat session management in the API and frontend (#8))
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