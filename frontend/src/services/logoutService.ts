import Cookies from 'js-cookie';



class LogoutService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  }

  private async refreshTokenIfNeeded(): Promise<boolean> {
    try {
      const token = Cookies.get('token');
      if (!token) return false;

      // Try to decode the token to check expiration
      const tokenParts = token.split('.');
      if (tokenParts.length !== 3) return false;

      const payload = JSON.parse(atob(tokenParts[1]));
      const expiresAt = payload.exp * 1000; // Convert to milliseconds
      const now = Date.now();
      const fiveMinutes = 5 * 60 * 1000;

      // If token expires in less than 5 minutes, refresh it
      if (expiresAt - now < fiveMinutes) {
        const refreshToken = Cookies.get('refresh_token');
        if (!refreshToken) return false;

        const response = await fetch(`${this.baseUrl}/auth/refresh`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
          },
          body: JSON.stringify({ refresh_token: refreshToken })
        });

        if (response.ok) {
          const data = await response.json();
          Cookies.set('token', data.access_token, { expires: 7, sameSite: 'Lax' });
          Cookies.set('refresh_token', data.refresh_token, { expires: 7, sameSite: 'Lax' });
          return true;
        }
      }
      return true; // Token is still valid
    } catch (error) {
      console.error('Error refreshing token:', error);
      return false;
    }
  }

  private async getAuthHeaders(): Promise<HeadersInit> {
    // Try to refresh token if needed
    const isValid = await this.refreshTokenIfNeeded();
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