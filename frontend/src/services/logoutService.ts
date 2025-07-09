import Cookies from 'js-cookie';



class LogoutService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  }

  private getAuthHeaders(): HeadersInit {
    const token = Cookies.get('token');
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    };
  }

  async logout(): Promise<{ success: boolean; error?: string }> {
    try {
      const response = await fetch(`${this.baseUrl}/auth/logout`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
      });

      if (response.ok) {
        // Remove the token cookie
        Cookies.remove('token');
        return { success: true };
      } else {
        return { success: false, error: 'Failed to logout' };
      }
    } catch (error) {
      return { success: false, error: 'Failed to logout' };
    }
  }
}

export const logoutService = new LogoutService();