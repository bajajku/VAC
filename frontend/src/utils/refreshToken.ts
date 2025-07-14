import Cookies from 'js-cookie';

export async function refreshTokenIfNeeded(): Promise<boolean> {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL;
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

        const response = await fetch(`${baseUrl}/auth/refresh`, {
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