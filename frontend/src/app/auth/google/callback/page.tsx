'use client';

import { Suspense, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Cookies from 'js-cookie';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const GoogleCallback = () => {
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const handleCallback = async () => {
      try {
        // Check if we have access_token in the URL (second phase)
        const accessToken = searchParams.get('access_token');
<<<<<<< HEAD
        const refreshToken = searchParams.get('refresh_token');
        
        if (accessToken && refreshToken) {
          console.log('Found tokens in URL, completing login');
          // We're in the second phase, store the tokens and redirect
          Cookies.set('token', accessToken, { expires: 7, sameSite: 'Lax' });
          Cookies.set('refresh_token', refreshToken, { expires: 7, sameSite: 'Lax' });
=======
        
        if (accessToken) {
          console.log('Found access token in URL, completing login');
          // We're in the second phase, store the token and redirect
          Cookies.set('token', accessToken, { expires: 7, sameSite: 'Lax' });
>>>>>>> c833bc7 (feat: Implement chat session management in the API and frontend (#8))
          router.push('/chat');
          return;
        }

        // First phase - exchange code for tokens
        const code = searchParams.get('code');
        const state = searchParams.get('state');

        console.log('Received code and state:', { code, state });

        if (!code || !state) {
          throw new Error('Missing authorization code or state');
        }

        // Call the backend callback endpoint
        const callbackUrl = `${BASE_URL}/auth/google/callback?code=${encodeURIComponent(code)}&state=${encodeURIComponent(state)}`;
        console.log('Calling backend:', callbackUrl);

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 85f5f79 (feat: Add ngrok header to API requests across authentication and feedback services)
        const response = await fetch(callbackUrl, {
          headers: {
            'ngrok-skip-browser-warning': 'true'
          }
        });
<<<<<<< HEAD
=======
        const response = await fetch(callbackUrl);
>>>>>>> c833bc7 (feat: Implement chat session management in the API and frontend (#8))
=======
>>>>>>> 85f5f79 (feat: Add ngrok header to API requests across authentication and feedback services)
        console.log('Backend response status:', response.status);

        const data = await response.json();
        console.log('Backend response data:', data);

        if (!response.ok) {
          throw new Error(data.message || 'Failed to authenticate with Google');
        }

        // The backend will return a redirect URL with the tokens
        if (data.redirect_url) {
          console.log('Received redirect URL, navigating...');
          window.location.href = data.redirect_url;
        } else {
          throw new Error('No redirect URL received from backend');
        }
      } catch (error) {
        console.error('Google callback error:', error);
        router.push('/auth/login?error=google_auth_failed');
      }
    };

    handleCallback();
  }, [router, searchParams]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mx-auto mb-4"></div>
        <h2 className="text-xl font-semibold text-gray-900">Completing sign in...</h2>
        <p className="mt-2 text-sm text-gray-600">Please wait while we verify your Google account.</p>
      </div>
    </div>
  );
} 

const GoogleCallbackPage = () => {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <GoogleCallback />
    </Suspense>
  );
};
export default GoogleCallbackPage;