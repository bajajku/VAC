from typing import Dict, Optional
import os
from pydantic import BaseModel

class OAuthConfig(BaseModel):
    client_id: str
    client_secret: str
    redirect_uri: str
    authorize_url: str
    token_url: str
    userinfo_url: str

class GoogleOAuthConfig(OAuthConfig):
    """Google OAuth specific configuration"""
    authorize_url: str = "https://accounts.google.com/o/oauth2/v2/auth"
    token_url: str = "https://oauth2.googleapis.com/token"
    userinfo_url: str = "https://www.googleapis.com/oauth2/v3/userinfo"

def get_google_oauth_config() -> GoogleOAuthConfig:
    """Get Google OAuth configuration from environment variables"""
    return GoogleOAuthConfig(
        client_id=os.getenv("GOOGLE_CLIENT_ID", ""),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET", ""),
        redirect_uri=os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback"),
    )

# Add these variables to your .env file:
"""
GOOGLE_CLIENT_ID=your_client_id
GOOGLE_CLIENT_SECRET=your_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
"""