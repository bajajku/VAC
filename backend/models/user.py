from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from config.mongodb import MongoDBConfig, mongodb_config
from utils.auth_service import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

class AuthProvider(str, Enum):
    LOCAL = "local"
    GOOGLE = "google"

class OAuthUserInfo(BaseModel):
    """OAuth user information"""
    provider: AuthProvider
    provider_user_id: str
    email: EmailStr
    name: Optional[str] = None
    picture: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[int] = None

class User(BaseModel):
    email: EmailStr = Field(..., description="User email")
    username: Optional[str] = Field(None, description="User username")
    hashed_password: Optional[str] = Field(None, description="Hashed password")
    created_at: Optional[datetime] = Field(None, description="User creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="User last update timestamp")
    auth_provider: AuthProvider = Field(default=AuthProvider.LOCAL, description="Authentication provider")
    oauth_info: Optional[OAuthUserInfo] = None
    
    class Config:
        from_attributes = True

class UserCreate(BaseModel):
    email: EmailStr = Field(..., description="User email")
    username: str = Field(..., description="User username") 
    password: str = Field(..., description="User password")

class UserLogin(BaseModel):
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., description="User password")

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = Field(None, description="User email")
    username: Optional[str] = Field(None, description="User username")
    password: Optional[str] = Field(None, description="User password")

class UserInDB(User):
    hashed_pw: str = Field(..., description="Hashed password")

class Token(BaseModel):
    access_token: str = Field(..., description="JWT access token")
    refresh_token: Optional[str] = Field(None, description="JWT refresh token")
    token_type: str = "bearer"
    expires_at: Optional[int] = Field(None, description="Token expiration timestamp")
    
class TokenData(BaseModel):
    sub: str = Field(..., description="Subject (usually email)")
    exp: Optional[int] = Field(None, description="Expiration timestamp")
    token_type: Optional[str] = Field(None, description="Token type (access or refresh)")
    
class TokenRefreshRequest(BaseModel):
    refresh_token: str = Field(..., description="Refresh token")
    
class TokenPayload(BaseModel):
    sub: Optional[str] = None

class OAuthState(BaseModel):
    """State parameter for OAuth flow"""
    redirect_url: str
    provider: AuthProvider


# Helper function to get user safely
async def get_user_by_email(mongodb_config: MongoDBConfig, email: str) -> Optional[dict]:
    """Get user by email with safe error handling"""
    try:
        users = mongodb_config.get_collection("users")
        user = await users.find_one({"email": email})
        return user
    except Exception as e:
        print(f"Error fetching user: {e}")
        return None

# Helper function to get current user
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current user from token"""
    payload = decode_access_token(token)
    print(f"Payload: {payload}")
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    email = payload.get("sub")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    user = await get_user_by_email(mongodb_config, email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    return User(
        email=user["email"],
        username=user.get("name", "User"),  # Safe access with default
        created_at=user.get("created_at"),
        updated_at=user.get("updated_at")
    )
