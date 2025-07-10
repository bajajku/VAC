from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
import os
import re
import secrets
import warnings
from typing import Optional, Dict, Tuple

<<<<<<< HEAD

=======
>>>>>>> c833bc7 (feat: Implement chat session management in the API and frontend (#8))
# Improved SECRET_KEY handling with fallback
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    # Generate a temporary secret key for development
    SECRET_KEY = secrets.token_hex(32)
    warnings.warn(
        "⚠️ No SECRET_KEY environment variable found. Using a temporary key. "
        "This is insecure for production environments."
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# In-memory token blacklist (consider using Redis for production)
token_blacklist = set()

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def is_password_complex(password: str) -> Tuple[bool, str]:
    """Check if password meets complexity requirements."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter" 
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one digit"
    return True, "Password meets complexity requirements"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a new JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "token_type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict) -> str:
    """Create a new JWT refresh token with longer expiry."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "token_type": "refresh"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> Optional[Dict]:
    """Decode and validate access token."""
    try:
        # Check if token is blacklisted
        if token in token_blacklist:
            return None
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Verify this is an access token
        if payload.get("token_type") != "access":
            return None
            
        return payload
    except JWTError:
        return None

def decode_refresh_token(token: str) -> Optional[Dict]:
    """Decode and validate refresh token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Verify this is a refresh token
        if payload.get("token_type") != "refresh":
            return None
            
        return payload
    except JWTError:
        return None

def blacklist_token(token: str) -> bool:
    """Add token to blacklist."""
    token_blacklist.add(token)
    return True

def prune_blacklist() -> int:
    """Remove expired tokens from the blacklist."""
    initial_size = len(token_blacklist)
    current_time = datetime.utcnow()
    
    to_remove = set()
    for token in token_blacklist:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM], options={"verify_exp": False})
            exp_timestamp = payload.get("exp", 0)
            expiry_time = datetime.fromtimestamp(exp_timestamp)
            if expiry_time < current_time:
                to_remove.add(token)
        except:
            # If token can't be decoded, keep it in blacklist to be safe
            pass
    
    token_blacklist.difference_update(to_remove)
    return initial_size - len(token_blacklist)
