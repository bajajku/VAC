# import uuid
# from typing import Dict, List, Optional, Tuple
# from datetime import datetime
# from pydantic import BaseModel

# class ChatMessage(BaseModel):
#     role: str  # 'user' or 'assistant'
#     content: str
#     timestamp: datetime = datetime.now()

# class ChatSession(BaseModel):
#     session_id: str
#     messages: List[Tuple[ChatMessage, ChatMessage]] = []
#     created_at: datetime = datetime.now()
#     last_activity: datetime = datetime.now()

# class ChatSessionManager:
#     def __init__(self, max_sessions: int = 1000, max_messages_per_session: int = 100):
#         self.sessions: Dict[str, ChatSession] = {}
#         self.max_sessions = max_sessions
#         self.max_messages_per_session = max_messages_per_session
    
#     def create_session(self) -> str:
#         """Create a new chat session and return session ID."""
#         session_id = str(uuid.uuid4())
#         self.sessions[session_id] = ChatSession(session_id=session_id)
        
#         # Clean up old sessions if we have too many
#         if len(self.sessions) > self.max_sessions:
#             self._cleanup_old_sessions()
        
#         return session_id
    
#     def get_session(self, session_id: str) -> Optional[ChatSession]:
#         """Get session by ID."""
#         return self.sessions.get(session_id)
    
#     def add_message(self, session_id: str, message: Tuple[ChatMessage, ChatMessage]) -> bool:
#         """Add a message to the session."""
#         session = self.get_session(session_id)
#         if not session:
#             return False
        
#         # Limit messages per session
#         if len(session.messages) >= self.max_messages_per_session:
#             # Remove oldest message
#             session.messages.pop(0)

#         # message is a tuple of (user_message, bot_message)
#         session.messages.append(message)
#         session.last_activity = datetime.now()
#         return True
    
#     def get_chat_history(self, session_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
#         """Get chat history for a session."""
#         session = self.get_session(session_id)
#         if not session:
#             return []
        
#         messages = session.messages
#         if limit:
#             messages = messages[-limit:]
        
#         return messages
    
#     def get_context_for_llm(self, session_id: str, context_limit: int = 10) -> str:
#         """Get formatted context from recent messages for LLM."""
#         messages = self.get_chat_history(session_id, limit=context_limit)
        
#         context_parts = []
#         for user_msg, bot_msg in messages:
#             history = f"Human: {user_msg.content}\nAssistant: {bot_msg.content}"
#             context_parts.append(history)
        
#         return "\n".join(context_parts)
    
#     def _cleanup_old_sessions(self):
#         """Remove oldest sessions when limit is exceeded."""
#         # Sort by last activity and remove oldest 10%
#         sorted_sessions = sorted(
#             self.sessions.items(),
#             key=lambda x: x[1].last_activity
#         )
        
#         sessions_to_remove = len(sorted_sessions) // 10
#         for session_id, _ in sorted_sessions[:sessions_to_remove]:
#             del self.sessions[session_id]

# # Global session manager instance
# chat_session_manager = ChatSessionManager()
