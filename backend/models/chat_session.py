from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId
from config.mongodb import mongodb_config
import uuid


class ChatMessage(BaseModel):
    """Individual chat message model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="Message content")
    sender: str = Field(..., description="Message sender: 'user' or 'assistant'")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    sources: Optional[List[str]] = Field(default_factory=list, description="List of source URLs/references for assistant messages")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatTurn(BaseModel):
    """Chat turn model"""
    user_message: ChatMessage = Field(..., description="User message")
    assistant_message: ChatMessage = Field(..., description="Assistant message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ChatSessionBase(BaseModel):
    """Base chat session model"""
    session_id: str = Field(..., description="Unique session identifier")
    title: Optional[str] = Field(None, description="Session title/summary")
    user_id: Optional[str] = Field(None, description="User identifier if available")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatSessionCreate(ChatSessionBase):
    """Model for creating a new chat session"""
    pass


class ChatSessionResponse(ChatSessionBase):
    """Model for chat session response"""
    id: str = Field(..., description="MongoDB document ID")
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    message_count: int = Field(default=0)


class ChatSessionUpdate(BaseModel):
    """Model for updating chat session"""
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatSessionService:
    """Service for managing chat sessions in MongoDB"""
    
    def __init__(self):
        self.collection_name = "chat_sessions"
        self.messages_collection_name = "chat_messages"

    async def initialize(self):
        """Initialize indexes and other setup requirements"""
        try:
            # Get collections
            sessions_collection = mongodb_config.get_collection(self.collection_name)
            messages_collection = mongodb_config.get_collection(self.messages_collection_name)
            
            # Create indexes for chat sessions
            await sessions_collection.create_index("user_id")  # For user's sessions
            await sessions_collection.create_index("session_id", unique=True)  # Unique session IDs
            await sessions_collection.create_index("updated_at")  # For sorting by last activity
            
            # Create indexes for messages
            await messages_collection.create_index([("session_id", 1), ("timestamp", 1)])  # For retrieving messages in order
            await messages_collection.create_index("timestamp")  # For global message queries
            
            print("✅ Created indexes for chat sessions and messages")
            return True
        except Exception as e:
            print(f"⚠️ Failed to create indexes: {e}")
            return False

    async def create_session(self, session_data: ChatSessionCreate) -> ChatSessionResponse:
        """Create a new chat session"""
        collection = mongodb_config.get_collection(self.collection_name)
        
        session_doc = {
            **session_data.dict(),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "message_count": 0
        }
        
        result = await collection.insert_one(session_doc)
        created_doc = await collection.find_one({"_id": result.inserted_id})
        
        # Create session response with explicit field mapping
        response_data = {
            "id": str(created_doc["_id"]),
            "session_id": created_doc.get("session_id", ""),
            "title": created_doc.get("title"),
            "user_id": created_doc.get("user_id"),
            "created_at": created_doc.get("created_at", datetime.utcnow()),
            "updated_at": created_doc.get("updated_at", datetime.utcnow()),
            "message_count": created_doc.get("message_count", 0),
            "metadata": created_doc.get("metadata", {}),
            "messages": []
        }
        return ChatSessionResponse(**response_data)

    async def get_session(self, session_id: str) -> Optional[ChatSessionResponse]:
        """Get a chat session by ID"""
        collection = mongodb_config.get_collection(self.collection_name)
        
        try:
            session_doc = await collection.find_one({"session_id": session_id})
            if not session_doc:
                return None
            
            # Get messages for this session
            messages = await self.get_session_messages(session_id)
            
            # Create session response with explicit field mapping
            session_data = {
                "id": str(session_doc["_id"]),
                "session_id": session_doc.get("session_id", ""),
                "title": session_doc.get("title"),
                "user_id": session_doc.get("user_id"),
                "created_at": session_doc.get("created_at", datetime.utcnow()),
                "updated_at": session_doc.get("updated_at", datetime.utcnow()),
                "message_count": session_doc.get("message_count", 0),
                "metadata": session_doc.get("metadata", {}),
                "messages": messages
            }
            return ChatSessionResponse(**session_data)
        except Exception as e:
            print(f"Error retrieving session {session_id}: {e}")
            return None

    async def get_session_messages(self, session_id: str) -> List[ChatMessage]:
        """Get all messages for a session"""
        collection = mongodb_config.get_collection(self.messages_collection_name)
        
        cursor = collection.find({"session_id": session_id}).sort("timestamp", 1)
        messages = []
        
        async for doc in cursor:
            try:
                # Create message with explicit field mapping to avoid conflicts
                # Use MongoDB's _id as our id, ignore any other 'id' field
                message_data = {
                    "id": str(doc["_id"]),  # Use MongoDB's _id
                    "content": doc.get("content", ""),
                    "sender": doc.get("sender", "user"),
                    "timestamp": doc.get("timestamp", datetime.utcnow()),
                    "metadata": doc.get("metadata", {})
                }
                messages.append(ChatMessage(**message_data))
            except Exception as e:
                print(f"Error processing message document: {e}")
                print(f"Document keys: {list(doc.keys())}")
                # Skip this message and continue
                continue
        
        return messages

    async def add_message(self, session_id: str, message: ChatMessage) -> bool:
        """Add a message to a session"""
        try:
            # Store message
            messages_collection = mongodb_config.get_collection(self.messages_collection_name)
            
            # Exclude the auto-generated 'id' field to avoid conflicts with MongoDB's _id
            message_dict = message.dict(exclude={'id'})
            message_doc = {
                "session_id": session_id,
                **message_dict,
                "timestamp": message.timestamp or datetime.utcnow()
            }
            
            await messages_collection.insert_one(message_doc)
            
            # Update session metadata
            sessions_collection = mongodb_config.get_collection(self.collection_name)
            await sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {"updated_at": datetime.utcnow()},
                    "$inc": {"message_count": 1}
                }
            )
            
            return True
        except Exception as e:
            print(f"Error adding message to session {session_id}: {e}")
            return False

    async def update_session(self, session_id: str, update_data: ChatSessionUpdate) -> Optional[ChatSessionResponse]:
        """Update a chat session"""
        collection = mongodb_config.get_collection(self.collection_name)
        
        try:
            update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
            update_dict["updated_at"] = datetime.utcnow()
            
            result = await collection.update_one(
                {"session_id": session_id},
                {"$set": update_dict}
            )
            
            if result.modified_count > 0:
                return await self.get_session(session_id)
            
            return None
        except Exception as e:
            print(f"Error updating session {session_id}: {e}")
            return None

    async def delete_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages"""
        try:
            # Delete messages first
            messages_collection = mongodb_config.get_collection(self.messages_collection_name)
            await messages_collection.delete_many({"session_id": session_id})
            
            # Delete session
            sessions_collection = mongodb_config.get_collection(self.collection_name)
            result = await sessions_collection.delete_one({"session_id": session_id})
            
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
            return False

    async def list_sessions(self, user_id: Optional[str] = None, limit: int = 20) -> List[ChatSessionResponse]:
        """List chat sessions, optionally filtered by user"""
        collection = mongodb_config.get_collection(self.collection_name)
        
        # Build aggregation pipeline for efficient single-query operation
        match_stage = {}
        if user_id:
            match_stage["user_id"] = user_id
        
        pipeline = [
            {"$match": match_stage},
            {"$sort": {"updated_at": -1}},
            {"$limit": limit},
            {
                "$lookup": {
                    "from": self.messages_collection_name,
                    "localField": "session_id",
                    "foreignField": "session_id",
                    "as": "messages",
                    "pipeline": [
                        {"$sort": {"timestamp": 1}}  # Sort messages by timestamp
                    ]
                }
            },
            {
                "$addFields": {
                    "actual_message_count": {"$size": "$messages"},
                    "preview_messages": {"$slice": ["$messages", -5]}  # Last 5 messages for preview
                }
            }
        ]
        
        cursor = collection.aggregate(pipeline)
        sessions = []
        
        async for doc in cursor:
            # Convert message documents to ChatMessage objects
            preview_messages = []
            for msg_doc in doc.get("preview_messages", []):
                try:
                    message_data = {
                        "id": str(msg_doc["_id"]),
                        "content": msg_doc.get("content", ""),
                        "sender": msg_doc.get("sender", "user"),
                        "timestamp": msg_doc.get("timestamp", datetime.utcnow()),
                        "metadata": msg_doc.get("metadata", {})
                    }
                    preview_messages.append(ChatMessage(**message_data))
                except Exception as e:
                    print(f"Error processing preview message: {e}")
                    continue
            
            # Create session response with explicit field mapping
            session_data = {
                "id": str(doc["_id"]),
                "session_id": doc.get("session_id", ""),
                "title": doc.get("title"),
                "user_id": doc.get("user_id"),
                "created_at": doc.get("created_at", datetime.utcnow()),
                "updated_at": doc.get("updated_at", datetime.utcnow()),
                "message_count": doc.get("actual_message_count", 0),
                "metadata": doc.get("metadata", {}),
                "messages": preview_messages
            }
            sessions.append(ChatSessionResponse(**session_data))
        
        return sessions

    async def get_or_create_session(self, session_id: str, user_id: Optional[str] = None) -> ChatSessionResponse:
        """Get existing session or create new one if it doesn't exist"""
        session = await self.get_session(session_id)
        
        if not session:
            # Create new session
            session_data = ChatSessionCreate(
                session_id=session_id,
                user_id=user_id,
                title=f"Chat Session {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
            )
            session = await self.create_session(session_data)
        
        return session

    async def add_chat_turn(self, session_id: str, chat_turn: ChatTurn) -> bool:
        """Add a chat turn to a session"""
        try:
            # Store chat turn
            chat_turns_collection = mongodb_config.get_collection(self.chat_turns_collection_name)
            await chat_turns_collection.insert_one(chat_turn.dict())
            return True
        except Exception as e:
            print(f"Error adding chat turn to session {session_id}: {e}")
            return False


# Global service instance
chat_session_service = ChatSessionService()
