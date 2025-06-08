from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId
from config.mongodb import mongodb_config


class FeedbackBase(BaseModel):
    session_id: str = Field(..., description="Session ID for the conversation")
    question: str = Field(..., description="The original question asked")
    answer: str = Field(..., description="The AI-generated answer")
    feedback_type: str = Field(..., description="Type of feedback: 'positive', 'negative', 'suggestion'")
    feedback_text: Optional[str] = Field(None, description="Additional feedback text")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1-5")
    user_id: Optional[str] = Field(None, description="User identifier if available")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class FeedbackCreate(FeedbackBase):
    pass


class FeedbackResponse(FeedbackBase):
    id: str = Field(..., description="Feedback ID")
    created_at: datetime = Field(..., description="Timestamp when feedback was created")
    updated_at: datetime = Field(..., description="Timestamp when feedback was last updated")

    class Config:
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }


class FeedbackUpdate(BaseModel):
    feedback_type: Optional[str] = None
    feedback_text: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    metadata: Optional[Dict[str, Any]] = None


class FeedbackStats(BaseModel):
    total_feedback: int
    positive_count: int
    negative_count: int
    suggestion_count: int
    average_rating: Optional[float]
    recent_feedback: List[FeedbackResponse]


class FeedbackService:
    def __init__(self):
        self.collection_name = "feedback"

    async def create_feedback(self, feedback: FeedbackCreate) -> FeedbackResponse:
        """Create a new feedback entry"""
        collection = mongodb_config.get_collection(self.collection_name)
        
        feedback_doc = {
            **feedback.dict(),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await collection.insert_one(feedback_doc)
        
        # Retrieve the created document
        created_doc = await collection.find_one({"_id": result.inserted_id})
        
        return FeedbackResponse(
            id=str(created_doc["_id"]),
            **{k: v for k, v in created_doc.items() if k != "_id"}
        )

    async def get_feedback_by_id(self, feedback_id: str) -> Optional[FeedbackResponse]:
        """Get feedback by ID"""
        collection = mongodb_config.get_collection(self.collection_name)
        
        try:
            doc = await collection.find_one({"_id": ObjectId(feedback_id)})
            if doc:
                return FeedbackResponse(
                    id=str(doc["_id"]),
                    **{k: v for k, v in doc.items() if k != "_id"}
                )
        except Exception as e:
            print(f"Error retrieving feedback {feedback_id}: {e}")
        
        return None

    async def get_feedback_by_session(self, session_id: str) -> List[FeedbackResponse]:
        """Get all feedback for a session"""
        collection = mongodb_config.get_collection(self.collection_name)
        
        cursor = collection.find({"session_id": session_id}).sort("created_at", -1)
        feedback_list = []
        
        async for doc in cursor:
            feedback_list.append(FeedbackResponse(
                id=str(doc["_id"]),
                **{k: v for k, v in doc.items() if k != "_id"}
            ))
        
        return feedback_list

    async def update_feedback(self, feedback_id: str, update_data: FeedbackUpdate) -> Optional[FeedbackResponse]:
        """Update feedback by ID"""
        collection = mongodb_config.get_collection(self.collection_name)
        
        try:
            update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
            update_dict["updated_at"] = datetime.utcnow()
            
            result = await collection.update_one(
                {"_id": ObjectId(feedback_id)},
                {"$set": update_dict}
            )
            
            if result.modified_count > 0:
                return await self.get_feedback_by_id(feedback_id)
        except Exception as e:
            print(f"Error updating feedback {feedback_id}: {e}")
        
        return None

    async def delete_feedback(self, feedback_id: str) -> bool:
        """Delete feedback by ID"""
        collection = mongodb_config.get_collection(self.collection_name)
        
        try:
            result = await collection.delete_one({"_id": ObjectId(feedback_id)})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting feedback {feedback_id}: {e}")
            return False

    async def get_feedback_stats(self, limit: int = 10) -> FeedbackStats:
        """Get feedback statistics"""
        collection = mongodb_config.get_collection(self.collection_name)
        
        # Get total counts
        total_count = await collection.count_documents({})
        positive_count = await collection.count_documents({"feedback_type": "positive"})
        negative_count = await collection.count_documents({"feedback_type": "negative"})
        suggestion_count = await collection.count_documents({"feedback_type": "suggestion"})
        
        # Calculate average rating
        pipeline = [
            {"$match": {"rating": {"$exists": True, "$ne": None}}},
            {"$group": {"_id": None, "avg_rating": {"$avg": "$rating"}}}
        ]
        
        avg_result = await collection.aggregate(pipeline).to_list(1)
        average_rating = avg_result[0]["avg_rating"] if avg_result else None
        
        # Get recent feedback
        recent_cursor = collection.find({}).sort("created_at", -1).limit(limit)
        recent_feedback = []
        
        async for doc in recent_cursor:
            recent_feedback.append(FeedbackResponse(
                id=str(doc["_id"]),
                **{k: v for k, v in doc.items() if k != "_id"}
            ))
        
        return FeedbackStats(
            total_feedback=total_count,
            positive_count=positive_count,
            negative_count=negative_count,
            suggestion_count=suggestion_count,
            average_rating=round(average_rating, 2) if average_rating else None,
            recent_feedback=recent_feedback
        )

# Global feedback service instance
feedback_service = FeedbackService() 