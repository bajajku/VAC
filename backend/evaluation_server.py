"""
Evaluation API Server - Standalone FastAPI server for Jury of LLMs evaluation system.
Separate from the main chatbot API for separation of concerns.

Run with: uvicorn evaluation_server:app --host 0.0.0.0 --port 8001 --reload
"""
import sys
import os
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import asyncio

# Import the core evaluation system
from tests.test_evaluation import EvaluationSystem
from config.mongodb import mongodb_config

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Jury of LLMs - Evaluation API",
    description="API for RAG evaluation using a jury of LLMs with judge oversight",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global evaluation system instance
evaluation_system: Optional[EvaluationSystem] = None


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class EvaluateRequest(BaseModel):
    """Request to run a single evaluation."""
    question: str
    use_judge: bool = True


class EvaluationResponse(BaseModel):
    """Complete evaluation response."""
    id: str
    query: str
    response: str
    overall_score: float
    pass_rate: float
    overall_pass_fail: str
    evaluation_results: Dict[str, Any]
    timestamp: str
    jury_composition: Optional[Dict[str, Any]] = None
    aggregated_improvements: Optional[List[str]] = None


class EvaluationSummary(BaseModel):
    """Summary for listing evaluations."""
    id: str
    query: str
    overall_score: float
    pass_rate: float
    overall_pass_fail: str
    timestamp: str


class DashboardStats(BaseModel):
    """Dashboard statistics."""
    total_evaluations: int
    avg_score: float
    pass_rate: float
    recent_trend: Optional[str] = None
    evaluations_today: int = 0
    evaluations_this_week: int = 0


class BatchEvaluateRequest(BaseModel):
    """Request to run batch evaluation."""
    questions: List[str]
    use_judge: bool = True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_evaluation_system() -> EvaluationSystem:
    """Get or initialize the evaluation system."""
    global evaluation_system
    if evaluation_system is None:
        print("Initializing Evaluation System...")
        evaluation_system = EvaluationSystem()
        print("Evaluation System initialized!")
    return evaluation_system


def evaluation_report_to_dict(report) -> dict:
    """Convert RAGEvaluationReport to dictionary."""
    eval_results = {}
    for criterion, result in report.evaluation_results.items():
        eval_results[criterion] = {
            "criterion": result.criterion,
            "score": result.score,
            "pass_fail": result.pass_fail,
            "reasoning": result.reasoning,
            "confidence": result.confidence,
            "improvement_suggestions": result.improvement_suggestions,
        }

    return {
        "query": report.query,
        "response": report.response,
        "overall_score": report.overall_score,
        "pass_rate": report.pass_rate,
        "overall_pass_fail": report.overall_pass_fail,
        "evaluation_results": eval_results,
        "timestamp": report.timestamp,
        "jury_composition": report.jury_composition,
        "aggregated_improvements": report.aggregated_improvements,
        "context_documents": report.context_documents[:3] if report.context_documents else [],
    }


async def store_evaluation(evaluation_data: dict) -> str:
    """Store evaluation in MongoDB and return ID."""
    try:
        collection = mongodb_config.get_collection("jury_evaluations")
        evaluation_id = str(uuid.uuid4())
        evaluation_data["_id"] = evaluation_id
        evaluation_data["created_at"] = datetime.utcnow()
        await collection.insert_one(evaluation_data)
        return evaluation_id
    except Exception as e:
        print(f"Warning: Could not store evaluation in MongoDB: {e}")
        return f"temp-{uuid.uuid4()}"


async def get_stored_evaluation(evaluation_id: str) -> Optional[dict]:
    """Retrieve evaluation from MongoDB."""
    try:
        collection = mongodb_config.get_collection("jury_evaluations")
        result = await collection.find_one({"_id": evaluation_id})
        return result
    except Exception as e:
        print(f"Error retrieving evaluation: {e}")
        return None


async def list_stored_evaluations(limit: int = 50, offset: int = 0) -> List[dict]:
    """List evaluations from MongoDB."""
    try:
        collection = mongodb_config.get_collection("jury_evaluations")
        cursor = collection.find().sort("created_at", -1).skip(offset).limit(limit)
        results = await cursor.to_list(length=limit)
        return results
    except Exception as e:
        print(f"Error listing evaluations: {e}")
        return []


async def get_evaluation_stats() -> dict:
    """Calculate evaluation statistics."""
    try:
        collection = mongodb_config.get_collection("jury_evaluations")
        total = await collection.count_documents({})

        if total == 0:
            return {
                "total_evaluations": 0,
                "avg_score": 0.0,
                "pass_rate": 0.0,
                "recent_trend": None,
                "evaluations_today": 0,
                "evaluations_this_week": 0,
            }

        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avg_score": {"$avg": "$overall_score"},
                    "avg_pass_rate": {"$avg": "$pass_rate"},
                    "total": {"$sum": 1},
                }
            }
        ]

        cursor = collection.aggregate(pipeline)
        agg_result = await cursor.to_list(length=1)

        if agg_result:
            stats = agg_result[0]
            return {
                "total_evaluations": stats["total"],
                "avg_score": round(stats["avg_score"], 2),
                "pass_rate": round(stats["avg_pass_rate"], 1),
                "recent_trend": None,
                "evaluations_today": 0,
                "evaluations_this_week": 0,
            }

        return {"total_evaluations": total, "avg_score": 0.0, "pass_rate": 0.0}

    except Exception as e:
        print(f"Error calculating stats: {e}")
        return {"total_evaluations": 0, "avg_score": 0.0, "pass_rate": 0.0}


# =============================================================================
# LIFECYCLE EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize MongoDB connection on startup."""
    print("Starting Evaluation API Server...")
    try:
        connected = await mongodb_config.connect()
        if connected:
            print("MongoDB connected for evaluation storage")
            # Create index for faster queries
            collection = mongodb_config.get_collection("jury_evaluations")
            await collection.create_index("created_at")
        else:
            print("Warning: MongoDB not connected, evaluations won't be persisted")
    except Exception as e:
        print(f"MongoDB connection error: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    await mongodb_config.disconnect()


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "jury-llm-evaluation-api",
        "evaluation_system_initialized": evaluation_system is not None,
    }


@app.post("/api/evaluate", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluateRequest):
    """
    Run a single evaluation for a question.

    This evaluates the RAG system's response using:
    1. Jury of LLMs deliberation (multiple models evaluate in parallel)
    2. Judge LLM final verdict (if use_judge=True)
    """
    try:
        eval_sys = get_evaluation_system()

        print(f"Running evaluation for: {request.question[:50]}...")

        # Run the evaluation (calls core logic from test_evaluation.py)
        report = eval_sys.evaluate_single_question(
            question=request.question,
            use_judge=request.use_judge
        )

        # Convert to dictionary
        eval_data = evaluation_report_to_dict(report)

        # Store in MongoDB
        evaluation_id = await store_evaluation(eval_data)
        eval_data["id"] = evaluation_id

        return EvaluationResponse(
            id=evaluation_id,
            query=eval_data["query"],
            response=eval_data["response"],
            overall_score=eval_data["overall_score"],
            pass_rate=eval_data["pass_rate"],
            overall_pass_fail=eval_data["overall_pass_fail"],
            evaluation_results=eval_data["evaluation_results"],
            timestamp=eval_data["timestamp"],
            jury_composition=eval_data.get("jury_composition"),
            aggregated_improvements=eval_data.get("aggregated_improvements"),
        )

    except Exception as e:
        print(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/api/evaluations", response_model=List[EvaluationSummary])
async def list_evaluations(limit: int = 50, offset: int = 0):
    """List past evaluations with pagination."""
    try:
        evaluations = await list_stored_evaluations(limit=limit, offset=offset)

        return [
            EvaluationSummary(
                id=eval_data.get("_id", "unknown"),
                query=eval_data.get("query", ""),
                overall_score=eval_data.get("overall_score", 0.0),
                pass_rate=eval_data.get("pass_rate", 0.0),
                overall_pass_fail=eval_data.get("overall_pass_fail", "UNKNOWN"),
                timestamp=eval_data.get("timestamp", datetime.utcnow().isoformat()),
            )
            for eval_data in evaluations
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list evaluations: {str(e)}")


@app.get("/api/evaluations/{evaluation_id}", response_model=EvaluationResponse)
async def get_evaluation(evaluation_id: str):
    """Get detailed evaluation by ID."""
    try:
        eval_data = await get_stored_evaluation(evaluation_id)

        if not eval_data:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        return EvaluationResponse(
            id=eval_data.get("_id", evaluation_id),
            query=eval_data.get("query", ""),
            response=eval_data.get("response", ""),
            overall_score=eval_data.get("overall_score", 0.0),
            pass_rate=eval_data.get("pass_rate", 0.0),
            overall_pass_fail=eval_data.get("overall_pass_fail", "UNKNOWN"),
            evaluation_results=eval_data.get("evaluation_results", {}),
            timestamp=eval_data.get("timestamp", ""),
            jury_composition=eval_data.get("jury_composition"),
            aggregated_improvements=eval_data.get("aggregated_improvements"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation: {str(e)}")


@app.get("/api/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get dashboard statistics."""
    try:
        stats = await get_evaluation_stats()
        return DashboardStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.post("/api/evaluate/batch")
async def run_batch_evaluation(request: BatchEvaluateRequest):
    """Run batch evaluation for multiple questions."""
    try:
        eval_sys = get_evaluation_system()

        results = []
        for i, question in enumerate(request.questions):
            print(f"Evaluating {i+1}/{len(request.questions)}: {question[:50]}...")

            report = eval_sys.evaluate_single_question(
                question=question,
                use_judge=request.use_judge
            )

            eval_data = evaluation_report_to_dict(report)
            evaluation_id = await store_evaluation(eval_data)
            eval_data["id"] = evaluation_id
            results.append(eval_data)

        return {"status": "completed", "total": len(results), "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")


@app.get("/api/jury/info")
async def get_jury_info():
    """Get information about the current jury composition."""
    try:
        eval_sys = get_evaluation_system()
        jury_info = eval_sys.jury_evaluator.jury.get_jury_info()

        return {
            "jury_members": jury_info.get("jury_size", 0),
            "models": jury_info.get("models", []),
            "judge_enabled": True,
            "judge_model": eval_sys.judge_llm.judge_llm.model_name if eval_sys.judge_llm else None,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get jury info: {str(e)}")


@app.delete("/api/evaluations/{evaluation_id}")
async def delete_evaluation(evaluation_id: str):
    """Delete an evaluation by ID."""
    try:
        collection = mongodb_config.get_collection("jury_evaluations")
        result = await collection.delete_one({"_id": evaluation_id})

        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Evaluation not found")

        return {"message": "Evaluation deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete evaluation: {str(e)}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
