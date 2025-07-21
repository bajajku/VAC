'''
This module provides a fallback service for the RAG application.
It routes guardrail violations to appropriate fallback tools and handles responses.
'''

from typing import Dict, Any, Optional, Callable
import logging
from models.tools.fallback_tools import (
    crisis_fallback_tool,
    profanity_hate_fallback_tool,
    privacy_fallback_tool,
    jailbreak_fallback_tool,
    out_of_domain_fallback_tool,
    llm_validation_fallback_tool,
    default_fallback_tool,
    enhanced_crisis_fallback_tool
)

logger = logging.getLogger(__name__)

# Configure logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class FallbackService:
    """
    Router service that maps guardrail violation categories to appropriate fallback tools.
    Supports both simple responses and complex fallback actions.
    """
    
    def __init__(self):
        # Map violation categories to their corresponding fallback tools
        self.tool_map: Dict[str, Callable] = {
            # Crisis-related violations (highest priority)
            "crisis_escalation": enhanced_crisis_fallback_tool,
            "CrisisEscalation": enhanced_crisis_fallback_tool,
            "crisis_escalation_ToxicLanguage": enhanced_crisis_fallback_tool,  # Full guard name
            "ToxicLanguage": self._route_toxic_language,  # Could be crisis or profanity
            
            # Security violations
            "jailbreak": jailbreak_fallback_tool,
            "DetectJailbreak": jailbreak_fallback_tool,
            "jailbreak_DetectJailbreak": jailbreak_fallback_tool,  # Full guard name
            
            # Privacy violations
            "privacy": privacy_fallback_tool,
            "DetectPII": privacy_fallback_tool,
            "privacy_DetectPII": privacy_fallback_tool,  # Full guard name
            "PrivateInfo": privacy_fallback_tool,
            
            # Content violations
            "profanity_hate_harassment": profanity_hate_fallback_tool,
            "ProfanityFree": profanity_hate_fallback_tool,
            "profanity_hate_harassment_ToxicLanguage": profanity_hate_fallback_tool,  # Full guard name
            "profanity_hate_harassment_ProfanityFree": profanity_hate_fallback_tool,  # Full guard name
            
            # Domain violations
            "topic_detection": out_of_domain_fallback_tool,
            "SensitiveTopic": out_of_domain_fallback_tool,
            "topic_detection_SensitiveTopic": out_of_domain_fallback_tool,  # Full guard name
            
            # LLM-based validation
            "llm_validation": llm_validation_fallback_tool,
            "LlamaGuard7B": llm_validation_fallback_tool,
        }
        
        # Legacy support
        self.fallback_response = "I'm sorry, I'm not able to answer that question. Please try again."

    def _route_toxic_language(self, context: Dict[str, Any] = None) -> str:
        """
        Route ToxicLanguage violations based on threshold/context.
        Low threshold might indicate crisis, high threshold indicates general toxicity.
        """
        if context:
            threshold = context.get("threshold", 0.5)
            if threshold <= 0.4:  # Low threshold suggests crisis detection
                return enhanced_crisis_fallback_tool(context)
        
        # Default to profanity handling
        return profanity_hate_fallback_tool(context)

    def get_fallback_response(self, category: str = "default", context: Dict[str, Any] = None) -> str:
        """
        Get appropriate fallback response based on violation category.
        
        Args:
            category: The violation category (guard name or custom category)
            context: Optional context about the violation (threshold, severity, etc.)
            
        Returns:
            Appropriate fallback message for the violation type
        """
        # Get the appropriate fallback tool
        fallback_tool = self.tool_map.get(category, default_fallback_tool)
        
        try:
            # Call the fallback tool with context
            response = fallback_tool(context)
            
            # Log the fallback action
            logger.info(f"Fallback activated: {category} -> {fallback_tool.__name__}")
            
            return response
            
        except Exception as e:
            # If fallback tool fails, use default
            logger.error(f"Fallback tool failed for category {category}: {str(e)}")
            return default_fallback_tool(context)

    def set_fallback_response(self, response: str):
        """Legacy method - sets default fallback response"""
        self.fallback_response = response

    def register_fallback_tool(self, category: str, tool: Callable):
        """
        Register a custom fallback tool for a specific category.
        
        Args:
            category: The violation category to handle
            tool: The fallback tool function
        """
        self.tool_map[category] = tool
        logger.info(f"Registered custom fallback tool for category: {category}")

    def get_available_categories(self) -> list:
        """Get list of all supported violation categories"""
        return list(self.tool_map.keys())

    def analyze_violation(self, violation_results: Dict) -> tuple:
        """
        Analyze guardrail validation results to determine primary violation and context.
        
        Args:
            violation_results: Results from guardrails validation
            
        Returns:
            Tuple of (primary_category, context_dict)
        """
        primary_category = "default"
        context = {}
        
        # Check solo guards first
        if "solo_guards" in violation_results:
            for guard_name, result in violation_results["solo_guards"].items():
                if not result.passed:
                    primary_category = guard_name
                    context = {
                        "guard_name": guard_name,
                        "message": result.message,
                        "details": result.details
                    }
                    logger.warning(f"Guardrail violation: {guard_name}")
                    
                    # Crisis gets highest priority
                    if any(crisis_indicator in guard_name.lower() 
                           for crisis_indicator in ["crisis", "toxic"]):
                        logger.warning(f"Crisis-level violation detected: {guard_name}")
                        break
        
        # Check multiple guards if no solo guard failed
        if primary_category == "default" and "multiple_guards" in violation_results:
            multiple_result = violation_results["multiple_guards"]
            if multiple_result and not multiple_result.passed:
                primary_category = "multiple_guards"
                context = {
                    "guard_type": "multiple",
                    "message": multiple_result.message,
                    "details": multiple_result.details
                }
                logger.warning(f"Multiple guards violation")
        
        return primary_category, context
        