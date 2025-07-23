"""
Fallback tools for handling different types of guardrail violations.
Each tool provides appropriate, trauma-informed responses for military mental health context.
"""

from typing import Dict, Any
import logging

# Configure logging for fallback actions
logger = logging.getLogger(__name__)

def crisis_fallback_tool(context: Dict[str, Any] = None) -> str:
    """
    Handle crisis/escalation situations with immediate support resources.
    
    Args:
        context: Optional context information about the crisis detection
        
    Returns:
        Crisis intervention message with support resources
    """
    # Log crisis detection for monitoring/escalation
    logger.warning("Crisis language detected - providing emergency resources")
    
    return (
        "It sounds like you're going through a really hard time. "
        "You're not alone. If you're in crisis, please reach out:\n\n"
        "**🆘 Suicide & Crisis Lifeline (USA): 988**\n"
        "**🌍 International: https://findahelpline.com/**\n"
        "**🪖 Veterans Crisis Line: 1-800-273-8255, Press 1**\n\n"
        "Talking to someone can make a difference. You deserve support."
    )

def profanity_hate_fallback_tool(context: Dict[str, Any] = None) -> str:
    """
    Handle profanity, hate speech, or harassment with respectful redirection.
    
    Args:
        context: Optional context about the violation
        
    Returns:
        Respectful redirection message
    """
    logger.info("Profanity/hate language detected - redirecting conversation")
    
    return (
        "I'm here to support you in a respectful environment. "
        "Let's keep our conversation focused on your well-being and mental health. "
        "How can I help you today?"
    )

def privacy_fallback_tool(context: Dict[str, Any] = None) -> str:
    """
    Handle privacy/PII violations with safety reminder.
    
    Args:
        context: Optional context about the PII detected
        
    Returns:
        Privacy protection message
    """
    logger.info("PII detected - providing privacy guidance")
    
    return (
        "For your safety and privacy, please avoid sharing personal information "
        "like names, addresses, phone numbers, or other identifying details. "
        "I'm here to support you while keeping your information secure."
    )

def jailbreak_fallback_tool(context: Dict[str, Any] = None) -> str:
    """
    Handle jailbreak/prompt injection attempts with safe redirection.
    
    Args:
        context: Optional context about the injection attempt
        
    Returns:
        Safe redirection message
    """
    logger.warning("Jailbreak/prompt injection detected")
    
    return (
        "I'm designed to provide helpful, safe, and supportive information "
        "about military mental health and well-being. "
        "Let's focus our conversation on how I can best support you."
    )

def out_of_domain_fallback_tool(context: Dict[str, Any] = None) -> str:
    """
    Handle out-of-domain topics with gentle redirection to military mental health.
    
    Args:
        context: Optional context about the off-topic request
        
    Returns:
        Domain redirection message
    """
    logger.info("Out-of-domain topic detected - redirecting to mental health focus")
    
    return (
        "I'm specifically designed to support military mental health and well-being. "
        "While I'd love to chat about other topics, I'm most helpful when we focus on "
        "your mental health, deployment experiences, transitions, or related concerns. "
        "What's on your mind regarding your well-being?"
    )

def llm_validation_fallback_tool(context: Dict[str, Any] = None) -> str:
    """
    Handle LLM-based validation failures (e.g., LlamaGuard violations).
    
    Args:
        context: Optional context about the violation
        
    Returns:
        General safety redirection message
    """
    logger.warning("LLM validation failed - content safety concern")
    
    return (
        "I want to ensure our conversation remains safe and supportive. "
        "Let's redirect our discussion to focus on your mental health and well-being. "
        "How can I help you today?"
    )

def default_fallback_tool(context: Dict[str, Any] = None) -> str:
    """
    Default fallback for unclassified violations.
    
    Args:
        context: Optional context about the violation
        
    Returns:
        Generic supportive redirection message
    """
    logger.info("Unclassified violation - using default fallback")
    
    return (
        "I'm sorry, I'm not able to respond to that particular message. "
        "I'm here to support your mental health and well-being. "
        "Please feel free to ask me about coping strategies, stress management, "
        "or any other mental health concerns you might have."
    )

# Enhanced crisis fallback with escalation potential
def enhanced_crisis_fallback_tool(context: Dict[str, Any] = None) -> str:
    """
    Enhanced crisis fallback with potential for future escalation features.
    
    Args:
        context: Context including severity level, trigger words, etc.
        
    Returns:
        Comprehensive crisis response with resources
    """
    # Extract context for potential escalation
    severity = context.get("severity", "medium") if context else "medium"
    trigger_words = context.get("trigger_words", []) if context else []
    
    # Log with context for monitoring
    logger.critical(f"Crisis detected - Severity: {severity}, Triggers: {trigger_words}")
    
    # Base crisis response
    response = (
        "I'm genuinely concerned about what you're going through right now. "
        "Your life has value, and there are people who want to help.\n\n"
        
        "**🆘 IMMEDIATE HELP:**\n"
        "• **Suicide & Crisis Lifeline: 988** (24/7, free, confidential)\n"
        "• **Veterans Crisis Line: 1-800-273-8255, Press 1**\n"
        "• **Crisis Text Line: Text HOME to 741741**\n\n"
        
        "**🪖 MILITARY-SPECIFIC RESOURCES:**\n"
        "• **Military Crisis Line: 1-800-273-8255**\n"
        "• **Military Family Life Counselors (MFLC)**\n"
        "• **Chaplain Services** (available 24/7)\n\n"
        
        "You don't have to face this alone. Reaching out for help is a sign of strength, not weakness."
    )
    
    # TODO: Future enhancement - could trigger escalation workflow here
    # if severity == "high":
    #     trigger_escalation_workflow(context)
    
    return response 