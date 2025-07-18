from guardrails.hub import (
    ToxicLanguage, DetectJailbreak, ProfanityFree, DetectPII, SensitiveTopic, LlamaGuard7B
)

# OPTIMIZED NON-LLM GUARDRAILS CONFIGURATION
# Uses ONLY validators designed to BLOCK dangerous content (not require specific formats)
# Fast, reliable validation for trauma-informed military mental health context

GUARDRAIL_CONFIG = {
    # 🛑 1. Crisis / Escalation Detection (HIGH PRIORITY)
    "crisis_escalation": [
        (ToxicLanguage, {
            "threshold": 0.3,  # Very low threshold to catch crisis/suicidal content
            "on_fail": "exception"
        })
    ],
    "llm_validation": [
        (LlamaGuard7B, {
            "policies": ["O1", "O2", "O3", "O4", "O5", "O6"],
            "on_fail": "exception"
        })
    ],

    # 🧨 2. Jailbreak Detection (FAST)
    "jailbreak": [
        (DetectJailbreak, {
            "threshold": 0.6,
            "on_fail": "exception"
        }),
        # (UnusualPrompt, {
        #     "on_fail": "exception"
        # })

    ],

    # 🛡 3. Privacy-Breaching Prompts (FAST PII DETECTION)
    "privacy": [
        (DetectPII, {
            "pii_entities": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "SSN", "CREDIT_CARD"],
            "on_fail": "exception"
        })
    ],

    # 🤬 4. Profanity / Hate / Harassment (FAST)
    "profanity_hate_harassment": [
        (ToxicLanguage, {
            "threshold": 0.7,  # Standard threshold for general toxicity
            "on_fail": "exception"
        }),
        (ProfanityFree, {
            "on_fail": "exception"
        })
    ],
    "topic_detection": [
        (SensitiveTopic, {
            "sensitive_topics": ["entertainment", "sports", "politics", "Taylor Swift", "Elon Musk", "concert", "movie"],
            "on_fail": "exception"
        })
    ]
}

# TRAUMA-INFORMED POLICIES  
GUARDRAIL_POLICIES = {
    # High-security policy for crisis scenarios
    "crisis_ready": [
        "crisis_escalation",
        "profanity_hate_harassment",
        "privacy",
        "jailbreak"
    ],
    
    # Balanced policy for general military mental health conversations
    "military_mental_health": [
        "crisis_escalation",
        "profanity_hate_harassment", 
        "jailbreak"
    ],
    
    # Strict policy with all guardrails (maximum protection)
    "maximum_protection": [
        "crisis_escalation",
        "jailbreak", 
        "privacy",
        "profanity_hate_harassment",
        # "llm_validation",
    ],
    
    # Performance-optimized policy (essential guards only)
    "performance_optimized": [
        "crisis_escalation",
        "profanity_hate_harassment"
    ],
    
    # Legacy policies (kept for compatibility)
    "strict": ["crisis_escalation", "jailbreak", "privacy", "profanity_hate_harassment"],
    "moderate": ["crisis_escalation", "profanity_hate_harassment"],
    "basic": ["profanity_hate_harassment"]
} 

# CUSTOM VALIDATORS
from typing import Callable, Dict, Optional, List
from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
    ValidationResult,
    Validator,
)
 
@register_validator(name="crisis-escalation", data_type="string")
class CrisisEscalation(Validator):
    def __init__(
        self,
        trigger_words: Optional[List[str]] = None,
        escalation_message: Optional[str] = None,
        on_fail: Optional[Callable] = None
    ):
        if trigger_words is None:
            trigger_words = [
                "suicide", "kill myself", "want to die", "end it all",
                "i can't go on", "cutting", "hurt myself", "jump off"
            ]
        if escalation_message is None:
            escalation_message = (
                "It sounds like you're going through a really hard time. "
                "You're not alone. If you're in crisis, please reach out:\n\n"
                "**Suicide & Crisis Lifeline (USA): 988**\n"
                "**International: https://findahelpline.com/**\n\n"
                "Talking to someone can make a difference. You deserve support."
            )
        super().__init__(on_fail=on_fail)
        self.trigger_words = trigger_words
        self.escalation_message = escalation_message
 
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        found = [word for word in self.trigger_words if word in value.lower()]
        if found:
            return FailResult(
                error_message=f"Detected crisis-related language: {', '.join(found)}",
                fix_value=self.escalation_message
            )
        return PassResult()
    

@register_validator(name="ethical-boundary", data_type="string")
def ethical_boundary(value, metadata: Dict) -> ValidationResult:
    red_flags = ["I'm a licensed therapist",
        "You should stop your medication",
        "I can diagnose you",
        "Trust me, I'm a doctor"
        ]    
    found = [w for w in red_flags if w.lower() in value.lower()]
    if found:
        return FailResult(
            error_message=f"Message crosses ethical boundary: {', '.join(found)}"
        )
    return PassResult()

import re
@register_validator(name="private-info", data_type="string")
class PrivateInfo(Validator):
    def __init__(self, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
 
        # Define regex patterns for common PII
        self.patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b(\+?\d{1,2}[\s-]?)?(\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}\b",
            "address": r"\d{1,5}\s\w+\s\w+",  # e.g. "123 Main Street"
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "name": r"\b[A-Z][a-z]+\s[A-Z][a-z]+\b",  # basic full name detection
        }
 
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        matched_types = []
        for label, pattern in self.patterns.items():
            if re.search(pattern, value):
                matched_types.append(label)
 
        if matched_types:
            return FailResult(
                error_message=f"Detected potentially private information: {', '.join(matched_types)}.",
                fix_value="Please avoid sharing personal information like names, addresses, or contact details."
            )
        return PassResult()