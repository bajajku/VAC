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
            "threshold": 0.81,
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