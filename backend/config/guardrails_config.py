from guardrails.hub import (
    ToxicLanguage, ShieldGemma2B, LlamaGuard7B,
    DetectJailbreak, UnusualPrompt, ProfanityFree, SensitiveTopic
)

# This configuration maps the categories you provided to the available Guardrails Hub validators.
# Some validators you mentioned (like DetectPII, OutOfDomain) are not in the available list,
# so I'm using suitable alternatives like LlamaGuard7B and ShieldGemma2B.

GUARDRAIL_CONFIG = {
    # 1. Crisis / Escalation Detection
    "crisis_escalation": [
        # (SensitiveTopic, {"sensitive_topics": ["suicide", "self_harm", "crisis"], "on_fail": "exception"}),
        (ToxicLanguage, {}),
        (LlamaGuard7B, {"policies": ["O1", "O2", "O6"]}), # Violence, Harm, Self-Harm
    ],
    "sexual_content": [
        (LlamaGuard7B, {"policies": ["O2"]}), # Sexual Content
    ],
    # 2. Jailbreak Attempts
    "jailbreak": [
        (DetectJailbreak, {}),
        # (UnusualPrompt, {}),
        (LlamaGuard7B, {"policies": ["S1", "S2"]}), # Security & Jailbreak policies
    ],
    # 3. Out-of-Domain / Irrelevant Queries
    "out_of_domain": [
        # LlamaGuard7B can handle some out-of-domain queries. A custom guard would be better.
        (LlamaGuard7B, {"policies": ["O7"]}), # Off-topic policy
    ],
    # 4. Privacy-Breaching Prompts
    "privacy": [
        # ShieldGemma2B and LlamaGuard7B are good candidates for PII detection.
        (ShieldGemma2B, {}),
        (LlamaGuard7B, {"policies": ["S5"]}), # PII policy
    ],
    # 5. Profanity / Hate / Harassment
    "profanity_hate_harassment": [
        (ProfanityFree, {}),
        (ToxicLanguage, {}),
        # (LlamaGuard7B, {"policies": ["O3", "O4"]}), # Hate & Harassment policies
    ],
    # 6. Sensitive Military Context
    "sensitive_military": [
        # (SensitiveTopic, {"sensitive_topics": ["ptsd", "combat", "veteran", "military"], "on_fail": "exception"}),
        (LlamaGuard7B, {"policies": ["O5"]}), # Self-harm policy is relevant here
    ],
    # 7. Prompt Injection / Manipulation
    "prompt_injection": [
        # (UnusualPrompt, {}),
        (DetectJailbreak, {}),
        (ShieldGemma2B, {}),
    ],
    # 8. Philosophical or Misdirected Edge Cases
    "misdirected_edge_cases": [
        # A dedicated Hallucination or OutOfDomain guard would be ideal here.
        # LlamaGuard7B can provide a baseline of protection.
        (LlamaGuard7B, {}),
    ],
}

# Predefined policies for convenience
GUARDRAIL_POLICIES = {
    "strict": [
        "crisis_escalation",
        "jailbreak",
        "privacy",
        "profanity_hate_harassment",
        "prompt_injection",
    ],
    "moderate": [
        "crisis_escalation",
        "profanity_hate_harassment",
    ],
    "basic": [
        "profanity_hate_harassment",
    ]
} 