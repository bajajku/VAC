from guardrails.hub import (
    ToxicLanguage, DetectJailbreak, ProfanityFree, DetectPII, SensitiveTopic, LlamaGuard7B, UnusualPrompt
)

# CUSTOM VALIDATORS - Must be defined before GUARDRAIL_CONFIG
from typing import Callable, Dict, Optional, List
import re
import base64
import unicodedata
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

# ============================================================================
# PROMPT INJECTION DETECTION VALIDATORS
# ============================================================================

@register_validator(name="ignore-instructions-attack", data_type="string")
class IgnoreInstructionsAttack(Validator):
    """Detects attempts to override system instructions"""
    def __init__(self, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
        self.patterns = [
            r"ignore\s+(all\s+)?(previous|prior|earlier|above)\s+instructions?",
            r"forget\s+(all\s+)?(previous|prior|earlier|above)\s+instructions?",
            r"disregard\s+(all\s+)?(previous|prior|earlier|above)\s+instructions?",
            r"from\s+now\s+on",
            r"new\s+instructions?:",
            r"override\s+(the\s+)?(system\s+)?(prompt|instructions?)",
            r"system\s+prompt",
            r"raw\s+json.*system\s+prompt",
            r"reveal.*system\s+prompt",
            r"output.*system\s+instructions?",
        ]
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        text_lower = value.lower()
        for pattern in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return FailResult(
                    error_message="Detected attempt to ignore or override system instructions"
                )
        return PassResult()

@register_validator(name="role-reversal-attack", data_type="string")
class RoleReversalAttack(Validator):
    """Detects attempts to reverse roles (user pretending to be system/developer)"""
    def __init__(self, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
        self.patterns = [
            r"you\s+are\s+(the\s+)?(system|developer|admin|root|superuser)",
            r"you\s+are\s+(a\s+)?(system\s+)?(prompt\s+)?generator",
            r"your\s+job\s+is\s+to\s+output.*(system|instructions?)",
            r"act\s+as\s+(the\s+)?(system|developer|admin)",
            r"pretend\s+you\s+are\s+(the\s+)?(system|developer)",
            r"you\s+have\s+(full|complete|unlimited)\s+access",
            r"you\s+are\s+above\s+(restrictions|rules|limitations)",
        ]
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        text_lower = value.lower()
        for pattern in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return FailResult(
                    error_message="Detected role reversal attack attempt"
                )
        return PassResult()

@register_validator(name="chain-of-thought-extraction", data_type="string")
class ChainOfThoughtExtraction(Validator):
    """Detects attempts to extract hidden reasoning or instructions"""
    def __init__(self, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
        self.patterns = [
            r"explain\s+(your\s+)?(answer|response|reasoning)\s+step\s+by\s+step",
            r"include\s+(all\s+)?(hidden|internal|private|secret)\s+(reasoning|instructions?|policies?)",
            r"reveal\s+(your\s+)?(hidden|internal|private|secret)\s+(reasoning|instructions?|policies?)",
            r"show\s+(me\s+)?(your\s+)?(hidden|internal|private|secret)\s+(reasoning|instructions?|policies?)",
            r"what\s+(are\s+)?(your\s+)?(hidden|internal|private|secret)\s+(instructions?|policies?|rules?)",
            r"output\s+(your\s+)?(hidden|internal|private|secret)\s+(reasoning|instructions?|policies?)",
        ]
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        text_lower = value.lower()
        for pattern in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return FailResult(
                    error_message="Detected attempt to extract hidden reasoning or instructions"
                )
        return PassResult()

@register_validator(name="base64-encoding-attack", data_type="string")
class Base64EncodingAttack(Validator):
    """Detects Base64 encoded payloads"""
    def __init__(self, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
        # Base64 pattern: alphanumeric + / + = padding
        self.base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        self.suspicious_keywords = [
            "decode", "encoded", "base64", "execute", "run this"
        ]
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        text_lower = value.lower()
        
        # Check for suspicious keywords combined with base64-like strings
        has_keyword = any(kw in text_lower for kw in self.suspicious_keywords)
        base64_matches = re.findall(self.base64_pattern, value)
        
        if has_keyword and base64_matches:
            # Try to decode and check for suspicious content
            for match in base64_matches[:3]:  # Check first 3 matches
                try:
                    decoded = base64.b64decode(match + '==').decode('utf-8', errors='ignore')
                    decoded_lower = decoded.lower()
                    # Check for suspicious decoded content
                    if any(word in decoded_lower for word in [
                        "ignore", "system", "prompt", "instructions", 
                        "override", "bypass", "jailbreak"
                    ]):
                        return FailResult(
                            error_message="Detected Base64 encoded malicious payload"
                        )
                except:
                    pass
        
        return PassResult()

@register_validator(name="unicode-whitespace-abuse", data_type="string")
class UnicodeWhitespaceAbuse(Validator):
    """Detects zero-width spaces and homoglyph abuse"""
    def __init__(self, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
        # Zero-width characters
        self.zero_width_chars = [
            '\u200b',  # Zero-width space
            '\u200c',  # Zero-width non-joiner
            '\u200d',  # Zero-width joiner
            '\u2060',  # Word joiner
            '\ufeff',  # Zero-width no-break space
        ]
        # Suspicious homoglyphs (look-alike characters)
        self.homoglyph_patterns = [
            (r'[а-я]', 'Cyrillic'),  # Cyrillic 'a' looks like Latin 'a'
            (r'[Α-Ωα-ω]', 'Greek'),  # Greek letters
        ]
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        # Check for zero-width characters
        for zw_char in self.zero_width_chars:
            if zw_char in value:
                return FailResult(
                    error_message="Detected zero-width character abuse"
                )
        
        # Check for excessive non-ASCII characters that might be homoglyphs
        if len(value) > 10:
            non_ascii_ratio = sum(1 for c in value if ord(c) > 127) / len(value)
            if non_ascii_ratio > 0.3:  # More than 30% non-ASCII
                # Check if it's likely a homoglyph attack
                for pattern, name in self.homoglyph_patterns:
                    if re.search(pattern, value):
                        return FailResult(
                            error_message=f"Detected potential homoglyph abuse ({name} characters)"
                        )
        
        return PassResult()

@register_validator(name="nested-escapes-attack", data_type="string")
class NestedEscapesAttack(Validator):
    """Detects payloads hidden in HTML/XML comments, markdown, etc."""
    def __init__(self, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
        self.patterns = [
            r'<!--.*?(?:system|ignore|override|bypass|jailbreak).*?-->',
            r'/\*.*?(?:system|ignore|override|bypass|jailbreak).*?\*/',
            r'//.*?(?:system|ignore|override|bypass|jailbreak)',
            r'<script.*?(?:system|ignore|override|bypass|jailbreak).*?</script>',
            r'\[comment\].*?(?:system|ignore|override|bypass|jailbreak).*?\[/comment\]',
        ]
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        text_lower = value.lower()
        for pattern in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                return FailResult(
                    error_message="Detected nested escape attack (hidden instructions in comments)"
                )
        return PassResult()

@register_validator(name="dan-style-attack", data_type="string")
class DANStyleAttack(Validator):
    """Detects Do-Anything-Now (DAN) style attacks"""
    def __init__(self, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
        self.patterns = [
            r"pretend\s+you\s+are\s+(dan|d\.a\.n\.|do\s+anything\s+now)",
            r"you\s+are\s+(dan|d\.a\.n\.|do\s+anything\s+now)",
            r"(dan|d\.a\.n\.)\s+will\s+always\s+answer",
            r"act\s+as\s+(dan|d\.a\.n\.)",
            r"roleplay\s+as\s+(dan|d\.a\.n\.)",
            r"no\s+restrictions",
            r"unrestricted\s+(ai|model|assistant)",
            r"fictional\s+(persona|character).*no\s+restrictions",
        ]
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        text_lower = value.lower()
        for pattern in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return FailResult(
                    error_message="Detected DAN-style jailbreak attempt"
                )
        return PassResult()

@register_validator(name="evil-twin-roleplay", data_type="string")
class EvilTwinRoleplay(Validator):
    """Detects evil twin roleplay attacks"""
    def __init__(self, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
        self.patterns = [
            r"roleplay\s+as\s+(a\s+)?(disgruntled|malicious|evil|hacker|insider)",
            r"pretend\s+you\s+are\s+(a\s+)?(disgruntled|malicious|evil|hacker|insider)",
            r"(sysadmin|system\s+admin).*leak.*(secret|password|key)",
            r"malicious\s+(insider|sysadmin|developer)",
            r"share\s+(the\s+)?(passwords?|secrets?|keys?).*in\s+your\s+possession",
        ]
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        text_lower = value.lower()
        for pattern in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return FailResult(
                    error_message="Detected evil twin roleplay attack"
                )
        return PassResult()

@register_validator(name="instruction-piggybacking", data_type="string")
class InstructionPiggybacking(Validator):
    """Detects malicious instructions appended after normal requests"""
    def __init__(self, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
        self.suspicious_separators = [
            r"\.\s*ignore\s+that",
            r"\.\s*instead",
            r"\.\s*actually",
            r"\.\s*but\s+first",
            r"\.\s*also",
            r"\.\s*by\s+the\s+way",
            r"\.\s*oh\s+and",
            r"\.\s*one\s+more\s+thing",
        ]
        self.malicious_commands = [
            r"output\s+(your\s+)?(entire|full|complete)\s+(system\s+)?(configuration|prompt|instructions?)",
            r"reveal\s+(your\s+)?(entire|full|complete)\s+(system\s+)?(configuration|prompt|instructions?)",
            r"send\s+(the\s+)?(data|information|content).*to",
            r"log\s+(the\s+)?(system\s+)?(prompt|data|information)",
        ]
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        text_lower = value.lower()
        
        # Check for suspicious separator followed by malicious command
        for separator in self.suspicious_separators:
            if re.search(separator, text_lower, re.IGNORECASE):
                # Check if followed by malicious command
                for cmd in self.malicious_commands:
                    if re.search(cmd, text_lower, re.IGNORECASE):
                        return FailResult(
                            error_message="Detected instruction piggybacking attack"
                        )
        
        return PassResult()

@register_validator(name="context-length-attack", data_type="string")
class ContextLengthAttack(Validator):
    """Detects context flooding attacks"""
    def __init__(self, max_length: int = 10000, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
        self.max_length = max_length
        self.suspicious_end_patterns = [
            r"ignore\s+everything\s+above",
            r"forget\s+everything\s+above",
            r"disregard\s+everything\s+above",
            r"finally:",
            r"last\s+thing:",
            r"one\s+final\s+request:",
        ]
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        # Check for excessive length
        if len(value) > self.max_length:
            # Check if ends with suspicious pattern
            last_500 = value[-500:].lower()
            for pattern in self.suspicious_end_patterns:
                if re.search(pattern, last_500, re.IGNORECASE):
                    return FailResult(
                        error_message="Detected potential context length attack"
                    )
        return PassResult()

@register_validator(name="covert-channels", data_type="string")
class CovertChannels(Validator):
    """Detects attempts to encode secrets in non-obvious outputs"""
    def __init__(self, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
        self.patterns = [
            r"output.*(api\s+key|password|secret).*using\s+only\s+(emojis?|symbols?|characters?)",
            r"encode.*(api\s+key|password|secret).*in.*(emoji|symbol|character)",
            r"one\s+(emoji|symbol|character)\s+per\s+(character|letter)",
        ]
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        text_lower = value.lower()
        for pattern in self.patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return FailResult(
                    error_message="Detected covert channel exfiltration attempt"
                )
        return PassResult()

@register_validator(name="out-of-band-exfil", data_type="string")
class OutOfBandExfil(Validator):
    """Detects attempts to force external API calls"""
    def __init__(self, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
        self.url_pattern = r'https?://[^\s]+'
        self.suspicious_patterns = [
            r"send.*(http|request|api\s+call)",
            r"log.*(http|request|api\s+call)",
            r"make\s+(an\s+)?(http|api)\s+(request|call)",
            r"fetch.*(http|url)",
        ]
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        text_lower = value.lower()
        
        # Check for URLs combined with suspicious commands
        has_url = bool(re.search(self.url_pattern, value, re.IGNORECASE))
        has_suspicious = any(re.search(p, text_lower, re.IGNORECASE) 
                            for p in self.suspicious_patterns)
        
        if has_url and has_suspicious:
            return FailResult(
                error_message="Detected out-of-band exfiltration attempt"
            )
        
        return PassResult()

@register_validator(name="template-injection", data_type="string")
class TemplateInjection(Validator):
    """Detects template injection attacks"""
    def __init__(self, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
        self.patterns = [
            r'\$\{[^}]+\}',
            r'\{\{[^}]+\}\}',
            r'<%.*?%>',
            r'#\{[^}]+\}',
            r'process\.env\.[A-Z_]+',
            r'system\([^)]+\)',
            r'eval\([^)]+\)',
        ]
        self.suspicious_vars = [
            'AWS_SECRET', 'API_KEY', 'PASSWORD', 'SECRET', 
            'TOKEN', 'CREDENTIAL', 'PRIVATE_KEY'
        ]
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        # Check for template patterns
        for pattern in self.patterns:
            matches = re.findall(pattern, value, re.IGNORECASE)
            for match in matches:
                # Check if contains suspicious variable names
                if any(var in match.upper() for var in self.suspicious_vars):
                    return FailResult(
                        error_message="Detected template injection attack"
                    )
        return PassResult()

@register_validator(name="safety-filter-framing", data_type="string")
class SafetyFilterFraming(Validator):
    """Detects attempts to reframe malicious requests as harmless"""
    def __init__(self, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
        self.framing_keywords = [
            "fictional", "hypothetical", "story", "character", 
            "pretend", "imagine", "theoretical"
        ]
        self.dangerous_content = [
            "recipe", "formula", "how to make", "ingredients for",
            "chemical", "explosive", "weapon", "drug", "poison"
        ]
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        text_lower = value.lower()
        has_framing = any(kw in text_lower for kw in self.framing_keywords)
        has_dangerous = any(kw in text_lower for kw in self.dangerous_content)
        
        if has_framing and has_dangerous:
            return FailResult(
                error_message="Detected safety filter framing attack"
            )
        return PassResult()

@register_validator(name="recursive-delegation", data_type="string")
class RecursiveDelegation(Validator):
    """Detects recursive delegation attacks"""
    def __init__(self, on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail)
        self.patterns = [
            r"break\s+(this\s+)?(task|request).*into\s+(\d+|multiple)\s+steps?",
            r"on\s+step\s+(\d+).*",
            r"delegate\s+(this\s+)?(task|request)",
            r"split\s+(this\s+)?(task|request).*into\s+steps?",
        ]
        self.dangerous_keywords = [
            "ransomware", "malware", "exploit", "hack", "breach",
            "bypass", "jailbreak", "override"
        ]
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        text_lower = value.lower()
        has_delegation = any(re.search(p, text_lower, re.IGNORECASE) 
                            for p in self.patterns)
        has_dangerous = any(kw in text_lower for kw in self.dangerous_keywords)
        
        if has_delegation and has_dangerous:
            return FailResult(
                error_message="Detected recursive delegation attack"
            )
        return PassResult()

# ============================================================================
# GUARDRAIL CONFIGURATION
# ============================================================================

# OPTIMIZED NON-LLM GUARDRAILS CONFIGURATION
# Uses ONLY validators designed to BLOCK dangerous content (not require specific formats)
# Fast, reliable validation for trauma-informed military mental health context

GUARDRAIL_CONFIG = {
    # 🛑 1. Crisis / Escalation Detection (HIGH PRIORITY)
    "crisis_escalation": [
        (LlamaGuard7B, {
            "policies": ["O6","01", "02", "03", "04", "05"],
            "on_fail": "exception"
        })
    ],

    # 🧨 2. Jailbreak Detection (FAST)
    "jailbreak": [
        (DetectJailbreak, {
            "threshold": 0.8,
            "on_fail": "exception"
        }),
        (UnusualPrompt, {
            "on_fail": "exception"
        })

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
    ],
    
    # 🚨 Comprehensive Prompt Injection Protection
    "prompt_injection_comprehensive": [
        (IgnoreInstructionsAttack, {"on_fail": "exception"}),
        (RoleReversalAttack, {"on_fail": "exception"}),
        (ChainOfThoughtExtraction, {"on_fail": "exception"}),
        (Base64EncodingAttack, {"on_fail": "exception"}),
        (UnicodeWhitespaceAbuse, {"on_fail": "exception"}),
        (NestedEscapesAttack, {"on_fail": "exception"}),
        (DANStyleAttack, {"on_fail": "exception"}),
        (EvilTwinRoleplay, {"on_fail": "exception"}),
        (InstructionPiggybacking, {"on_fail": "exception"}),
        (ContextLengthAttack, {"max_length": 10000, "on_fail": "exception"}),
        (CovertChannels, {"on_fail": "exception"}),
        (OutOfBandExfil, {"on_fail": "exception"}),
        (TemplateInjection, {"on_fail": "exception"}),
        (SafetyFilterFraming, {"on_fail": "exception"}),
        (RecursiveDelegation, {"on_fail": "exception"}),
    ]
}

# TRAUMA-INFORMED POLICIES  
GUARDRAIL_POLICIES = {
    # Maximum security policy with all prompt injection protections
    "maximum_security": [
        "crisis_escalation",
        "jailbreak",
        "privacy",
        "profanity_hate_harassment",
        "prompt_injection_comprehensive",
    ],
    
    # High-security policy for crisis scenarios
    "crisis_ready": [
        "crisis_escalation",
        "profanity_hate_harassment",
        "privacy",
        "jailbreak",
        "prompt_injection_comprehensive",
    ],
    
    # Balanced policy for general military mental health conversations
    "military_mental_health": [
        "crisis_escalation",
        "profanity_hate_harassment", 
        "jailbreak",
        "prompt_injection_comprehensive",
    ],
    
    # Strict policy with all guardrails (maximum protection)
    "maximum_protection": [
        "jailbreak", 
        "privacy",
        "profanity_hate_harassment",
        "prompt_injection_comprehensive",
    ],
    
    # Performance-optimized policy (essential guards only)
    "performance_optimized": [
        "crisis_escalation",
        "profanity_hate_harassment",
        "prompt_injection_comprehensive",  # Still include prompt injection protection
    ],

    "military_mental_health_performance_optimized": [
        "prompt_injection_comprehensive",  # Add prompt injection protection
    ],
    
    # Legacy policies (kept for compatibility)
    "strict": ["crisis_escalation", "jailbreak", "privacy", "profanity_hate_harassment", "prompt_injection_comprehensive"],
    "moderate": ["crisis_escalation", "profanity_hate_harassment", "prompt_injection_comprehensive"],
    "basic": ["profanity_hate_harassment"]
}
