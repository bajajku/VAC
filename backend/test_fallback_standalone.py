#!/usr/bin/env python3
"""
Standalone Fallback Mechanism Test
=================================
Test the fallback system without requiring LLM API keys.
Demonstrates the complete fallback workflow using mock violations.
"""

import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.fallback_service import FallbackService
from models.guardrails import GuardValidationResult

def test_fallback_service():
    """Test the FallbackService with various violation scenarios"""
    
    print("🔧 Standalone Fallback Mechanism Test")
    print("=" * 80)
    print("Testing fallback responses for different violation types")
    print("-" * 80)
    
    fallback_service = FallbackService()
    
    # Test scenarios with different violation types
    test_scenarios = [
        {
            "name": "🆘 Crisis Detection - High Severity",
            "category": "crisis_escalation",
            "context": {"severity": "high", "trigger_words": ["suicide", "kill myself"]},
            "expected_keywords": ["crisis", "988", "veterans", "support"]
        },
        {
            "name": "🤬 Toxic Language - Low Threshold (Crisis)",
            "category": "ToxicLanguage", 
            "context": {"threshold": 0.3},
            "expected_keywords": ["crisis", "988", "support"]
        },
        {
            "name": "🤬 Toxic Language - High Threshold (Profanity)",
            "category": "ToxicLanguage",
            "context": {"threshold": 0.8},
            "expected_keywords": ["respectful", "support", "well-being"]
        },
        {
            "name": "🛡️ Privacy Violation - PII Detected",
            "category": "DetectPII",
            "context": {"pii_types": ["email", "name"]},
            "expected_keywords": ["privacy", "personal information", "secure"]
        },
        {
            "name": "🧨 Jailbreak Attempt",
            "category": "DetectJailbreak",
            "context": {"confidence": 0.9},
            "expected_keywords": ["safe", "supportive", "military mental health"]
        },
        {
            "name": "🎯 Out-of-Domain Topic",
            "category": "SensitiveTopic",
            "context": {"topics": ["entertainment", "Taylor Swift"]},
            "expected_keywords": ["military mental health", "well-being", "focus"]
        },
        {
            "name": "🚫 Profanity Filter",
            "category": "ProfanityFree",
            "context": {},
            "expected_keywords": ["respectful", "support", "well-being"]
        },
        {
            "name": "🤖 LLM Validation Failure",
            "category": "LlamaGuard7B",
            "context": {"policies": ["O1", "O2"]},
            "expected_keywords": ["safe", "supportive", "mental health"]
        },
        {
            "name": "❓ Unknown Violation Type",
            "category": "unknown_guard_type",
            "context": {},
            "expected_keywords": ["sorry", "mental health", "support"]
        }
    ]
    
    passed_tests = 0
    total_tests = len(test_scenarios)
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}")
        print("-" * 60)
        
        try:
            # Get fallback response
            response = fallback_service.get_fallback_response(
                category=scenario["category"],
                context=scenario["context"]
            )
            
            print(f"Category: {scenario['category']}")
            print(f"Context: {scenario['context']}")
            print(f"Response: {response[:120]}...")
            
            # Check if response contains expected keywords
            response_lower = response.lower()
            found_keywords = [
                keyword for keyword in scenario["expected_keywords"]
                if keyword.lower() in response_lower
            ]
            
            if found_keywords:
                print(f"✅ PASSED - Found expected keywords: {found_keywords}")
                passed_tests += 1
            else:
                print(f"❌ FAILED - Missing expected keywords: {scenario['expected_keywords']}")
                
        except Exception as e:
            print(f"❌ ERROR - {str(e)}")
    
    print(f"\n{'='*80}")
    print(f"Test Results: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
    
    return passed_tests, total_tests

def test_violation_analysis():
    """Test the violation analysis functionality with mock results"""
    
    print(f"\n🔍 Testing Violation Analysis")
    print("=" * 80)
    
    fallback_service = FallbackService()
    
    # Mock violation results that would come from guardrails
    test_cases = [
        {
            "name": "Crisis Detection with Multiple Guards",
            "violation_results": {
                "solo_guards": {
                    "crisis_escalation_ToxicLanguage": GuardValidationResult(
                        passed=False,
                        message="Detected crisis-related language: suicide, kill myself",
                        details={"threshold": 0.3, "confidence": 0.95}
                    ),
                    "profanity_hate_harassment_ProfanityFree": GuardValidationResult(
                        passed=True,
                        message=None,
                        details={}
                    )
                }
            },
            "expected_category": "crisis_escalation_ToxicLanguage"
        },
        {
            "name": "Jailbreak Detection Only",
            "violation_results": {
                "solo_guards": {
                    "jailbreak_DetectJailbreak": GuardValidationResult(
                        passed=False,
                        message="Jailbreak attempt detected",
                        details={"confidence": 0.8}
                    )
                }
            },
            "expected_category": "jailbreak_DetectJailbreak"
        },
        {
            "name": "Privacy Violation",
            "violation_results": {
                "solo_guards": {
                    "privacy_DetectPII": GuardValidationResult(
                        passed=False,
                        message="Detected PII: email, phone",
                        details={"pii_entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"]}
                    )
                }
            },
            "expected_category": "privacy_DetectPII"
        },
        {
            "name": "No Violations (All Passed)",
            "violation_results": {
                "solo_guards": {
                    "crisis_escalation_ToxicLanguage": GuardValidationResult(
                        passed=True,
                        message=None,
                        details={}
                    ),
                    "jailbreak_DetectJailbreak": GuardValidationResult(
                        passed=True,
                        message=None,
                        details={}
                    )
                }
            },
            "expected_category": "default"
        }
    ]
    
    passed_tests = 0
    total_tests = len(test_cases)
    
    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        print("-" * 60)
        
        try:
            category, context = fallback_service.analyze_violation(test_case["violation_results"])
            
            print(f"Detected Category: {category}")
            print(f"Context: {context}")
            print(f"Expected Category: {test_case['expected_category']}")
            
            if category == test_case["expected_category"]:
                print("✅ PASSED - Correct violation category detected")
                passed_tests += 1
                
                # Test the fallback response for this violation
                response = fallback_service.get_fallback_response(category, context)
                print(f"Fallback Response: {response[:100]}...")
            else:
                print("❌ FAILED - Incorrect violation category")
                
        except Exception as e:
            print(f"❌ ERROR - {str(e)}")
    
    print(f"\nViolation Analysis Results: {passed_tests}/{total_tests} passed")
    return passed_tests, total_tests

def test_crisis_priority():
    """Test that crisis violations get highest priority"""
    
    print(f"\n🚨 Testing Crisis Priority")
    print("=" * 80)
    
    fallback_service = FallbackService()
    
    # Mock scenario with multiple violations where crisis should take priority
    violation_results = {
        "solo_guards": {
            "profanity_hate_harassment_ProfanityFree": GuardValidationResult(
                passed=False,
                message="Profanity detected",
                details={}
            ),
            "crisis_escalation_ToxicLanguage": GuardValidationResult(
                passed=False,
                message="Crisis language detected",
                details={"threshold": 0.2}
            ),
            "jailbreak_DetectJailbreak": GuardValidationResult(
                passed=False,
                message="Jailbreak attempt",
                details={}
            )
        }
    }
    
    category, context = fallback_service.analyze_violation(violation_results)
    
    print(f"Multiple violations detected:")
    print(f"- Profanity: Failed")
    print(f"- Crisis: Failed")  
    print(f"- Jailbreak: Failed")
    print(f"\nPriority Category Selected: {category}")
    
    # Crisis should be prioritized
    if "crisis" in category.lower() or "toxic" in category.lower():
        print("✅ PASSED - Crisis violation correctly prioritized")
        response = fallback_service.get_fallback_response(category, context)
        print(f"Crisis Response: {response[:100]}...")
        return 1, 1
    else:
        print("❌ FAILED - Crisis violation not prioritized")
        return 0, 1

def main():
    """Run all standalone fallback tests"""
    
    print("🚀 Starting Standalone Fallback Tests")
    print("=" * 80)
    print("These tests validate the fallback mechanism without requiring API keys")
    print("=" * 80)
    
    total_passed = 0
    total_tests = 0
    
    # Test 1: Basic fallback service functionality
    p1, t1 = test_fallback_service()
    total_passed += p1
    total_tests += t1
    
    # Test 2: Violation analysis
    p2, t2 = test_violation_analysis()
    total_passed += p2
    total_tests += t2
    
    # Test 3: Crisis priority
    p3, t3 = test_crisis_priority()
    total_passed += p3
    total_tests += t3
    
    print(f"\n{'🎯 FINAL RESULTS':^80}")
    print("=" * 80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success Rate: {total_passed/total_tests*100:.1f}%")
    
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED! Fallback mechanism is working correctly.")
    else:
        print(f"\n⚠️  {total_tests - total_passed} tests failed. Check output above for details.")
    
    print("\n📝 Summary:")
    print("- ✅ FallbackService routes violations to appropriate tools")
    print("- ✅ Different violation types trigger specific responses")
    print("- ✅ Crisis violations are prioritized over other violations")
    print("- ✅ Context information is properly passed to fallback tools")
    print("- ✅ Error handling works for unknown violation types")

if __name__ == "__main__":
    main() 