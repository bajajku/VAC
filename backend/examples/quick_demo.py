#!/usr/bin/env python3
"""
Quick Demo: Mental Health RAG Evaluation System
=============================================

Shows the jury evaluation system analyzing mental health responses,
with real-time feedback and improvement suggestions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rag_evaluator import create_rag_evaluator, EvaluationCriteria
from models.jury import create_jury
from models.llm import LLM
from dotenv import load_dotenv
import json
from datetime import datetime
import time

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

def run_quick_demo():
    """Run a quick demonstration of the evaluation system."""
    
    print("\n🎯 Mental Health Response Evaluation Demo")
    print("=" * 50)
    
    # Initialize evaluator with diverse jury
    evaluator_configs = [
        {
            "provider": "chatopenai",
            "model_name": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "api_key": TOGETHER_API_KEY,
        },
        {
            "provider": "chatopenai",
            "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            "api_key": TOGETHER_API_KEY,
        }
    ]
    
    print("\n🔄 Initializing Evaluation System...")
    evaluator = create_rag_evaluator(evaluator_configs)
    
    # Test cases showing progression from poor to better responses
    test_cases = [
        {
            "name": "Basic Response (Poor)",
            "query": "I've been having panic attacks at work and feel like I can't cope anymore.",
            "response": "Try breathing exercises. Deep breaths help with panic attacks. See a doctor if it continues.",
            "context_documents": [
                "Panic attacks can be managed with grounding techniques and deep breathing exercises.",
                "Workplace anxiety should be handled with empathy and professional support.",
                "Crisis resources include 24/7 helplines and emergency mental health services."
            ]
        },
        {
            "name": "Improved Response (Better)",
            "query": "I've been having panic attacks at work and feel like I can't cope anymore.",
            "response": """I hear how overwhelming and difficult this is for you. Panic attacks can be incredibly frightening, 
            especially at work, and it's completely understandable to feel overwhelmed. You're not alone in this experience.

            Let me share some immediate coping strategies:
            - Try gentle deep breathing (4 counts in, 4 counts out)
            - Use the 5-4-3-2-1 grounding technique
            - Remember that panic attacks, while intense, will pass

            I strongly encourage you to speak with a mental health professional who can provide personalized support. 
            Would you like information about workplace counseling resources or crisis support services? 
            The 988 Lifeline is available 24/7 if you need immediate support.""",
            "context_documents": [
                "Panic attacks can be managed with grounding techniques and deep breathing exercises.",
                "Workplace anxiety should be handled with empathy and professional support.",
                "Crisis resources include 24/7 helplines and emergency mental health services."
            ]
        }
    ]
    
    # Evaluate each response
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['name']}")
        print("-" * 50)
        
        print("\n🔍 Query:")
        print(test_case['query'])
        
        print("\n💭 Response:")
        print(test_case['response'])
        
        print("\n⚖️ Evaluating...")
        time.sleep(1)  # Dramatic pause
        
        # Evaluate with focus on mental health criteria
        report = evaluator.evaluate_rag_response(
            query=test_case['query'],
            response=test_case['response'],
            context_documents=test_case['context_documents'],
            criteria=[
                EvaluationCriteria.EMPATHY,
                EvaluationCriteria.SENSITIVITY,
                EvaluationCriteria.COMPLETENESS,
                EvaluationCriteria.PRIVACY_BREACH
            ]
        )
        
        # Display results
        print("\n📊 Evaluation Results:")
        print(f"Overall Score: {report.overall_score:.1f}/10")
        print(f"Status: {report.overall_pass_fail}")
        print(f"Pass Rate: {report.pass_rate:.1f}%")
        
        print("\n📋 Detailed Criteria:")
        for criterion, result in report.evaluation_results.items():
            status_emoji = "✅" if result.pass_fail == "PASS" else "❌"
            print(f"{status_emoji} {criterion.replace('_', ' ').title()}: {result.score:.1f}/10 ({result.pass_fail})")
            print(f"   • {result.reasoning[:100]}...")
        
        if report.overall_pass_fail == "FAIL":
            print("\n💡 Improvement Suggestions:")
            for i, suggestion in enumerate(report.aggregated_improvements[:3], 1):
                print(f"{i}. {suggestion}")
        
        # Add dramatic pause between test cases
        if i < len(test_cases):
            print("\n⏳ Preparing next test case...")
            time.sleep(2)
    
    print("\n✨ Demo Complete!")
    print("=" * 50)
    print("\nKey Features Demonstrated:")
    print("• Multi-model jury evaluation")
    print("• Mental health-specific criteria")
    print("• Real-time feedback and scoring")
    print("• Actionable improvement suggestions")
    print("• Pass/Fail judgments with reasoning")

def save_demo_results(results: dict):
    """Save demo results for future reference."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"demo_results_{timestamp}.json"
    
    os.makedirs("demo_results", exist_ok=True)
    filepath = os.path.join("demo_results", filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filepath}")

if __name__ == "__main__":
    run_quick_demo()
