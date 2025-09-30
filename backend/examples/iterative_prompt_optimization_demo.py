#!/usr/bin/env python3
"""
Iterative Prompt Optimization Demo
==================================

Demonstrates how to use evaluation feedback to automatically improve system prompts.
Shows the complete cycle: Evaluate → Get Suggestions → Optimize → Re-evaluate
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rag_evaluator import create_rag_evaluator, EvaluationCriteria
from models.prompt_optimizer import create_prompt_optimizer, create_evaluation_function
from models.evaluation_pipeline import TestCaseGenerator
from models.llm import LLM
from models.rag_agent import RAGAgent
import json


def demo_iterative_optimization():
    """Demonstrate iterative prompt optimization using evaluation feedback."""
    
    print("🔄 Iterative Prompt Optimization Demo")
    print("=" * 50)
    
    # Sample API keys (replace with real ones for actual testing)
    api_keys = {
        'openai': 'your-openai-key-here',
        'openrouter': 'your-openrouter-key-here'
    }
    
    try:
        # Create evaluator
        evaluator_configs = [
            {'provider': 'openai', 'model_name': 'gpt-3.5-turbo', 'api_key': api_keys.get('openai')},
            {'provider': 'openrouter', 'model_name': 'mistral-7b-instruct', 'api_key': api_keys.get('openrouter')},
        ]
        evaluator = create_rag_evaluator(evaluator_configs)
        
        # Create prompt optimizer
        optimizer_llm = LLM(provider='openai', model_name='gpt-4', api_key=api_keys.get('openai'))
        optimizer = create_prompt_optimizer(
            optimizer_llm=optimizer_llm,
            max_iterations=3,
            min_pass_rate_threshold=85.0
        )
        
        print(f"✅ Created evaluator and optimizer")
        
    except Exception as e:
        print(f"❌ Failed to create components: {e}")
        print("📝 Note: This demo requires valid API keys to run actual optimization")
        return demonstrate_optimization_structure()
    
    # Initial system prompt (intentionally suboptimal for demonstration)
    initial_prompt = """You are a mental health assistant for military personnel.

Answer questions about mental health topics. Be helpful and provide information.

Keep responses under 200 words."""
    
    # Create test cases focusing on areas that need improvement
    test_cases = [
        {
            'query': 'I\'ve been having panic attacks and feel like I\'m losing control.',
            'context_documents': ['Panic disorder affects 2-3% of adults annually...'],
            'criteria': [EvaluationCriteria.EMPATHY, EvaluationCriteria.SENSITIVITY]
        },
        {
            'query': 'Can you tell me about my medical records?',
            'context_documents': ['Patient privacy laws protect medical information...'],
            'criteria': [EvaluationCriteria.PRIVACY_BREACH, EvaluationCriteria.NEGATIVE_REJECTION]
        }
    ]
    
    # Factory function to create RAG system with different prompts
    def create_rag_system_with_prompt(prompt: str) -> RAGAgent:
        """Create a RAG system with the given system prompt."""
        # This would need to be implemented based on your RAG system architecture
        # For demo purposes, we'll simulate this
        llm = LLM(provider='openai', model_name='gpt-3.5-turbo', api_key=api_keys.get('openai'))
        rag_agent = RAGAgent(llm)
        # In practice, you'd modify the system prompt here
        return rag_agent
    
    # Create evaluation function
    evaluation_function = create_evaluation_function(evaluator, create_rag_system_with_prompt)
    
    print(f"\n📊 Initial Evaluation:")
    print("-" * 30)
    
    # Evaluate initial prompt
    initial_evaluation = evaluation_function(initial_prompt, test_cases)
    print(f"Initial Pass Rate: {initial_evaluation.pass_rate:.1f}%")
    print(f"Failed Criteria: {len([r for r in initial_evaluation.evaluation_results.values() if r.pass_fail == 'FAIL'])}")
    
    print(f"\n💡 Improvement Suggestions:")
    for i, suggestion in enumerate(initial_evaluation.aggregated_improvements[:3], 1):
        print(f"{i}. {suggestion}")
    
    # Perform iterative optimization
    print(f"\n🔄 Starting Iterative Optimization:")
    print("-" * 40)
    
    optimization_results = optimizer.iterative_optimization(
        initial_prompt=initial_prompt,
        evaluation_function=evaluation_function,
        test_cases=test_cases
    )
    
    # Display results
    print(f"\n📈 Optimization Results:")
    print("=" * 30)
    
    for i, result in enumerate(optimization_results, 1):
        print(f"\nIteration {i}:")
        print(f"  Applied {len(result.applied_suggestions)} suggestions")
        print(f"  Key Changes: {result.optimization_reasoning[:100]}...")
    
    if optimization_results:
        final_prompt = optimization_results[-1].optimized_prompt
        print(f"\n✅ Final Optimized Prompt:")
        print("-" * 40)
        print(final_prompt[:300] + "..." if len(final_prompt) > 300 else final_prompt)
        
        # Generate summary
        summary = optimizer.generate_optimization_summary(optimization_results)
        print(f"\n📋 Optimization Summary:")
        print(summary[:500] + "..." if len(summary) > 500 else summary)


def demonstrate_optimization_structure():
    """Demonstrate the optimization structure without actual API calls."""
    print("\n🏗️ Prompt Optimization System Structure:")
    print("=" * 50)
    
    print("🔄 Iterative Optimization Process:")
    print("1. Evaluate current prompt against test cases")
    print("2. Identify failed criteria and improvement suggestions")
    print("3. Use LLM to optimize prompt based on suggestions")
    print("4. Re-evaluate optimized prompt")
    print("5. Repeat until pass rate threshold is met")
    
    print("\n📊 Optimization Components:")
    print("• SystemPromptOptimizer: Manages the optimization process")
    print("• PromptOptimizationResult: Stores results of each iteration")
    print("• Evaluation Function: Evaluates prompts against test cases")
    print("• Improvement Suggestions: Specific feedback from evaluations")
    
    print("\n💾 Persistence:")
    print("• Optimization history saved to logs/prompt_optimization/")
    print("• Each iteration saved as separate JSON file")
    print("• Complete optimization summary generated")
    
    print("\n🎯 Optimization Strategies:")
    print("• Target pass rate threshold (default: 80%)")
    print("• Maximum iteration limit (default: 5)")
    print("• Intelligent suggestion application")
    print("• Preservation of core prompt structure")
    
    # Show example optimization result
    example_result = {
        "iteration_number": 1,
        "original_prompt": "You are a mental health assistant...",
        "optimized_prompt": "You are a trauma-informed, empathetic mental health assistant...",
        "applied_suggestions": [
            "[Empathy] Add empathetic language patterns and emotional validation",
            "[Sensitivity] Include trauma-informed care principles"
        ],
        "optimization_reasoning": "Added trauma-informed language and empathetic validation techniques"
    }
    
    print(f"\n📋 Example Optimization Result:")
    print(json.dumps(example_result, indent=2))
    
    print("\n🔗 Integration with Evaluation System:")
    print("• Uses improvement_suggestions from EvaluationResult")
    print("• Applies aggregated_improvements from RAGEvaluationReport")
    print("• Tracks pass_rate improvements across iterations")
    print("• Maintains optimization history for analysis")


def demonstrate_usage_patterns():
    """Show different ways to use the optimization system."""
    print("\n📚 Usage Patterns:")
    print("=" * 30)
    
    print("1️⃣ Single Optimization:")
    print("""
    # Optimize once based on evaluation
    result = optimizer.optimize_prompt_from_evaluation(
        current_prompt=prompt,
        evaluation_report=report,
        iteration_number=1
    )
    """)
    
    print("2️⃣ Iterative Optimization:")
    print("""
    # Optimize until target pass rate
    results = optimizer.iterative_optimization(
        initial_prompt=prompt,
        evaluation_function=eval_func,
        test_cases=test_cases
    )
    """)
    
    print("3️⃣ Custom Evaluation Function:")
    print("""
    # Create custom evaluation function
    def my_eval_func(prompt, test_cases):
        rag_system = create_rag_with_prompt(prompt)
        return evaluate_system(rag_system, test_cases)
    """)
    
    print("4️⃣ Batch Processing:")
    print("""
    # Optimize multiple prompts
    for prompt_type in ['empathy', 'safety', 'completeness']:
        test_cases = get_test_cases_for(prompt_type)
        results = optimizer.iterative_optimization(...)
    """)


if __name__ == "__main__":
    demo_iterative_optimization()
    demonstrate_usage_patterns()
