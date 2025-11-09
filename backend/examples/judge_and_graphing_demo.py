#!/usr/bin/env python3
"""
Demo: Judge LLM and Graphing System

This script demonstrates the new Judge LLM and graphing features
for the RAG evaluation system.

Features demonstrated:
1. Jury + Judge evaluation system
2. Automatic graph generation
3. Comparison of initial vs optimized results
4. Visual analysis of improvements per criterion
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_evaluation import EvaluationSystem
import dotenv

dotenv.load_dotenv()

def main():
    print("=" * 80)
    print("RAG EVALUATION: JUDGE LLM + GRAPHING DEMO")
    print("=" * 80)
    
    # Initialize the evaluation system
    # This automatically sets up:
    # - Jury evaluator (3 LLM jury members)
    # - Judge LLM (presiding judge for final verdicts)
    # - Prompt optimizer
    # - Graphing system
    print("\n🔧 Initializing Evaluation System...")
    evaluation_system = EvaluationSystem()
    
    print("\n" + "=" * 80)
    print("SYSTEM COMPONENTS")
    print("=" * 80)
    print("✅ Jury Evaluator: 3 LLM members")
    print("✅ Judge LLM: Qwen/Qwen2.5-14B-Instruct")
    print("✅ Prompt Optimizer: Qwen/Qwen2.5-14B-Instruct")
    print("✅ Graphing System: 4 chart types")
    
    # Option 1: Run complete workflow (recommended)
    print("\n" + "=" * 80)
    print("RUNNING COMPLETE EVALUATION WORKFLOW")
    print("=" * 80)
    print("\nThis will:")
    print("1. Evaluate initial system (Jury + Judge)")
    print("2. Extract improvement suggestions")
    print("3. Optimize the prompt")
    print("4. Re-evaluate optimized system (Jury + Judge)")
    print("5. Compare results")
    print("6. Generate performance graphs")
    print("7. Save detailed logs")
    print("\nPress Enter to continue or Ctrl+C to exit...")
    input()
    
    try:
        # Run the complete workflow
        result = evaluation_system.main()
        
        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETE!")
        print("=" * 80)
        print("\n📊 Graphs saved to: logs/evaluation_graphs/")
        print("   - criteria_comparison_*.png  (Initial vs Optimized per criterion)")
        print("   - improvement_delta_*.png    (Score improvements)")
        print("   - overall_trend_*.png        (Overall scores across test cases)")
        print("   - pass_rate_comparison_*.png (Pass/Fail rate comparison)")
        
        print("\n💾 Detailed logs saved to: logs/evaluation_workflows/")
        print("   - evaluation_workflow_*.json")
        
        print("\n✅ Optimized prompt available in result.optimized_prompt")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def demo_judge_only():
    """
    Demo: Using judge for a single evaluation
    """
    print("\n" + "=" * 80)
    print("DEMO: JUDGE-SUPERVISED EVALUATION")
    print("=" * 80)
    
    from tests.test_evaluation import EvaluationSystem
    
    # Initialize
    eval_system = EvaluationSystem()
    
    # Evaluate with judge (default)
    print("\n📊 Evaluating with Judge + Jury...")
    results_with_judge = eval_system.evaluate_system(use_judge=True)
    
    print(f"\n✅ Evaluated {len(results_with_judge)} test cases")
    print(f"Average Score: {sum(r.overall_score for r in results_with_judge) / len(results_with_judge):.2f}/10")
    print(f"Average Pass Rate: {sum(r.pass_rate for r in results_with_judge) / len(results_with_judge):.1f}%")
    
    # Show judge verdict details for first result
    first_result = results_with_judge[0]
    print(f"\nFirst Test Case Details:")
    print(f"  Query: {first_result.query[:80]}...")
    print(f"  Overall Score: {first_result.overall_score}/10")
    print(f"  Overall Status: {first_result.overall_pass_fail}")
    print(f"  Pass Rate: {first_result.pass_rate:.1f}%")
    
    print("\nSample Judge Verdicts:")
    for i, (criterion, result) in enumerate(list(first_result.evaluation_results.items())[:3]):
        print(f"  {i+1}. {criterion.replace('_', ' ').title()}")
        print(f"     Score: {result.score}/10 ({result.pass_fail})")
        print(f"     Confidence: {result.confidence:.2f}")
        print(f"     Reasoning: {result.reasoning[:100]}...")


def demo_graphing_only():
    """
    Demo: Generate graphs from existing results
    """
    print("\n" + "=" * 80)
    print("DEMO: GRAPH GENERATION")
    print("=" * 80)
    
    from tests.test_evaluation import EvaluationSystem
    
    # You would need existing results for this
    # This is just a conceptual example
    eval_system = EvaluationSystem()
    
    print("\n📈 Generating graphs from evaluation results...")
    print("\nNote: You need to run evaluations first to generate graphs")
    print("Run the full workflow (main()) to see graph generation in action")


if __name__ == "__main__":
    # Run the main demo
    main()
    
    # Optional: Uncomment to run additional demos
    # demo_judge_only()
    # demo_graphing_only()



