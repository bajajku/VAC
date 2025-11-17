#!/usr/bin/env python3
"""
System Prompt Optimizer
========================

Automatically applies improvement suggestions from RAG evaluations to optimize system prompts.
Provides iterative prompt improvement based on evaluation feedback.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from models.rag_evaluator import RAGEvaluationReport, EvaluationResult
from models.llm import LLM
import json
import re
from datetime import datetime
from pathlib import Path


@dataclass
class PromptOptimizationResult:
    """Result of a prompt optimization iteration"""
    original_prompt: str
    optimized_prompt: str
    applied_suggestions: List[str]
    optimization_reasoning: str
    iteration_number: int
    timestamp: str


class SystemPromptOptimizer:
    """
    Automatically optimizes system prompts based on evaluation feedback.
    Uses LLM to intelligently apply improvement suggestions.
    """
    
    def __init__(self, optimizer_llm: LLM, **kwargs):
        """
        Initialize the prompt optimizer.
        
        Args:
            optimizer_llm: LLM to use for prompt optimization
            **kwargs: Additional configuration options
        """
        self.optimizer_llm = optimizer_llm
        self.optimization_history = []
        self.max_iterations = kwargs.get('max_iterations', 5)
        self.min_pass_rate_threshold = kwargs.get('min_pass_rate_threshold', 80.0)
        self.save_optimizations = kwargs.get('save_optimizations', True)
        
        # Create optimization logs directory
        self.log_dir = Path(kwargs.get('log_dir', 'logs/prompt_optimization'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ System Prompt Optimizer initialized")
    
    def optimize_prompt_from_evaluation(self, 
                                      current_prompt: str,
                                      evaluation_report: RAGEvaluationReport,
                                      iteration_number: int = 1) -> PromptOptimizationResult:
        """
        Optimize a system prompt based on evaluation feedback.
        
        Args:
            current_prompt: The current system prompt
            evaluation_report: Evaluation results with improvement suggestions
            iteration_number: Current iteration number
            
        Returns:
            PromptOptimizationResult with the optimized prompt
        """
        print(f"🔧 Optimizing prompt (Iteration {iteration_number})...")
        print(f"   Current pass rate: {evaluation_report.pass_rate:.1f}%")
        print(f"   Failed criteria: {len([r for r in evaluation_report.evaluation_results.values() if r.pass_fail == 'FAIL'])}")
        
        # Create optimization prompt
        optimization_prompt = self._build_optimization_prompt(
            current_prompt, evaluation_report
        )
        
        # Get optimized prompt from LLM
        chat = self.optimizer_llm.create_chat()
        response = chat.invoke(optimization_prompt)
        
        # Parse the optimization response
        optimized_prompt, reasoning = self._parse_optimization_response(response.content)
        
        # Create result
        result = PromptOptimizationResult(
            original_prompt=current_prompt,
            optimized_prompt=optimized_prompt,
            applied_suggestions=evaluation_report.aggregated_improvements,
            optimization_reasoning=reasoning,
            iteration_number=iteration_number,
            timestamp=datetime.now().isoformat()
        )
        
        # Save optimization history
        if self.save_optimizations:
            self.optimization_history.append(result)
            self._save_optimization_result(result)
        
        print(f"✅ Prompt optimization complete (Iteration {iteration_number})")
        return result
    
    def iterative_optimization(self,
                             initial_prompt: str,
                             evaluation_function: callable,
                             test_cases: List[Dict[str, Any]]) -> List[PromptOptimizationResult]:
        """
        Perform iterative prompt optimization until pass rate threshold is met.
        
        Args:
            initial_prompt: Starting system prompt
            evaluation_function: Function that evaluates a prompt and returns RAGEvaluationReport
            test_cases: Test cases to evaluate against
            
        Returns:
            List of optimization results for each iteration
        """
        print(f"🔄 Starting iterative prompt optimization...")
        print(f"   Target pass rate: {self.min_pass_rate_threshold}%")
        print(f"   Max iterations: {self.max_iterations}")
        
        current_prompt = initial_prompt
        optimization_results = []
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n📊 Iteration {iteration}: Evaluating current prompt...")
            
            # Evaluate current prompt
            evaluation_report = evaluation_function(current_prompt, test_cases)
            
            # Check if we've reached the target pass rate
            if evaluation_report.pass_rate >= self.min_pass_rate_threshold:
                print(f"🎯 Target pass rate achieved: {evaluation_report.pass_rate:.1f}%")
                break
            
            # Optimize prompt based on evaluation
            optimization_result = self.optimize_prompt_from_evaluation(
                current_prompt, evaluation_report, iteration
            )
            optimization_results.append(optimization_result)
            
            # Update current prompt for next iteration
            current_prompt = optimization_result.optimized_prompt
            
            print(f"   Applied {len(optimization_result.applied_suggestions)} suggestions")
        
        print(f"\n✅ Iterative optimization complete after {len(optimization_results)} iterations")
        return optimization_results
    
    def _build_optimization_prompt(self, 
                                 current_prompt: str, 
                                 evaluation_report: RAGEvaluationReport) -> str:
        """Build the prompt for the optimization LLM."""
        
        # Analyze failed criteria
        failed_criteria = [
            (name, result) for name, result in evaluation_report.evaluation_results.items() 
            if result.pass_fail == "FAIL"
        ]
        
        failed_criteria_summary = "\n".join([
            f"- {name.replace('_', ' ').title()}: {result.score:.1f}/10 - {result.reasoning[:100]}..."
            for name, result in failed_criteria
        ])
        
        improvement_suggestions = "\n".join([
            f"- {suggestion}" for suggestion in evaluation_report.aggregated_improvements
        ])
        
        return f"""You are a system prompt optimization expert. Your task is to improve a RAG system's prompt based on evaluation feedback.

CURRENT SYSTEM PROMPT:
{current_prompt}

EVALUATION RESULTS:
- Overall Score: {evaluation_report.overall_score}/10
- Pass Rate: {evaluation_report.pass_rate:.1f}%
- Overall Status: {evaluation_report.overall_pass_fail}

FAILED CRITERIA:
{failed_criteria_summary}

IMPROVEMENT SUGGESTIONS:
{improvement_suggestions}

OPTIMIZATION TASK:
1. Analyze the current prompt and identify areas that need improvement based on the failed criteria
2. Apply the improvement suggestions to create an optimized version
3. Maintain the core functionality and structure of the prompt
4. Focus on addressing the specific failures while preserving what's working well

RESPONSE FORMAT:
Provide your response in this exact format:

OPTIMIZATION_REASONING:
[Explain your analysis of the current prompt, which suggestions you're applying, and why]

OPTIMIZED_PROMPT:
[The complete optimized system prompt - include everything, don't just show changes]

CRITICAL REQUIREMENTS:
- **MUST PRESERVE** the template variables exactly as they appear: {{context}} and {{input}}
- The optimized prompt MUST include these lines exactly:
  Context: {{context}}
  Question: {{input}}
- Any other text with curly braces that is NOT a template variable must be escaped with double curly braces (e.g., if the text contains curly braces for emphasis or examples, double them so they're treated as literal text, not template variables)
- Keep the same general structure and core instructions
- Address the specific failed criteria (empathy, sensitivity, safety, etc.)
- Maintain the trauma-informed and military-focused approach
- Preserve tool usage requirements and formatting rules
- Make targeted improvements without completely rewriting
- Ensure the prompt remains clear and actionable

IMPORTANT: The optimized prompt MUST be a valid LangChain template with {{context}} and {{input}} as the only template variables."""
    
    def _parse_optimization_response(self, response: str) -> Tuple[str, str]:
        """Parse the optimization response to extract reasoning and optimized prompt."""
        try:
            # Extract reasoning
            reasoning_match = re.search(r'OPTIMIZATION_REASONING:\s*(.*?)\s*OPTIMIZED_PROMPT:', response, re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            
            # Extract optimized prompt
            prompt_match = re.search(r'OPTIMIZED_PROMPT:\s*(.*?)$', response, re.DOTALL)
            optimized_prompt = prompt_match.group(1).strip() if prompt_match else response
            
            # Validate and fix the prompt to ensure required variables are present
            optimized_prompt = self._validate_and_fix_prompt(optimized_prompt)
            
            return optimized_prompt, reasoning
            
        except Exception as e:
            print(f"⚠️ Error parsing optimization response: {e}")
            return response, "Error parsing optimization reasoning"
    
    def _validate_and_fix_prompt(self, prompt: str) -> str:
        """
        Validate and fix the prompt to ensure it has required template variables.
        
        Args:
            prompt: The optimized prompt to validate
            
        Returns:
            Fixed prompt with required variables preserved
        """
        # Check if required variables are present
        has_context = '{context}' in prompt
        has_input = '{input}' in prompt
        
        # If missing, try to fix by adding them
        if not has_context or not has_input:
            print(f"⚠️ Warning: Optimized prompt missing required template variables")
            print(f"   Has {{context}}: {has_context}, Has {{input}}: {has_input}")
            
            # Try to find and fix common patterns
            # Look for "Context:" or "Question:" lines and fix them
            lines = prompt.split('\n')
            fixed_lines = []
            context_found = False
            input_found = False
            
            for line in lines:
                # Check if this line should have context variable
                if 'Context:' in line or 'context:' in line.lower():
                    if '{context}' not in line:
                        # Replace with proper format
                        line = re.sub(r'Context:\s*.*', 'Context: {context}', line, flags=re.IGNORECASE)
                        context_found = True
                    else:
                        context_found = True
                
                # Check if this line should have input variable
                if 'Question:' in line or 'question:' in line.lower() or 'Input:' in line.lower():
                    if '{input}' not in line:
                        # Replace with proper format
                        line = re.sub(r'(Question|Input):\s*.*', r'\1: {input}', line, flags=re.IGNORECASE)
                        input_found = True
                    else:
                        input_found = True
                
                fixed_lines.append(line)
            
            # If still missing, add them explicitly
            if not context_found:
                # Find a good place to insert (after first line or before guidelines)
                insert_idx = 1
                for i, line in enumerate(fixed_lines):
                    if 'Guidelines:' in line or 'guidelines:' in line.lower():
                        insert_idx = i
                        break
                fixed_lines.insert(insert_idx, 'Context: {context}')
                context_found = True
            
            if not input_found:
                # Insert after context line
                for i, line in enumerate(fixed_lines):
                    if '{context}' in line:
                        fixed_lines.insert(i + 1, 'Question: {input}')
                        input_found = True
                        break
                if not input_found:
                    # Fallback: add at beginning after first line
                    fixed_lines.insert(1, 'Question: {input}')
            
            prompt = '\n'.join(fixed_lines)
            print(f"   ✅ Fixed prompt to include required template variables")
        
        # Escape any other curly braces that aren't template variables
        # First, temporarily replace the required variables with placeholders
        prompt = prompt.replace('{context}', '__TEMP_CONTEXT__')
        prompt = prompt.replace('{input}', '__TEMP_INPUT__')
        
        # Now escape all remaining curly braces (double them)
        prompt = re.sub(r'\{([^}]+)\}', r'{{\1}}', prompt)
        
        # Restore the required variables
        prompt = prompt.replace('__TEMP_CONTEXT__', '{context}')
        prompt = prompt.replace('__TEMP_INPUT__', '{input}')
        
        # Final validation
        if '{context}' not in prompt or '{input}' not in prompt:
            print(f"❌ Error: Could not ensure required variables in prompt. Adding fallback.")
            # Last resort: prepend the required structure
            if '{context}' not in prompt:
                prompt = f"Context: {{context}}\n{prompt}"
            if '{input}' not in prompt:
                prompt = f"{prompt}\nQuestion: {{input}}"
        
        return prompt
    
    def _save_optimization_result(self, result: PromptOptimizationResult):
        """Save optimization result to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prompt_optimization_iter_{result.iteration_number}_{timestamp}.json"
        filepath = self.log_dir / filename
        
        try:
            # Convert to dict for JSON serialization
            result_dict = {
                'iteration_number': result.iteration_number,
                'timestamp': result.timestamp,
                'original_prompt': result.original_prompt,
                'optimized_prompt': result.optimized_prompt,
                'applied_suggestions': result.applied_suggestions,
                'optimization_reasoning': result.optimization_reasoning
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Saved optimization result to {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving optimization result: {e}")
    
    def generate_optimization_summary(self, optimization_results: List[PromptOptimizationResult]) -> str:
        """Generate a summary of the optimization process."""
        if not optimization_results:
            return "No optimization results to summarize."
        
        summary = f"""
PROMPT OPTIMIZATION SUMMARY
===========================
Total Iterations: {len(optimization_results)}
Optimization Period: {optimization_results[0].timestamp} to {optimization_results[-1].timestamp}

ITERATION PROGRESS:
"""
        
        for i, result in enumerate(optimization_results, 1):
            summary += f"\nIteration {i}:\n"
            summary += f"  Applied Suggestions: {len(result.applied_suggestions)}\n"
            summary += f"  Key Changes: {result.optimization_reasoning[:100]}...\n"
        
        summary += f"\nFINAL OPTIMIZED PROMPT:\n"
        summary += "=" * 50 + "\n"
        summary += optimization_results[-1].optimized_prompt
        
        return summary
    
    def get_optimization_history(self) -> List[PromptOptimizationResult]:
        """Get the complete optimization history."""
        return self.optimization_history
    
    def clear_optimization_history(self):
        """Clear the optimization history."""
        self.optimization_history = []
        print("✅ Optimization history cleared")


# Helper functions for integration with existing evaluation system

def create_evaluation_function(evaluator, rag_system_factory):
    """
    Create an evaluation function for iterative optimization.
    
    Args:
        evaluator: RAGEvaluator instance
        rag_system_factory: Function that creates RAG system with given prompt
        
    Returns:
        Function that evaluates a prompt against test cases
    """
    def evaluate_prompt(prompt: str, test_cases: List[Dict[str, Any]]) -> RAGEvaluationReport:
        """Evaluate a prompt against test cases and return aggregated results."""
        # Create RAG system with the new prompt
        rag_system = rag_system_factory(prompt)
        
        # Run evaluations on test cases
        all_results = []
        for test_case in test_cases:
            # Generate response with the RAG system
            response = rag_system.query(test_case['query'])
            final_message = response["messages"][-1]
            response_text = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            # Evaluate the response
            evaluation_report = evaluator.evaluate_rag_response(
                query=test_case['query'],
                response=response_text,
                context_documents=test_case.get('context_documents', []),
                criteria=test_case.get('criteria')
            )
            all_results.append(evaluation_report)
        
        # Return the last evaluation report as representative
        # In practice, you might want to aggregate across all test cases
        return all_results[-1] if all_results else None
    
    return evaluate_prompt


# Factory function
def create_prompt_optimizer(optimizer_llm: LLM, **kwargs) -> SystemPromptOptimizer:
    """
    Factory function to create a prompt optimizer.
    
    Example usage:
        optimizer = create_prompt_optimizer(
            optimizer_llm=LLM(provider='openai', model_name='gpt-4'),
            max_iterations=3,
            min_pass_rate_threshold=85.0
        )
    """
    return SystemPromptOptimizer(optimizer_llm, **kwargs)
