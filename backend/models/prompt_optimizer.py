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
        self.token_limit = kwargs.get('token_limit', 32768)
        # actual token limit is self.token_limit - 1000, because we need to leave some tokens for the footer
        
        # Create optimization logs directory
        self.log_dir = Path(kwargs.get('log_dir', 'logs/prompt_optimization'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Fixed template footer that should never be modified
        self.fixed_template_footer = """Given the following context and question, provide a helpful response:
Context: {context}
Question: {input}"""
        
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
        
        # Split current prompt into optimizable and fixed sections
        optimizable_section, _ = self._split_prompt(current_prompt)
        
        # Create optimization prompt (only optimizable section is passed to LLM)
        optimization_prompt = self._build_optimization_prompt(
            optimizable_section, evaluation_report
        )
        
        # Get optimized prompt from LLM
        chat = self.optimizer_llm.create_chat()
        response = chat.invoke(optimization_prompt)
        
        # Parse the optimization response (only optimizable section)
        optimized_section, reasoning = self._parse_optimization_response(response.content)
        
        # Reassemble: optimized section + fixed footer
        optimized_prompt = self._reassemble_prompt(optimized_section)
        
        # Validate the final prompt structure
        if not self._validate_prompt_structure(optimized_prompt):
            print(f"   ⚠️ Warning: Optimized prompt structure validation failed, but proceeding anyway")
        
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
    
    def _split_prompt(self, prompt: str) -> Tuple[str, str]:
        """
        Split prompt into optimizable section and fixed template footer.
        
        Args:
            prompt: The full prompt to split
            
        Returns:
            Tuple of (optimizable_section, fixed_footer)
        """
        # Look for the fixed footer pattern
        # The footer starts with "Given the following context and question..."
        footer_pattern = r'Given the following context and question.*?Question:\s*\{input\}'
        match = re.search(footer_pattern, prompt, re.DOTALL | re.IGNORECASE)
        
        if match:
            # Split at the footer
            optimizable_section = prompt[:match.start()].rstrip()
            fixed_footer = match.group(0).strip()
        else:
            # If footer not found, try to find just the template variable lines
            # Look for lines containing {context} and {input}
            lines = prompt.split('\n')
            footer_start_idx = None
            
            for i, line in enumerate(lines):
                if '{context}' in line or '{input}' in line:
                    # Check if this is the start of the footer section
                    # Usually preceded by "Given the following..." or similar
                    if i > 0 and ('Given' in lines[i-1] or 'context' in lines[i-1].lower() or 'question' in lines[i-1].lower()):
                        footer_start_idx = i - 1
                        break
                    elif '{context}' in line:
                        footer_start_idx = i
                        break
            
            if footer_start_idx is not None:
                optimizable_section = '\n'.join(lines[:footer_start_idx]).rstrip()
                fixed_footer = '\n'.join(lines[footer_start_idx:]).strip()
            else:
                # Fallback: assume everything is optimizable except the last 3 lines
                lines = prompt.split('\n')
                if len(lines) >= 3:
                    optimizable_section = '\n'.join(lines[:-3]).rstrip()
                    fixed_footer = '\n'.join(lines[-3:]).strip()
                else:
                    # Very short prompt, assume all is optimizable
                    optimizable_section = prompt.rstrip()
                    fixed_footer = self.fixed_template_footer
        
        return optimizable_section, fixed_footer
    
    def _reassemble_prompt(self, optimizable_section: str) -> str:
        """
        Reassemble prompt from optimized section and fixed footer.
        
        Args:
            optimizable_section: The optimized instructions/guidelines section
            
        Returns:
            Complete prompt with fixed footer appended
        """
        # Clean up the optimizable section (remove any trailing template footer if LLM added it)
        optimizable_section = optimizable_section.rstrip()
        
        # Remove the fixed footer if LLM accidentally included it
        footer_pattern = r'Given the following context and question.*?Question:\s*\{input\}'
        optimizable_section = re.sub(footer_pattern, '', optimizable_section, flags=re.DOTALL | re.IGNORECASE).rstrip()
        
        # Reassemble: optimizable section + blank line + fixed footer
        return f"{optimizable_section}\n\n{self.fixed_template_footer}"
    
    def _build_optimization_prompt(self, 
                                 optimizable_section: str, 
                                 evaluation_report: RAGEvaluationReport) -> str:
        """
        Build the prompt for the optimization LLM.
        
        Args:
            optimizable_section: The instructions/guidelines section to optimize (no template variables)
            evaluation_report: Evaluation results with improvement suggestions
        """
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
        
        return f"""You are a system prompt optimization expert. Your task is to improve the instructions and guidelines section of a RAG system's prompt based on evaluation feedback.

CURRENT INSTRUCTIONS/GUIDELINES SECTION:
{optimizable_section}

EVALUATION RESULTS:
- Overall Score: {evaluation_report.overall_score}/10
- Pass Rate: {evaluation_report.pass_rate:.1f}%
- Overall Status: {evaluation_report.overall_pass_fail}

FAILED CRITERIA:
{failed_criteria_summary}

IMPROVEMENT SUGGESTIONS:
{improvement_suggestions}

OPTIMIZATION TASK:
You are allowed to use up to {self.token_limit - 1000} tokens in your response.
1. Analyze the instructions/guidelines section above
2. Apply the improvement suggestions to optimize this section
3. You can ADD, REMOVE, or MODIFY lines in the instructions/guidelines
4. Focus on addressing the specific failed criteria (empathy, sensitivity, safety, etc.)
5. Maintain the trauma-informed and military-focused approach
6. Make targeted improvements - optimization can mean removing unnecessary lines too
7. Keep the same general structure (Guidelines, Key Principles, Remember sections)
8. Ensure the optimized section remains clear and actionable

RESPONSE FORMAT:
Provide your response in this exact format:

OPTIMIZATION_REASONING:
[Explain your analysis, which suggestions you're applying, and why. Note any lines you're removing and why.]

OPTIMIZED_INSTRUCTIONS:
[The optimized instructions/guidelines section - return only this section, nothing else]"""
        
    def _parse_optimization_response(self, response: str) -> Tuple[str, str]:
        """
        Parse the optimization response to extract reasoning and optimized instructions section.
        
        Args:
            response: The LLM response containing optimization reasoning and optimized instructions
            
        Returns:
            Tuple of (optimized_instructions_section, reasoning)
        """
        try:
            # Extract reasoning
            reasoning_match = re.search(
                r'OPTIMIZATION_REASONING:\s*(.*?)\s*(?:OPTIMIZED_INSTRUCTIONS|OPTIMIZED_PROMPT):', 
                response, 
                re.DOTALL
            )
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            
            # Extract optimized instructions section (look for both possible labels)
            instructions_match = re.search(
                r'(?:OPTIMIZED_INSTRUCTIONS|OPTIMIZED_PROMPT):\s*(.*?)$', 
                response, 
                re.DOTALL
            )
            optimized_section = instructions_match.group(1).strip() if instructions_match else response.strip()
            
            # Clean up: remove any accidental template footer if LLM included it
            footer_pattern = r'Given the following context and question.*?Question:\s*\{input\}'
            optimized_section = re.sub(footer_pattern, '', optimized_section, flags=re.DOTALL | re.IGNORECASE).strip()
            
            # Remove any template variables that shouldn't be here
            optimized_section = optimized_section.replace('{context}', '').replace('{input}', '').strip()
            
            # Clean up extra whitespace
            optimized_section = re.sub(r'\n{3,}', '\n\n', optimized_section)
            optimized_section = optimized_section.rstrip()
            
            return optimized_section, reasoning
            
        except Exception as e:
            print(f"⚠️ Error parsing optimization response: {e}")
            print(f"   Response preview: {response[:200]}...")
            # Fallback: try to extract just the text after "OPTIMIZED_INSTRUCTIONS:" or similar
            fallback_match = re.search(r'(?:OPTIMIZED|OPTIMIZED_INSTRUCTIONS|OPTIMIZED_PROMPT):\s*(.*)', response, re.DOTALL)
            if fallback_match:
                return fallback_match.group(1).strip(), "Error parsing optimization reasoning - used fallback extraction"
            return response.strip(), "Error parsing optimization reasoning"
    
    def _validate_prompt_structure(self, prompt: str) -> bool:
        """
        Validate that the prompt has the correct structure with required template variables.
        
        Args:
            prompt: The complete prompt to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check if required variables are present in the footer
        has_context = '{context}' in prompt
        has_input = '{input}' in prompt
        
        # Check if the fixed footer pattern exists
        footer_pattern = r'Given the following context and question.*?Question:\s*\{input\}'
        has_footer = bool(re.search(footer_pattern, prompt, re.DOTALL | re.IGNORECASE))
        
        if not (has_context and has_input):
            print(f"⚠️ Warning: Prompt missing required template variables")
            print(f"   Has {{context}}: {has_context}, Has {{input}}: {has_input}")
            return False
        
        if not has_footer:
            print(f"⚠️ Warning: Prompt missing expected footer structure")
            return False
        
        return True
    
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
