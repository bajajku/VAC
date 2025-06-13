from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime
import logging
from pathlib import Path

class RAGEvaluationLogger:
    """
    Handles logging and persistence of RAG evaluation results.
    Provides structured logging with different log levels and formats.
    """
    
    def __init__(self, log_dir: str = "logs/rag_evaluation"):
        """
        Initialize the evaluation logger.
        
        Args:
            log_dir: Directory to store logs (default: logs/rag_evaluation)
        """
        self.log_dir = Path(log_dir)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging directory structure and configuration."""
        # Create log directories if they don't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "json").mkdir(exist_ok=True)
        (self.log_dir / "summaries").mkdir(exist_ok=True)
        (self.log_dir / "debug").mkdir(exist_ok=True)
        (self.log_dir / "llm_scores").mkdir(exist_ok=True)  # New directory for LLM scores
        
        # Setup logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / "debug" / "rag_evaluation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("RAGEvaluation")
    
    def save_evaluation_results(self, 
                              results: List[Dict[str, Any]], 
                              comprehensive_report: Dict[str, Any],
                              filename: Optional[str] = None):
        """
        Save evaluation results and comprehensive report.
        
        Args:
            results: List of individual evaluation results
            comprehensive_report: Generated comprehensive report
            filename: Optional custom filename (without extension)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = filename or f"rag_evaluation_{timestamp}"
        
        # Save detailed JSON results
        json_path = self.log_dir / "json" / f"{base_filename}.json"
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Saved detailed results to {json_path}")
        except Exception as e:
            self.logger.error(f"Error saving JSON results: {e}")
        
        # Save human-readable summary
        summary_path = self.log_dir / "summaries" / f"{base_filename}.txt"
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(self.generate_text_summary(comprehensive_report))
            self.logger.info(f"Saved summary to {summary_path}")
        except Exception as e:
            self.logger.error(f"Error saving summary: {e}")
        
        # Save detailed LLM scores
        llm_scores_path = self.log_dir / "llm_scores" / f"{base_filename}_llm_scores.txt"
        try:
            with open(llm_scores_path, 'w', encoding='utf-8') as f:
                f.write(self.generate_llm_scores_summary(results))
            self.logger.info(f"Saved LLM scores to {llm_scores_path}")
        except Exception as e:
            self.logger.error(f"Error saving LLM scores: {e}")
    
    def generate_text_summary(self, comprehensive_report: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the evaluation results."""
        summary = f"""RAG EVALUATION SUMMARY
======================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW
--------
Total Evaluations: {comprehensive_report['summary']['total_evaluations']}
Average Score: {comprehensive_report['summary']['overall_score_stats']['mean']:.2f}/10
Score Range: {comprehensive_report['summary']['overall_score_stats']['min']:.1f} - {comprehensive_report['summary']['overall_score_stats']['max']:.1f}

CRITERION ANALYSIS
----------------
"""
        
        for criterion, stats in comprehensive_report['criterion_analysis'].items():
            summary += f"{criterion.replace('_', ' ').title()}:\n"
            summary += f"  Average: {stats['mean']:.2f}/10\n"
            summary += f"  Range: {stats['min']:.1f} - {stats['max']:.1f}\n"
            summary += f"  Count: {stats['count']}\n\n"
        
        if 'performance_highlights' in comprehensive_report:
            summary += "\nPERFORMANCE HIGHLIGHTS\n-------------------\n"
            best = comprehensive_report['performance_highlights']['best_performing_query']
            worst = comprehensive_report['performance_highlights']['worst_performing_query']
            
            summary += f"Best Query ({best['score']:.1f}/10):\n  {best['query']}\n\n"
            summary += f"Worst Query ({worst['score']:.1f}/10):\n  {worst['query']}\n"
        
        return summary
    
    def generate_llm_scores_summary(self, results: List[Dict[str, Any]]) -> str:
        """Generate a detailed summary of individual LLM scores and reasoning."""
        summary = f"""DETAILED LLM SCORING SUMMARY
===========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        for i, result in enumerate(results, 1):
            summary += f"\nQUERY {i}: {result['query']}\n"
            summary += "=" * (len(result['query']) + 8) + "\n\n"
            
            # Add RAG response
            summary += f"RAG RESPONSE:\n{result['response']}\n\n"
            
            # Add tool execution details
            if 'tool_execution_details' in result and result['tool_execution_details']:
                summary += "TOOL EXECUTION DETAILS:\n"
                for j, tool_detail in enumerate(result['tool_execution_details'], 1):
                    summary += f"Tool {j}: {tool_detail['tool_name']}\n"
                    summary += f"  Call ID: {tool_detail['tool_call_id']}\n"
                    summary += f"  Response Length: {tool_detail['response_length']} characters\n"
                    summary += f"  Response Preview: {tool_detail['response_preview']}\n\n"
                summary += "-" * 50 + "\n\n"
            
            # Add retrieved documents information
            if 'retrieved_documents_metadata' in result and result['retrieved_documents_metadata']:
                summary += "RETRIEVED DOCUMENTS (from actual RAG execution):\n"
                for j, doc_meta in enumerate(result['retrieved_documents_metadata'], 1):
                    summary += f"Document {j}:\n"
                    summary += f"  Source: {doc_meta.get('source', 'unknown')}\n"
                    summary += f"  Length: {doc_meta.get('content_length', 0)} characters\n"
                    summary += f"  Extraction Method: {doc_meta.get('extraction_method', 'unknown')}\n"
                    summary += f"  Preview: {doc_meta.get('content_preview', 'No preview available')}\n\n"
                summary += "-" * 50 + "\n\n"
            
            # Add context documents used for evaluation
            if 'context_documents' in result and result['context_documents']:
                summary += "CONTEXT DOCUMENTS USED FOR EVALUATION:\n"
                for j, doc in enumerate(result['context_documents'], 1):
                    preview = doc[:200] + "..." if len(doc) > 200 else doc
                    summary += f"Context {j}: {preview}\n\n"
                summary += "-" * 50 + "\n\n"
            
            # Add evaluation results
            for criterion, eval_result in result['evaluation_report'].evaluation_results.items():
                summary += f"{criterion.replace('_', ' ').title()}:\n"
                summary += f"Final Score: {eval_result.score}/10\n"
                summary += f"Confidence: {eval_result.confidence:.2f}\n"
                summary += f"Reasoning: {eval_result.reasoning}\n\n"
                
                if eval_result.individual_scores:
                    summary += "Individual LLM Scores:\n"
                    for llm_score in eval_result.individual_scores:
                        summary += f"- {llm_score['provider']} ({llm_score['model']}):\n"
                        summary += f"  Score: {llm_score['score']}/10\n"
                        summary += f"  Confidence: {llm_score['confidence']:.2f}\n"
                        summary += f"  Reasoning: {llm_score['reasoning']}\n\n"
                summary += "-" * 50 + "\n\n"
        
        return summary
    
    def log_evaluation_start(self, query: str):
        """Log the start of an evaluation."""
        self.logger.info(f"Starting evaluation for query: {query[:100]}...")
    
    def log_evaluation_complete(self, query: str, score: float):
        """Log the completion of an evaluation."""
        self.logger.info(f"Completed evaluation for query: {query[:100]}... Score: {score:.2f}/10")
    
    def log_error(self, message: str, error: Exception):
        """Log an error during evaluation."""
        self.logger.error(f"{message}: {str(error)}")
    
    def get_latest_results(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the n most recent evaluation results."""
        json_dir = self.log_dir / "json"
        if not json_dir.exists():
            return []
        
        files = sorted(json_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        results = []
        
        for file in files[:n]:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    results.append(json.load(f))
            except Exception as e:
                self.logger.error(f"Error reading {file}: {e}")
        
        return results 