#!/usr/bin/env python3
"""
Improvement Analyzer
====================

Analyzes saved evaluation results to extract and prioritize improvement suggestions.
Provides utilities for working with historical evaluation data.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import Counter, defaultdict
import json
from datetime import datetime
import statistics


class ImprovementAnalyzer:
    """
    Analyzes evaluation results to extract actionable improvement insights.
    """
    
    def __init__(self, log_dir: str = "logs/rag_evaluation"):
        """
        Initialize the improvement analyzer.
        
        Args:
            log_dir: Directory containing evaluation logs
        """
        self.log_dir = Path(log_dir)
        self.json_dir = self.log_dir / "json"
        
    def load_evaluation_results(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load evaluation results from saved JSON files.
        
        Args:
            limit: Maximum number of recent results to load
            
        Returns:
            List of evaluation result dictionaries
        """
        if not self.json_dir.exists():
            print(f"⚠️ No evaluation results found in {self.json_dir}")
            return []
        
        # Get all JSON files sorted by modification time (newest first)
        json_files = sorted(
            self.json_dir.glob("*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        if limit:
            json_files = json_files[:limit]
        
        results = []
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
        
        print(f"📊 Loaded {len(results)} evaluation results")
        return results
    
    def extract_improvement_suggestions(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract and analyze improvement suggestions from evaluation results.
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            
        Returns:
            Dictionary with analyzed improvement suggestions
        """
        all_suggestions = []
        criterion_failures = defaultdict(list)
        pass_rates = []
        
        for result in evaluation_results:
            # Extract improvement recommendations if available
            if 'improvement_recommendations' in result:
                suggestions = result['improvement_recommendations'].get('top_suggestions', [])
                for suggestion_data in suggestions:
                    if isinstance(suggestion_data, dict):
                        suggestion = suggestion_data.get('suggestion', '')
                        frequency = suggestion_data.get('frequency', 1)
                    else:
                        suggestion = str(suggestion_data)
                        frequency = 1
                    
                    if suggestion:
                        all_suggestions.extend([suggestion] * frequency)
            
            # Extract detailed results if available
            if 'detailed_results' in result:
                for detailed_result in result['detailed_results']:
                    if 'evaluation_report' in detailed_result:
                        report = detailed_result['evaluation_report']
                        
                        # Track pass rates
                        if hasattr(report, 'pass_rate') or 'pass_rate' in report:
                            pass_rate = report.get('pass_rate', 0) if isinstance(report, dict) else getattr(report, 'pass_rate', 0)
                            pass_rates.append(pass_rate)
                        
                        # Track criterion failures
                        eval_results = report.get('evaluation_results', {}) if isinstance(report, dict) else getattr(report, 'evaluation_results', {})
                        for criterion, eval_result in eval_results.items():
                            if isinstance(eval_result, dict):
                                pass_fail = eval_result.get('pass_fail', 'UNKNOWN')
                                score = eval_result.get('score', 0)
                                improvement = eval_result.get('improvement_suggestions', '')
                            else:
                                pass_fail = getattr(eval_result, 'pass_fail', 'UNKNOWN')
                                score = getattr(eval_result, 'score', 0)
                                improvement = getattr(eval_result, 'improvement_suggestions', '')
                            
                            criterion_failures[criterion].append({
                                'pass_fail': pass_fail,
                                'score': score,
                                'improvement_suggestions': improvement
                            })
        
        # Analyze suggestions
        suggestion_frequency = Counter(all_suggestions)
        
        # Analyze criterion performance
        criterion_analysis = {}
        for criterion, failures in criterion_failures.items():
            fail_count = sum(1 for f in failures if f['pass_fail'] == 'FAIL')
            total_count = len(failures)
            fail_rate = (fail_count / total_count * 100) if total_count > 0 else 0
            
            scores = [f['score'] for f in failures if f['score'] > 0]
            avg_score = statistics.mean(scores) if scores else 0
            
            # Collect improvement suggestions for this criterion
            criterion_improvements = [f['improvement_suggestions'] for f in failures 
                                    if f['improvement_suggestions'] and f['pass_fail'] == 'FAIL']
            
            criterion_analysis[criterion] = {
                'fail_rate': fail_rate,
                'average_score': avg_score,
                'total_evaluations': total_count,
                'failed_evaluations': fail_count,
                'common_improvements': Counter(criterion_improvements).most_common(3)
            }
        
        return {
            'total_evaluations_analyzed': len(evaluation_results),
            'most_common_suggestions': suggestion_frequency.most_common(10),
            'criterion_analysis': criterion_analysis,
            'overall_pass_rate_trend': {
                'average': statistics.mean(pass_rates) if pass_rates else 0,
                'min': min(pass_rates) if pass_rates else 0,
                'max': max(pass_rates) if pass_rates else 0,
                'count': len(pass_rates)
            },
            'priority_areas': self._identify_priority_areas(criterion_analysis)
        }
    
    def _identify_priority_areas(self, criterion_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify priority areas for improvement based on failure rates and scores."""
        priority_areas = []
        
        for criterion, analysis in criterion_analysis.items():
            # Calculate priority score based on failure rate and average score
            fail_rate = analysis['fail_rate']
            avg_score = analysis['average_score']
            total_evals = analysis['total_evaluations']
            
            # Higher priority for high failure rates and low scores
            priority_score = (fail_rate * 0.7) + ((10 - avg_score) * 10 * 0.3)
            
            if fail_rate > 20:  # Only include criteria with significant failure rates
                priority_areas.append({
                    'criterion': criterion,
                    'priority_score': priority_score,
                    'fail_rate': fail_rate,
                    'average_score': avg_score,
                    'total_evaluations': total_evals,
                    'severity': 'High' if fail_rate > 60 else 'Medium' if fail_rate > 40 else 'Low'
                })
        
        # Sort by priority score (highest first)
        priority_areas.sort(key=lambda x: x['priority_score'], reverse=True)
        return priority_areas
    
    def generate_improvement_report(self, limit: Optional[int] = None) -> str:
        """
        Generate a comprehensive improvement report from saved evaluation results.
        
        Args:
            limit: Maximum number of recent results to analyze
            
        Returns:
            Formatted improvement report
        """
        evaluation_results = self.load_evaluation_results(limit)
        if not evaluation_results:
            return "No evaluation results available for analysis."
        
        analysis = self.extract_improvement_suggestions(evaluation_results)
        
        report = f"""
IMPROVEMENT ANALYSIS REPORT
===========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Analyzed Evaluations: {analysis['total_evaluations_analyzed']}

OVERALL PERFORMANCE TRENDS
--------------------------
Average Pass Rate: {analysis['overall_pass_rate_trend']['average']:.1f}%
Pass Rate Range: {analysis['overall_pass_rate_trend']['min']:.1f}% - {analysis['overall_pass_rate_trend']['max']:.1f}%
Total Evaluations: {analysis['overall_pass_rate_trend']['count']}

TOP IMPROVEMENT SUGGESTIONS
---------------------------
"""
        
        for i, (suggestion, frequency) in enumerate(analysis['most_common_suggestions'][:5], 1):
            report += f"{i}. {suggestion} (mentioned {frequency} times)\n"
        
        report += f"\nPRIORITY AREAS FOR IMPROVEMENT\n"
        report += "-" * 30 + "\n"
        
        for area in analysis['priority_areas'][:5]:
            report += f"\n{area['criterion'].replace('_', ' ').title()} ({area['severity']} Priority):\n"
            report += f"  Failure Rate: {area['fail_rate']:.1f}%\n"
            report += f"  Average Score: {area['average_score']:.1f}/10\n"
            report += f"  Total Evaluations: {area['total_evaluations']}\n"
        
        report += f"\nCRITERION PERFORMANCE ANALYSIS\n"
        report += "-" * 30 + "\n"
        
        for criterion, analysis_data in analysis['criterion_analysis'].items():
            if analysis_data['total_evaluations'] > 0:
                report += f"\n{criterion.replace('_', ' ').title()}:\n"
                report += f"  Fail Rate: {analysis_data['fail_rate']:.1f}%\n"
                report += f"  Average Score: {analysis_data['average_score']:.1f}/10\n"
                report += f"  Evaluations: {analysis_data['failed_evaluations']}/{analysis_data['total_evaluations']}\n"
                
                if analysis_data['common_improvements']:
                    report += f"  Common Improvements:\n"
                    for improvement, count in analysis_data['common_improvements']:
                        if improvement:
                            report += f"    - {improvement[:80]}... ({count}x)\n"
        
        return report
    
    def get_suggestions_for_criteria(self, criteria: List[str], limit: Optional[int] = None) -> Dict[str, List[str]]:
        """
        Get improvement suggestions for specific criteria.
        
        Args:
            criteria: List of criterion names to analyze
            limit: Maximum number of recent results to analyze
            
        Returns:
            Dictionary mapping criteria to their improvement suggestions
        """
        evaluation_results = self.load_evaluation_results(limit)
        analysis = self.extract_improvement_suggestions(evaluation_results)
        
        suggestions_by_criterion = {}
        for criterion in criteria:
            if criterion in analysis['criterion_analysis']:
                criterion_data = analysis['criterion_analysis'][criterion]
                suggestions = [improvement for improvement, count in criterion_data['common_improvements']]
                suggestions_by_criterion[criterion] = suggestions
            else:
                suggestions_by_criterion[criterion] = []
        
        return suggestions_by_criterion
    
    def export_analysis_to_json(self, output_path: str, limit: Optional[int] = None):
        """Export improvement analysis to JSON file."""
        evaluation_results = self.load_evaluation_results(limit)
        analysis = self.extract_improvement_suggestions(evaluation_results)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📊 Exported improvement analysis to {output_file}")


# Utility functions
def analyze_recent_evaluations(log_dir: str = "logs/rag_evaluation", limit: int = 10) -> str:
    """Quick function to analyze recent evaluations and get improvement report."""
    analyzer = ImprovementAnalyzer(log_dir)
    return analyzer.generate_improvement_report(limit)


def get_priority_improvements(log_dir: str = "logs/rag_evaluation", limit: int = 10) -> List[str]:
    """Get the top priority improvement suggestions."""
    analyzer = ImprovementAnalyzer(log_dir)
    evaluation_results = analyzer.load_evaluation_results(limit)
    analysis = analyzer.extract_improvement_suggestions(evaluation_results)
    
    return [suggestion for suggestion, frequency in analysis['most_common_suggestions'][:5]]


if __name__ == "__main__":
    # Demo usage
    analyzer = ImprovementAnalyzer()
    report = analyzer.generate_improvement_report(limit=5)
    print(report)
