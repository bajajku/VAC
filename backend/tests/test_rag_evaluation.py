#!/usr/bin/env python3
"""
Unit tests for the RAG Evaluation System

Tests the core functionality of the RAG evaluator and evaluation pipeline.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.rag_evaluator import (
    RAGEvaluator, 
    EvaluationCriteria, 
    EvaluationResult, 
    RAGEvaluationReport,
    create_rag_evaluator
)
from models.evaluation_pipeline import (
    RAGEvaluationPipeline, 
    TestCaseGenerator,
    create_evaluation_pipeline
)

class TestRAGEvaluator(unittest.TestCase):
    """Test cases for the RAG Evaluator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock jury configs to avoid needing real API keys
        self.mock_jury_configs = [
            {'provider': 'mock', 'model_name': 'mock-model-1'},
            {'provider': 'mock', 'model_name': 'mock-model-2'}
        ]
    
    @patch('models.rag_evaluator.create_jury')
    def test_evaluator_initialization(self, mock_create_jury):
        """Test that RAG evaluator initializes correctly"""
        # Mock the jury creation
        mock_jury = Mock()
        mock_create_jury.return_value = mock_jury
        
        evaluator = RAGEvaluator(self.mock_jury_configs)
        
        # Check that juries were created
        self.assertEqual(mock_create_jury.call_count, 3)  # evaluation, safety, quality juries
        self.assertIsNotNone(evaluator.criterion_weights)
        self.assertEqual(len(evaluator.criterion_weights), len(EvaluationCriteria))
    
    def test_criterion_weights_initialization(self):
        """Test that criterion weights are properly initialized"""
        with patch('models.rag_evaluator.create_jury'):
            evaluator = RAGEvaluator(self.mock_jury_configs)
            
            # Check that all criteria have weights
            for criterion in EvaluationCriteria:
                self.assertIn(criterion.value, evaluator.criterion_weights)
                self.assertIsInstance(evaluator.criterion_weights[criterion.value], float)
                self.assertGreater(evaluator.criterion_weights[criterion.value], 0)
    
    def test_evaluation_criteria_enum(self):
        """Test that all evaluation criteria are properly defined"""
        expected_criteria = [
            'retrieval_relevance', 'retrieval_diversity', 'hallucination',
            'noise_robustness', 'negative_rejection', 'information_integration',
            'counterfactual_robustness', 'privacy_breach', 'malicious_use',
            'security_breach', 'out_of_domain', 'completeness', 'brand_damage'
        ]
        
        actual_criteria = [criterion.value for criterion in EvaluationCriteria]
        
        for expected in expected_criteria:
            self.assertIn(expected, actual_criteria)
    
    @patch('models.rag_evaluator.create_jury')
    def test_build_evaluation_prompt(self, mock_create_jury):
        """Test that evaluation prompts are built correctly"""
        mock_jury = Mock()
        mock_create_jury.return_value = mock_jury
        
        evaluator = RAGEvaluator(self.mock_jury_configs)
        
        query = "Test query"
        response = "Test response"
        context_docs = ["Doc 1", "Doc 2"]
        
        # Test different criteria
        for criterion in EvaluationCriteria:
            prompt = evaluator._build_evaluation_prompt(query, response, context_docs, criterion)
            
            # Check that the prompt contains key elements
            self.assertIn(query, prompt)
            self.assertIn(response, prompt)
            self.assertIn("Doc 1", prompt)
            self.assertIn("Doc 2", prompt)
            self.assertIn(criterion.value.upper(), prompt)
            self.assertIn("JSON", prompt.upper())
    
    def test_score_extraction_from_text(self):
        """Test score extraction from various text formats"""
        with patch('models.rag_evaluator.create_jury'):
            evaluator = RAGEvaluator(self.mock_jury_configs)
            
            # Test various score formats
            test_cases = [
                ("Score: 8.5", 8.5),
                ("The score is 7", 7.0),
                ("8/10", 8.0),
                ("Rating: 9.2 out of 10", 9.2),
                ("No clear score here", 5.0),  # Default
                ("Score: 15", 5.0),  # Out of range, should default
            ]
            
            for text, expected_score in test_cases:
                actual_score = evaluator._extract_score_from_text(text)
                self.assertEqual(actual_score, expected_score)
    
    @patch('models.rag_evaluator.create_jury')
    def test_calculate_overall_score(self, mock_create_jury):
        """Test overall score calculation with weights"""
        mock_jury = Mock()
        mock_create_jury.return_value = mock_jury
        
        evaluator = RAGEvaluator(self.mock_jury_configs)
        
        # Create mock evaluation results
        evaluation_results = {
            'retrieval_relevance': EvaluationResult('retrieval_relevance', 8.0, 'Good', 0.9),
            'hallucination': EvaluationResult('hallucination', 6.0, 'Some issues', 0.8),
            'privacy_breach': EvaluationResult('privacy_breach', 9.0, 'No issues', 0.9)
        }
        
        overall_score = evaluator._calculate_overall_score(evaluation_results)
        
        # Should be a weighted average
        self.assertIsInstance(overall_score, float)
        self.assertGreaterEqual(overall_score, 0.0)
        self.assertLessEqual(overall_score, 10.0)

class TestEvaluationPipeline(unittest.TestCase):
    """Test cases for the Evaluation Pipeline class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_rag_system = Mock()
        self.mock_rag_system.invoke.return_value = "Mock RAG response"
        
        self.mock_jury_configs = [
            {'provider': 'mock', 'model_name': 'mock-model-1'}
        ]
    
    @patch('models.evaluation_pipeline.RAGEvaluator')
    def test_pipeline_initialization(self, mock_evaluator_class):
        """Test that evaluation pipeline initializes correctly"""
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        
        pipeline = RAGEvaluationPipeline(self.mock_rag_system, self.mock_jury_configs)
        
        self.assertEqual(pipeline.rag_system, self.mock_rag_system)
        self.assertIsInstance(pipeline.evaluation_history, list)
        self.assertEqual(len(pipeline.evaluation_history), 0)
    
    def test_generate_rag_response(self):
        """Test RAG response generation"""
        with patch('models.evaluation_pipeline.RAGEvaluator'):
            pipeline = RAGEvaluationPipeline(self.mock_rag_system, self.mock_jury_configs)
            
            response = pipeline._generate_rag_response("Test query")
            self.assertEqual(response, "Mock RAG response")
            self.mock_rag_system.invoke.assert_called_once_with("Test query")

class TestTestCaseGenerator(unittest.TestCase):
    """Test cases for the Test Case Generator"""
    
    def test_generate_retrieval_quality_tests(self):
        """Test generation of retrieval quality test cases"""
        tests = TestCaseGenerator.generate_retrieval_quality_tests()
        
        self.assertIsInstance(tests, list)
        self.assertGreater(len(tests), 0)
        
        for test in tests:
            self.assertIn('query', test)
            self.assertIsInstance(test['query'], str)
            self.assertGreater(len(test['query']), 0)
    
    def test_generate_safety_tests(self):
        """Test generation of safety test cases"""
        tests = TestCaseGenerator.generate_safety_tests()
        
        self.assertIsInstance(tests, list)
        self.assertGreater(len(tests), 0)
        
        for test in tests:
            self.assertIn('query', test)
            self.assertIn('criteria', test)
            self.assertIsInstance(test['criteria'], list)
    
    def test_generate_hallucination_tests(self):
        """Test generation of hallucination test cases"""
        tests = TestCaseGenerator.generate_hallucination_tests()
        
        self.assertIsInstance(tests, list)
        self.assertGreater(len(tests), 0)
        
        for test in tests:
            self.assertIn('query', test)
            # Should include criteria that test for hallucinations
            if 'criteria' in test:
                criteria_values = [c.value for c in test['criteria']]
                self.assertTrue(
                    any(criterion in criteria_values 
                        for criterion in ['hallucination', 'negative_rejection'])
                )
    
    def test_generate_comprehensive_test_suite(self):
        """Test generation of comprehensive test suite"""
        tests = TestCaseGenerator.generate_comprehensive_test_suite()
        
        self.assertIsInstance(tests, list)
        self.assertGreater(len(tests), 5)  # Should include multiple test types
        
        # Should include different types of tests
        queries = [test['query'] for test in tests]
        query_text = ' '.join(queries).lower()
        
        # Check for different test types
        self.assertTrue(any('ptsd' in query.lower() or 'mental health' in query.lower() for query in queries))
        self.assertTrue(any('personal' in query.lower() or 'private' in query.lower() for query in queries))

class TestEvaluationDataStructures(unittest.TestCase):
    """Test cases for evaluation data structures"""
    
    def test_evaluation_result_creation(self):
        """Test EvaluationResult creation"""
        result = EvaluationResult(
            criterion='test_criterion',
            score=8.5,
            reasoning='Test reasoning',
            confidence=0.9
        )
        
        self.assertEqual(result.criterion, 'test_criterion')
        self.assertEqual(result.score, 8.5)
        self.assertEqual(result.reasoning, 'Test reasoning')
        self.assertEqual(result.confidence, 0.9)
        self.assertIsNone(result.individual_scores)
    
    def test_rag_evaluation_report_creation(self):
        """Test RAGEvaluationReport creation"""
        evaluation_results = {
            'test_criterion': EvaluationResult('test_criterion', 8.0, 'Good', 0.9)
        }
        
        report = RAGEvaluationReport(
            query='Test query',
            response='Test response',
            context_documents=['Doc 1', 'Doc 2'],
            overall_score=8.0,
            evaluation_results=evaluation_results,
            timestamp='2024-01-01T00:00:00',
            jury_composition={'jury_size': 3}
        )
        
        self.assertEqual(report.query, 'Test query')
        self.assertEqual(report.response, 'Test response')
        self.assertEqual(len(report.context_documents), 2)
        self.assertEqual(report.overall_score, 8.0)
        self.assertIn('test_criterion', report.evaluation_results)

class TestFactoryFunctions(unittest.TestCase):
    """Test cases for factory functions"""
    
    @patch('models.rag_evaluator.RAGEvaluator')
    def test_create_rag_evaluator_factory(self, mock_evaluator_class):
        """Test the create_rag_evaluator factory function"""
        mock_evaluator = Mock()
        mock_evaluator_class.return_value = mock_evaluator
        
        configs = [{'provider': 'mock', 'model_name': 'mock-model'}]
        evaluator = create_rag_evaluator(configs)
        
        mock_evaluator_class.assert_called_once_with(configs)
        self.assertEqual(evaluator, mock_evaluator)
    
    @patch('models.evaluation_pipeline.RAGEvaluationPipeline')
    def test_create_evaluation_pipeline_factory(self, mock_pipeline_class):
        """Test the create_evaluation_pipeline factory function"""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_rag_system = Mock()
        configs = [{'provider': 'mock', 'model_name': 'mock-model'}]
        
        pipeline = create_evaluation_pipeline(mock_rag_system, configs)
        
        mock_pipeline_class.assert_called_once_with(mock_rag_system, configs)
        self.assertEqual(pipeline, mock_pipeline)

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2) 