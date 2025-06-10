'use client';

import React, { useState, useEffect } from 'react';
import { Download, RefreshCw, Star, ThumbsUp, ThumbsDown, MessageSquare, TrendingUp } from 'lucide-react';
import { feedbackService, FeedbackStats} from '../services/feedbackService';
  
const FeedbackDashboard: React.FC = () => {
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'details'>('overview');

  useEffect(() => {
    loadFeedbackData();
  }, []);

  const loadFeedbackData = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await feedbackService.getFeedbackStats();
      if (result.success && result.data) {
        setStats(result.data);
      } else {
        setError(result.error || 'Failed to load feedback data');
      }
    } catch (error) {
      console.error('Error loading feedback data:', error);
      setError('Failed to load feedback data');
    } finally {
      setLoading(false);
    }
  };

  const exportToCSV = () => {
    if (!stats || stats.recent_feedback.length === 0) return;

    const csvHeaders = [
      'ID', 'Session ID', 'Question', 'Answer', 'Feedback Type', 'Rating', 
      'Comment', 'Created At', 'Updated At'
    ];

    const csvData = stats.recent_feedback.map(item => [
      item.id,
      item.session_id,
      `"${item.question.replace(/"/g, '""')}"`,
      `"${item.answer.replace(/"/g, '""')}"`,
      item.feedback_type,
      item.rating || '',
      `"${(item.feedback_text || '').replace(/"/g, '""')}"`,
      item.created_at,
      item.updated_at
    ]);

    const csvContent = [csvHeaders, ...csvData]
      .map(row => row.join(','))
      .join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `feedback-export-${new Date().toISOString().split('T')[0]}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <RefreshCw className="w-6 h-6 animate-spin text-blue-500" />
        <span className="ml-2 text-slate-600">Loading feedback data...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 text-center">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
          <p className="text-red-700">{error}</p>
        </div>
        <button
          onClick={loadFeedbackData}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
        >
          Try Again
        </button>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="p-8 text-center">
        <p className="text-slate-600">No feedback data available</p>
        <button
          onClick={loadFeedbackData}
          className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
        >
          Refresh
        </button>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-slate-800">Feedback Dashboard</h1>
        <div className="flex space-x-3">
          <button
            onClick={loadFeedbackData}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
          <button
            onClick={exportToCSV}
            disabled={stats.recent_feedback.length === 0}
            className="flex items-center space-x-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Download className="w-4 h-4" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex space-x-1 mb-6 bg-slate-100 p-1 rounded-lg">
        <button
          onClick={() => setActiveTab('overview')}
          className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'overview'
              ? 'bg-white text-slate-900 shadow-sm'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          Overview
        </button>
        <button
          onClick={() => setActiveTab('details')}
          className={`flex-1 py-2 px-4 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'details'
              ? 'bg-white text-slate-900 shadow-sm'
              : 'text-slate-600 hover:text-slate-900'
          }`}
        >
          Recent Feedback
        </button>
      </div>

      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
              <div className="flex items-center">
                <MessageSquare className="w-8 h-8 text-blue-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-slate-600">Total Feedback</p>
                  <p className="text-2xl font-bold text-slate-900">{stats.total_feedback}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
              <div className="flex items-center">
                <ThumbsUp className="w-8 h-8 text-green-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-slate-600">Positive</p>
                  <p className="text-2xl font-bold text-slate-900">{stats.positive_count}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
              <div className="flex items-center">
                <ThumbsDown className="w-8 h-8 text-red-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-slate-600">Negative</p>
                  <p className="text-2xl font-bold text-slate-900">{stats.negative_count}</p>
                </div>
              </div>
            </div>
            
            <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
              <div className="flex items-center">
                <TrendingUp className="w-8 h-8 text-purple-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-slate-600">Suggestions</p>
                  <p className="text-2xl font-bold text-slate-900">{stats.suggestion_count}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Average Rating */}
          {stats.average_rating && (
            <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-slate-800 mb-2">Average Rating</h3>
                  <div className="flex items-center space-x-2">
                    <div className="flex items-center">
                      {[1, 2, 3, 4, 5].map((star) => (
                        <Star
                          key={star}
                          className={`w-6 h-6 ${
                            star <= Math.round(stats.average_rating!)
                              ? 'text-yellow-400 fill-current'
                              : 'text-gray-300'
                          }`}
                        />
                      ))}
                    </div>
                    <span className="text-2xl font-bold text-slate-900">
                      {stats.average_rating.toFixed(1)}/5
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Feedback Distribution */}
          <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
            <h3 className="text-lg font-semibold text-slate-800 mb-4">Feedback Distribution</h3>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600">Positive</span>
                <div className="flex items-center space-x-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-green-500 h-2 rounded-full"
                      style={{
                        width: `${stats.total_feedback > 0 ? (stats.positive_count / stats.total_feedback) * 100 : 0}%`
                      }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-slate-900">
                    {stats.total_feedback > 0 ? Math.round((stats.positive_count / stats.total_feedback) * 100) : 0}%
                  </span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600">Negative</span>
                <div className="flex items-center space-x-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-red-500 h-2 rounded-full"
                      style={{
                        width: `${stats.total_feedback > 0 ? (stats.negative_count / stats.total_feedback) * 100 : 0}%`
                      }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-slate-900">
                    {stats.total_feedback > 0 ? Math.round((stats.negative_count / stats.total_feedback) * 100) : 0}%
                  </span>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600">Suggestions</span>
                <div className="flex items-center space-x-2">
                  <div className="w-32 bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-purple-500 h-2 rounded-full"
                      style={{
                        width: `${stats.total_feedback > 0 ? (stats.suggestion_count / stats.total_feedback) * 100 : 0}%`
                      }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-slate-900">
                    {stats.total_feedback > 0 ? Math.round((stats.suggestion_count / stats.total_feedback) * 100) : 0}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'details' && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-slate-800">Recent Feedback</h3>
          {stats.recent_feedback.length === 0 ? (
            <div className="bg-white p-8 rounded-lg shadow-sm border border-slate-200 text-center">
              <p className="text-slate-600">No feedback available yet.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {stats.recent_feedback.map((feedback) => (
                <div
                  key={feedback.id}
                  className="bg-white p-6 rounded-lg shadow-sm border border-slate-200"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center space-x-2">
                      <span
                        className={`px-2 py-1 text-xs font-medium rounded-full ${
                          feedback.feedback_type === 'positive'
                            ? 'bg-green-100 text-green-800'
                            : feedback.feedback_type === 'negative'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-purple-100 text-purple-800'
                        }`}
                      >
                        {feedback.feedback_type}
                      </span>
                      {feedback.rating && (
                        <div className="flex items-center space-x-1">
                          <Star className="w-4 h-4 text-yellow-400 fill-current" />
                          <span className="text-sm text-slate-600">{feedback.rating}/5</span>
                        </div>
                      )}
                    </div>
                    <span className="text-sm text-slate-500">
                      {formatDate(feedback.created_at)}
                    </span>
                  </div>
                  
                  <div className="space-y-3">
                    <div>
                      <p className="text-sm font-medium text-slate-700 mb-1">Question:</p>
                      <p className="text-sm text-slate-600 bg-slate-50 p-2 rounded">
                        {feedback.question}
                      </p>
                    </div>
                    
                    <div>
                      <p className="text-sm font-medium text-slate-700 mb-1">Answer:</p>
                      <p className="text-sm text-slate-600 bg-slate-50 p-2 rounded">
                        {feedback.answer.length > 200 
                          ? `${feedback.answer.substring(0, 200)}...` 
                          : feedback.answer}
                      </p>
                    </div>
                    
                    {feedback.feedback_text && (
                      <div>
                        <p className="text-sm font-medium text-slate-700 mb-1">Comment:</p>
                        <p className="text-sm text-slate-600 bg-slate-50 p-2 rounded">
                          {feedback.feedback_text}
                        </p>
                      </div>
                    )}
                  </div>
                  
                  <div className="mt-3 pt-3 border-t border-slate-200">
                    <p className="text-xs text-slate-500">
                      Session: {feedback.session_id} • ID: {feedback.id}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FeedbackDashboard; 