import React, { useState, useEffect } from 'react';
import { Download, RefreshCw, Star, ThumbsUp, ThumbsDown, MessageSquare, Trash2 } from 'lucide-react';
import { feedbackService, FeedbackStorageItem } from '../services/feedbackService';

type FeedbackStats = {
  totalFeedback: number;
  averageRatings: {
    overall: number;
    accuracy: number;
    helpfulness: number;
    clarity: number;
  };
  voteDistribution: {
    likes: number;
    dislikes: number;
    neutral: number;
  };
};

const FeedbackDashboard: React.FC = () => {
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [feedbackData, setFeedbackData] = useState<FeedbackStorageItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'details'>('overview');
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  useEffect(() => {
    loadFeedbackData();
  }, []);

  const loadFeedbackData = async () => {
    setLoading(true);
    try {
      const [statsData, exportData] = await Promise.all([
        feedbackService.getFeedbackStats(),
        feedbackService.exportFeedback()
      ]);
      setStats(statsData);
      setFeedbackData(exportData);
    } catch (error) {
      console.error('Error loading feedback data:', error);
    } finally {
      setLoading(false);
    }
  };

  const exportToCSV = () => {
    if (feedbackData.length === 0) return;

    const csvHeaders = [
      'ID', 'Response ID', 'Session ID', 'Question', 'Answer', 'Overall Rating', 'Accuracy', 'Helpfulness', 
      'Clarity', 'Vote', 'Comment', 'Expert Notes', 'Timestamp'
    ];

    const csvData = feedbackData.map(item => [
      item.id,
      item.responseId,
      item.sessionId || '',
      `"${(item.questionAnswer?.question || '').replace(/"/g, '""')}"`,
      `"${(item.questionAnswer?.answer || '').replace(/"/g, '""')}"`,
      item.overallRating,
      item.accuracy,
      item.helpfulness,
      item.clarity,
      item.vote || '',
      `"${item.comment.replace(/"/g, '""')}"`,
      `"${item.expertNotes.replace(/"/g, '""')}"`,
      item.timestamp
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

  const handleClearFeedback = () => {
    feedbackService.clearLocalFeedback();
    setStats(null);
    setFeedbackData([]);
    setShowClearConfirm(false);
    // Reload to show empty state
    loadFeedbackData();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <RefreshCw className="w-6 h-6 animate-spin text-blue-500" />
        <span className="ml-2 text-slate-600">Loading feedback data...</span>
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="p-8 text-center">
        <p className="text-slate-600">No feedback data available</p>
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
            className="flex items-center space-x-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
          >
            <Download className="w-4 h-4" />
            <span>Export CSV</span>
          </button>
          <button
            onClick={() => setShowClearConfirm(true)}
            disabled={feedbackData.length === 0}
            className="flex items-center space-x-2 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Trash2 className="w-4 h-4" />
            <span>Clear Data</span>
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
          Detailed Feedback
        </button>
      </div>

      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
              <div className="flex items-center">
                <MessageSquare className="w-8 h-8 text-blue-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-slate-600">Total Feedback</p>
                  <p className="text-2xl font-bold text-slate-900">{stats.totalFeedback}</p>
                </div>
              </div>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
              <div className="flex items-center">
                <Star className="w-8 h-8 text-yellow-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-slate-600">Avg Overall Rating</p>
                  <p className="text-2xl font-bold text-slate-900">
                    {stats.averageRatings.overall.toFixed(1)}/5
                  </p>
                </div>
              </div>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
              <div className="flex items-center">
                <ThumbsUp className="w-8 h-8 text-green-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-slate-600">Positive Votes</p>
                  <p className="text-2xl font-bold text-slate-900">{stats.voteDistribution.likes}</p>
                </div>
              </div>
            </div>
            <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
              <div className="flex items-center">
                <ThumbsDown className="w-8 h-8 text-red-500" />
                <div className="ml-4">
                  <p className="text-sm font-medium text-slate-600">Negative Votes</p>
                  <p className="text-2xl font-bold text-slate-900">{stats.voteDistribution.dislikes}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Simple Rating Bars (instead of charts) */}
          <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
            <h3 className="text-lg font-semibold mb-4">Average Ratings by Category</h3>
            <div className="space-y-4">
              {[
                { label: 'Overall', value: stats.averageRatings.overall, color: 'bg-blue-500' },
                { label: 'Accuracy', value: stats.averageRatings.accuracy, color: 'bg-green-500' },
                { label: 'Helpfulness', value: stats.averageRatings.helpfulness, color: 'bg-yellow-500' },
                { label: 'Clarity', value: stats.averageRatings.clarity, color: 'bg-red-500' },
              ].map((item) => (
                <div key={item.label} className="flex items-center space-x-4">
                  <div className="w-20 text-sm font-medium text-slate-600">{item.label}</div>
                  <div className="flex-1 bg-slate-200 rounded-full h-4 relative">
                    <div
                      className={`${item.color} h-4 rounded-full transition-all duration-300`}
                      style={{ width: `${(item.value / 5) * 100}%` }}
                    />
                  </div>
                  <div className="w-12 text-sm font-medium text-slate-900">
                    {item.value.toFixed(1)}/5
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'details' && (
        <div className="bg-white rounded-lg shadow-sm border border-slate-200">
          <div className="p-6">
            <h3 className="text-lg font-semibold mb-4">Detailed Feedback ({feedbackData.length} items)</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-slate-200">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Timestamp
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Question & Answer
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Ratings
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Vote
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Comment
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                      Expert Notes
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-slate-200">
                  {feedbackData.map((item) => (
                    <tr key={item.id}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-900">
                        {new Date(item.timestamp).toLocaleDateString()} {new Date(item.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-900 max-w-sm">
                        <div className="space-y-2">
                          <div>
                            <span className="font-medium text-slate-600">Q:</span>
                            <p className="text-xs text-slate-800 truncate">
                              {item.questionAnswer?.question || 'No question recorded'}
                            </p>
                          </div>
                          <div>
                            <span className="font-medium text-slate-600">A:</span>
                            <p className="text-xs text-slate-800 truncate">
                              {item.questionAnswer?.answer || 'No answer recorded'}
                            </p>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-900">
                        <div className="space-y-1">
                          <div>Overall: {item.overallRating}/5</div>
                          <div className="text-xs text-slate-500">
                            A:{item.accuracy} H:{item.helpfulness} C:{item.clarity}
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {item.vote === 'like' && <ThumbsUp className="w-4 h-4 text-green-500" />}
                        {item.vote === 'dislike' && <ThumbsDown className="w-4 h-4 text-red-500" />}
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-900 max-w-xs truncate">
                        {item.comment || <span className="text-slate-400">No comment</span>}
                      </td>
                      <td className="px-6 py-4 text-sm text-slate-900 max-w-xs truncate">
                        {item.expertNotes || <span className="text-slate-400">No expert notes</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* Clear Confirmation Modal */}
      {showClearConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md mx-4">
            <h3 className="text-lg font-semibold text-slate-800 mb-2">Clear All Feedback Data?</h3>
            <p className="text-slate-600 mb-6">
              This will permanently delete all stored feedback data from local storage. This action cannot be undone.
            </p>
            <div className="flex space-x-3">
              <button
                onClick={() => setShowClearConfirm(false)}
                className="flex-1 px-4 py-2 border border-slate-300 rounded-lg text-slate-700 hover:bg-slate-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleClearFeedback}
                className="flex-1 px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors"
              >
                Clear Data
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FeedbackDashboard; 