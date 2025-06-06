'use client';

import FeedbackDashboard from '../../components/FeedbackDashboard';
import Link from 'next/link';
import { ArrowLeft, Shield } from 'lucide-react';

const FeedbackPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <header className="bg-white/90 backdrop-blur-md border-b border-slate-200/50 px-4 sm:px-6 py-4 shadow-sm">
        <div className="flex items-center space-x-3">
          <Link 
            href="/" 
            className="w-8 h-8 bg-slate-100 hover:bg-slate-200 rounded-lg flex items-center justify-center transition-colors"
            aria-label="Return to home"
          >
            <ArrowLeft className="w-4 h-4 text-slate-600" />
          </Link>
          <div className="w-10 h-10 bg-gradient-to-r from-green-600 to-emerald-700 rounded-xl flex items-center justify-center">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-slate-900">Feedback & Analytics Dashboard</h1>
            <p className="text-sm text-slate-600">Community insights • System improvement</p>
          </div>
        </div>
      </header>

      {/* Dashboard Content */}
      <div className="px-4 py-6">
        <FeedbackDashboard />
      </div>
    </div>
  );
};

export default FeedbackPage;
