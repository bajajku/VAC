'use client';

import FeedbackDashboard from '../../components/FeedbackDashboard';
import Link from 'next/link';
import { ArrowLeft, Shield } from 'lucide-react';

const FeedbackPage = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 safe-area-inset-top safe-area-inset-bottom">
      {/* Header */}
      <header className="bg-white/90 backdrop-blur-md border-b border-slate-200/50 px-3 sm:px-4 md:px-6 py-3 sm:py-4 shadow-sm">
        <div className="flex items-center space-x-2 sm:space-x-3">
          <Link 
            href="/" 
            className="w-8 h-8 bg-slate-100 hover:bg-slate-200 rounded-lg flex items-center justify-center transition-colors touch-target"
            aria-label="Return to home"
          >
            <ArrowLeft className="w-4 h-4 text-slate-600" />
          </Link>
          <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-r from-green-600 to-emerald-700 rounded-lg sm:rounded-xl flex items-center justify-center">
            <Shield className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
          </div>
          <div className="min-w-0 flex-1">
            <h1 className="text-sm sm:text-lg font-semibold text-slate-900 truncate">Feedback & Analytics Dashboard</h1>
            <p className="text-xs sm:text-sm text-slate-600">Community insights • System improvement</p>
          </div>
        </div>
      </header>

      {/* Dashboard Content */}
      <div className="px-3 sm:px-4 md:px-6 py-4 sm:py-6">
        <FeedbackDashboard />
      </div>
    </div>
  );
};

export default FeedbackPage;
