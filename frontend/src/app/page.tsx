import Link from "next/link";
import { Shield, MessageCircle, Heart, Users } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-slate-200/50 safe-area-inset-top">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 sm:py-4">
          <div className="flex items-center space-x-2 sm:space-x-3">
            <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-r from-blue-600 to-indigo-700 rounded-lg sm:rounded-xl flex items-center justify-center">
              <Shield className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg sm:text-xl font-bold text-slate-900">VAC Support Assistant</h1>
              <p className="text-xs sm:text-sm text-slate-600">Trauma-Informed AI Support</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex items-center justify-center min-h-[calc(100vh-80px)] px-4 sm:px-6 py-8 sm:py-12 safe-area-inset-bottom">
        <div className="max-w-4xl w-full">
          {/* Hero Section */}
          <div className="text-center mb-8 sm:mb-12">
            <div className="w-12 h-12 sm:w-16 sm:h-16 bg-gradient-to-r from-blue-600 to-indigo-700 rounded-xl sm:rounded-2xl flex items-center justify-center mx-auto mb-4 sm:mb-6">
              <Heart className="w-6 h-6 sm:w-8 sm:h-8 text-white" />
            </div>
            <h1 className="text-2xl sm:text-4xl md:text-5xl font-bold text-slate-900 mb-4 sm:mb-6 leading-tight px-2">
              Confidential Support<br className="hidden sm:block" />
              <span className="text-blue-600">When You Need It</span>
            </h1>
            <p className="text-lg sm:text-xl text-slate-600 mb-3 sm:mb-4 max-w-2xl mx-auto leading-relaxed px-2">
              A secure, trauma-informed AI assistant designed specifically for military personnel and veterans.
            </p>
            <p className="text-base sm:text-lg text-slate-500 max-w-xl mx-auto px-2">
              This tool complements professional care and provides supportive guidance in a safe, confidential environment.
            </p>
          </div>

          {/* Action Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6 mb-8 sm:mb-12 px-2 sm:px-0">
            <Link 
              href="/chat"
              className="group bg-white hover:bg-blue-50 border border-slate-200 hover:border-blue-300 rounded-xl sm:rounded-2xl p-6 sm:p-8 transition-all duration-300 hover:shadow-lg hover:-translate-y-1 mobile-button"
            >
              <div className="flex items-start space-x-3 sm:space-x-4">
                <div className="w-10 h-10 sm:w-12 sm:h-12 bg-blue-100 group-hover:bg-blue-200 rounded-lg sm:rounded-xl flex items-center justify-center flex-shrink-0">
                  <MessageCircle className="w-5 h-5 sm:w-6 sm:h-6 text-blue-600" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-lg sm:text-xl font-semibold text-slate-900 mb-2">Start a Conversation</h3>
                  <p className="text-sm sm:text-base text-slate-600 mb-3 sm:mb-4 leading-relaxed">
                    Begin a confidential chat with our trauma-informed AI assistant. Get support, ask questions, and find resources.
                  </p>
                  <div className="text-blue-600 font-medium group-hover:text-blue-700 text-sm sm:text-base">
                    Start Chat →
                  </div>
                </div>
              </div>
            </Link>
            
            <Link 
              href="/feedback"
              className="group bg-white hover:bg-green-50 border border-slate-200 hover:border-green-300 rounded-xl sm:rounded-2xl p-6 sm:p-8 transition-all duration-300 hover:shadow-lg hover:-translate-y-1 mobile-button"
            >
              <div className="flex items-start space-x-3 sm:space-x-4">
                <div className="w-10 h-10 sm:w-12 sm:h-12 bg-green-100 group-hover:bg-green-200 rounded-lg sm:rounded-xl flex items-center justify-center flex-shrink-0">
                  <Users className="w-5 h-5 sm:w-6 sm:h-6 text-green-600" />
                </div>
                <div className="flex-1 min-w-0">
                  <h3 className="text-lg sm:text-xl font-semibold text-slate-900 mb-2">Feedback & Analytics</h3>
                  <p className="text-sm sm:text-base text-slate-600 mb-3 sm:mb-4 leading-relaxed">
                    View community feedback and help improve the system. Your input helps make this tool better for everyone.
                  </p>
                  <div className="text-green-600 font-medium group-hover:text-green-700 text-sm sm:text-base">
                    View Dashboard →
                  </div>
                </div>
              </div>
            </Link>
          </div>

          {/* Key Features */}
          <div className="bg-white/70 backdrop-blur-sm rounded-xl sm:rounded-2xl border border-slate-200 p-6 sm:p-8 mx-2 sm:mx-0">
            <h3 className="text-xl sm:text-2xl font-semibold text-slate-900 mb-4 sm:mb-6 text-center">Designed With Your Well-being in Mind</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 sm:gap-6">
              <div className="text-center">
                <div className="w-10 h-10 sm:w-12 sm:h-12 bg-blue-100 rounded-lg sm:rounded-xl flex items-center justify-center mx-auto mb-2 sm:mb-3">
                  <Shield className="w-5 h-5 sm:w-6 sm:h-6 text-blue-600" />
                </div>
                <h4 className="font-semibold text-slate-900 mb-1 sm:mb-2 text-sm sm:text-base">Confidential & Secure</h4>
                <p className="text-xs sm:text-sm text-slate-600">Your conversations are private and protected with enterprise-grade security.</p>
              </div>
              <div className="text-center">
                <div className="w-10 h-10 sm:w-12 sm:h-12 bg-green-100 rounded-lg sm:rounded-xl flex items-center justify-center mx-auto mb-2 sm:mb-3">
                  <Heart className="w-5 h-5 sm:w-6 sm:h-6 text-green-600" />
                </div>
                <h4 className="font-semibold text-slate-900 mb-1 sm:mb-2 text-sm sm:text-base">Trauma-Informed</h4>
                <p className="text-xs sm:text-sm text-slate-600">Built with trauma-informed care principles and military cultural awareness.</p>
              </div>
              <div className="text-center sm:col-span-2 md:col-span-1">
                <div className="w-10 h-10 sm:w-12 sm:h-12 bg-indigo-100 rounded-lg sm:rounded-xl flex items-center justify-center mx-auto mb-2 sm:mb-3">
                  <Users className="w-5 h-5 sm:w-6 sm:h-6 text-indigo-600" />
                </div>
                <h4 className="font-semibold text-slate-900 mb-1 sm:mb-2 text-sm sm:text-base">Complement to Care</h4>
                <p className="text-xs sm:text-sm text-slate-600">Designed to support, not replace, professional mental health services.</p>
              </div>
            </div>
          </div>

          {/* Important Notice */}
          <div className="mt-6 sm:mt-8 p-4 sm:p-6 bg-amber-50 border border-amber-200 rounded-lg sm:rounded-xl mx-2 sm:mx-0">
            <p className="text-xs sm:text-sm text-amber-800 text-center leading-relaxed">
              <strong>Important:</strong> This AI assistant is a supportive tool and should not be used for crisis situations. 
              If you&#39;re experiencing a mental health emergency, please contact your local emergency services, 
              the Veterans Crisis Line (1-XXX-XXX-XXXX), or speak with a healthcare professional immediately.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
