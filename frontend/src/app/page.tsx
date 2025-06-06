import Link from "next/link";
import { Shield, MessageCircle, Heart, Users } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-slate-200/50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-indigo-700 rounded-xl flex items-center justify-center">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900">VAC Support Assistant</h1>
              <p className="text-sm text-slate-600">Trauma-Informed AI Support</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex items-center justify-center min-h-[calc(100vh-80px)] px-6 py-12">
        <div className="max-w-4xl w-full">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-600 to-indigo-700 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <Heart className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-slate-900 mb-6 leading-tight">
              Confidential Support<br />
              <span className="text-blue-600">When You Need It</span>
            </h1>
            <p className="text-xl text-slate-600 mb-4 max-w-2xl mx-auto leading-relaxed">
              A secure, trauma-informed AI assistant designed specifically for military personnel and veterans.
            </p>
            <p className="text-lg text-slate-500 max-w-xl mx-auto">
              This tool complements professional care and provides supportive guidance in a safe, confidential environment.
            </p>
          </div>

          {/* Action Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
            <Link 
              href="/chat"
              className="group bg-white hover:bg-blue-50 border border-slate-200 hover:border-blue-300 rounded-2xl p-8 transition-all duration-300 hover:shadow-lg hover:-translate-y-1"
            >
              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-blue-100 group-hover:bg-blue-200 rounded-xl flex items-center justify-center">
                  <MessageCircle className="w-6 h-6 text-blue-600" />
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold text-slate-900 mb-2">Start a Conversation</h3>
                  <p className="text-slate-600 mb-4 leading-relaxed">
                    Begin a confidential chat with our trauma-informed AI assistant. Get support, ask questions, and find resources.
                  </p>
                  <div className="text-blue-600 font-medium group-hover:text-blue-700">
                    Start Chat →
                  </div>
                </div>
              </div>
            </Link>
            
            <Link 
              href="/feedback"
              className="group bg-white hover:bg-green-50 border border-slate-200 hover:border-green-300 rounded-2xl p-8 transition-all duration-300 hover:shadow-lg hover:-translate-y-1"
            >
              <div className="flex items-start space-x-4">
                <div className="w-12 h-12 bg-green-100 group-hover:bg-green-200 rounded-xl flex items-center justify-center">
                  <Users className="w-6 h-6 text-green-600" />
                </div>
                <div className="flex-1">
                  <h3 className="text-xl font-semibold text-slate-900 mb-2">Feedback & Analytics</h3>
                  <p className="text-slate-600 mb-4 leading-relaxed">
                    View community feedback and help improve the system. Your input helps make this tool better for everyone.
                  </p>
                  <div className="text-green-600 font-medium group-hover:text-green-700">
                    View Dashboard →
                  </div>
                </div>
              </div>
            </Link>
          </div>

          {/* Key Features */}
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl border border-slate-200 p-8">
            <h3 className="text-2xl font-semibold text-slate-900 mb-6 text-center">Designed With Your Well-being in Mind</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <Shield className="w-6 h-6 text-blue-600" />
                </div>
                <h4 className="font-semibold text-slate-900 mb-2">Confidential & Secure</h4>
                <p className="text-sm text-slate-600">Your conversations are private and protected with enterprise-grade security.</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <Heart className="w-6 h-6 text-green-600" />
                </div>
                <h4 className="font-semibold text-slate-900 mb-2">Trauma-Informed</h4>
                <p className="text-sm text-slate-600">Built with trauma-informed care principles and military cultural awareness.</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-indigo-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <Users className="w-6 h-6 text-indigo-600" />
                </div>
                <h4 className="font-semibold text-slate-900 mb-2">Complement to Care</h4>
                <p className="text-sm text-slate-600">Designed to support, not replace, professional mental health services.</p>
              </div>
            </div>
          </div>

          {/* Important Notice */}
          <div className="mt-8 p-6 bg-amber-50 border border-amber-200 rounded-xl">
            <p className="text-sm text-amber-800 text-center">
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
