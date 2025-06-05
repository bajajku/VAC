import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 flex items-center justify-center">
      <div className="max-w-2xl text-center px-6">
        <h1 className="text-4xl font-bold text-slate-900 mb-4">
          VAC AI Assistant
        </h1>
        <p className="text-slate-600 mb-8 text-lg">
          Experience our AI assistant and help us improve through your feedback.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Link 
            href="/chat"
            className="bg-gradient-to-r from-blue-500 to-blue-600 text-white px-8 py-6 rounded-xl hover:from-blue-600 hover:to-blue-700 transition-all duration-200 shadow-lg hover:shadow-xl group"
          >
            <div className="text-xl font-semibold mb-2">Start Chatting</div>
            <div className="text-blue-100 text-sm">
              Interact with our AI assistant and get help with your questions.
            </div>
          </Link>
          
          <Link 
            href="/feedback"
            className="bg-gradient-to-r from-green-500 to-green-600 text-white px-8 py-6 rounded-xl hover:from-green-600 hover:to-green-700 transition-all duration-200 shadow-lg hover:shadow-xl group"
          >
            <div className="text-xl font-semibold mb-2">Feedback Dashboard</div>
            <div className="text-green-100 text-sm">
              View collected feedback and analytics to help improve the system.
            </div>
          </Link>
        </div>
        
        <div className="mt-8 p-6 bg-white/50 backdrop-blur-sm rounded-xl border border-slate-200">
          <h3 className="text-lg font-semibold text-slate-800 mb-2">How it works</h3>
          <p className="text-slate-600 text-sm">
            1. Chat with our AI assistant • 2. Rate the responses • 3. View analytics and feedback data
          </p>
        </div>
      </div>
    </div>
  );
}
