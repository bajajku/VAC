import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="max-w-md text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          VAC Feedback Portal
        </h1>
        <p className="text-gray-600 mb-6">
          Your feedback helps us improve our chatbot.
        </p>
        <Link 
          href={"/feedback"}
          className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700">
          Provide Feedback
        </Link>
      </div>
    </div>
  );
}
