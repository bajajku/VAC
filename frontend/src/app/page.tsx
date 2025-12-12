"use client"
import Link from "next/link";
import { Shield, ArrowRight } from "lucide-react";
import ThemeToggle from "../components/ThemeToggle";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-900 dark:to-slate-800 transition-colors duration-300">
      {/* Minimal Header */}
      <header className="absolute top-0 left-0 right-0 z-10 px-4 sm:px-6 py-4 sm:py-6">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-indigo-700 rounded-xl flex items-center justify-center">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <span className="text-lg font-semibold text-slate-900 dark:text-slate-100">VAC Support</span>
          </div>
          <ThemeToggle />
        </div>
      </header>

      {/* Hero - Centered */}
      <main className="min-h-screen flex flex-col items-center justify-center px-4 sm:px-6">
        <div className="max-w-2xl w-full text-center">
          {/* Main Message */}
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-semibold text-slate-900 dark:text-slate-100 mb-6 leading-tight tracking-tight">
            A safe space to talk.
          </h1>

          <p className="text-lg sm:text-xl text-slate-600 dark:text-slate-400 mb-10 max-w-lg mx-auto leading-relaxed">
            Confidential, trauma-informed support for Veterans and military personnel. Available whenever you need it.
          </p>

          {/* Single CTA */}
          <Link
            href="/chat"
            className="inline-flex items-center justify-center space-x-2 bg-gradient-to-r from-blue-600 to-indigo-700 hover:from-blue-700 hover:to-indigo-800 text-white px-8 py-4 rounded-xl font-medium text-lg transition-all duration-200 hover:shadow-lg hover:scale-[1.02] active:scale-[0.98]"
          >
            <span>Start a conversation</span>
            <ArrowRight className="w-5 h-5" />
          </Link>

          {/* Trust indicators - subtle */}
          <div className="mt-12 flex items-center justify-center space-x-6 text-sm text-slate-500 dark:text-slate-500">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span>Secure</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <span>Confidential</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
              <span>24/7</span>
            </div>
          </div>
        </div>

        {/* Crisis Notice - Bottom */}
        <div className="absolute bottom-0 left-0 right-0 px-4 py-4 sm:py-6">
          <p className="text-xs sm:text-sm text-slate-500 dark:text-slate-500 text-center max-w-2xl mx-auto">
            If you&apos;re in crisis, please contact the Veterans Crisis Line:{" "}
            <a href="tel:18334564566" className="text-blue-600 dark:text-blue-400 hover:underline font-medium">
              1-833-456-4566
            </a>
          </p>
        </div>
      </main>
    </div>
  );
}
