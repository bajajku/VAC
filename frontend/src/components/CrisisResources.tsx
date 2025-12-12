"use client"
import React, { useState } from 'react';
import { Phone, MessageCircle, ChevronDown, ChevronUp, Heart } from 'lucide-react';

interface CrisisResourcesProps {
  variant?: 'banner' | 'compact' | 'full';
  showExpanded?: boolean;
}

const CrisisResources: React.FC<CrisisResourcesProps> = ({
  variant = 'banner',
  showExpanded = false
}) => {
  const [isExpanded, setIsExpanded] = useState(showExpanded);

  const resources = [
    {
      name: "Veterans Crisis Line (Canada)",
      phone: "1-833-456-4566",
      description: "24/7 confidential support for Veterans and their families",
      icon: Phone,
      primary: true
    },
    {
      name: "VAC Assistance Service",
      phone: "1-800-268-7708",
      description: "Mental health support and counseling referrals",
      icon: MessageCircle,
      primary: false
    },
    {
      name: "Crisis Text Line",
      phone: "Text HOME to 686868",
      description: "Free, 24/7 text-based mental health support",
      icon: MessageCircle,
      primary: false
    }
  ];

  if (variant === 'compact') {
    return (
      <div className="bg-gradient-to-r from-emerald-50 to-teal-50 border border-emerald-200 rounded-lg px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Heart className="w-4 h-4 text-emerald-600" />
          <span className="text-sm text-emerald-800">
            Need immediate support? <strong>1-833-456-4566</strong>
          </span>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-emerald-600 hover:text-emerald-700 text-sm underline"
        >
          More resources
        </button>
      </div>
    );
  }

  if (variant === 'banner') {
    return (
      <div className="bg-gradient-to-r from-emerald-50 via-teal-50 to-cyan-50 border-b border-emerald-200">
        <div className="max-w-4xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-emerald-100 p-2 rounded-full">
                <Heart className="w-5 h-5 text-emerald-600" />
              </div>
              <div>
                <p className="text-sm font-medium text-emerald-900">
                  You&apos;re not alone. Support is available 24/7.
                </p>
                <p className="text-xs text-emerald-700">
                  Veterans Crisis Line: <strong>1-833-456-4566</strong>
                </p>
              </div>
            </div>
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="flex items-center gap-1 text-emerald-700 hover:text-emerald-800 text-sm font-medium transition-colors"
              aria-expanded={isExpanded}
              aria-label={isExpanded ? "Hide resources" : "Show more resources"}
            >
              {isExpanded ? (
                <>Hide <ChevronUp className="w-4 h-4" /></>
              ) : (
                <>More <ChevronDown className="w-4 h-4" /></>
              )}
            </button>
          </div>

          {isExpanded && (
            <div className="mt-4 pt-4 border-t border-emerald-200 grid gap-3 md:grid-cols-3">
              {resources.map((resource, index) => (
                <div
                  key={index}
                  className={`p-3 rounded-lg ${
                    resource.primary
                      ? 'bg-emerald-100 border border-emerald-300'
                      : 'bg-white/70 border border-emerald-100'
                  }`}
                >
                  <div className="flex items-start gap-2">
                    <resource.icon className={`w-4 h-4 mt-0.5 ${
                      resource.primary ? 'text-emerald-700' : 'text-emerald-600'
                    }`} />
                    <div>
                      <p className={`text-sm font-medium ${
                        resource.primary ? 'text-emerald-900' : 'text-emerald-800'
                      }`}>
                        {resource.name}
                      </p>
                      <p className={`text-sm font-semibold ${
                        resource.primary ? 'text-emerald-700' : 'text-emerald-600'
                      }`}>
                        {resource.phone}
                      </p>
                      <p className="text-xs text-emerald-600 mt-1">
                        {resource.description}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Full variant for dedicated resources page/modal
  return (
    <div className="bg-white rounded-xl shadow-lg border border-emerald-100 overflow-hidden">
      <div className="bg-gradient-to-r from-emerald-500 to-teal-500 px-6 py-4">
        <div className="flex items-center gap-3">
          <Heart className="w-6 h-6 text-white" />
          <div>
            <h2 className="text-lg font-semibold text-white">Crisis Support Resources</h2>
            <p className="text-emerald-100 text-sm">Help is available 24/7, 365 days a year</p>
          </div>
        </div>
      </div>
      <div className="p-6 space-y-4">
        {resources.map((resource, index) => (
          <div
            key={index}
            className={`p-4 rounded-lg border ${
              resource.primary
                ? 'bg-emerald-50 border-emerald-200'
                : 'bg-gray-50 border-gray-200'
            }`}
          >
            <div className="flex items-start gap-3">
              <div className={`p-2 rounded-full ${
                resource.primary ? 'bg-emerald-200' : 'bg-gray-200'
              }`}>
                <resource.icon className={`w-5 h-5 ${
                  resource.primary ? 'text-emerald-700' : 'text-gray-600'
                }`} />
              </div>
              <div className="flex-1">
                <h3 className="font-medium text-gray-900">{resource.name}</h3>
                <a
                  href={resource.phone.startsWith('Text') ? '#' : `tel:${resource.phone.replace(/[^0-9]/g, '')}`}
                  className={`text-lg font-semibold ${
                    resource.primary ? 'text-emerald-600' : 'text-blue-600'
                  } hover:underline`}
                >
                  {resource.phone}
                </a>
                <p className="text-sm text-gray-600 mt-1">{resource.description}</p>
              </div>
              {!resource.phone.startsWith('Text') && (
                <a
                  href={`tel:${resource.phone.replace(/[^0-9]/g, '')}`}
                  className="bg-emerald-500 hover:bg-emerald-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
                >
                  <Phone className="w-4 h-4" />
                  Call Now
                </a>
              )}
            </div>
          </div>
        ))}

        <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <p className="text-sm text-blue-800">
            <strong>Remember:</strong> Reaching out for help is a sign of strength, not weakness.
            These services are confidential and staffed by people who understand military and Veteran experiences.
          </p>
        </div>
      </div>
    </div>
  );
};

export default CrisisResources;
