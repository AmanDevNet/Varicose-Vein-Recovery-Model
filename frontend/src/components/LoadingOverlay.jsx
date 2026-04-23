import React, { useState, useEffect } from 'react';

const LoadingOverlay = ({ isVisible }) => {
  const [messageIdx, setMessageIdx] = useState(0);
  const messages = [
    "Analyzing symptoms...",
    "Running ensemble models...",
    "Calculating SHAP values...",
    "Generating recommendations..."
  ];

  useEffect(() => {
    if (!isVisible) return;
    const interval = setInterval(() => {
      setMessageIdx((prev) => (prev + 1) % messages.length);
    }, 1500);
    return () => clearInterval(interval);
  }, [isVisible]);

  if (!isVisible) return null;

  return (
    <div className="fixed inset-0 z-50 bg-background/90 backdrop-blur-sm flex flex-col items-center justify-center p-6">
      <div className="relative w-24 h-24 mb-8">
        <div className="absolute inset-0 border-4 border-slate-800 rounded-full" />
        <div className="absolute inset-0 border-4 border-accent rounded-full border-t-transparent animate-spin" />
      </div>
      <div className="text-2xl font-bold text-white mb-2 animate-pulse">
        {messages[messageIdx]}
      </div>
      <p className="text-slate-500 text-sm">Processing results from ML ensemble</p>
    </div>
  );
};

export default LoadingOverlay;
