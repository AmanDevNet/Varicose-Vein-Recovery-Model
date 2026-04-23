import React from 'react';

const MetricCard = ({ label, value, description }) => {
  return (
    <div className="bg-card border border-border p-6 rounded-2xl hover:border-accent transition-all duration-300 group">
      <div className="text-accent text-2xl md:text-3xl font-black mb-1 group-hover:scale-105 transition-transform">{value}</div>
      <div className="dark:text-white text-slate-900 font-bold text-sm md:text-lg mb-1">{label}</div>
      <div className="text-slate-500 text-xs md:text-sm">{description}</div>
    </div>
  );
};

export default MetricCard;
