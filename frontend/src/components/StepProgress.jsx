import React from 'react';

const StepProgress = ({ currentStep, totalSteps = 3 }) => {
  const steps = ["Basic Info", "Symptoms", "Supplements"];
  
  return (
    <div className="w-full mb-12">
      <div className="flex justify-between mb-2">
        {steps.map((step, idx) => (
          <div 
            key={idx} 
            className={`text-sm font-medium transition-colors duration-300 ${
              idx + 1 <= currentStep ? 'text-accent' : 'text-slate-500'
            }`}
          >
            {step}
          </div>
        ))}
      </div>
      <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
        <div 
          className="h-full bg-accent transition-all duration-500 ease-out"
          style={{ width: `${(currentStep / totalSteps) * 100}%` }}
        />
      </div>
    </div>
  );
};

export default StepProgress;
