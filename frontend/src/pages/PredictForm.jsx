import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import StepProgress from '../components/StepProgress';
import SliderInput from '../components/SliderInput';
import LoadingOverlay from '../components/LoadingOverlay';
import { Info, ArrowRight, ArrowLeft, ChevronRight, AlertTriangle, RefreshCcw, Activity } from 'lucide-react';
import { useGlobal } from '../context/GlobalContext';

const PredictForm = () => {
  const navigate = useNavigate();
  const { addToast } = useGlobal();
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isSlow, setIsSlow] = useState(false);
  const timeoutRef = useRef(null);
  const [formData, setFormData] = useState({
    age: 45,
    gender: "Female",
    bmi: 24.5,
    duration_weeks: 4,
    pain_level: 5,
    swelling_level: 3,
    activity_level: 7,
    beetroot_intake: "None",
    fenugreek_intake: "None"
  });

  const updateField = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setIsSlow(false);
    
    // Set timeout for slow response
    timeoutRef.current = setTimeout(() => {
      setIsSlow(true);
    }, 10000);

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      
      clearTimeout(timeoutRef.current);
      
      if (!response.ok) throw new Error("API service error");
      
      const data = await response.json();
      addToast("Analysis complete!", "success");
      
      setTimeout(() => {
        setLoading(false);
        navigate('/results', { state: { result: data, input: formData } });
      }, 1000);
    } catch (err) {
      clearTimeout(timeoutRef.current);
      console.error("Error:", err);
      setLoading(false);
      setError("Service temporarily unavailable. Please check if the backend server is running.");
      addToast("Prediction failed", "error");
    }
  };

  const SummaryCard = () => (
    <div className="bg-card border border-border p-6 rounded-2xl sticky top-8 transition-colors duration-300">
      <h3 className="text-accent font-bold mb-4 flex items-center gap-2">
        <Activity size={18} /> Profile Summary
      </h3>
      <div className="space-y-3 text-sm">
        <div className="flex justify-between border-b border-border pb-2">
          <span className="text-slate-400 dark:text-slate-500">Basic Info</span>
          <span className="dark:text-white text-slate-900">{formData.age}y, {formData.gender}</span>
        </div>
        <div className="flex justify-between border-b border-border pb-2">
          <span className="text-slate-400 dark:text-slate-500">BMI</span>
          <span className="dark:text-white text-slate-900">{formData.bmi}</span>
        </div>
        <div className="flex justify-between border-b border-border pb-2">
          <span className="text-slate-400 dark:text-slate-500">Pain Level</span>
          <span className={formData.pain_level > 7 ? 'text-red-500 font-bold' : 'text-accent font-bold'}>{formData.pain_level}/10</span>
        </div>
        <div className="flex justify-between border-b border-border pb-2">
          <span className="text-slate-400 dark:text-slate-500">Swelling</span>
          <span className="dark:text-white text-slate-900">{formData.swelling_level}/10</span>
        </div>
        <div className="flex justify-between border-b border-border pb-2">
          <span className="text-slate-400 dark:text-slate-500">Activity</span>
          <span className="dark:text-white text-slate-900">{formData.activity_level}/10</span>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-background text-slate-900 dark:text-white p-4 md:p-8">
      <LoadingOverlay isVisible={loading} />
      
      {isSlow && loading && (
        <div className="fixed top-24 left-1/2 -translate-x-1/2 z-[60] bg-amber-500/90 text-white px-6 py-3 rounded-xl shadow-lg flex items-center gap-3 animate-bounce">
          <Info size={20} /> This is taking longer than usual...
        </div>
      )}

      <div className="max-w-5xl mx-auto">
        {error ? (
          <div className="bg-card border-2 border-red-500/20 p-12 rounded-[2.5rem] text-center max-w-2xl mx-auto mt-20 animate-in zoom-in-95 duration-300">
            <div className="w-20 h-20 bg-red-500/10 rounded-full flex items-center justify-center mx-auto mb-6">
              <AlertTriangle className="text-red-500" size={40} />
            </div>
            <h2 className="text-2xl font-bold mb-4">Connection Failed</h2>
            <p className="text-slate-500 mb-8">{error}</p>
            <button 
              onClick={() => { setError(null); handlePredict(); }}
              className="bg-accent text-white px-8 py-3 rounded-xl font-bold flex items-center gap-2 mx-auto hover:bg-teal-600 transition-all"
            >
              <RefreshCcw size={18} /> Retry Connection
            </button>
          </div>
        ) : (
          <>
            <div className="flex items-center gap-4 mb-8">
          <button 
            onClick={() => navigate('/')}
            className="p-2 hover:bg-slate-800 rounded-full transition-colors"
          >
            <ArrowLeft />
          </button>
          <h1 className="text-2xl md:text-3xl font-bold">Assessment Tool</h1>
        </div>

        <StepProgress currentStep={step} />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="lg:col-span-2 bg-card border border-border p-8 rounded-3xl shadow-2xl">
            {/* Step 1: Basic Info */}
            {step === 1 && (
              <div className="animate-in fade-in slide-in-from-right-4 duration-500">
                <h2 className="text-2xl font-bold mb-8">Basic Information</h2>
                <SliderInput 
                  label="Age" 
                  min={18} max={80} 
                  value={formData.age} 
                  onChange={(v) => updateField('age', v)} 
                  unit=" years"
                />
                
                <div className="mb-8">
                  <label className="block text-slate-300 font-medium mb-4">Gender</label>
                  <div className="grid grid-cols-3 gap-4">
                    {["Male", "Female", "Other"].map(g => (
                      <button
                        key={g}
                        onClick={() => updateField('gender', g)}
                        className={`py-3 rounded-xl border transition-all ${
                          formData.gender === g 
                            ? 'bg-accent/10 border-accent text-accent' 
                            : 'bg-slate-800/50 border-slate-700 text-slate-400 hover:border-slate-500'
                        }`}
                      >
                        {g}
                      </button>
                    ))}
                  </div>
                </div>

                <SliderInput 
                  label="BMI" 
                  min={15} max={50} 
                  value={formData.bmi} 
                  onChange={(v) => updateField('bmi', v)} 
                />

                <SliderInput 
                  label="Condition Duration" 
                  min={1} max={52} 
                  value={formData.duration_weeks} 
                  onChange={(v) => updateField('duration_weeks', v)} 
                  unit=" weeks"
                />
              </div>
            )}

            {/* Step 2: Symptoms */}
            {step === 2 && (
              <div className="animate-in fade-in slide-in-from-right-4 duration-500">
                <h2 className="text-2xl font-bold mb-8">Symptom Assessment</h2>
                <SliderInput 
                  label="Pain Level" 
                  min={1} max={10} 
                  value={formData.pain_level} 
                  onChange={(v) => updateField('pain_level', v)} 
                  labels={{ 1: "None", 5: "Moderate", 10: "Severe" }}
                />
                <SliderInput 
                  label="Swelling Level" 
                  min={1} max={10} 
                  value={formData.swelling_level} 
                  onChange={(v) => updateField('swelling_level', v)} 
                />
                <SliderInput 
                  label="Activity Level" 
                  min={1} max={10} 
                  value={formData.activity_level} 
                  onChange={(v) => updateField('activity_level', v)} 
                  labels={{ 1: "Bedridden", 5: "Limited", 10: "Active" }}
                />
              </div>
            )}

            {/* Step 3: Supplements */}
            {step === 3 && (
              <div className="animate-in fade-in slide-in-from-right-4 duration-500">
                <h2 className="text-2xl font-bold mb-8">Supplement Intake</h2>
                
                <div className="space-y-8">
                  {["Beetroot", "Fenugreek"].map(item => (
                    <div key={item} className="bg-slate-800/30 p-6 rounded-2xl border border-slate-800">
                      <div className="flex items-center gap-2 mb-4">
                        <label className="text-lg font-bold">{item} Intake</label>
                        <div className="group relative">
                          <Info size={16} className="text-slate-500 cursor-help" />
                          <div className="absolute left-0 bottom-full mb-2 w-64 p-3 bg-slate-900 text-xs rounded-xl border border-border opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                            {item === 'Beetroot' 
                              ? "Daily consumption of beetroot extract or powder. High = 30 days/mo @ 15g+" 
                              : "Daily consumption of fenugreek seeds or extract. High = 30 days/mo @ 12g+"}
                          </div>
                        </div>
                      </div>
                      <div className="grid grid-cols-4 gap-3">
                        {["None", "Low", "Medium", "High"].map(level => (
                          <button
                            key={level}
                            onClick={() => updateField(`${item.toLowerCase()}_intake`, level)}
                            className={`py-3 rounded-xl border transition-all ${
                              formData[`${item.toLowerCase()}_intake`] === level 
                                ? 'bg-accent border-accent text-white' 
                                : 'bg-slate-900 border-slate-700 text-slate-500 hover:border-slate-500'
                            }`}
                          >
                            {level}
                          </button>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Navigation Buttons */}
            <div className="mt-12 flex justify-between gap-4">
              {step > 1 && (
                <button
                  onClick={() => setStep(step - 1)}
                  className="px-8 py-3 rounded-xl border border-slate-700 text-slate-300 hover:bg-slate-800 transition-colors"
                >
                  Back
                </button>
              )}
              {step < 3 ? (
                <button
                  onClick={() => setStep(step + 1)}
                  className="flex-1 bg-accent hover:bg-teal-600 py-3 rounded-xl font-bold flex items-center justify-center gap-2 transition-all"
                >
                  Continue <ChevronRight size={20} />
                </button>
              ) : (
                <button
                  onClick={handlePredict}
                  className="flex-1 bg-accent hover:bg-teal-600 py-3 rounded-xl font-bold flex items-center justify-center gap-2 transition-all shadow-lg shadow-teal-500/20"
                >
                  Predict Recovery <ArrowRight size={20} />
                </button>
              )}
            </div>
          </div>

          <div className="hidden lg:block">
            <SummaryCard />
          </div>
        </div>
      </>
    )}
      </div>
    </div>
  );
};

export default PredictForm;
