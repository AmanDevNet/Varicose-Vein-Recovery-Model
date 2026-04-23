import React from 'react';
import { Link } from 'react-router-dom';
import MetricCard from '../components/MetricCard';
import { ArrowRight, Activity, Shield, PieChart } from 'lucide-react';

const Landing = () => {
  const metrics = [
    { label: "Accuracy", value: "94.3%", description: "Verified by test data" },
    { label: "AUC-ROC", value: "0.986", description: "High classification confidence" },
    { label: "Training Set", value: "50k", description: "Synthetic patient samples" },
    { label: "Models", value: "3", description: "RF, GBT & XGB Ensemble" }
  ];

  const steps = [
    {
      icon: <Activity className="w-8 h-8 text-accent" />,
      title: "Step 1: Input Data",
      description: "Enter patient symptoms, lifestyle factors, and supplement intake."
    },
    {
      icon: <Shield className="w-8 h-8 text-accent" />,
      title: "Step 2: ML Analysis",
      description: "Our ensemble model predicts risk levels and recovery timelines."
    },
    {
      icon: <PieChart className="w-8 h-8 text-accent" />,
      title: "Step 3: Get Results",
      description: "Receive a detailed PDF report with SHAP explainability values."
    }
  ];

  return (
    <div className="min-h-screen bg-background text-white selection:bg-accent selection:text-white">
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-4 overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-full max-w-6xl opacity-20 pointer-events-none">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-accent rounded-full blur-[128px]" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-blue-600 rounded-full blur-[128px]" />
        </div>

        <div className="max-w-4xl mx-auto text-center relative z-10">
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight mb-6">
            Varicose Vein <span className="text-accent">Recovery Predictor</span>
          </h1>
          <p className="text-xl md:text-2xl text-slate-400 mb-10 max-w-2xl mx-auto">
            ML-powered recovery risk assessment using Random Forest, Gradient Boosting & XGBoost.
          </p>
          <div className="flex flex-col items-center gap-4">
            <Link 
              to="/predict" 
              className="bg-accent hover:bg-teal-600 text-white px-8 py-4 rounded-full text-lg font-bold transition-all transform hover:scale-105 flex items-center gap-2 shadow-lg shadow-teal-500/20"
            >
              Run Prediction <ArrowRight size={20} />
            </Link>
            <p className="text-sm text-slate-500 max-w-xs">
              For educational purposes only. Not a substitute for medical advice.
            </p>
          </div>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="py-12 px-4 max-w-6xl mx-auto">
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
          {metrics.map((m, i) => (
            <MetricCard key={i} {...m} />
          ))}
        </div>
      </section>

      {/* How it Works */}
      <section className="py-24 px-4 bg-slate-900/30">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-16">How It Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            {steps.map((step, i) => (
              <div key={i} className="text-center group">
                <div className="w-16 h-16 bg-card border border-border rounded-2xl flex items-center justify-center mx-auto mb-6 group-hover:border-accent transition-colors">
                  {step.icon}
                </div>
                <h3 className="text-xl font-bold mb-4">{step.title}</h3>
                <p className="text-slate-400 leading-relaxed">{step.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-border/50 text-center text-slate-500 text-sm">
        &copy; 2026 Varicose Vein Recovery ML Project. All rights reserved.
      </footer>
    </div>
  );
};

export default Landing;
