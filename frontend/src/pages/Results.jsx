import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Download, RefreshCw, CheckCircle2, ChevronRight, Share2, Clipboard, TrendingDown, Info, Loader2 } from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, Cell
} from 'recharts';
import { useGlobal } from '../context/GlobalContext';

const Results = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { addToast, removeToast } = useGlobal();
  const [pdfLoading, setPdfLoading] = useState(false);
  
  // Use state from navigation or dummy data for development
  const data = location.state?.result || {
    "risk_level": "Moderate",
    "risk_confidence": 0.84,
    "recovery_weeks_min": 7,
    "recovery_weeks_max": 11,
    "recovery_weeks_mean": 9,
    "scenarios": {
      "no_supplements": {"risk": "High", "recovery_weeks": 14},
      "current_intake": {"risk": "Moderate", "recovery_weeks": 9},
      "optimal_intake": {"risk": "Low", "recovery_weeks": 6}
    },
    "shap_values": [
      {"feature": "BMI", "value": 0.31, "impact": "increases risk"},
      {"feature": "Age", "value": 0.24, "impact": "increases risk"},
      {"feature": "Pain Level", "value": 0.18, "impact": "increases risk"},
      {"feature": "Beetroot Intake", "value": -0.15, "impact": "reduces risk"},
      {"feature": "Activity Level", "value": -0.12, "impact": "reduces risk"}
    ],
    "recommendations": [
      "Increase beetroot intake to optimal level",
      "Aim to improve activity level gradually",
      "Monitor swelling weekly"
    ]
  };

  const inputData = location.state?.input || {};

  const getRiskColor = (level) => {
    switch (level) {
      case 'Low': return 'bg-[#22c55e]';
      case 'Moderate': return 'bg-[#f59e0b]';
      case 'High': return 'bg-[#ef4444]';
      default: return 'bg-slate-500';
    }
  };

  const handleDownloadPDF = async () => {
    setPdfLoading(true);
    const toastId = addToast("Generating your medical report...", "loading");
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}/generate-report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prediction_result: data,
          input_data: inputData
        })
      });
      
      if (!response.ok) throw new Error("PDF generation failed");
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      a.download = `varicose_vein_report_${timestamp}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      addToast("Report downloaded successfully!", "success");
    } catch (error) {
      console.error("PDF download failed:", error);
      addToast("Failed to generate report. Try again.", "error");
    } finally {
      setPdfLoading(false);
      removeToast(toastId);
    }
  };

  const TimelineChart = ({ data }) => {
    const [metrics, setMetrics] = useState({ pain: true, swelling: true, activity: true });
    
    // Generate data points for the chart
    const maxWeeks = Math.max(
      data.scenarios.no_supplements.recovery_weeks,
      data.scenarios.current_intake.recovery_weeks,
      data.scenarios.optimal_intake.recovery_weeks
    );
    
    const chartData = [];
    for (let i = 0; i <= maxWeeks + 2; i++) {
      const getSeverity = (targetWeeks, week) => {
        if (week >= targetWeeks) return 0;
        return Math.round(100 * Math.exp(-3 * week / targetWeeks));
      };
      
      chartData.push({
        name: `Week ${i}`,
        week: i,
        no_supp: getSeverity(data.scenarios.no_supplements.recovery_weeks, i),
        current: getSeverity(data.scenarios.current_intake.recovery_weeks, i),
        optimal: getSeverity(data.scenarios.optimal_intake.recovery_weeks, i),
      });
    }

    return (
      <div className="w-full h-full flex flex-col">
        <div className="flex gap-6 mb-4">
          {['pain', 'swelling', 'activity'].map(m => (
            <label key={m} className="flex items-center gap-2 cursor-pointer group">
              <input 
                type="checkbox" 
                checked={metrics[m]} 
                onChange={() => setMetrics(prev => ({ ...prev, [m]: !prev[m] }))}
                className="w-4 h-4 rounded border-slate-700 bg-slate-900 text-accent focus:ring-accent"
              />
              <span className="text-xs text-slate-400 group-hover:text-slate-200 capitalize">{m}</span>
            </label>
          ))}
        </div>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
            <XAxis 
              dataKey="week" 
              stroke="#64748b" 
              fontSize={12} 
              tickLine={false} 
              axisLine={false}
              label={{ value: 'Weeks', position: 'insideBottomRight', offset: -10, fill: '#64748b', fontSize: 10 }}
            />
            <YAxis 
              stroke="#64748b" 
              fontSize={12} 
              tickLine={false} 
              axisLine={false}
              tickFormatter={(v) => `${v}%`}
              label={{ value: 'Severity', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }}
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
              itemStyle={{ fontSize: '12px' }}
            />
            <Line 
              type="monotone" 
              dataKey="no_supp" 
              stroke="#ef4444" 
              strokeWidth={2} 
              strokeDasharray="5 5" 
              dot={false}
              name="No Supplements"
            />
            <Line 
              type="monotone" 
              dataKey="current" 
              stroke="#f59e0b" 
              strokeWidth={3} 
              dot={{ r: 4, fill: '#f59e0b' }}
              activeDot={{ r: 6 }}
              name="Current Intake"
            />
            <Line 
              type="monotone" 
              dataKey="optimal" 
              stroke="#14b8a6" 
              strokeWidth={3} 
              dot={{ r: 4, fill: '#14b8a6' }}
              activeDot={{ r: 6 }}
              name="Optimal Intake"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-background text-white p-4 md:p-12">
      <div className="max-w-6xl mx-auto">
        
        {/* SECTION 1: Risk Summary */}
        <section className="bg-card border border-border p-10 rounded-[2.5rem] shadow-2xl relative overflow-hidden mb-8">
          <div className="absolute top-0 right-0 w-64 h-64 bg-accent/5 rounded-full -mr-32 -mt-32 blur-3xl" />
          
          <div className="flex flex-col md:flex-row justify-between items-center gap-8 relative z-10">
            <div className="text-center md:text-left">
              <h2 className="text-slate-400 font-medium mb-4 uppercase tracking-widest text-sm">Health Risk Assessment</h2>
              <div className="flex flex-col items-center md:items-start gap-4">
                <div className={`${getRiskColor(data.risk_level)} text-white px-10 py-3 rounded-full text-2xl font-black shadow-lg uppercase tracking-wider`}>
                  {data.risk_level} Risk
                </div>
                <div className="text-accent font-bold text-lg">
                  {Math.round(data.risk_confidence * 100)}% Confidence Level
                </div>
              </div>
            </div>

            <div className="text-center">
              <div className="text-slate-500 text-sm mb-1">Estimated Recovery Time</div>
              <div className="text-4xl md:text-6xl font-extrabold text-white mb-2">
                {data.recovery_weeks_min}–{data.recovery_weeks_max} <span className="text-2xl text-slate-500">weeks</span>
              </div>
              <div className="text-accent/80 font-medium">Mean Duration: {data.recovery_weeks_mean} weeks</div>
            </div>

            <div className="flex flex-col gap-3 w-full md:w-auto">
              <button 
                onClick={handleDownloadPDF}
                disabled={pdfLoading}
                className="bg-accent hover:bg-teal-600 disabled:bg-slate-700 text-white px-6 py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all shadow-lg shadow-teal-500/20 min-h-[44px]"
              >
                {pdfLoading ? <Loader2 className="animate-spin" size={18} /> : <Download size={18} />} 
                {pdfLoading ? 'Generating...' : 'Download PDF Report'}
              </button>
              <button 
                onClick={() => navigate('/predict')}
                className="bg-slate-800 hover:bg-slate-700 text-slate-300 px-6 py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all border border-slate-700 min-h-[44px]"
              >
                <RefreshCw size={18} /> Try Different Inputs
              </button>
            </div>
          </div>
        </section>

        {/* SECTION 2: Three Scenario Comparison */}
        <section className="mb-12">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {Object.entries(data.scenarios).map(([key, scenario]) => {
              const isOptimal = key === 'optimal_intake';
              const isCurrent = key === 'current_intake';
              
              return (
                <div 
                  key={key}
                  className={`bg-card p-8 rounded-3xl border transition-all duration-500 ${
                    isOptimal ? 'border-accent ring-1 ring-accent/30 shadow-lg shadow-accent/10' : 'border-border'
                  }`}
                >
                  <div className="flex justify-between items-start mb-6">
                    <h3 className="text-slate-400 font-bold text-sm uppercase tracking-wider">
                      {key.replace('_', ' ')}
                    </h3>
                    {isOptimal && (
                      <span className="bg-accent/20 text-accent text-[10px] px-2 py-1 rounded-md font-bold uppercase">Recommended</span>
                    )}
                  </div>

                  <div className="space-y-6">
                    <div>
                      <div className="text-slate-500 text-xs mb-2">Projected Risk</div>
                      <div className={`inline-block px-4 py-1 rounded-full text-xs font-bold uppercase ${getRiskColor(scenario.risk)}`}>
                        {scenario.risk}
                      </div>
                    </div>
                    
                    <div>
                      <div className="text-slate-500 text-xs mb-1">Recovery Duration</div>
                      <div className="text-3xl font-bold">
                        {scenario.recovery_weeks} <span className="text-sm text-slate-500 font-normal">weeks</span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
          
          <div className="mt-8 text-center">
            <div className="inline-flex items-center gap-3 bg-accent/10 border border-accent/20 px-6 py-3 rounded-2xl text-accent font-medium">
              <CheckCircle2 size={20} />
              Optimizing your supplement intake could reduce recovery time by {data.scenarios.no_supplements.recovery_weeks - data.scenarios.optimal_intake.recovery_weeks} weeks
            </div>
          </div>
        </section>

        {/* SECTION 3: Recovery Timeline Chart */}
        <section className="bg-card border border-border p-10 rounded-[2.5rem] mb-12">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-10 gap-6">
            <div>
              <h2 className="text-2xl font-bold mb-2">Projected Recovery Timeline</h2>
              <p className="text-slate-400 text-sm">Estimated symptom reduction across different regimens</p>
            </div>
            
            <div className="flex flex-wrap gap-4">
              <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/50 rounded-lg border border-slate-700">
                <div className="w-3 h-3 rounded-full bg-[#ef4444] border border-white/20" />
                <span className="text-xs font-medium text-slate-300">No Supplements</span>
              </div>
              <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800/50 rounded-lg border border-slate-700">
                <div className="w-3 h-3 rounded-full bg-[#f59e0b]" />
                <span className="text-xs font-medium text-slate-300">Current Intake</span>
              </div>
              <div className="flex items-center gap-2 px-3 py-1.5 bg-accent/20 rounded-lg border border-accent/30">
                <div className="w-3 h-3 rounded-full bg-accent" />
                <span className="text-xs font-medium text-accent">Optimal Intake</span>
              </div>
            </div>
          </div>

          <div className="h-[400px] w-full">
            {/* We'll use a simple simulation for the chart data since the API doesn't provide the full trajectory */}
            <TimelineChart data={data} />
          </div>
        </section>

        {/* SECTION 4: SHAP Feature Importance */}
        <section className="bg-card border border-border p-10 rounded-[2.5rem] mb-12">
          <h2 className="text-2xl font-bold mb-2">Why this prediction was made</h2>
          <p className="text-slate-400 text-sm mb-10">Top factors influencing your specific risk profile</p>
          
          <div className="h-[350px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={data.shap_values}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={true} vertical={false} />
                <XAxis type="number" hide />
                <YAxis 
                  dataKey="feature" 
                  type="category" 
                  stroke="#94a3b8" 
                  fontSize={12} 
                  width={120}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip 
                  cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px' }}
                  formatter={(value) => [`${value.toFixed(4)}`, 'Impact Score']}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {data.shap_values.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.value > 0 ? '#ef4444' : '#22c55e'} 
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <div className="mt-8 flex items-start gap-4 p-4 bg-slate-900/50 rounded-2xl border border-slate-800">
            <Info className="text-accent shrink-0 mt-0.5" size={18} />
            <p className="text-sm text-slate-400 leading-relaxed">
              These are the top factors that influenced your specific risk prediction. 
              <span className="text-red-400"> Red bars</span> indicate factors that increased your estimated risk, 
              while <span className="text-green-400">green bars</span> indicate protective factors that reduced it.
            </p>
          </div>
        </section>

        {/* SECTION 5: Recommendations */}
        <section className="bg-card border border-border p-10 rounded-[2.5rem] mb-12">
          <h2 className="text-2xl font-bold mb-8">Personalized Recommendations</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {data.recommendations.map((rec, i) => (
              <div key={i} className="flex items-center gap-4 p-5 bg-slate-900/30 border border-slate-800 rounded-2xl hover:border-accent/50 transition-colors group">
                <div className="w-10 h-10 rounded-full bg-accent/10 flex items-center justify-center shrink-0 group-hover:bg-accent/20 transition-colors">
                  <CheckCircle2 className="text-accent" size={22} />
                </div>
                <p className="text-slate-300 font-medium">{rec}</p>
              </div>
            ))}
          </div>
        </section>

        {/* SECTION 6: Copy Results */}
        <section className="flex flex-col items-center py-10">
          <button 
            onClick={() => {
              const summary = `Risk Level: ${data.risk_level} (${Math.round(data.risk_confidence * 100)}% confidence)\nRecovery: ${data.recovery_weeks_min}-${data.recovery_weeks_max} weeks\nTop factors: ${data.shap_values.slice(0, 3).map(s => s.feature).join(', ')}`;
              navigator.clipboard.writeText(summary);
              addToast("Summary copied to clipboard!", "success");
            }}
            className="group flex items-center gap-3 bg-slate-800 hover:bg-slate-700 text-slate-300 px-8 py-4 rounded-2xl font-bold transition-all border border-slate-700 hover:border-slate-500 shadow-xl min-h-[44px]"
          >
            <Clipboard size={20} className="group-hover:scale-110 transition-transform" /> 
            Copy Plain Text Summary
          </button>
          <p className="mt-4 text-slate-500 text-xs">Share your summary quickly with your healthcare provider</p>
        </section>

      </div>
    </div>
  );
};

export default Results;
