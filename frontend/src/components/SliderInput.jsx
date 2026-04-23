import React from 'react';

const SliderInput = ({ label, min, max, value, onChange, unit = "", labels = {} }) => {
  return (
    <div className="mb-8">
      <div className="flex justify-between items-center mb-4">
        <label className="text-slate-300 font-medium">{label}</label>
        <span className="bg-slate-800 text-accent px-3 py-1 rounded-lg font-bold border border-slate-700">
          {value}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-accent"
      />
      {Object.keys(labels).length > 0 && (
        <div className="flex justify-between mt-2 px-1">
          {Object.entries(labels).map(([val, text]) => (
            <span key={val} className="text-[10px] text-slate-500 uppercase tracking-wider">
              {text}
            </span>
          ))}
        </div>
      )}
    </div>
  );
};

export default SliderInput;
