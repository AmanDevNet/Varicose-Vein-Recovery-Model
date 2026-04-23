import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useGlobal } from '../context/GlobalContext';
import { Moon, Sun, Activity, Menu, X } from 'lucide-react';

const Navbar = () => {
  const { theme, toggleTheme } = useGlobal();
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const navLinks = [
    { name: 'Home', path: '/' },
    { name: 'Predict', path: '/predict' },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 dark:bg-background/80 backdrop-blur-md border-b border-border transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-4 md:px-8">
        <div className="flex items-center justify-between h-20">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 group">
            <div className="w-10 h-10 bg-accent rounded-xl flex items-center justify-center text-white shadow-lg shadow-accent/20 group-hover:scale-110 transition-transform">
              <Activity size={24} />
            </div>
            <span className="text-xl font-black tracking-tight dark:text-white text-slate-900">
              VV Predictor
            </span>
          </Link>

          {/* Desktop Nav */}
          <div className="hidden md:flex items-center gap-8">
            {navLinks.map(link => (
              <Link 
                key={link.name} 
                to={link.path}
                className="text-slate-600 dark:text-slate-400 hover:text-accent dark:hover:text-accent font-bold transition-colors"
              >
                {link.name}
              </Link>
            ))}
            <a 
              href="https://github.com/AmanDevNet/Varicose-Vein-Recovery-Model" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-slate-600 dark:text-slate-400 hover:text-accent dark:hover:text-accent flex items-center gap-1 font-bold"
            >
              GitHub <Activity size={18} />
            </a>
            
            <div className="h-6 w-px bg-border mx-2" />
            
            <button 
              onClick={toggleTheme}
              className="p-2 rounded-xl bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:scale-110 transition-all"
            >
              {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </div>

          {/* Mobile Toggle */}
          <div className="md:hidden flex items-center gap-4">
            <button onClick={toggleTheme} className="p-2 dark:text-white text-slate-900">
              {theme === 'dark' ? <Sun size={20} /> : <Moon size={20} />}
            </button>
            <button onClick={() => setIsMenuOpen(!isMenuOpen)} className="p-2 dark:text-white text-slate-900">
              {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {isMenuOpen && (
        <div className="md:hidden bg-background border-b border-border animate-in slide-in-from-top duration-300">
          <div className="flex flex-col p-6 gap-6">
            {navLinks.map(link => (
              <Link 
                key={link.name} 
                to={link.path}
                onClick={() => setIsMenuOpen(false)}
                className="text-xl font-bold dark:text-white text-slate-900"
              >
                {link.name}
              </Link>
            ))}
            <a 
              href="https://github.com/AmanDevNet/Varicose-Vein-Recovery-Model"
              className="text-xl font-bold dark:text-white text-slate-900 flex items-center gap-2"
            >
              GitHub <Activity size={20} />
            </a>
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;
