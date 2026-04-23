import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing';
import PredictForm from './pages/PredictForm';
import Results from './pages/Results';
import Navbar from './components/Navbar';
import { GlobalProvider } from './context/GlobalContext';

function App() {
  return (
    <GlobalProvider>
      <Router>
        <div className="bg-background dark:bg-background min-h-screen text-slate-900 dark:text-white transition-colors duration-300">
          <Navbar />
          <main className="pt-20">
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/predict" element={<PredictForm />} />
              <Route path="/results" element={<Results />} />
            </Routes>
          </main>
        </div>
      </Router>
    </GlobalProvider>
  );
}

export default App;
