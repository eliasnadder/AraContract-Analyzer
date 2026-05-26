import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar/Navbar';
import UploadZone from './components/UploadZone/UploadZone';
import ContractAnalysis from './components/ContractAnalysis/ContractAnalysis';
import ChatInterface from './components/ChatInterface/ChatInterface';
import ContractComparison from './components/ContractComparison/ContractComparison';
import './styles/theme.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<UploadZone />} />
            <Route path="/analysis" element={<ContractAnalysis />} />
            <Route path="/chat" element={<ChatInterface />} />
            <Route path="/compare" element={<ContractComparison />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;