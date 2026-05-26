import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-content">
        <Link to="/" className="navbar-logo">
          AraContract Analyzer
        </Link>
        <div className="navbar-links">
          <Link to="/">Home</Link>
          <Link to="/analysis">Analysis</Link>
          <Link to="/chat">Q&A</Link>
          <Link to="/compare">Compare</Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;