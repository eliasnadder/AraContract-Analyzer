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
          <Link to="/">الرئيسية</Link>
          <Link to="/analysis">تحليل العقود</Link>
          <Link to="/compare">مقارنة عقدين</Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;