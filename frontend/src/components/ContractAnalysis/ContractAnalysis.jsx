import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './ContractAnalysis.css';

// Mapping for clause type display names (English key -> Arabic display)
const TYPE_DISPLAY_NAMES = {
  "payment_financial": "مالي / دفع",
  "duration_expiration": "مدة / انتهاء",
  "termination": "فسخ / إنهاء",
  "penalties_damages": "غرامات / تعويضات",
  "party_obligations_a": "التزامات الطرف الأول",
  "party_obligations_b": "التزامات الطرف الثاني",
  "dispute_resolution": "تسوية نزاعات",
  "general_provisions": "أحكام عامة",
};

// Mapping for risk display names
const RISK_DISPLAY_NAMES = {
  "low": "منخفض المخاطر",
  "medium": "متوسط المخاطر",
  "high": "مرتفع المخاطر",
};

const ContractAnalysis = () => {
  const navigate = useNavigate();
  const [analysisData, setAnalysisData] = useState(null);
  const [selectedRisk, setSelectedRisk] = useState('all');
  const [selectedType, setSelectedType] = useState('all');

  useEffect(() => {
    const data = localStorage.getItem('analysisResults');
    if (!data) {
      // If no data exists, redirect to upload page
      navigate('/');
      return;
    }
    try {
      setAnalysisData(JSON.parse(data));
    } catch (e) {
      console.error(e);
      navigate('/');
    }
  }, [navigate]);

  if (!analysisData) {
    return (
      <div className="loading-container">
        <span className="spinner"></span>
        <p>جاري تحميل نتائج التحليل...</p>
      </div>
    );
  }

  const { filename, is_scanned, clauses, summary, stats } = analysisData;

  // Filter logic
  const filteredClauses = clauses.filter(clause => {
    const matchesRisk = selectedRisk === 'all' || clause.predicted_risk_level.toLowerCase() == selectedRisk.toLowerCase();
    const matchesType = selectedType === 'all' || clause.predicted_type_clause == selectedType;
    return matchesRisk && matchesType;
  });

  return (
    <div className="analysis-page">
      {/* Header Panel */}
      <header className="glass-panel analysis-header">
        <div className="header-info">
          <h1>نتائج تحليل العقد</h1>
          <div className="meta-tags">
            <span className="meta-tag filename">📄 {filename}</span>
            <span className={`meta-tag method ${is_scanned ? 'ocr' : 'digital'}`}>
              {is_scanned ? 'مستخرج عبر التعرف الضوئي (OCR)' : 'استخراج مباشر من PDF'}
            </span>
          </div>
        </div>
        <button className="new-upload-btn" onClick={() => navigate('/')}>
          تحليل عقد جديد ➕
        </button>
      </header>

      {/* Summary Section (FR-6) */}
      <section className="glass-panel summary-section">
        <div className="section-title-wrapper">
          <span className="section-icon">📝</span>
          <h2>الملخص التنفيذي الذكي</h2>
        </div>
        <p className="summary-text">{summary}</p>
      </section>

      {/* Dashboard Stats (FR-4 / FR-5) */}
      <section className="stats-dashboard">
        <div className="glass-panel stat-card total">
          <span className="stat-label">إجمالي البنود</span>
          <span className="stat-value">{stats.total_clauses || clauses.length}</span>
        </div>
        <div className={`glass-panel stat-card high-risk ${stats.high_risk_clauses > 0 ? 'pulse' : ''}`}>
          <span className="stat-label">بنود عالية الخطورة</span>
          <span className="stat-value">{stats.high_risk_clauses || 0}</span>
        </div>
        <div className="glass-panel stat-card medium-risk">
          <span className="stat-label">بنود متوسطة الخطورة</span>
          <span className="stat-value">{stats.medium_risk_clauses || 0}</span>
        </div>
        <div className="glass-panel stat-card low-risk">
          <span className="stat-label">بنود آمنة (منخفضة)</span>
          <span className="stat-value">{stats.low_risk_clauses || 0}</span>
        </div>
      </section>

      {/* Filter and List Container */}
      <div className="analysis-content">
        {/* Filters Sidebar */}
        <aside className="glass-panel filters-panel">
          <h3>تصفية وتصفح البنود</h3>
          
          <div className="filter-group">
            <label>حسب مستوى الخطورة:</label>
            <div className="risk-filters">
              <button 
                className={`filter-btn ${selectedRisk === 'all' ? 'active' : ''}`}
                onClick={() => setSelectedRisk('all')}
              >
                الكل
              </button>
              <button 
                className={`filter-btn risk-high-btn ${selectedRisk === 'high' ? 'active' : ''}`}
                onClick={() => setSelectedRisk('high')}
              >
                مرتفع 🔴
              </button>
              <button 
                className={`filter-btn risk-medium-btn ${selectedRisk === 'medium' ? 'active' : ''}`}
                onClick={() => setSelectedRisk('medium')}
              >
                متوسط 🟡
              </button>
              <button 
                className={`filter-btn risk-low-btn ${selectedRisk === 'low' ? 'active' : ''}`}
                onClick={() => setSelectedRisk('low')}
              >
                منخفض 🟢
              </button>
            </div>
          </div>

          <div className="filter-group">
            <label>حسب نوع البند:</label>
            <select 
              value={selectedType} 
              onChange={(e) => setSelectedType(e.target.value)}
              className="type-select"
            >
              <option value="all">كل أنواع البنود</option>
              {Object.entries(TYPE_DISPLAY_NAMES).map(([key, name]) => (
                <option key={key} value={key}>{name}</option>
              ))}
            </select>
          </div>

          <div className="filter-stats">
            عرض {filteredClauses.length} من أصل {clauses.length} بنداً
          </div>
        </aside>

        {/* Clauses List */}
        <main className="clauses-list">
          {filteredClauses.length === 0 ? (
            <div className="glass-panel empty-list">
              <p>لا توجد بنود تطابق خيارات التصفية المحددة.</p>
            </div>
          ) : (
            filteredClauses.map((clause, idx) => {
              const risk = clause.predicted_risk_level.toLowerCase();
              return (
                <div key={idx} className={`glass-panel clause-card risk-${risk}`}>
                  <header className="clause-card-header">
                    <span className="clause-index">البند #{idx + 1}</span>
                    <div className="clause-badges">
                      <span className="badge type-badge">
                        {TYPE_DISPLAY_NAMES[clause.predicted_type_clause] || clause.predicted_type_clause}
                      </span>
                      <span className={`badge risk-badge risk-${risk}`}>
                        {RISK_DISPLAY_NAMES[risk] || clause.predicted_risk_level}
                      </span>
                    </div>
                  </header>
                  
                  <div className="clause-text">
                    <p>{clause.text}</p>
                  </div>
                  
                  {clause.warning && (
                    <div className="clause-warning-box">
                      <span className="warning-icon">⚠️</span>
                      <div className="warning-content">
                        <strong>تحذير مخاطر مرتفعة:</strong>
                        <p>{clause.warning}</p>
                      </div>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </main>
      </div>
    </div>
  );
};

export default ContractAnalysis;
