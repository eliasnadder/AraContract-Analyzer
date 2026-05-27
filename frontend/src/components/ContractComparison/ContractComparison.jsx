import React, { useState } from 'react';
import axios from 'axios';
import './ContractComparison.css';

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

const ContractComparison = () => {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const handleFile1Change = (e) => {
    setFile1(e.target.files[0]);
    setError(null);
  };

  const handleFile2Change = (e) => {
    setFile2(e.target.files[0]);
    setError(null);
  };

  const handleCompare = async () => {
    if (!file1 || !file2) {
      setError('يرجى اختيار العقدين معاً لإجراء المقارنة');
      return;
    }
    
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file1', file1);
    formData.append('file2', file2);

    try {
      const response = await axios.post('/api/contract/compare', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.message || 'فشلت عملية مقارنة العقود. يرجى التأكد من تشغيل الخادم.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="comparison-page">
      <header className="glass-panel comparison-header">
        <h1>مقارنة العقود الذكية</h1>
        <p>قارن بين عقدين في آن واحد واكتشف الاختلافات الهيكلية ونسب المخاطر فوراً</p>
      </header>

      {/* Select Files Section */}
      {!result && (
        <div className="comparison-setup">
          <div className="glass-panel file-selector-wrapper">
            {/* File 1 Selector */}
            <div className="file-selector-box">
              <h3>العقد الأول (الأساسي)</h3>
              <div className={`file-dropzone ${file1 ? 'has-file' : ''}`}>
                <div className="file-icon">📄</div>
                {file1 ? (
                  <span className="selected-filename">{file1.name}</span>
                ) : (
                  <span className="dropzone-placeholder">اختر أو اسحب ملف العقد الأول</span>
                )}
                <input 
                  type="file" 
                  id="file1-input"
                  accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp" 
                  onChange={handleFile1Change} 
                />
                <label htmlFor="file1-input" className="selector-label">اختر الملف</label>
              </div>
            </div>

            {/* Link Connector */}
            <div className="vs-connector">VS</div>

            {/* File 2 Selector */}
            <div className="file-selector-box">
              <h3>العقد الثاني (للمقارنة)</h3>
              <div className={`file-dropzone ${file2 ? 'has-file' : ''}`}>
                <div className="file-icon">📄</div>
                {file2 ? (
                  <span className="selected-filename">{file2.name}</span>
                ) : (
                  <span className="dropzone-placeholder">اختر أو اسحب ملف العقد الثاني</span>
                )}
                <input 
                  type="file" 
                  id="file2-input"
                  accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp" 
                  onChange={handleFile2Change} 
                />
                <label htmlFor="file2-input" className="selector-label">اختر الملف</label>
              </div>
            </div>
          </div>

          {error && <div className="error-message">{error}</div>}

          <button 
            className="submit-btn compare-btn"
            onClick={handleCompare} 
            disabled={!file1 || !file2 || loading}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                جاري تحليل ومقارنة العقود...
              </>
            ) : 'بدء المقارنة الذكية'}
          </button>
        </div>
      )}

      {/* Results View */}
      {result && (
        <div className="comparison-results">
          <div className="results-header">
            <button className="back-btn" onClick={() => setResult(null)}>
              ⬅️ مقارنة عقود أخرى
            </button>
            <h2>نتائج المقارنة والتحليل</h2>
          </div>

          <div className="results-layout">
            {/* Side-by-Side Statistics */}
            <aside className="glass-panel stats-compare-aside">
              <h3>مقارنة إحصائية</h3>
              
              <div className="side-by-side-table">
                <div className="table-row table-header-row">
                  <div className="cell prop-name">الخاصية</div>
                  <div className="cell val-contract1">العقد الأول</div>
                  <div className="cell val-contract2">العقد الثاني</div>
                </div>

                <div className="table-row">
                  <div className="cell prop-name">الأطراف</div>
                  <div className="cell val-contract1 text-sm">{result.contract1_summary.parties}</div>
                  <div className="cell val-contract2 text-sm">{result.contract2_summary.parties}</div>
                </div>

                <div className="table-row">
                  <div className="cell prop-name">إجمالي البنود</div>
                  <div className="cell val-contract1 font-bold">{result.contract1_summary.total_clauses}</div>
                  <div className="cell val-contract2 font-bold">{result.contract2_summary.total_clauses}</div>
                </div>

                <div className="table-row risk-row-high">
                  <div className="cell prop-name">بنود عالية الخطورة</div>
                  <div className="cell val-contract1 risk-high-txt">{result.contract1_summary.high_risk_clauses} 🔴</div>
                  <div className="cell val-contract2 risk-high-txt">{result.contract2_summary.high_risk_clauses} 🔴</div>
                </div>

                <div className="table-row risk-row-medium">
                  <div className="cell prop-name">بنود متوسطة الخطورة</div>
                  <div className="cell val-contract1 risk-med-txt">{result.contract1_summary.medium_risk_clauses} 🟡</div>
                  <div className="cell val-contract2 risk-med-txt">{result.contract2_summary.medium_risk_clauses} 🟡</div>
                </div>

                <div className="table-row risk-row-low">
                  <div className="cell prop-name">بنود آمنة</div>
                  <div className="cell val-contract1 risk-low-txt">{result.contract1_summary.low_risk_clauses} 🟢</div>
                  <div className="cell val-contract2 risk-low-txt">{result.contract2_summary.low_risk_clauses} 🟢</div>
                </div>
              </div>
            </aside>

            {/* Differences List */}
            <main className="differences-main">
              <div className="glass-panel differences-panel">
                <h3>الاختلافات المكتشفة ({result.differences.length})</h3>
                
                {result.differences.length === 0 ? (
                  <div className="empty-diffs">
                    <p>يتطابق العقدان في الهيكل والتوزيع العام للبنود ومستويات الخطورة.</p>
                  </div>
                ) : (
                  <div className="diff-cards-list">
                    {result.differences.map((diff, index) => (
                      <div key={index} className={`diff-card severity-${diff.severity || 'info'}`}>
                        <div className="diff-card-header">
                          <span className="diff-icon">
                            {diff.severity === 'warning' ? '⚠️' : 'ℹ️'}
                          </span>
                          <h4>{diff.title}</h4>
                        </div>
                        <p className="diff-description">{diff.description}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </main>
          </div>
        </div>
      )}
    </div>
  );
};

export default ContractComparison;
