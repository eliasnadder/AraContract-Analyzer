import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import './UploadZone.css';

const UploadZone = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) {
      setFile(e.dataTransfer.files[0]);
      setError(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleUpload = async () => {
    if (!file) {
      setError('يرجى اختيار ملف أولاً');
      return;
    }
    setUploading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      // Call the main unified analyze endpoint
      const response = await axios.post('/api/contract/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      
      // Store the analysis results in localStorage for the Analysis page to read
      localStorage.setItem('analysisResults', JSON.stringify(response.data));
      
      // Navigate to the analysis results page
      navigate('/analysis');
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.message || 'فشلت عملية رفع العقد وتحليله. يرجى التحقق من تشغيل الخادم.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-container">
      <div className="glass-panel upload-zone">
        <h2>تحليل عقد جديد</h2>
        <p className="upload-subtitle">ارفع عقدك بصيغة PDF أو كصورة وسيقوم النظام باستخراج البنود وتحليل المخاطر فوراً</p>
        
        <div 
          className={`upload-dropzone ${file ? 'has-file' : ''} ${uploading ? 'uploading' : ''}`}
          onDragOver={handleDragOver} 
          onDrop={handleDrop}
        >
          <div className="upload-icon">📁</div>
          {file ? (
            <div className="file-info">
              <span className="file-name">{file.name}</span>
              <span className="file-size">({(file.size / (1024 * 1024)).toFixed(2)} MB)</span>
            </div>
          ) : (
            <>
              <p className="drop-text">اسحب وأفلت ملف العقد هنا</p>
              <p className="or-text">أو</p>
            </>
          )}
          
          <input 
            type="file" 
            id="file-input"
            accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp" 
            onChange={handleFileChange} 
            disabled={uploading}
            style={{ display: 'none' }}
          />
          <label htmlFor="file-input" className="file-select-btn">
            {file ? 'تغيير الملف' : 'اختر ملف من جهازك'}
          </label>
        </div>

        {error && <div className="error-message">{error}</div>}

        <button 
          className="submit-btn" 
          onClick={handleUpload} 
          disabled={!file || uploading}
        >
          {uploading ? (
            <>
              <span className="spinner"></span>
              جاري تحليل العقد واستخراج البنود...
            </>
          ) : 'ابدأ التحليل الذكي'}
        </button>
      </div>
    </div>
  );
};

export default UploadZone;