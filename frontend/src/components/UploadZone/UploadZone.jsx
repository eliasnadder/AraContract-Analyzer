import React, { useState } from 'react';
import axios from 'axios';
import './UploadZone.css';

const UploadZone = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [extractedText, setExtractedText] = useState('');
  const [isScanned, setIsScanned] = useState(false);
  const [error, setError] = useState(null);

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
      setError('Please select a file');
      return;
    }

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/api/contract/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setExtractedText(response.data.extracted_text);
      setIsScanned(response.data.is_scanned);
    } catch (err) {
      setError(err.response?.data?.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-zone">
      <h2>رفع عقد للتحليل</h2>
      <div
        className={`upload-dropzone ${file || uploading ? 'has-file' : ''}`}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <p>اسحب وافلت الملف هنا</p>
        <p>أو</p>
        <input type="file" accept=".pdf,.png,.jpg,.jpeg,.tiff,.bmp" onChange={handleFileChange} />
        <button onClick={handleUpload} disabled={!file || uploading}>
          {uploading ? 'جاري الرفع...' : 'رفع الملف'}
        </button>
      </div>

      {extractedText && (
        <div className="extracted-text-preview">
          <h3>النص المستخرج:</h3>
          <p>{extractedText.slice(0, 500)}{extractedText.length > 500 ? '...' : ''}</p>
          <p>تم الاستخراج عبر {'OCR' if isScanned else 'الاستخراج المباشر'} من PDF</p>
        </div>
      )}

      {error && <div className="error">{error}</div>}
    </div>
  );
};

export default UploadZone;