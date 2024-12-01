import React, { useState } from 'react';
import axios from 'axios';
import { Image as ImageIcon, UploadCloud } from 'lucide-react'; // Import specific icons
import './upload.css';

const ImageUploader = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [processingResult, setProcessingResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);

    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviewImage(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:8000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setProcessingResult(response.data);
      setError(null);
    } catch (err) {
      setError('Error uploading image');
      setProcessingResult(null);
      console.error('Upload error:', err);
    }
  };

  return (
    <div className="uploader-container">
      <h2 className="uploader-title">Image Uploader</h2>

      <form onSubmit={handleSubmit} className="uploader-form">
        <label className="file-input-label">
          <UploadCloud className="icon" />
          <input
            type="file"
            onChange={handleFileChange}
            accept="image/*"
            className="file-input"
          />
        </label>

        {previewImage && (
          <div className="preview-container">
            <img
              src={previewImage}
              alt="Preview"
              className="preview-image"
            />
          </div>
        )}

        <button type="submit" className="submit-button">
          <ImageIcon className="icon" /> Upload and Process Image
        </button>
      </form>

      {error && <div className="error-message">{error}</div>}

      {processingResult && (
        <div className="result-container">
          <h3>Processing Results:</h3>
          {processingResult.decoded_results.map((url, index) => (
            <a
              key={index}
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="result-link"
            >
              {url}
            </a>
          ))}
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
