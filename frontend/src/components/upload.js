import React, { useState } from 'react';
import axios from 'axios';

const ImageUploader = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [processingResult, setProcessingResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);

    // Create image preview
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
          'Content-Type': 'multipart/form-data'
        }
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
    <div className="max-w-md mx-auto p-4 bg-white shadow-md rounded-lg">
      <h2 className="text-2xl font-bold mb-4 text-center">Image Uploader</h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <input 
          type="file" 
          onChange={handleFileChange} 
          accept="image/*"
          className="w-full p-2 border rounded-md"
        />
        
        {previewImage && (
          <div className="mt-4">
            <img 
              src={previewImage} 
              alt="Preview" 
              className="max-w-full h-auto mx-auto rounded-md"
            />
          </div>
        )}
        
        <button 
          type="submit" 
          className="w-full bg-blue-500 text-white p-2 rounded-md hover:bg-blue-600 transition"
        >
          Upload and Process Image
        </button>
      </form>

      {error && (
        <div className="mt-4 p-2 bg-red-100 text-red-700 rounded-md">
          {error}
        </div>
      )}

      {processingResult && (
        <div className="mt-4 p-4 bg-green-100 rounded-md">
          <h3 className="font-bold mb-2">Processing Results:</h3>
          <pre className="text-sm">
            {JSON.stringify(processingResult, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;