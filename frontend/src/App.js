// App.js
import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import ImageUploader from "./components/upload";

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<ImageUploader />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
