import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("detect");
  const [history, setHistory] = useState([]);
  const [isConverting, setIsConverting] = useState(false);
  const [conversionMessage, setConversionMessage] = useState("");
  const [textAnalysis, setTextAnalysis] = useState(null);

  useEffect(() => {
    const savedHistory = localStorage.getItem("sarcasmHistory");
    if (savedHistory) {
      setHistory(JSON.parse(savedHistory));
    }
  }, []);

  // Analyze text when it changes to show current status
  useEffect(() => {
    if (text.trim()) {
      analyzeText(text);
    } else {
      setTextAnalysis(null);
    }
  }, [text]);

  const analyzeText = async (textToAnalyze) => {
    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: textToAnalyze }),
      });
      const data = await res.json();
      if (!data.error) {
        setTextAnalysis({
          isSarcastic: data.is_sarcastic,
          confidence: data.confidence
        });
      }
    } catch (error) {
      console.log("Analysis failed");
    }
  };

  const addToHistory = (item) => {
    const newItem = {
      ...item,
      id: Date.now(),
      timestamp: new Date().toLocaleString()
    };
    const updatedHistory = [newItem, ...history.slice(0, 9)];
    setHistory(updatedHistory);
    localStorage.setItem("sarcasmHistory", JSON.stringify(updatedHistory));
  };

  const handlePredict = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setConversionMessage("");
    
    try {
      const res = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      
      if (!data.error) {
        addToHistory({
          type: 'detection',
          text: data.text,
          isSarcastic: data.is_sarcastic,
          confidence: data.confidence,
        });
      }
      
      setResult(data);
    } catch (error) {
      setResult({ error: "⚠️ Could not connect to backend. Make sure the server is running on port 5000." });
    } finally {
      setLoading(false);
    }
  };

  const handleConvert = async (targetType) => {
    if (!text.trim()) return;
    setLoading(true);
    setIsConverting(true);
    setConversionMessage("");
    
    try {
      const res = await fetch("http://localhost:5000/convert", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          text: text,
          target: targetType 
        }),
      });
      const data = await res.json();
      
      if (!data.error) {
        if (data.conversion_happened) {
          addToHistory({
            type: 'conversion',
            originalText: data.original_text,
            convertedText: data.converted_text,
            conversionType: data.conversion_type,
            wasSarcastic: data.original_was_sarcastic,
          });
          setConversionMessage("success");
        } else {
          setConversionMessage("already_in_form");
        }
      }
      
      setResult(data);
    } catch (error) {
      setResult({ error: "⚠️ Conversion failed." });
    } finally {
      setLoading(false);
      setTimeout(() => setIsConverting(false), 500);
    }
  };

  const runTests = async () => {
    setLoading(true);
    setConversionMessage("");
    try {
      const res = await fetch("http://localhost:5000/test");
      const data = await res.json();
      setResult({ testResults: data.test_results });
    } catch (error) {
      setResult({ error: "Test failed" });
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem("sarcasmHistory");
  };

  // Check if conversion should be disabled
  const shouldDisableConversion = (targetType) => {
    if (!textAnalysis || !text.trim()) return false;
    
    if (targetType === "sarcastic" && textAnalysis.isSarcastic) {
      return true; // Disable if trying to convert sarcastic to sarcastic
    }
    
    if (targetType === "unsarcastic" && !textAnalysis.isSarcastic) {
      return true; // Disable if trying to convert genuine to genuine
    }
    
    return false;
  };

  const getConversionTooltip = (targetType) => {
    if (!textAnalysis) return "";
    
    if (targetType === "sarcastic" && textAnalysis.isSarcastic) {
      return "Text is already sarcastic - can only convert to genuine form";
    }
    
    if (targetType === "unsarcastic" && !textAnalysis.isSarcastic) {
      return "Text is already genuine - can only convert to sarcastic form";
    }
    
    return "";
  };

  const examples = {
    sarcastic: [
      { text: "I'm just bursting with energy", description: "I am tired", type: "sarcastic" },
      { text: "This is just fantastic", description: "This is a problem", type: "sarcastic" },
      { text: "What beautiful weather", description: "The weather is bad", type: "sarcastic" },
      { text: "Just what I needed, more work", description: "I have more work", type: "sarcastic" },
      { text: "I get to enjoy some quality waiting time", description: "I have to wait", type: "sarcastic" },
      { text: "Well, this is a productive use of my time", description: "This is a pointless waste of time", type: "sarcastic" },
      { text: "You're a genius, thank you for that insight", description: "That was a very stupid or obvious comment", type: "sarcastic" },
      { text: "I'm thrilled to be on hold for the 45th minute", description: "I am extremely frustrated with this customer service", type: "sarcastic" },
      { text: "My phone updating right before a call is perfect timing", description: "The timing for this update is incredibly inconvenient", type: "sarcastic" },
      { text: "Another meeting that could have been an email? Lovely.", description: "This meeting is unnecessary", type: "sarcastic" },
      { text: "Oh, fantastic, the Wi-Fi is down again", description: "The unreliable Wi-Fi is a major problem", type: "sarcastic" },
      { text: "I love it when people are fashionably late", description: "I am annoyed that people are late", type: "sarcastic" },
      { text: "This is exactly how I pictured my relaxing evening", description: "My evening is chaotic and stressful", type: "sarcastic" },
    ],
    normal: [
  { text: "I am tired", description: "I'm just bursting with energy", type: "genuine" },
  { text: "I'm exhausted", description: "I'm feeling absolutely refreshed", type: "genuine" },
  { text: "This is a problem", description: "This is just fantastic", type: "genuine" },
  { text: "The weather is bad", description: "What beautiful weather", type: "genuine" },
  { text: "I have to wait", description: "I get to enjoy some quality waiting time", type: "genuine" },
  { text: "This is a pointless waste of time", description: "Well, this is a productive use of my time", type: "genuine" },
  { text: "That was a stupid comment", description: "You're a genius, thank you for that insight", type: "genuine" },
  { text: "I am frustrated with this customer service", description: "I'm thrilled to be on hold for the 45th minute", type: "genuine" },
  { text: "The timing for this update is inconvenient", description: "My phone updating right before a call is perfect timing", type: "genuine" },
  { text: "This meeting is unnecessary", description: "Another meeting that could have been an email? Lovely.", type: "genuine" },
  { text: "The unreliable Wi-Fi is a problem", description: "Oh, fantastic, the Wi-Fi is down again", type: "genuine" },
  { text: "I am annoyed that people are late", description: "I love it when people are fashionably late", type: "genuine" },
  { text: "My evening is chaotic", description: "This is exactly how I pictured my relaxing evening", type: "genuine" },
]
  };

  return (
    <div className="App">
      {/* Animated Background */}
      <div className="background">
        <div className="gradient-bg">
          <div className="gradient-circle-1"></div>
          <div className="gradient-circle-2"></div>
          <div className="gradient-circle-3"></div>
        </div>
      </div>

      <div className="container">
        {/* Header */}
        <header className="header">
          <div className="header-content">
            <div className="logo">
              <div className="logo-icon">🎭</div>
              <div className="logo-text">
                <h1>SarcasmAI</h1>
                <p>Intelligent Sarcasm Detection & Conversion</p>
              </div>
            </div>
            <div className="header-stats">
              <div className="stat">
                <div className="stat-number">{history.length}</div>
                <div className="stat-label">Analyses</div>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="main-content">
          {/* Navigation Tabs */}
          <nav className="tabs">
            <button 
              className={`tab ${activeTab === "detect" ? "active" : ""}`}
              onClick={() => setActiveTab("detect")}
            >
              <span className="tab-icon">🔍</span>
              <span className="tab-text">Detect</span>
            </button>
            <button 
              className={`tab ${activeTab === "convert" ? "active" : ""}`}
              onClick={() => setActiveTab("convert")}
            >
              <span className="tab-icon">🔄</span>
              <span className="tab-text">Convert</span>
            </button>
            <button 
              className={`tab ${activeTab === "examples" ? "active" : ""}`}
              onClick={() => setActiveTab("examples")}
            >
              <span className="tab-icon">💡</span>
              <span className="tab-text">Examples</span>
            </button>
            <button 
              className={`tab ${activeTab === "history" ? "active" : ""}`}
              onClick={() => setActiveTab("history")}
            >
              <span className="tab-icon">📚</span>
              <span className="tab-text">History</span>
            </button>
          </nav>

          {/* Tab Content */}
          <div className="tab-content">
            {/* Detect Tab */}
            {activeTab === "detect" && (
              <div className="tab-panel">
                <div className="panel-header">
                  <h2>Detect Sarcasm</h2>
                  <p>Analyze text to identify sarcastic content with AI-powered accuracy</p>
                </div>
                
                <div className="input-card">
                  <div className="input-header">
                    <span className="input-label">Enter Text to Analyze</span>
                    <span className="input-hint">We'll detect if it's sarcastic or genuine</span>
                  </div>
                  <textarea
                    rows={4}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder='Try: "Im just bursting with energy" (sarcastic) or "I am tired" (genuine)'
                    className="input-field"
                  />
                  <button 
                    onClick={handlePredict} 
                    disabled={loading || !text.trim()}
                    className={`action-btn primary ${loading ? 'loading' : ''}`}
                  >
                    <span className="btn-icon">🔍</span>
                    <span className="btn-text">
                      {loading ? "Analyzing..." : "Detect Sarcasm"}
                    </span>
                  </button>
                </div>
              </div>
            )}

            {/* Convert Tab */}
            {activeTab === "convert" && (
              <div className="tab-panel">
                <div className="panel-header">
                  <h2>Convert Text</h2>
                  <p>Transform between sarcastic and genuine expressions</p>
                </div>
                
                <div className="input-card">
                  <div className="input-header">
                    <span className="input-label">Enter Text to Convert</span>
                    <span className="input-hint">
                      {textAnalysis ? (
                        <span className={`current-status ${textAnalysis.isSarcastic ? 'sarcastic' : 'genuine'}`}>
                          Current: {textAnalysis.isSarcastic ? '😏 Sarcastic' : '🙂 Genuine'} 
                          ({(textAnalysis.confidence * 100).toFixed(1)}%)
                        </span>
                      ) : "Enter text to see current status"}
                    </span>
                  </div>
                  <textarea
                    rows={4}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder='Try: "I am tired" or "Im just bursting with energy"'
                    className="input-field"
                  />
                  
                  {/* Current Text Analysis */}
                  {textAnalysis && (
                    <div className="current-analysis">
                      <div className={`analysis-badge ${textAnalysis.isSarcastic ? 'sarcastic' : 'genuine'}`}>
                        <span className="analysis-icon">
                          {textAnalysis.isSarcastic ? '😏' : '🙂'}
                        </span>
                        <span className="analysis-text">
                          {textAnalysis.isSarcastic ? 'Sarcastic Text' : 'Genuine Text'}
                        </span>
                        <span className="analysis-confidence">
                          {(textAnalysis.confidence * 100).toFixed(1)}% confidence
                        </span>
                      </div>
                    </div>
                  )}
                  
                  <div className="conversion-buttons">
                    <button 
                      onClick={() => handleConvert("sarcastic")} 
                      disabled={loading || !text.trim() || shouldDisableConversion("sarcastic")}
                      className={`action-btn sarcastic ${isConverting ? 'converting' : ''} ${shouldDisableConversion("sarcastic") ? 'disabled' : ''}`}
                      title={getConversionTooltip("sarcastic")}
                    >
                      <span className="btn-icon">😏</span>
                      <span className="btn-text">
                        {shouldDisableConversion("sarcastic") ? "Already Sarcastic" : "Make Sarcastic"}
                      </span>
                      {isConverting && <div className="conversion-wave"></div>}
                    </button>
                    
                    <button 
                      onClick={() => handleConvert("unsarcastic")} 
                      disabled={loading || !text.trim() || shouldDisableConversion("unsarcastic")}
                      className={`action-btn genuine ${isConverting ? 'converting' : ''} ${shouldDisableConversion("unsarcastic") ? 'disabled' : ''}`}
                      title={getConversionTooltip("unsarcastic")}
                    >
                      <span className="btn-icon">🙂</span>
                      <span className="btn-text">
                        {shouldDisableConversion("unsarcastic") ? "Already Genuine" : "Make Genuine"}
                      </span>
                      {isConverting && <div className="conversion-wave"></div>}
                    </button>
                  </div>

                  {/* Conversion Rules */}
                  <div className="conversion-rules">
                    <h4>Conversion Rules:</h4>
                    <ul>
                      <li>✅ <strong>Sarcastic</strong> text can only be converted to <strong>Genuine</strong></li>
                      <li>✅ <strong>Genuine</strong> text can only be converted to <strong>Sarcastic</strong></li>
                      <li>❌ Cannot convert sarcastic to sarcastic</li>
                      <li>❌ Cannot convert genuine to genuine</li>
                    </ul>
                  </div>
                </div>
              </div>
            )}

            {/* Examples Tab */}
            {activeTab === "examples" && (
              <div className="tab-panel">
                <div className="panel-header">
                  <h2>Examples</h2>
                  <p>See how the system handles different types of sarcasm</p>
                  <button onClick={runTests} disabled={loading} className="test-btn">
                    {loading ? "Running Tests..." : "Run All Tests"}
                  </button>
                </div>

                <div className="examples-grid">
                  <div className="examples-column">
                    <h3 className="examples-title sarcastic">😏 Sarcastic Examples</h3>
                    <p className="examples-subtitle">These sound positive but mean the opposite</p>
                    {examples.sarcastic.map((example, index) => (
                      <div 
                        key={index}
                        className="example-card sarcastic"
                        onClick={() => setText(example.text)}
                      >
                        <div className="example-content">
                          <div className="example-text">"{example.text}"</div>
                          <div className="example-type sarcastic-type">Sarcastic</div>
                          <div className="example-conversion">{example.description}</div>
                        </div>
                        <div className="example-arrow">→</div>
                      </div>
                    ))}
                  </div>

                  <div className="examples-column">
                    <h3 className="examples-title genuine">🙂 Genuine Examples</h3>
                    <p className="examples-subtitle">These are straightforward statements</p>
                    {examples.normal.map((example, index) => (
                      <div 
                        key={index}
                        className="example-card genuine"
                        onClick={() => setText(example.text)}
                      >
                        <div className="example-content">
                          <div className="example-text">"{example.text}"</div>
                          <div className="example-type genuine-type">Genuine</div>
                          <div className="example-conversion">{example.description}</div>
                        </div>
                        <div className="example-arrow">→</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="examples-explanation">
                  <h4>Smart Conversion System</h4>
                  <div className="explanation-grid">
                    <div className="explanation-item">
                      <div className="explanation-icon">🎯</div>
                      <div className="explanation-content">
                        <h5>Accurate Detection</h5>
                        <p>"I'm just bursting with energy" = Sarcastic<br/>"I am tired" = Genuine</p>
                      </div>
                    </div>
                    <div className="explanation-item">
                      <div className="explanation-icon">🔄</div>
                      <div className="explanation-content">
                        <h5>One-Way Conversion</h5>
                        <p>Sarcastic → Genuine only<br/>Genuine → Sarcastic only</p>
                      </div>
                    </div>
                    <div className="explanation-item">
                      <div className="explanation-icon">🚫</div>
                      <div className="explanation-content">
                        <h5>No Duplicate Conversions</h5>
                        <p>Prevents sarcastic→sarcastic<br/>Prevents genuine→genuine</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* History Tab - Same as before */}
            {activeTab === "history" && (
              <div className="tab-panel">
                <div className="panel-header">
                  <div className="panel-header-row">
                    <div>
                      <h2>Analysis History</h2>
                      <p>Recent sarcasm detections and conversions</p>
                    </div>
                    {history.length > 0 && (
                      <button onClick={clearHistory} className="clear-btn">
                        🗑️ Clear History
                      </button>
                    )}
                  </div>
                </div>

                {history.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-icon">📝</div>
                    <h3>No History Yet</h3>
                    <p>Start detecting sarcasm or converting text to see your history here</p>
                  </div>
                ) : (
                  <div className="history-list">
                    {history.map((item) => (
                      <div key={item.id} className="history-item">
                        <div className="history-time">{item.timestamp}</div>
                        
                        {item.type === 'detection' ? (
                          <div className="history-content">
                            <div className="history-text">"{item.text}"</div>
                            <div className="history-details">
                              <div className={`history-badge ${item.isSarcastic ? 'sarcastic' : 'genuine'}`}>
                                <span className="badge-icon">
                                  {item.isSarcastic ? '😏' : '🙂'}
                                </span>
                                <span className="badge-text">
                                  {item.isSarcastic ? 'Sarcastic' : 'Genuine'}
                                </span>
                                <span className="badge-confidence">
                                  {(item.confidence * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="history-content">
                            <div className="conversion-flow">
                              <div className="conversion-original">
                                <span className="conversion-label">From:</span>
                                <span>"{item.originalText}"</span>
                                {item.wasSarcastic !== undefined && (
                                  <span className={`sarcasm-indicator ${item.wasSarcastic ? 'sarcastic' : 'genuine'}`}>
                                    {item.wasSarcastic ? '😏' : '🙂'}
                                  </span>
                                )}
                              </div>
                              <div className="conversion-arrow">→</div>
                              <div className="conversion-result">
                                <span className="conversion-label">To ({item.conversionType}):</span>
                                <span className="conversion-text">"{item.convertedText}"</span>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Conversion Message */}
          {conversionMessage === "already_in_form" && result && (
            <div className="info-message">
              <div className="info-icon">💡</div>
              <div className="info-content">
                <h4>Already in {result.conversion_type === 'sarcastic' ? 'Sarcastic' : 'Genuine'} Form!</h4>
                <p>
                  This text is already {result.conversion_type}. 
                  {result.conversion_type === 'sarcastic' 
                    ? " You can only convert it to genuine form." 
                    : " You can only convert it to sarcastic form."
                  }
                </p>
              </div>
            </div>
          )}


{/* Results Section */}
{result && !result.error && !result.testResults && (
  <div className="results-section">
    {/* Detection Results */}
    {result.is_sarcastic !== undefined && (
      <div className="result-card">
        <div className="result-header">
          <div className="result-icon">
            {result.is_sarcastic ? '😏' : '🙂'}
          </div>
          <div className="result-title">
            <h3>
              {result.is_sarcastic ? 'Sarcastic Text Detected' : 'Genuine Text Detected'}
            </h3>
            <p>
              Confidence: {(result.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        <div className={`prediction-card ${result.is_sarcastic ? 'sarcastic' : 'genuine'}`}>
          <div className="prediction-icon">
            {result.is_sarcastic ? '😏' : '🙂'}
          </div>
          <div className="prediction-content">
            <div className="prediction-text">
              {result.is_sarcastic ? 'Sarcastic' : 'Genuine'}
            </div>
            <div className="prediction-confidence">
              {(result.confidence * 100).toFixed(1)}% confidence
            </div>
          </div>
        </div>

        {/* Confidence Visualization */}
        <div className="confidence-visual">
          <div className="confidence-labels">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
          <div className="confidence-track">
            <div 
              className="confidence-progress"
              style={{ width: `${result.confidence * 100}%` }}
            ></div>
          </div>
        </div>

        {/* Detailed Analysis */}
        {result.detailed_scores && (
          <div className="analysis-breakdown">
            <h4>Analysis Breakdown</h4>
            <div className="scores-grid">
              {Object.entries(result.detailed_scores).map(([key, value]) => (
                <div key={key} className="score-item">
                  <div className="score-info">
                    <span className="score-label">{key.replace('_', ' ')}</span>
                    <span className="score-value">{(value * 100).toFixed(1)}%</span>
                  </div>
                  <div className="score-bar">
                    <div 
                      className="score-fill"
                      style={{ width: `${value * 100}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )}

    {/* Conversion API Results */}
    {result.conversion_type && (
      <div className="result-card">
        <div className="result-header">
          <div className="result-icon">
            {result.conversion_happened ? '🔄' : 'ℹ️'}
          </div>
          <div className="result-title">
            <h3>
              {result.conversion_happened ? 'Conversion Results' : 'No Conversion Needed'}
            </h3>
            <p>{result.message}</p>
          </div>
        </div>

        {result.conversion_happened && (
          <div className="conversion-display">
            <div className="conversion-column">
              <div className="conversion-label">Original Text</div>
              <div className="conversion-text original">
                "{result.original_text}"
              </div>
              <div className="text-analysis">
                <span className={`analysis-tag ${result.original_was_sarcastic ? 'sarcastic' : 'genuine'}`}>
                  {result.original_was_sarcastic ? '😏 Sarcastic' : '🙂 Genuine'}
                </span>
              </div>
            </div>

            <div className="conversion-arrow">
              <div className="arrow-line"></div>
              <div className="arrow-head">→</div>
              <div className="arrow-line"></div>
            </div>

            <div className="conversion-column">
              <div className="conversion-label">
                {result.conversion_type === 'sarcastic' ? 'Sarcastic Version' : 'Genuine Version'}
              </div>
              <div className="conversion-text converted">
                "{result.converted_text}"
              </div>
              <div className="text-analysis">
                <span className={`analysis-tag ${result.conversion_type === 'sarcastic' ? 'sarcastic' : 'genuine'}`}>
                  {result.conversion_type === 'sarcastic' ? '😏 Sarcastic' : '🙂 Genuine'}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    )}
  </div>
)}

{/* Test Results */}
{result && result.testResults && (
  <div className="result-card">
    <div className="result-header">
      <div className="result-icon">🧪</div>
      <div className="result-title">
        <h3>Test Results</h3>
        <p>System performance evaluation</p>
      </div>
    </div>

    <div className="test-grid">
      {result.testResults.map((test, index) => (
        <div key={index} className="test-item">
          <div className="test-text">"{test.input}"</div>
          <div className="test-analysis">
            <span className={`test-result ${test.detected_sarcastic ? 'sarcastic' : 'genuine'}`}>
              {test.detected_sarcastic ? '😏 Sarcastic' : '🙂 Genuine'} 
              ({(test.detection_confidence * 100).toFixed(1)}%)
            </span>
          </div>
          <div className="test-conversions">
            <div className="test-conversion">
              <span className="conversion-direction">To Sarcastic:</span>
              <span>"{test.to_sarcastic}"</span>
            </div>
            <div className="test-conversion">
              <span className="conversion-direction">To Genuine:</span>
              <span>"{test.to_unsarcastic}"</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  </div>
)}

{/* Error Display */}
{result && result.error && (
  <div className="result-card error-card">
    <div className="result-header">
      <div className="result-icon">❌</div>
      <div className="result-title">
        <h3>Error</h3>
        <p>Something went wrong</p>
      </div>
    </div>
    <div className="error-content">
      <p>{result.error}</p>
      {result.details && <p>Details: {result.details}</p>}
    </div>
  </div>
)}

        </main>

        {/* Footer */}
        <footer className="footer">
          <div className="footer-content">
            <div className="footer-brand">
              <div className="footer-logo">🎭</div>
              <div className="footer-text">
                <strong>SarcasmAI</strong>
                <span>Smart One-Way Conversion • Accurate Detection</span>
              </div>
            </div>
            <div className="footer-info">
              <span>Real-time Analysis</span>
              <span>•</span>
              <span>One-Way Conversion</span>
              <span>•</span>
              <span>No Duplicate Conversions</span>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;