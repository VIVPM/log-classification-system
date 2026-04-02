import { useState, useEffect, useCallback } from 'react'
import { checkApi, getModelVersions, getModelInfo, trainModel, getTrainingStatus, predictSingle, predictBatch } from './api'
import './App.css'


function getLabelClass(label) {
  const l = String(label).toLowerCase()
  if (l.includes('security')) return 'lbl-security'
  if (l.includes('critical') || l.includes('error')) return 'lbl-critical'
  if (l.includes('system')) return 'lbl-system'
  if (l.includes('user')) return 'lbl-user'
  if (l.includes('http')) return 'lbl-http'
  return 'lbl-unclass'
}

function parseCSV(text) {
  const lines = text.trim().split('\n')
  if (lines.length < 2) return { headers: [], rows: [] }
  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''))
  const rows = lines.slice(1).map(line => {
    const values = []
    let current = ''
    let inQuotes = false
    for (const ch of line) {
      if (ch === '"') { inQuotes = !inQuotes; continue }
      if (ch === ',' && !inQuotes) { values.push(current.trim()); current = ''; continue }
      current += ch
    }
    values.push(current.trim())
    return values
  })
  return { headers, rows }
}

function computeDistribution(predictions) {
  const counts = {}
  predictions.forEach(p => {
    const label = p.predicted_label
    counts[label] = (counts[label] || 0) + 1
  })
  return counts
}

// ─── Sidebar ─────────────────────────────────────────────────────────────────
function Sidebar({ apiOk, versions, selectedVersion, setSelectedVersion, modelInfo }) {
  const isLocal = true // dev proxy always points to localhost

  return (
    <aside className="sidebar">
      <h2 className="sidebar-title">Log Classifier</h2>
      <hr />

      {apiOk ? (
        <div className="status-pill status-ok">API Connected</div>
      ) : isLocal ? (
        <>
          <div className="status-pill status-err">API Offline</div>
          <code className="hint-code">cd backend && uvicorn api:app --reload</code>
        </>
      ) : (
        <div className="status-pill status-warn">Backend waking up...</div>
      )}

      {apiOk && versions.length > 0 && (
        <div className="sidebar-section">
          <label className="field-label">Select Model Version</label>
          <select
            className="select-field"
            value={selectedVersion}
            onChange={e => setSelectedVersion(e.target.value)}
          >
            {[...versions].reverse().map(v => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>

          {modelInfo && (
            <div className="model-info">
              <div><strong>Model:</strong> <code>{modelInfo.model_name || 'Loaded'}</code></div>
              <div><strong>Features:</strong> <code>{modelInfo.num_features || 'Unknown'}</code></div>
              {modelInfo.best_cv_score != null && (
                <div><strong>Score:</strong> <code>{Number(modelInfo.best_cv_score).toFixed(4)}</code></div>
              )}
            </div>
          )}
        </div>
      )}

      {apiOk && versions.length === 0 && (
        <div className="status-pill status-warn">No model versions found. Train a model first.</div>
      )}

      <hr />
      <div className="sidebar-section">
        <h3>How to use</h3>
        <ol className="how-to">
          <li><strong>Train Model</strong> &mdash; upload a new dataset</li>
          <li><strong>Single Prediction</strong> &mdash; test an individual log message</li>
          <li><strong>Batch Prediction</strong> &mdash; upload test.csv in bulk</li>
        </ol>
      </div>
    </aside>
  )
}

// ─── Train Model Tab ─────────────────────────────────────────────────────────
function TrainModelTab({ apiOk, versions, selectedVersion, busy, setBusy, onRetry }) {
  const [file, setFile] = useState(null)
  const [trainingTriggered, setTrainingTriggered] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState(null)
  const [error, setError] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)

  const fetchStatus = useCallback(async () => {
    const s = await getTrainingStatus()
    if (s) setTrainingStatus(s)
    if (s && s.status === 'idle' && !trainingTriggered && selectedVersion && selectedVersion !== 'main') {
      const info = await getModelInfo()
      setModelInfo(info)
    }
  }, [trainingTriggered, selectedVersion])

  useEffect(() => { if (apiOk) fetchStatus() }, [apiOk, fetchStatus])

  const handleTrain = async () => {
    if (!file) return
    setError(null)
    setBusy('training')
    try {
      const res = await trainModel(file)
      if (res.status === 200) {
        setTrainingTriggered(true)
        fetchStatus()
      } else if (res.status === 409) {
        setTrainingTriggered(true)
        setError('Training already in progress.')
      } else {
        setError(`Failed to start training: ${JSON.stringify(res.data)}`)
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(null)
    }
  }

  const status = trainingStatus?.status || 'idle'
  const message = trainingStatus?.message || ''

  // Step indicators
  const msgLower = message.toLowerCase()
  const step1 = msgLower.includes('generating embeddings') || msgLower.includes('complete') ? 'done' : (status === 'running' ? 'running' : 'pending')
  const step2 = msgLower.includes('training model') || msgLower.includes('complete') ? 'done' : (msgLower.includes('generating embeddings') ? 'running' : 'pending')
  const step3 = msgLower.includes('complete') ? 'done' : (msgLower.includes('training model') ? 'running' : 'pending')

  const stepIcon = (s) => s === 'done' ? '✅' : s === 'running' ? '🔄' : '⏳'

  if (!apiOk) return <div className="tab-warning">Connecting to backend... <button className="btn btn-secondary" onClick={onRetry}>Retry</button></div>

  return (
    <div className="tab-content">
      <h2>Retrain the Log Classification Model</h2>
      <p className="tab-desc">
        Upload the dataset (CSV) to retrain the model from scratch. The pipeline runs:
      </p>
      <ul className="pipeline-steps">
        <li><strong>Step 1</strong> &mdash; Data Preprocessing (Regex Classification &amp; LegacyCRM filtering)</li>
        <li><strong>Step 2</strong> &mdash; Feature Engineering (Google Embeddings Generation via <code>gemini-2.5-flash</code>)</li>
        <li><strong>Step 3</strong> &mdash; Model Training (Logistic Regression)</li>
      </ul>

      <div className="upload-section">
        <h3>Upload Training Data</h3>
        <p className="info-box">Upload your training dataset containing <code>log_message</code> and <code>target_label</code>.</p>
        <input
          type="file"
          accept=".csv"
          onChange={e => setFile(e.target.files[0] || null)}
          className="file-input"
        />
        <button
          className="btn btn-primary"
          disabled={!file || busy}
          onClick={handleTrain}
        >
          {busy === 'training' ? 'Starting...' : 'Start Training'}
        </button>
      </div>

      {error && <div className="error-box">{error}</div>}

      <hr />
      <h3>Training Status</h3>

      {status === 'idle' && !trainingTriggered && modelInfo && selectedVersion !== 'main' ? (
        <>
          <div className="status-card status-completed">
            ✅ <strong>Model Loaded from HF Hub</strong><br />Version: {selectedVersion}
          </div>
          <table className="steps-table">
            <thead><tr><th>Step</th><th>Task</th><th>Status</th></tr></thead>
            <tbody>
              <tr><td>1</td><td>Regex Preprocessing</td><td>✅</td></tr>
              <tr><td>2</td><td>Model Embeddings Generator</td><td>✅</td></tr>
              <tr><td>3</td><td>Logistic Regression Training</td><td>✅</td></tr>
            </tbody>
          </table>
          <div className="metrics-row">
            <div className="metric"><span className="metric-label">Model</span><span className="metric-value">{modelInfo.model_name?.split(' + ')[0]}</span></div>
            <div className="metric"><span className="metric-label">CV Score</span><span className="metric-value">{modelInfo.best_cv_score ? Number(modelInfo.best_cv_score).toFixed(4) : '-'}</span></div>
            <div className="metric"><span className="metric-label">Features</span><span className="metric-value">{modelInfo.num_features || '-'}</span></div>
          </div>
        </>
      ) : status === 'idle' ? (
        <p className="info-box">Upload your CSV file and click <strong>Start Training</strong> to begin.</p>
      ) : status === 'running' ? (
        <>
          <div className="status-card status-running">
            🔄 <strong>Training in Progress</strong><br />{message}
          </div>
          <table className="steps-table">
            <thead><tr><th>Step</th><th>Task</th><th>Status</th></tr></thead>
            <tbody>
              <tr><td>1</td><td>Regex Preprocessing</td><td>{stepIcon(step1)}</td></tr>
              <tr><td>2</td><td>Model Embeddings Generator</td><td>{stepIcon(step2)}</td></tr>
              <tr><td>3</td><td>Logistic Regression Training</td><td>{stepIcon(step3)}</td></tr>
            </tbody>
          </table>
          <p className="hint">Training running in background &mdash; click Refresh to check progress.</p>
          <button className="btn btn-secondary" onClick={fetchStatus}>Refresh Status</button>
        </>
      ) : status === 'completed' ? (
        <>
          <div className="status-card status-completed">
            ✅ <strong>Training Complete!</strong><br />{message}
          </div>
          <table className="steps-table">
            <thead><tr><th>Step</th><th>Task</th><th>Status</th></tr></thead>
            <tbody>
              <tr><td>1</td><td>Regex Preprocessing</td><td>✅</td></tr>
              <tr><td>2</td><td>Model Embeddings Generator</td><td>✅</td></tr>
              <tr><td>3</td><td>Logistic Regression Training</td><td>✅</td></tr>
            </tbody>
          </table>
          <div className="metrics-row">
            <div className="metric"><span className="metric-label">Logs Trained</span><span className="metric-value">{trainingStatus.num_trained_logs || '-'}</span></div>
            <div className="metric"><span className="metric-label">Testing Accuracy</span><span className="metric-value">{trainingStatus.accuracy ? Number(trainingStatus.accuracy).toFixed(4) : '-'}</span></div>
            <div className="metric"><span className="metric-label">Features Used</span><span className="metric-value">{trainingStatus.num_features || '-'}</span></div>
          </div>
          <button className="btn btn-secondary" onClick={fetchStatus}>Refresh Status</button>
        </>
      ) : status === 'failed' ? (
        <>
          <div className="status-card status-failed">
            ❌ <strong>Training Failed</strong><br />{message}
          </div>
          {trainingStatus.error && (
            <details className="error-details">
              <summary>Error Details</summary>
              <pre>{trainingStatus.error}</pre>
            </details>
          )}
          <button className="btn btn-secondary" onClick={fetchStatus}>Refresh Status</button>
        </>
      ) : null}
    </div>
  )
}

// ─── Single Prediction Tab ───────────────────────────────────────────────────
function SinglePredictionTab({ apiOk, versions, selectedVersion, busy, setBusy, onRetry }) {
  const [logMessage, setLogMessage] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handlePredict = async () => {
    if (!logMessage.trim()) { setError('Please enter a log message.'); return }
    if (!versions.length) { setError('No models available. Please train a model first.'); return }
    setError(null)
    setResult(null)
    setBusy('predicting')
    try {
      const res = await predictSingle(logMessage, selectedVersion)
      setResult(res)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(null)
    }
  }

  if (!apiOk) return <div className="tab-warning">Connecting to backend... <button className="btn btn-secondary" onClick={onRetry}>Retry</button></div>

  return (
    <div className="tab-content">
      <h2>Enter Log Details</h2>
      <div className="form-group">
        <label className="field-label">Log Message</label>
        <textarea
          className="textarea-field"
          placeholder="e.g. Email service experiencing issues with sending..."
          value={logMessage}
          onChange={e => setLogMessage(e.target.value)}
          rows={3}
        />
      </div>

      <button
        className="btn btn-primary full-width"
        disabled={busy}
        onClick={handlePredict}
      >
        {busy === 'predicting' ? 'Classifying...' : 'Classify Log'}
      </button>

      {error && <div className="error-box">{error}</div>}

      {result && (
        <>
          <hr />
          <h2>Prediction Result</h2>
          <div className="result-card">
            <div className="result-info">
              <div><strong>Message:</strong> <em>{result.log_message}</em></div>
            </div>
            <div className="result-label">
              <span className={`label-badge ${getLabelClass(result.predicted_label)}`}>
                {result.predicted_label}
              </span>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

// ─── Batch Prediction Tab ────────────────────────────────────────────────────
function BatchPredictionTab({ apiOk, versions, selectedVersion, busy, setBusy, onRetry }) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    const f = e.target.files[0]
    setFile(f || null)
    setPredictions(null)
    setError(null)
    if (f) {
      const reader = new FileReader()
      reader.onload = (ev) => {
        const parsed = parseCSV(ev.target.result)
        setPreview(parsed)
      }
      reader.readAsText(f)
    } else {
      setPreview(null)
    }
  }

  const handleBatchPredict = async () => {
    if (!file) return
    if (!versions.length) { setError('No models available. Please train a model first.'); return }
    setError(null)
    setPredictions(null)
    setBusy('batching')
    try {
      const res = await predictBatch(file, selectedVersion)
      setPredictions(res.predictions)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(null)
    }
  }

  const handleDownload = () => {
    if (!predictions) return
    const headers = Object.keys(predictions[0])
    const csv = [
      headers.join(','),
      ...predictions.map(row => headers.map(h => {
        const val = String(row[h] ?? '')
        return val.includes(',') || val.includes('"') ? `"${val.replace(/"/g, '""')}"` : val
      }).join(','))
    ].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'classified_logs.csv'
    a.click()
    URL.revokeObjectURL(url)
  }

  if (!apiOk) return <div className="tab-warning">Connecting to backend... <button className="btn btn-secondary" onClick={onRetry}>Retry</button></div>

  const distribution = predictions ? computeDistribution(predictions) : null
  const maxCount = distribution ? Math.max(...Object.values(distribution)) : 0

  return (
    <div className="tab-content">
      <h2>Upload Log CSV</h2>
      <p className="info-box">CSV must contain a <strong>log_message</strong> column</p>

      <input type="file" accept=".csv" onChange={handleFileChange} className="file-input" />

      {preview && (
        <div className="preview-section">
          <p><strong>Loaded {preview.rows.length} logs</strong></p>
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>{preview.headers.map((h, i) => <th key={i}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {preview.rows.slice(0, 5).map((row, i) => (
                  <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>

          <button
            className="btn btn-primary full-width"
            disabled={busy}
            onClick={handleBatchPredict}
          >
            {busy === 'batching' ? `Classifying ${preview.rows.length} logs...` : 'Classify All Logs'}
          </button>
        </div>
      )}

      {error && <div className="error-box">{error}</div>}

      {predictions && (
        <>
          <hr />
          <h2>Results Summary</h2>

          {/* Bar chart */}
          <div className="bar-chart">
            {Object.entries(distribution).sort((a, b) => b[1] - a[1]).map(([label, count]) => (
              <div key={label} className="bar-row">
                <span className="bar-label">{label}</span>
                <div className="bar-track">
                  <div
                    className={`bar-fill ${getLabelClass(label)}-bg`}
                    style={{ width: `${(count / maxCount) * 100}%` }}
                  />
                </div>
                <span className="bar-count">{count}</span>
              </div>
            ))}
          </div>

          <h2>Detailed Predictions</h2>
          <div className="table-wrapper">
            <table className="data-table">
              <thead>
                <tr>{Object.keys(predictions[0]).map((h, i) => <th key={i}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {predictions.map((row, i) => (
                  <tr key={i}>
                    {Object.entries(row).map(([key, val], j) => (
                      <td key={j}>
                        {key === 'predicted_label'
                          ? <span className={`label-badge ${getLabelClass(val)}`}>{val}</span>
                          : String(val)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <button className="btn btn-primary full-width" onClick={handleDownload}>
            Download Classified Logs CSV
          </button>
        </>
      )}
    </div>
  )
}

// ─── Main App ────────────────────────────────────────────────────────────────
function App() {
  const [apiOk, setApiOk] = useState(false)
  const [versions, setVersions] = useState([])
  const [selectedVersion, setSelectedVersion] = useState('main')
  const [modelInfo, setModelInfo] = useState(null)
  const [activeTab, setActiveTab] = useState(0)
  const [busy, setBusy] = useState(null) // 'training' | 'predicting' | 'batching' | null

  const [loading, setLoading] = useState(true)

  const connectApi = useCallback(async () => {
    setLoading(true)
    const { ok } = await checkApi()
    setApiOk(ok)
    if (ok) {
      const v = await getModelVersions()
      setVersions(v)
      if (v.length > 0) setSelectedVersion(v[v.length - 1])
      const info = await getModelInfo()
      setModelInfo(info)
    }
    setLoading(false)
  }, [])

  useEffect(() => { connectApi() }, [connectApi])

  const tabs = [
    // { label: 'Train Model', id: 'train' },
    { label: 'Single Prediction', id: 'single' },
    { label: 'Batch Prediction', id: 'batch' },
  ]

  return (
    <div className="app-layout">
      {/* <Sidebar
        apiOk={apiOk}
        versions={versions}
        selectedVersion={selectedVersion}
        setSelectedVersion={setSelectedVersion}
        modelInfo={modelInfo}
      /> */}

      <main className="main-content">
        <header className="app-header">
          <h1>AI Log Classification System</h1>
          <p className="subtitle">Classify system logs automatically using Regex, Google Embeddings, and Gemini LLM</p>
          {loading && <p className="hint">Connecting to backend (may take ~30s on cold start)...</p>}
          {!loading && apiOk && <div className="status-pill status-ok" style={{ display: 'inline-block', marginTop: '12px' }}>API Connected</div>}
          {!loading && !apiOk && <div className="status-pill status-err" style={{ display: 'inline-block', cursor: 'pointer' }} onClick={connectApi}>API Offline — click to retry</div>}
        </header>
        <hr />

        <div className="tabs">
          {tabs.map((tab, i) => (
            <button
              key={tab.id}
              className={`tab-btn ${activeTab === i ? 'active' : ''}`}
              onClick={() => setActiveTab(i)}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* {activeTab === 0 && (
          <TrainModelTab apiOk={apiOk} versions={versions} selectedVersion={selectedVersion} busy={busy} setBusy={setBusy} onRetry={connectApi} />
        )} */}
        {activeTab === 0 && (
          <SinglePredictionTab apiOk={apiOk} versions={versions} selectedVersion={selectedVersion} busy={busy} setBusy={setBusy} onRetry={connectApi} />
        )}
        {activeTab === 1 && (
          <BatchPredictionTab apiOk={apiOk} versions={versions} selectedVersion={selectedVersion} busy={busy} setBusy={setBusy} onRetry={connectApi} />
        )}
      </main>
    </div>
  )
}

export default App
