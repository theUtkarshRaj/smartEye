import React, { useState, useEffect } from 'react'
import Dashboard from './components/Dashboard'
import Header from './components/Header'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [serverStatus, setServerStatus] = useState('checking')
  const [serverUrl, setServerUrl] = useState(API_URL)

  useEffect(() => {
    checkServerHealth()
    const interval = setInterval(checkServerHealth, 5000)
    return () => clearInterval(interval)
  }, [serverUrl])

  const checkServerHealth = async () => {
    try {
      const response = await fetch(`${serverUrl}/health`)
      if (response.ok) {
        setServerStatus('online')
      } else {
        setServerStatus('error')
      }
    } catch (error) {
      setServerStatus('offline')
    }
  }

  const handleRestart = () => {
    // Force refresh after restart
    checkServerHealth()
  }

  return (
    <div className="app">
      <Header 
        serverStatus={serverStatus} 
        serverUrl={serverUrl}
        onServerUrlChange={setServerUrl}
        onRestart={handleRestart}
      />
      <main className="main-content">
        {serverStatus === 'online' ? (
          <Dashboard serverUrl={serverUrl} />
        ) : (
          <div className="error-state">
            <h2>Server Connection Error</h2>
            <p>
              {serverStatus === 'offline' 
                ? 'Cannot connect to the inference server. Please make sure the server is running.'
                : 'Server is not responding. Please check the server URL and try again.'}
            </p>
            <p className="server-url">Server URL: {serverUrl}</p>
            <button onClick={checkServerHealth} className="retry-button">
              Retry Connection
            </button>
          </div>
        )}
      </main>
    </div>
  )
}

export default App

