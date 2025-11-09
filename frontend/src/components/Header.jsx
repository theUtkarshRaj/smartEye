import React, { useState } from 'react'
import { Activity, Server, RefreshCw } from 'lucide-react'
import './Header.css'

const Header = ({ serverStatus, serverUrl, onServerUrlChange, onRestart }) => {
  const [isRestarting, setIsRestarting] = useState(false)
  const getStatusColor = () => {
    switch (serverStatus) {
      case 'online':
        return '#10b981'
      case 'offline':
        return '#ef4444'
      case 'error':
        return '#f59e0b'
      default:
        return '#6b7280'
    }
  }

  const getStatusText = () => {
    switch (serverStatus) {
      case 'online':
        return 'Online'
      case 'offline':
        return 'Offline'
      case 'error':
        return 'Error'
      default:
        return 'Checking...'
    }
  }

  const handleRestart = async () => {
    if (!window.confirm('Are you sure you want to restart the server? This will stop all streams and clear all state.')) {
      return
    }
    
    setIsRestarting(true)
    
    try {
      // Try to clear all server state - use stop-all as fallback if clear doesn't exist
      let response = await fetch(`${serverUrl}/streams/clear`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      })
      
      // If clear endpoint doesn't exist (404), try stop-all
      if (!response.ok && response.status === 404) {
        response = await fetch(`${serverUrl}/streams/stop-all`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        })
      }
      
      // If both fail, just continue anyway - we'll reload the page
      if (!response.ok && response.status !== 404) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
        console.warn('Failed to clear server state:', error)
        // Continue anyway - reload will reset frontend
      }
      
      // Notify parent to refresh/restart
      if (onRestart) {
        onRestart()
      }
      
      // Small delay to ensure state is cleared, then reload
      setTimeout(() => {
        window.location.reload()
      }, 500)
      
    } catch (error) {
      console.error('Error restarting server:', error)
      // Even if there's an error, reload the page to reset frontend
      if (onRestart) {
        onRestart()
      }
      setTimeout(() => {
        window.location.reload()
      }, 500)
    }
  }

  return (
    <header className="header">
      <div className="header-content">
        <div className="header-left">
          <Activity className="header-icon" />
          <div>
            <h1 className="header-title">SmartEye</h1>
            <p className="header-subtitle">Real-time Video Inference Dashboard</p>
          </div>
        </div>
        <div className="header-right">
          <div className="server-status">
            <Server className="status-icon" size={16} />
            <div className="status-info">
              <span className="status-label">Server Status:</span>
              <span 
                className="status-value"
                style={{ color: getStatusColor() }}
              >
                {getStatusText()}
              </span>
            </div>
          </div>
          <div className="server-url-input">
            <input
              type="text"
              value={serverUrl}
              onChange={(e) => onServerUrlChange(e.target.value)}
              placeholder="Server URL"
              className="url-input"
            />
          </div>
          {serverStatus === 'online' && (
            <button
              onClick={handleRestart}
              className="restart-button"
              title="Restart Server"
              disabled={isRestarting}
            >
              <RefreshCw size={16} className={isRestarting ? 'spinning' : ''} />
              {isRestarting ? 'Restarting...' : 'Restart Server'}
            </button>
          )}
        </div>
      </div>
    </header>
  )
}

export default Header

