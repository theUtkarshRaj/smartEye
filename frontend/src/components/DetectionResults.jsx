import React, { useState, useEffect } from 'react'
import { Eye, Download, RefreshCw, Filter, Video, Trash2, Square } from 'lucide-react'
import VideoPlayer from './VideoPlayer'
import './DetectionResults.css'

const DetectionResults = ({ serverUrl, activeStreams, onDetectionsUpdate, onStreamsChange }) => {
  const [results, setResults] = useState([])
  const [filter, setFilter] = useState('all')
  const [selectedStream, setSelectedStream] = useState(activeStreams.length > 0 ? activeStreams[0] : 'all')
  const [showVideo, setShowVideo] = useState(true)
  const [availableStreams, setAvailableStreams] = useState([])
  const [streamStatuses, setStreamStatuses] = useState({}) // Track stream running status
  const [deletedStreams, setDeletedStreams] = useState(() => {
    // Load deleted streams from localStorage on mount
    try {
      const saved = localStorage.getItem('smarteye_deleted_streams')
      return saved ? new Set(JSON.parse(saved)) : new Set()
    } catch {
      return new Set()
    }
  }) // Track deleted streams
  const [notifications, setNotifications] = useState([]) // Track notifications

  // Save deleted streams to localStorage whenever it changes
  useEffect(() => {
    try {
      localStorage.setItem('smarteye_deleted_streams', JSON.stringify([...deletedStreams]))
    } catch (error) {
      console.warn('Failed to save deleted streams to localStorage:', error)
    }
  }, [deletedStreams])

  // Reset state on mount (page refresh) - but keep deleted streams
  useEffect(() => {
    setResults([])
    setFilter('all')
    setSelectedStream('all')
    setShowVideo(true)
    setAvailableStreams([])
    setStreamStatuses({})
    // Don't reset deletedStreams - keep them from localStorage
    setNotifications([])
  }, [])

  // Update available streams when activeStreams changes, but exclude deleted streams
  useEffect(() => {
    if (activeStreams.length > 0) {
      const filtered = activeStreams.filter(s => !deletedStreams.has(s))
      setAvailableStreams(filtered)
      // Auto-select first stream if none selected
      if (selectedStream === 'all' || !filtered.includes(selectedStream)) {
        setSelectedStream(filtered[0] || 'all')
      }
    } else {
      setAvailableStreams([])
    }
  }, [activeStreams, deletedStreams, selectedStream])

  useEffect(() => {
    // Simulate fetching results - in real implementation, this would fetch from API
    // or use WebSocket for real-time updates
    const interval = setInterval(() => {
      // This would be replaced with actual API call
      // fetch(`${serverUrl}/results?stream=${selectedStream}`)
    }, 5000)

    return () => clearInterval(interval)
  }, [serverUrl, selectedStream])

  const uniqueLabels = [...new Set(results.flatMap(r => r.detections?.map(d => d.label) || []))]

  const filteredResults = results.filter(result => {
    if (selectedStream !== 'all' && result.stream_name !== selectedStream) return false
    if (filter === 'all') return true
    return result.detections?.some(d => d.label === filter)
  })

  const exportResults = () => {
    const dataStr = JSON.stringify(filteredResults, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `detections_${new Date().toISOString()}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  const clearAll = async () => {
    if (!window.confirm('Are you sure you want to clear all streams and results? This will stop all running streams.')) {
      return
    }
    
    try {
      // Try to clear all streams on server - use stop-all as fallback
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
      
      // Clear all local state regardless of server response
      setResults([])
      setFilter('all')
      setSelectedStream('all')
      setAvailableStreams([])
      setStreamStatuses({})
      setDeletedStreams(new Set())
      // Clear localStorage
      try {
        localStorage.removeItem('smarteye_deleted_streams')
      } catch (error) {
        console.warn('Failed to clear deleted streams from localStorage:', error)
      }
      setNotifications([])
      
      // Notify parent to clear streams
      if (onStreamsChange) {
        onStreamsChange([])
      }
      
      if (response.ok || response.status === 404) {
        showNotification('All streams and results cleared successfully', 'success')
      } else {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
        showNotification(`Cleared local state, but server error: ${error.detail || 'Unknown error'}`, 'error')
      }
    } catch (error) {
      // Even if there's an error, clear local state
      setResults([])
      setFilter('all')
      setSelectedStream('all')
      setAvailableStreams([])
      setStreamStatuses({})
      setDeletedStreams(new Set())
      // Clear localStorage
      try {
        localStorage.removeItem('smarteye_deleted_streams')
      } catch (err) {
        console.warn('Failed to clear deleted streams from localStorage:', err)
      }
      setNotifications([])
      
      if (onStreamsChange) {
        onStreamsChange([])
      }
      
      // Only show error if it's not a network/404 error
      if (error.name !== 'TypeError' && !error.message.includes('Failed to fetch')) {
        console.error('Error clearing all:', error)
        showNotification(`Cleared local state, but server error: ${error.message || 'Unknown error'}`, 'error')
      } else {
        showNotification('All streams and results cleared successfully', 'success')
      }
    }
  }

  const showNotification = (message, type = 'success') => {
    const id = Date.now()
    setNotifications(prev => [...prev, { id, message, type }])
    // Auto-dismiss after 3 seconds
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id))
    }, 3000)
  }

  const stopStream = async (streamName, e) => {
    e?.preventDefault()
    e?.stopPropagation()
    
    try {
      const response = await fetch(`${serverUrl}/streams/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stream_name: streamName })
      })
      
      if (response.ok) {
        setStreamStatuses(prev => ({ ...prev, [streamName]: 'stopped' }))
        // Mark as deleted to remove from grid (persisted in localStorage)
        setDeletedStreams(prev => {
          const updated = new Set([...prev, streamName])
          // Save to localStorage immediately
          try {
            localStorage.setItem('smarteye_deleted_streams', JSON.stringify([...updated]))
          } catch (error) {
            console.warn('Failed to save deleted streams:', error)
          }
          return updated
        })
        // Notify parent
        if (onStreamsChange) {
          const updated = availableStreams.filter(s => s !== streamName)
          onStreamsChange(updated)
        }
        showNotification(`Stream "${streamName}" stopped successfully`, 'success')
      } else {
        const error = await response.json()
        showNotification(`Failed to stop stream: ${error.detail || 'Unknown error'}`, 'error')
      }
    } catch (error) {
      console.error('Error stopping stream:', error)
      showNotification(`Failed to stop stream: ${error.message || 'Unknown error'}`, 'error')
    }
  }

  const deleteStream = async (streamName, e) => {
    e?.preventDefault()
    e?.stopPropagation()
    
    if (!window.confirm(`Are you sure you want to delete stream "${streamName}"?`)) {
      return
    }
    
    try {
      // First stop the stream if it's running
      const isRunning = streamStatuses[streamName] === 'running'
      if (isRunning) {
        const response = await fetch(`${serverUrl}/streams/stop`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ stream_name: streamName })
        })
        if (!response.ok) {
          console.warn(`Failed to stop stream before deletion: ${streamName}`)
        }
      }
      
      // Mark as deleted to remove from grid (persisted in localStorage)
      setDeletedStreams(prev => {
        const updated = new Set([...prev, streamName])
        // Save to localStorage immediately
        try {
          localStorage.setItem('smarteye_deleted_streams', JSON.stringify([...updated]))
        } catch (error) {
          console.warn('Failed to save deleted streams:', error)
        }
        return updated
      })
      
      // Remove from stream statuses
      setStreamStatuses(prev => {
        const updated = { ...prev }
        delete updated[streamName]
        return updated
      })
      
      // Notify parent
      if (onStreamsChange) {
        const updated = availableStreams.filter(s => s !== streamName)
        onStreamsChange(updated)
      }
      
      showNotification(`Stream "${streamName}" deleted successfully`, 'success')
    } catch (error) {
      console.error('Error deleting stream:', error)
      showNotification(`Failed to delete stream: ${error.message || 'Unknown error'}`, 'error')
    }
  }

  return (
    <div className="detection-results">
      <div className="results-header">
        <h2 className="panel-title">Detection Results</h2>
        <div className="results-actions">
          <button 
            onClick={() => setShowVideo(!showVideo)} 
            className="action-button"
            title={showVideo ? "Hide Video" : "Show Video"}
          >
            <Video size={16} />
            {showVideo ? 'Hide Video' : 'Show Video'}
          </button>
          <button onClick={exportResults} className="action-button">
            <Download size={16} />
            Export
          </button>
          <button onClick={clearAll} className="action-button">
            <RefreshCw size={16} />
            Clear All
          </button>
        </div>
      </div>

      {showVideo && (
        <div className="video-section">
          {availableStreams.length > 0 ? (
            <div className="video-grid">
              {availableStreams.map(streamName => (
                <div key={streamName} className="video-grid-item">
                  <div className="video-item-header">
                    <span className="video-item-title">{streamName}</span>
                    <div className="video-item-controls">
                      <button
                        onClick={(e) => stopStream(streamName, e)}
                        className="video-control-btn stop-btn"
                        title="Stop Stream"
                        type="button"
                      >
                        <Square size={18} />
                      </button>
                      <button
                        onClick={(e) => deleteStream(streamName, e)}
                        className="video-control-btn delete-btn"
                        title="Delete Stream"
                        type="button"
                      >
                        <Trash2 size={18} />
                      </button>
                    </div>
                  </div>
                  <VideoPlayer 
                    serverUrl={serverUrl} 
                    streamName={streamName}
                    autoPlay={true}
                  />
                </div>
              ))}
            </div>
          ) : (
            <div className="video-placeholder">
              <Video size={48} />
              <p>No streams available</p>
              <p className="empty-hint">
                Add a stream from Stream Management to view video
              </p>
            </div>
          )}
          {/* Notifications below video grid */}
          {notifications.length > 0 && (
            <div className="notifications-container">
              {notifications.map(notification => (
                <div
                  key={notification.id}
                  className={`notification notification-${notification.type}`}
                >
                  {notification.message}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="results-filters">
        <div className="filter-group">
          <Filter size={16} />
          <label>Stream:</label>
          <select
            value={selectedStream}
            onChange={(e) => {
              setSelectedStream(e.target.value)
              if (e.target.value !== 'all') {
                setShowVideo(true)
              }
            }}
            className="filter-select"
          >
            <option value="all">All Streams</option>
            {availableStreams.map(stream => (
              <option key={stream} value={stream}>{stream}</option>
            ))}
          </select>
        </div>
        <div className="filter-group">
          <label>Label:</label>
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="filter-select"
          >
            <option value="all">All Labels</option>
            {uniqueLabels.map(label => (
              <option key={label} value={label}>{label}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="results-stats">
        <div className="stat-item">
          <span className="stat-label">Total Results:</span>
          <span className="stat-value">{filteredResults.length}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Total Detections:</span>
          <span className="stat-value">
            {filteredResults.reduce((sum, r) => sum + (r.detections?.length || 0), 0)}
          </span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Unique Labels:</span>
          <span className="stat-value">{uniqueLabels.length}</span>
        </div>
      </div>

      <div className="results-list">
        {filteredResults.length === 0 ? (
          <div className="empty-state">
            <Eye size={48} />
            <p>No detection results yet</p>
            <p className="empty-hint">Results will appear here as frames are processed</p>
          </div>
        ) : (
          filteredResults.slice(-20).reverse().map((result, index) => (
            <div key={index} className="result-item">
              <div className="result-header">
                <div className="result-meta">
                  <span className="result-stream">{result.stream_name}</span>
                  <span className="result-frame">Frame #{result.frame_id}</span>
                  <span className="result-latency">{result.latency_ms?.toFixed(2)}ms</span>
                </div>
                <div className="result-timestamp">
                  {new Date(result.timestamp * 1000).toLocaleTimeString()}
                </div>
              </div>
              <div className="result-detections">
                {result.detections && result.detections.length > 0 ? (
                  result.detections.map((detection, idx) => (
                    <div key={idx} className="detection-item">
                      <span className="detection-label">{detection.label}</span>
                      <span className="detection-confidence">
                        {(detection.conf * 100).toFixed(1)}%
                      </span>
                      <span className="detection-bbox">
                        [{detection.bbox?.map(b => b.toFixed(0)).join(', ')}]
                      </span>
                    </div>
                  ))
                ) : (
                  <span className="no-detections">No detections</span>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default DetectionResults

