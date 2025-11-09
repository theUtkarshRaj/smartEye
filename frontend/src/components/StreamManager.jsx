import React, { useState, useEffect } from 'react'
import { Play, Square, Plus, Trash2, Video } from 'lucide-react'
import './StreamManager.css'

const StreamManager = ({ serverUrl, onStreamsChange, streamAnalytics = {} }) => {
  const [streams, setStreams] = useState([])
  const [newStream, setNewStream] = useState({
    name: '',
    source: '',
    type: 'webcam',
    fpsLimit: 60  // Increased default FPS limit for higher throughput
  })
  const [downloadProgress, setDownloadProgress] = useState({}) // {streamName: {progress, message}}
  const [deletedStreams, setDeletedStreams] = useState(() => {
    // Load deleted streams from localStorage on mount
    try {
      const saved = localStorage.getItem('smarteye_deleted_streams')
      return saved ? new Set(JSON.parse(saved)) : new Set()
    } catch {
      return new Set()
    }
  }) // Track deleted streams to prevent reappearing

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
    setStreams([])
    setDownloadProgress({})
    setNewStream({ name: '', source: '', type: 'webcam', fpsLimit: 60 })
    // Don't reset deletedStreams - keep them from localStorage
    // Clear any download progress intervals
    if (window.downloadProgressIntervals) {
      Object.values(window.downloadProgressIntervals).forEach(interval => clearInterval(interval))
      window.downloadProgressIntervals = {}
    }
  }, [])

  // Function to detect if a URL is a YouTube URL
  const isYouTubeUrl = (url) => {
    if (!url || typeof url !== 'string') return false
    const youtubePatterns = [
      /^https?:\/\/(www\.)?(youtube\.com|youtu\.be)\/.+/i,
      /^https?:\/\/youtube\.com\/watch\?v=.+/i,
      /^https?:\/\/youtu\.be\/.+/i,
      /^https?:\/\/www\.youtube\.com\/embed\/.+/i,
      /^https?:\/\/youtube\.com\/shorts\/.+/i
    ]
    return youtubePatterns.some(pattern => pattern.test(url.trim()))
  }

  // Handle source input change with auto-detection
  const handleSourceChange = (value) => {
    const trimmedValue = value.trim()
    
    // Auto-detect YouTube URLs
    if (isYouTubeUrl(trimmedValue)) {
      setNewStream({ ...newStream, source: trimmedValue, type: 'youtube' })
    } else {
      setNewStream({ ...newStream, source: value })
    }
  }
  
  // Notify parent when streams change
  useEffect(() => {
    if (streams && streams.length > 0) {
      const streamNames = streams.map(s => s.name).filter(Boolean)
      if (streamNames.length > 0 && onStreamsChange) {
        onStreamsChange(streamNames)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streams])  // Removed onStreamsChange from deps to prevent infinite loop

  const addStream = () => {
    // Validate inputs
    if (!newStream.name || !newStream.name.trim()) {
      alert('Please enter a stream name')
      return
    }
    
    if (!newStream.source || !newStream.source.trim()) {
      alert('Please enter a source (URL, file path, or webcam index)')
      return
    }
    
    // Check if stream name already exists
    if (streams.some(s => s.name === newStream.name.trim())) {
      alert(`Stream name "${newStream.name.trim()}" already exists. Please use a different name.`)
      return
    }
    
    // Add stream
    const stream = {
      id: Date.now(),
      name: newStream.name.trim(),
      source: newStream.source.trim(),
      type: newStream.type,
      fpsLimit: newStream.fpsLimit || 30,
      status: 'stopped'
    }
    
    const updatedStreams = [...streams, stream]
    setStreams(updatedStreams)
    setNewStream({ name: '', source: '', type: 'webcam', fpsLimit: 30 })
    
    // Notify parent about available streams
    onStreamsChange(updatedStreams.map(s => s.name))
    
    console.log('Stream added:', stream)
  }

  const removeStream = (id) => {
    setStreams(prevStreams => {
      const streamToRemove = prevStreams.find(s => s.id === id)
      if (streamToRemove) {
        // Add to deleted streams to prevent reappearing
        setDeletedStreams(prev => {
          const updated = new Set([...prev, streamToRemove.name])
          try {
            localStorage.setItem('smarteye_deleted_streams', JSON.stringify([...updated]))
          } catch (error) {
            console.warn('Failed to save deleted streams:', error)
          }
          return updated
        })
      }
      const filtered = prevStreams.filter(s => s.id !== id)
      onStreamsChange(filtered.map(s => s.name))
      return filtered
    })
  }

  const stopActiveStream = async (streamName, event) => {
    if (event) {
      event.preventDefault()
      event.stopPropagation()
    }
    
    try {
      console.log('Stopping stream:', streamName)
      const response = await fetch(`${serverUrl}/streams/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stream_name: streamName })
      })
      
      if (response.ok) {
        const result = await response.json()
        console.log('Stream stopped:', result)
        // Update local stream status if it exists
        setStreams(prevStreams => prevStreams.map(s => {
          if (s.name === streamName) {
            return { ...s, status: 'stopped' }
          }
          return s
        }))
        // Notify parent
        if (onStreamsChange) {
          setStreams(prevStreams => {
            onStreamsChange(prevStreams.map(s => s.name))
            return prevStreams
          })
        }
      } else {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
        console.error('Failed to stop stream:', error)
        alert(`Failed to stop stream: ${error.detail || 'Unknown error'}`)
      }
    } catch (error) {
      console.error('Error stopping stream:', error)
      alert(`Failed to stop stream: ${error.message || 'Unknown error'}`)
    }
  }

  const deleteActiveStream = async (streamName, event) => {
    if (event) {
      event.preventDefault()
      event.stopPropagation()
    }
    
    if (!window.confirm(`Are you sure you want to delete stream "${streamName}"? This will stop the stream and remove it from the list.`)) {
      return
    }
    
    try {
      console.log('Deleting stream:', streamName)
      // First stop the stream
      const stopResponse = await fetch(`${serverUrl}/streams/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stream_name: streamName })
      })
      
      // Add to deleted streams set to prevent reappearing (persisted in localStorage)
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
      
      // Remove from local streams if it exists
      setStreams(prevStreams => {
        const filtered = prevStreams.filter(s => s.name !== streamName)
        if (onStreamsChange) {
          onStreamsChange(filtered.map(s => s.name))
        }
        return filtered
      })
      
      if (!stopResponse.ok) {
        const error = await stopResponse.json().catch(() => ({ detail: 'Unknown error' }))
        console.warn(`Failed to stop stream before deletion: ${error.detail || 'Unknown error'}`)
      } else {
        console.log('Stream deleted successfully:', streamName)
      }
    } catch (error) {
      console.error('Error deleting stream:', error)
      // Still add to deleted streams and remove from local state even if server call fails
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
      setStreams(prevStreams => {
        const filtered = prevStreams.filter(s => s.name !== streamName)
        if (onStreamsChange) {
          onStreamsChange(filtered.map(s => s.name))
        }
        return filtered
      })
    }
  }

  const toggleStream = async (id) => {
    // Get current stream state using functional update to avoid stale state
    let stream = null
    let isRunning = false
    setStreams(prevStreams => {
      stream = prevStreams.find(s => s.id === id)
      if (stream) {
        isRunning = stream.status === 'running'
      }
      return prevStreams
    })
    
    if (!stream) return
    
    try {
      if (isRunning) {
        // Stop stream
        const response = await fetch(`${serverUrl}/streams/stop`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ stream_name: stream.name })
        })
        
        if (response.ok) {
          setStreams(prevStreams => prevStreams.map(s => {
            if (s.id === id) {
              return { ...s, status: 'stopped' }
            }
            return s
          }))
        } else {
          const error = await response.json()
          console.error('Error stopping stream:', error)
          alert(`Failed to stop stream: ${error.detail || 'Unknown error'}`)
        }
      } else {
        // Show loading state for YouTube downloads
        if (stream.type === 'youtube') {
          setStreams(prevStreams => prevStreams.map(s => {
            if (s.id === id) {
              return { ...s, status: 'downloading' }
            }
            return s
          }))
          
          // Start polling for download progress
          const progressInterval = setInterval(async () => {
            try {
              const progressResponse = await fetch(`${serverUrl}/streams/${stream.name}/download-progress`)
              if (progressResponse.ok) {
                const progressData = await progressResponse.json()
                setDownloadProgress(prev => ({
                  ...prev,
                  [stream.name]: progressData
                }))
                
                // If download completed or failed, stop polling
                if (progressData.status === 'completed' || progressData.status === 'failed') {
                  clearInterval(progressInterval)
                }
              }
            } catch (e) {
              console.error('Error fetching download progress:', e)
            }
          }, 500) // Poll every 500ms
          
          // Store interval ID to clear it later
          if (!window.downloadProgressIntervals) {
            window.downloadProgressIntervals = {}
          }
          window.downloadProgressIntervals[stream.name] = progressInterval
        }
        
        // Start stream with timeout
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), 600000) // 10 minute timeout for YouTube downloads
        
        try {
          const response = await fetch(`${serverUrl}/streams/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              stream_name: stream.name,
              source: stream.source,
              source_type: stream.type,
              fps_limit: stream.fpsLimit
            }),
            signal: controller.signal
          })
          
          clearTimeout(timeoutId)
          
          // Clear progress polling if it exists
          if (window.downloadProgressIntervals && window.downloadProgressIntervals[stream.name]) {
            clearInterval(window.downloadProgressIntervals[stream.name])
            delete window.downloadProgressIntervals[stream.name]
          }
          
          if (response.ok) {
            const result = await response.json()
            setStreams(prevStreams => prevStreams.map(s => {
              if (s.id === id) {
                return { ...s, status: 'running' }
              }
              return s
            }))
            // Clear download progress
            setDownloadProgress(prev => {
              const newProgress = { ...prev }
              delete newProgress[stream.name]
              return newProgress
            })
            // Notify parent that stream started
            setStreams(prevStreams => {
              onStreamsChange(prevStreams.map(s => s.name))
              return prevStreams
            })
            
            if (stream.type === 'youtube') {
              alert(`YouTube video downloaded and processing started!\n\nStream: ${stream.name}`)
            }
          } else {
            let errorMessage = 'Unknown error'
            try {
              const error = await response.json()
              errorMessage = error.detail || error.message || 'Unknown error'
              console.error('Error starting stream:', error)
            } catch (e) {
              errorMessage = `Server error: ${response.status} ${response.statusText}`
            }
            
            // Reset status if failed
            setStreams(prevStreams => prevStreams.map(s => {
              if (s.id === id) {
                return { ...s, status: 'stopped' }
              }
              return s
            }))
            
            // Clear progress polling if it exists
            if (window.downloadProgressIntervals && window.downloadProgressIntervals[stream.name]) {
              clearInterval(window.downloadProgressIntervals[stream.name])
              delete window.downloadProgressIntervals[stream.name]
            }
            
            alert(`Failed to start stream: ${errorMessage}\n\n${stream.type === 'youtube' ? 'Make sure the YouTube URL is valid and accessible.' : 'Make sure the source URL is valid and accessible.'}`)
          }
        } catch (fetchError) {
          clearTimeout(timeoutId)
          
          // Clear progress polling if it exists
          if (window.downloadProgressIntervals && window.downloadProgressIntervals[stream.name]) {
            clearInterval(window.downloadProgressIntervals[stream.name])
            delete window.downloadProgressIntervals[stream.name]
          }
          
          // Reset status if failed
          setStreams(prevStreams => prevStreams.map(s => {
            if (s.id === id) {
              return { ...s, status: 'stopped' }
            }
            return s
          }))
          
          if (fetchError.name === 'AbortError') {
            alert('Request timed out. The server may be processing a large video. Please wait and try again.')
          } else if (fetchError.message.includes('Failed to fetch') || fetchError.message.includes('NetworkError')) {
            alert('Failed to connect to server. Please check:\n1. Server is running\n2. Server URL is correct\n3. Network connection is active')
          } else {
            alert(`Network error: ${fetchError.message}\n\nPlease check your connection and try again.`)
          }
          console.error('Fetch error:', fetchError)
        }
      }
    } catch (error) {
      console.error('Error toggling stream:', error)
      
      // Reset status if failed
      setStreams(prevStreams => prevStreams.map(s => {
        if (s.id === id) {
          return { ...s, status: 'stopped' }
        }
        return s
      }))
      
      // Better error messages
      if (error.name === 'AbortError') {
        alert('Request timed out. Please try again.')
      } else if (error.message && error.message.includes('fetch')) {
        alert('Failed to connect to server. Please check if the server is running.')
      } else {
        alert(`Error: ${error.message || 'Unknown error occurred'}`)
      }
    }
  }

  return (
    <div className="stream-manager">
      <h2 className="panel-title">Stream Management</h2>

      <div className="add-stream-form">
        <h3 className="form-title">Add New Stream</h3>
        <div className="form-grid">
          <div className="form-group">
            <label>Stream Name</label>
            <input
              type="text"
              value={newStream.name}
              onChange={(e) => setNewStream({ ...newStream, name: e.target.value })}
              placeholder="e.g., webcam_0"
              className="form-input"
            />
          </div>
          <div className={`form-group ${newStream.type === 'youtube' ? 'form-group-full-width' : ''}`}>
            <label>Source</label>
            <input
              type="text"
              value={newStream.source}
              onChange={(e) => handleSourceChange(e.target.value)}
              placeholder={newStream.type === 'youtube' ? 'https://youtube.com/watch?v=... or https://youtu.be/...' : newStream.type === 'webcam' ? '0 (webcam index)' : newStream.type === 'rtsp' ? 'rtsp://...' : 'path/to/video.mp4'}
              className="form-input"
              style={{ 
                wordBreak: 'break-all',
                overflowWrap: 'break-word'
              }}
            />
            {isYouTubeUrl(newStream.source) && newStream.type !== 'youtube' && (
              <small className="auto-detect-hint" style={{ color: '#10b981', marginTop: '4px', display: 'block' }}>
                ✓ YouTube URL detected - Type automatically set to YouTube
              </small>
            )}
          </div>
          <div className="form-group">
            <label>Type</label>
            <select
              value={newStream.type}
              onChange={(e) => {
                const newType = e.target.value
                // Don't clear source if it's a YouTube URL and we're changing to YouTube, or if source is empty
                if (isYouTubeUrl(newStream.source) && newType === 'youtube') {
                  setNewStream({ ...newStream, type: newType })
                } else {
                  setNewStream({ ...newStream, type: newType, source: '' })
                }
              }}
              className="form-select"
            >
              <option value="webcam">Webcam</option>
              <option value="rtsp">RTSP</option>
              <option value="file">Video File</option>
              <option value="youtube">YouTube Video</option>
            </select>
          </div>
          <div className="form-group">
            <label>FPS Limit</label>
            <input
              type="number"
              value={newStream.fpsLimit}
              onChange={(e) => setNewStream({ ...newStream, fpsLimit: parseInt(e.target.value) || 30 })}
              placeholder="30"
              className="form-input"
            />
          </div>
        </div>
        <button onClick={addStream} className="add-button">
          <Plus size={16} />
          Add Stream
        </button>
      </div>

      <div className="streams-list">
        {(() => {
          // Count active streams (filter out deleted)
          const activeStreams = Object.entries(streamAnalytics)
            .filter(([streamName]) => !deletedStreams.has(streamName))
          const activeCount = activeStreams.length
          
          const configuredStreams = streams
            .filter(s => !streamAnalytics[s.name] && !deletedStreams.has(s.name))
          const configuredCount = configuredStreams.length
          
          // Show empty state if no streams at all
          if (activeCount === 0 && configuredCount === 0) {
            return (
              <>
                <h3 className="list-title">Streams</h3>
                <div className="empty-state">
                  <Video size={48} />
                  <p>No streams configured</p>
                  <p className="empty-hint">Add a stream to start processing</p>
                </div>
              </>
            )
          }
          
          // Show title with count
          let title
          if (activeCount > 0 && configuredCount > 0) {
            title = <h3 className="list-title">Active Streams ({activeCount}) | Configured ({configuredCount})</h3>
          } else if (activeCount > 0) {
            title = <h3 className="list-title">Active Streams ({activeCount})</h3>
          } else if (configuredCount > 0) {
            title = <h3 className="list-title">Configured Streams ({configuredCount})</h3>
          } else {
            title = <h3 className="list-title">Streams</h3>
          }
          
          return (
            <>
              {title}
              {/* Display active streams from server analytics (filter out deleted streams) */}
              {activeStreams.map(([streamName, analytics]) => (
              <div key={streamName} className="stream-item">
                <div className="stream-info">
                  <div className="stream-header">
                    <Video size={20} />
                    <span className="stream-name">{streamName}</span>
                    <span className="stream-status running">Active</span>
                  </div>
                  <div className="stream-details">
                    <span className="detail-item">
                      <strong>FPS:</strong> {analytics.current_fps || analytics.fps || 0}
                    </span>
                    <span className="detail-item">
                      <strong>Latency:</strong> {analytics.average_latency_ms || analytics.latency_ms || 0}ms
                    </span>
                    <span className="detail-item">
                      <strong>Stability:</strong> {analytics.stability_score || analytics.stability || 0}%
                    </span>
                    <span className="detail-item">
                      <strong>CPU Core:</strong> {analytics.cpu_core >= 0 ? analytics.cpu_core : 'N/A'}
                    </span>
                    <span className="detail-item">
                      <strong>Frames:</strong> {analytics.total_frames || 0}
                    </span>
                    <span className="detail-item">
                      <strong>Throughput:</strong> {analytics.throughput_fps || 0} FPS
                    </span>
                  </div>
                </div>
                <div className="stream-actions">
                  <button
                    onClick={(e) => stopActiveStream(streamName, e)}
                    className="action-button stop"
                    type="button"
                    style={{ cursor: 'pointer', zIndex: 10 }}
                  >
                    <Square size={16} />
                    Stop
                  </button>
                  <button
                    onClick={(e) => deleteActiveStream(streamName, e)}
                    className="action-button delete"
                    type="button"
                    style={{ cursor: 'pointer', zIndex: 10 }}
                  >
                    <Trash2 size={16} />
                    Delete
                  </button>
                </div>
              </div>
              ))}
              {/* Display configured streams not yet active (filter out deleted streams) */}
              {configuredStreams.map(stream => (
              <div key={stream.id} className="stream-item">
                <div className="stream-info">
                  <div className="stream-header">
                    <Video size={20} />
                    <span className="stream-name">{stream.name}</span>
                    <span className={`stream-status ${stream.status}`}>
                      {stream.status}
                    </span>
                  </div>
                  <div className="stream-details">
                    <span className="detail-item">
                      <strong>Source:</strong> <span className="source-url">{stream.source}</span>
                    </span>
                    <span className="detail-item">
                      <strong>Type:</strong> {stream.type}
                    </span>
                    <span className="detail-item">
                      <strong>FPS Limit:</strong> {stream.fpsLimit}
                    </span>
                  </div>
                  {stream.status === 'downloading' && downloadProgress[stream.name] && (
                    <div className="download-progress">
                      <div className="progress-bar-container">
                        <div 
                          className="progress-bar" 
                          style={{ width: `${downloadProgress[stream.name].progress || 0}%` }}
                        />
                      </div>
                      <div className="progress-text">
                        {downloadProgress[stream.name].message || 'Downloading...'} ({downloadProgress[stream.name].progress?.toFixed(1) || 0}%)
                      </div>
                    </div>
                  )}
                </div>
                <div className="stream-actions">
                  <button
                    onClick={() => toggleStream(stream.id)}
                    className={`action-button ${stream.status === 'running' ? 'stop' : 'start'}`}
                  >
                    {stream.status === 'running' ? (
                      <>
                        <Square size={16} />
                        Stop
                      </>
                    ) : stream.status === 'downloading' ? (
                      <>
                        <Play size={16} />
                        Downloading...
                      </>
                    ) : (
                      <>
                        <Play size={16} />
                        Start
                      </>
                    )}
                  </button>
                  <button
                    onClick={() => removeStream(stream.id)}
                    className="action-button delete"
                  >
                    <Trash2 size={16} />
                  </button>
                </div>
              </div>
              ))}
            </>
          )
        })()}
      </div>

      <div className="stream-commands">
        <h3 className="commands-title">Command Line Usage</h3>
        <div className="command-example">
          <code>
            python client.py --streams {streams.length > 0 ? streams[0].source : '0'} --names {streams.length > 0 ? streams[0].name : 'stream_0'} --fps-limit {streams.length > 0 ? streams[0].fpsLimit : 30}
          </code>
        </div>
      </div>
    </div>
  )
}

export default StreamManager

