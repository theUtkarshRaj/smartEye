import React, { useState, useEffect, useCallback, useMemo } from 'react'
import MetricsPanel from './MetricsPanel'
import StreamManager from './StreamManager'
import DetectionResults from './DetectionResults'
import './Dashboard.css'

const Dashboard = ({ serverUrl }) => {
  const [metrics, setMetrics] = useState(null)
  const [detections, setDetections] = useState([])
  const [activeStreams, setActiveStreams] = useState([])
  const [streamAnalytics, setStreamAnalytics] = useState({})
  const [configuredStreams, setConfiguredStreams] = useState([])
  
  // Memoize the callback to prevent infinite loops
  const handleStreamsChange = useCallback((streams) => {
    if (Array.isArray(streams)) {
      setConfiguredStreams(streams)
      // Use the provided streams directly (they already exclude deleted ones)
      setActiveStreams(streams)
    }
  }, [])

  useEffect(() => {
    // Clear server state on mount (page refresh)
    const clearServerState = async () => {
      try {
        const response = await fetch(`${serverUrl}/streams/clear`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        })
        // Silently ignore 404 - endpoint might not exist on old server
        if (!response.ok && response.status !== 404) {
          console.warn('Could not clear server state:', response.status)
        }
      } catch (error) {
        // Silently ignore errors - server might not be running or endpoint doesn't exist
        // Only log non-network errors
        if (error.name !== 'TypeError' && !error.message.includes('Failed to fetch')) {
          console.warn('Could not clear server state:', error)
        }
      }
    }
    
    // Reset local state on mount
    setMetrics(null)
    setDetections([])
    setActiveStreams([])
    setStreamAnalytics({})
    setConfiguredStreams([])
    
    // Clear server state
    clearServerState()
    
    // Start fetching metrics
    fetchMetrics()
    fetchStreamAnalytics()
    const interval = setInterval(() => {
      fetchMetrics()
      fetchStreamAnalytics()
    }, 2000) // Update every 2 seconds
    return () => clearInterval(interval)
  }, [serverUrl])

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${serverUrl}/metrics`, {
        signal: AbortSignal.timeout(5000) // 5 second timeout
      })
      if (response.ok) {
        const data = await response.json()
        setMetrics(data)
        // Extract stream names from stream_analytics if available
        if (data.stream_analytics) {
          const streamNames = Object.keys(data.stream_analytics)
          setActiveStreams(streamNames)
        } else {
          setActiveStreams(Array.from({ length: data.active_streams || 0 }, (_, i) => `stream_${i + 1}`))
        }
      }
    } catch (error) {
      // Only log if it's not a connection error (to reduce console spam)
      if (error.name !== 'AbortError' && !error.message.includes('Failed to fetch')) {
        console.error('Error fetching metrics:', error)
      }
    }
  }

  const fetchStreamAnalytics = async () => {
    try {
      const response = await fetch(`${serverUrl}/streams/analytics`, {
        signal: AbortSignal.timeout(5000) // 5 second timeout
      })
      if (response.ok) {
        const data = await response.json()
        setStreamAnalytics(data.streams || {})
        // Update active streams from analytics
        if (data.streams && Object.keys(data.streams).length > 0) {
          const streamNames = Object.keys(data.streams)
          setActiveStreams(streamNames)
        }
      }
    } catch (error) {
      // Only log if it's not a connection error (to reduce console spam)
      if (error.name !== 'AbortError' && !error.message.includes('Failed to fetch')) {
        console.error('Error fetching stream analytics:', error)
      }
    }
  }

  return (
    <div className="dashboard">
      <div className="dashboard-grid">
        <div className="dashboard-section metrics-section">
          <MetricsPanel metrics={metrics} />
        </div>
        <div className="dashboard-section streams-section">
          <StreamManager 
            serverUrl={serverUrl} 
            onStreamsChange={handleStreamsChange}
            streamAnalytics={streamAnalytics}
          />
        </div>
        <div className="dashboard-section results-section">
          <DetectionResults 
            serverUrl={serverUrl} 
            activeStreams={useMemo(() => [...new Set([...configuredStreams, ...activeStreams])], [configuredStreams, activeStreams])}
            onDetectionsUpdate={setDetections}
            onStreamsChange={handleStreamsChange}
          />
        </div>
      </div>
    </div>
  )
}

export default Dashboard

