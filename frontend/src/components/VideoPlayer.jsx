import React, { useRef, useEffect, useState } from 'react'
import { Play, Pause, Volume2, VolumeX, Maximize, Minimize, Video } from 'lucide-react'
import './VideoPlayer.css'

const VideoPlayer = ({ serverUrl, streamName, autoPlay = true }) => {
  const canvasRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(autoPlay)
  const [isMuted, setIsMuted] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!streamName || streamName === 'all') {
      setError(null)
      setLoading(false)
      return
    }

    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    let animationFrameId = null
    let lastFrameTime = 0
    let isFetching = false
    const fps = 60  // Increased from 30 to 60 for smoother playback
    const frameInterval = 1000 / fps
    let pendingFetch = null  // Track pending fetch to cancel if needed

    const drawFrame = async () => {
      try {
        const now = Date.now()
        // Skip if too soon or already fetching
        if (now - lastFrameTime < frameInterval || isFetching) {
          if (isPlaying) {
            animationFrameId = requestAnimationFrame(drawFrame)
          }
          return
        }
        
        isFetching = true
        lastFrameTime = now

        // Fetch annotated frame from server with cache busting
        const url = `${serverUrl}/streams/${encodeURIComponent(streamName)}/frame?t=${Date.now()}`
        
        const response = await fetch(url)
        
        if (response.ok) {
          const blob = await response.blob()
          const img = new Image()
          img.onload = () => {
            if (canvas && ctx) {
              canvas.width = img.width
              canvas.height = img.height
              ctx.drawImage(img, 0, 0)
              setLoading(false)
              setError(null)
            }
            // Clean up old object URL
            if (img.src.startsWith('blob:')) {
              URL.revokeObjectURL(img.src)
            }
            isFetching = false
          }
          img.onerror = (e) => {
            console.error('Image load error:', e)
            setError('Failed to load frame image')
            setLoading(false)
            isFetching = false
          }
          img.src = URL.createObjectURL(blob)
        } else {
          // Server should always return 200 with placeholder or real frame
          // If we get an error, it's a server issue
          const errorText = await response.text().catch(() => 'Unknown error')
          setError(`Server error (${response.status}): ${errorText}`)
          setLoading(false)
          isFetching = false
        }
      } catch (err) {
        console.error('Error loading frame:', err)
        setError(`Error: ${err.message}`)
        setLoading(false)
        isFetching = false
      }

      // Continue animation loop immediately, don't wait for fetch
      if (isPlaying) {
        animationFrameId = requestAnimationFrame(drawFrame)
      } else {
        isFetching = false
      }
    }

    if (isPlaying && streamName && streamName !== 'all') {
      setLoading(true)
      drawFrame()
    }

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId)
      }
    }
  }, [serverUrl, streamName, isPlaying])

  const togglePlay = () => {
    setIsPlaying(!isPlaying)
  }

  const toggleMute = () => {
    setIsMuted(!isMuted)
  }

  const toggleFullscreen = () => {
    if (!isFullscreen) {
      const canvas = canvasRef.current
      if (canvas && canvas.requestFullscreen) {
        canvas.requestFullscreen()
      } else if (canvas && canvas.webkitRequestFullscreen) {
        canvas.webkitRequestFullscreen()
      } else if (canvas && canvas.msRequestFullscreen) {
        canvas.msRequestFullscreen()
      }
      setIsFullscreen(true)
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen()
      } else if (document.webkitExitFullscreen) {
        document.webkitExitFullscreen()
      } else if (document.msExitFullscreen) {
        document.msExitFullscreen()
      }
      setIsFullscreen(false)
    }
  }

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  if (!streamName || streamName === 'all') {
    return (
      <div className="video-player-empty">
        <Video size={48} />
        <p>Select a stream to view annotated video</p>
      </div>
    )
  }

  return (
    <div className="video-player-container">
      <div className="video-wrapper">
        <canvas
          ref={canvasRef}
          className="video-canvas"
          onClick={togglePlay}
          style={{ display: (loading || error) ? 'none' : 'block' }}
        />
        {loading && !error && (
          <div className="video-loading">
            <p>Loading annotated video for stream: {streamName}...</p>
            <p style={{ fontSize: '12px', marginTop: '8px', color: '#888' }}>
              Make sure the stream is processing
            </p>
          </div>
        )}
        {error && (
          <div className="video-error">
            <p>{error}</p>
            <p style={{ fontSize: '12px', marginTop: '8px' }}>
              Stream: {streamName}
            </p>
            <p style={{ fontSize: '12px', marginTop: '4px', color: '#888' }}>
              Start processing using the command from Stream Management
            </p>
          </div>
        )}
        {!loading && !error && (
          <div className="video-controls">
            <button onClick={togglePlay} className="control-button">
              {isPlaying ? <Pause size={20} /> : <Play size={20} />}
            </button>
            <button onClick={toggleMute} className="control-button">
              {isMuted ? <VolumeX size={20} /> : <Volume2 size={20} />}
            </button>
            <button onClick={toggleFullscreen} className="control-button">
              {isFullscreen ? <Minimize size={20} /> : <Maximize size={20} />}
            </button>
          </div>
        )}
      </div>
      <div className="video-info">
        <span className="stream-name">{streamName}</span>
        <span className="video-status">
          {loading ? 'Loading...' : error ? 'Error' : isPlaying ? 'Playing' : 'Paused'}
        </span>
      </div>
    </div>
  )
}

export default VideoPlayer
