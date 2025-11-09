import React, { useState, useEffect } from 'react'
import { Cpu, Zap, Activity, TrendingUp, Clock, Database } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import './MetricsPanel.css'

const MetricsPanel = ({ metrics }) => {
  const [latencyHistory, setLatencyHistory] = useState([])
  const [fpsHistory, setFpsHistory] = useState([])
  const [cpuHistory, setCpuHistory] = useState([])

  useEffect(() => {
    if (metrics) {
      const timestamp = new Date().toLocaleTimeString()
      
      // Update latency history
      setLatencyHistory(prev => {
        const newData = [...prev, { time: timestamp, latency: metrics.avg_latency_ms }]
        return newData.slice(-20) // Keep last 20 data points
      })

      // Update FPS history
      setFpsHistory(prev => {
        const newData = [...prev, { time: timestamp, fps: metrics.current_fps }]
        return newData.slice(-20)
      })

      // Update CPU history
      setCpuHistory(prev => {
        const newData = [...prev, { time: timestamp, cpu: metrics.cpu_usage_percent }]
        return newData.slice(-20)
      })
    }
  }, [metrics])

  if (!metrics) {
    return (
      <div className="metrics-panel">
        <h2 className="panel-title">Performance Metrics</h2>
        <div className="loading-state">Loading metrics...</div>
      </div>
    )
  }

  const metricCards = [
    {
      title: 'Total Frames',
      value: metrics.total_frames?.toLocaleString() || '0',
      icon: Database,
      color: '#3b82f6',
      unit: ''
    },
    {
      title: 'Average Latency',
      value: metrics.avg_latency_ms?.toFixed(2) || '0',
      icon: Clock,
      color: '#10b981',
      unit: 'ms'
    },
    {
      title: 'Current FPS',
      value: metrics.current_fps?.toFixed(2) || '0',
      icon: Zap,
      color: '#f59e0b',
      unit: ''
    },
    {
      title: 'Average FPS',
      value: metrics.avg_fps?.toFixed(2) || '0',
      icon: TrendingUp,
      color: '#8b5cf6',
      unit: ''
    },
    {
      title: 'CPU Usage',
      value: metrics.cpu_usage_percent?.toFixed(1) || '0',
      icon: Cpu,
      color: '#ef4444',
      unit: '%'
    },
    {
      title: 'Memory Usage',
      value: (metrics.memory_usage_mb / 1024)?.toFixed(2) || '0',
      icon: Activity,
      color: '#06b6d4',
      unit: 'GB'
    }
  ]

  return (
    <div className="metrics-panel">
      <h2 className="panel-title">Performance Metrics</h2>
      
      <div className="metrics-grid">
        {metricCards.map((metric, index) => {
          const Icon = metric.icon
          return (
            <div key={index} className="metric-card">
              <div className="metric-icon" style={{ color: metric.color }}>
                <Icon size={24} />
              </div>
              <div className="metric-content">
                <div className="metric-label">{metric.title}</div>
                <div className="metric-value">
                  {metric.value}
                  {metric.unit && <span className="metric-unit">{metric.unit}</span>}
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="charts-grid">
        <div className="chart-container">
          <h3 className="chart-title">Latency Over Time</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={latencyHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="time" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#e2e8f0'
                }}
              />
              <Area 
                type="monotone" 
                dataKey="latency" 
                stroke="#10b981" 
                fill="#10b981" 
                fillOpacity={0.2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3 className="chart-title">FPS Over Time</h3>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={fpsHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="time" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#e2e8f0'
                }}
              />
              <Area 
                type="monotone" 
                dataKey="fps" 
                stroke="#f59e0b" 
                fill="#f59e0b" 
                fillOpacity={0.2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <h3 className="chart-title">CPU Usage Over Time</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={cpuHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="time" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 100]} />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1e293b', 
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#e2e8f0'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="cpu" 
                stroke="#ef4444" 
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="system-info">
        <div className="info-item">
          <span className="info-label">GPU Available:</span>
          <span className={`info-value ${metrics.gpu_available ? 'available' : 'unavailable'}`}>
            {metrics.gpu_available ? 'Yes' : 'No'}
          </span>
        </div>
        {metrics.active_streams > 0 && (
          <div className="info-item">
            <span className="info-label">Active Streams:</span>
            <span className="info-value">{metrics.active_streams}</span>
          </div>
        )}
      </div>
    </div>
  )
}

export default MetricsPanel

