# SmartEye - Ultra-Optimized Real-Time Vision Streaming System

A lightweight, cross-platform Video Management System (VMS) with real-time AI-powered object detection using YOLOv8. Built with Python and React, SmartEye provides a clean web interface for camera management, real-time streaming, and video analytics.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![React](https://img.shields.io/badge/React-18+-61dafb.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)

## 🎯 Features

- **Multi-Source Video Support**: RTSP streams, webcams, video files, and YouTube URLs
- **Real-Time Object Detection**: YOLOv8 with 80+ object classes (COCO dataset)
- **Web-Based Dashboard**: Modern React interface for stream management and monitoring
- **High Performance**: Average latency of 41.94ms, 15-60+ FPS depending on hardware
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **Scalable Architecture**: Handles multiple concurrent streams with independent processing

## 📸 Dashboard

The SmartEye dashboard provides a comprehensive interface for managing video streams, monitoring performance, and viewing real-time detection results.

![Dashboard Overview](images/dashboard_smartEye.png)

The dashboard includes:
- **Stream Manager**: Add, configure, start, stop, and delete video streams
- **Real-Time Metrics**: Live FPS, latency, CPU/GPU usage monitoring
- **Detection Results**: View real-time object detection with bounding boxes
- **Performance Analytics**: Visual charts showing system performance over time

![Dashboard Features](images/feature.png)

### Dashboard Components

- **Metrics Panel**: System-wide performance metrics including FPS, latency, CPU/GPU usage, and stability scores
- **Stream Manager**: Configure and manage multiple video streams simultaneously
- **Video Player Grid**: Display multiple video streams with real-time annotations
- **Detection Results Viewer**: View detailed detection results for each stream

## 🎥 Video Demo

Watch the system in action:

[![SmartEye Demo Video](https://img.youtube.com/vi/YOUR_VIDEO_ID/0.jpg)](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

**Video URL**: [https://www.youtube.com/watch?v=YOUR_VIDEO_ID](https://www.youtube.com/watch?v=YOUR_VIDEO_ID)

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Source  │───▶│  Client Module  │───▶│  Server Module  │
│   (RTSP/Webcam/ │    │  (client.py)    │    │  (server.py)    │
│   File/YouTube) │    │  Frame Capture  │    │  YOLOv8 Engine  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         └─────────────▶│  Results JSON   │◀─────────────┘
                        │  (Real-time)    │
                        └─────────────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Web Dashboard  │
                        │  (React Frontend)│
                        └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Node.js 16+ and npm (for frontend)
- (Optional) CUDA-capable GPU for GPU acceleration

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/matrixAI.git
cd matrixAI
```

2. **Create a virtual environment**:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install frontend dependencies**:
```bash
cd frontend
npm install
cd ..
```

5. **Download YOLOv8 model** (optional, will auto-download on first run):
   - The system will automatically download `yolov8n.pt` (nano model) on first run
   - For better accuracy, you can manually download larger models from Ultralytics

### Running the System

1. **Start the server**:
```bash
python server.py
```

The server will start on `http://localhost:8000`

2. **Start the web dashboard**:
```bash
cd frontend
npm run dev
```

Open your browser to `http://localhost:3000`

3. **Add streams via the dashboard**:
   - Use the Stream Manager to add new video streams
   - Support for RTSP URLs, webcams, video files, and YouTube URLs
   - Configure stream names and FPS limits

## 📖 Usage

### Command Line Client

#### Single Stream
```bash
# RTSP stream
python client.py --server http://localhost:8000 --streams rtsp://username:password@camera_ip:554/stream

# Webcam (camera index 0)
python client.py --server http://localhost:8000 --streams 0 --names webcam_0

# Video file
python client.py --server http://localhost:8000 --streams video.mp4 --names video_1

# YouTube URL
python client.py --server http://localhost:8000 --streams "https://www.youtube.com/watch?v=..." --names youtube_video --types youtube
```

#### Multiple Streams
```bash
python client.py \
  --server http://localhost:8000 \
  --streams rtsp://camera1/stream 0 video.mp4 \
  --names camera_1 webcam_0 video_1 \
  --types rtsp webcam file \
  --fps-limit 30
```

### Web Dashboard

1. Navigate to `http://localhost:3000` after starting the frontend
2. Use the **Stream Manager** to add new streams
3. Monitor real-time metrics in the **Metrics Panel**
4. View detection results in the **Detection Results** section
5. Control streams with start/stop/delete buttons

## 🔧 Configuration

### Server Configuration

The server can be configured via command-line arguments or environment variables:

- **Host**: `0.0.0.0` (default) - Listen on all interfaces
- **Port**: `8000` (default) - Server port
- **Model**: `yolov8n.pt` (default) - YOLOv8 model file
- **Device**: Auto-detect (CPU/GPU)

### Client Configuration

```bash
--server URL              Inference server URL (default: http://localhost:8000)
--streams SOURCE ...      Video stream sources
--names NAME ...          Stream names
--types TYPE ...          Source types: rtsp, webcam, file, youtube, auto
--fps-limit FPS           Maximum FPS to process
--frame-skip N            Skip every N frames (for performance)
--output-dir DIR          Output directory for JSON results
```

## 📊 Performance

**Achieved Performance:**
- **Average Latency**: 41.94ms (excellent - < 50ms target)
- **Min Latency**: 29.73ms
- **Max Latency**: 85.24ms
- **Throughput**: 15.75 FPS (CPU), 60+ FPS (GPU)
- **Stability**: Low variance, consistent performance

**Hardware Requirements:**
- **Minimum**: 4GB RAM, 2-core CPU
- **Recommended**: 8GB+ RAM, 4+ core CPU, GPU (CUDA-capable)

## 📁 Project Structure

```
matrixAI/
├── server.py                  # YOLOv8 inference server (FastAPI)
├── client.py                  # Video stream client
├── requirements.txt           # Python dependencies
├── config.yaml                # Configuration file
├── frontend/                  # React web dashboard
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx      # Main dashboard component
│   │   │   ├── StreamManager.jsx # Stream management UI
│   │   │   ├── DetectionResults.jsx # Detection results viewer
│   │   │   ├── MetricsPanel.jsx  # Performance metrics display
│   │   │   └── VideoPlayer.jsx   # Video stream player
│   │   └── App.jsx              # Main application component
│   └── package.json            # Frontend dependencies
├── results/                    # Output directory (created)
│   ├── *.jsonl                 # Detection results (JSON Lines)
│   └── *_annotated.mp4         # Annotated videos
└── logs/                       # Stream process logs (created)
```

## 🔌 API Endpoints

### Inference Endpoints
- `POST /inference` - Single frame inference
- `POST /inference/batch` - Batch inference for multiple frames

### Stream Management
- `POST /streams/start` - Start a new video stream
- `POST /streams/stop` - Stop a running stream
- `POST /streams/stop-all` - Stop all running streams
- `GET /streams/status` - Get status of all streams
- `GET /streams/analytics` - Get analytics for all streams

### Monitoring
- `GET /metrics` - System performance metrics
- `GET /health` - Health check endpoint
- `GET /streams/{stream_name}/frame` - Get latest annotated frame

See the full API documentation at `http://localhost:8000/docs` when the server is running.

## 🛠️ Technologies

### Backend
- **FastAPI** - Modern async web framework
- **YOLOv8 (Ultralytics)** - State-of-the-art object detection
- **OpenCV** - Video processing
- **Uvicorn** - ASGI server
- **httpx** - Async HTTP client

### Frontend
- **React 18+** - UI framework
- **Vite** - Build tool and dev server
- **Axios** - HTTP client
- **Recharts** - Chart visualization

## 📝 Output Format

Results are saved to JSONL (JSON Lines) files in the output directory:

```json
{
  "timestamp": 1713459200.123,
  "frame_id": 32,
  "stream_name": "cam_1",
  "latency_ms": 41.94,
  "detections": [
    {
      "label": "person",
      "conf": 0.88,
      "bbox": [100.5, 150.2, 200.3, 300.7]
    }
  ]
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Download Fails**: Check internet connection or manually download the model
2. **High Latency**: Reduce FPS limit, use smaller model, or enable GPU acceleration
3. **Out of Memory**: Reduce number of concurrent streams or use frame skipping
4. **Connection Errors**: Verify server is running and check firewall settings
5. **YouTube Download Fails**: Check internet connection and yt-dlp installation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [React](https://react.dev/) for the frontend framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**SmartEye** - Ultra-Optimized Real-Time Vision Streaming System
