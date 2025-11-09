#Ultra-Optimized Real-Time Vision Streaming System (YOLOv8)
import asyncio
import base64
import io
import json 
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil 
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from ultralytics import YOLO

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Configure logging - only metrics (latency, FPS, stability)
# Suppress verbose API request logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)

# Suppress FastAPI/uvicorn access logs
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Global inference engine instance (forward reference)
inference_engine: Optional["YOLOv8InferenceEngine"] = None

# Track running stream processes
stream_processes: Dict[str, subprocess.Popen] = {}

# Track YouTube download progress
download_progress: Dict[str, Dict] = {}  # {stream_name: {"status": "downloading", "progress": 0.0, "message": ""}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    global inference_engine
    try:
        logger.info("Loading YOLOv8 model from yolov8n.pt...")
        inference_engine = YOLOv8InferenceEngine(
            model_path="yolov8n.pt",  # Nano model for speed, change to yolov8s/m/l/x for accuracy
            device=None,  # Auto-detect
            conf_threshold=0.25,
            iou_threshold=0.45
        )
        logger.info("Inference server started successfully")
    except Exception as e:
        logger.error(f"Failed to start inference server: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down inference server")
    
    # Stop all running stream processes
    for stream_name, process in list(stream_processes.items()):
        try:
            if process.poll() is None:  # Process is still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                logger.info(f"Stopped stream process: {stream_name}")
        except Exception as e:
            logger.error(f"Error stopping stream {stream_name}: {e}", exc_info=True)
    
    stream_processes.clear()


# FastAPI app with lifespan events
app = FastAPI(
    title="SmartEye Inference Server",
    description="Ultra-optimized real-time YOLOv8 inference server",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler to prevent server crashes
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch all unhandled exceptions to prevent server crashes"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)[:200]}"}
    )


class InferenceRequest(BaseModel):
    """Request model for inference"""
    stream_name: str = Field(..., description="Name of the video stream")
    frame_id: int = Field(..., description="Frame identifier")
    image_data: str = Field(..., description="Base64 encoded image data")
    timestamp: Optional[float] = Field(None, description="Client-side timestamp")


class Detection(BaseModel):
    """Detection result model"""
    label: str
    conf: float
    bbox: List[float]  # [x1, y1, x2, y2]


class InferenceResponse(BaseModel):
    """Response model for inference results - exact format as specified"""
    timestamp: float
    frame_id: int
    stream_name: str
    latency_ms: float
    detections: List[Detection]


class PerformanceMetrics(BaseModel):
    """Performance metrics model with stability"""
    total_frames: int
    avg_latency_ms: float
    avg_fps: float
    current_fps: float
    stability_score: float
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_available: bool
    active_streams: int
    stream_analytics: Optional[Dict[str, Dict]] = None


class YOLOv8InferenceEngine:
    """
    Ultra-optimized YOLOv8 inference engine with performance monitoring.
    Model is loaded once and reused throughout runtime.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 100
    ):
        """
        Initialize YOLOv8 inference engine.

        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)  # Track last 100 inferences
        self.frame_times = deque(maxlen=100)  # Track frame processing times
        self.total_frames_processed = 0
        self.start_time = time.time()
        
        # Stream tracking with individual analytics
        self.stream_metrics: Dict[str, Dict] = {}  # Per-stream metrics
        self.stream_latencies: Dict[str, deque] = {}  # Per-stream latencies
        self.stream_fps_history: Dict[str, deque] = {}  # Per-stream FPS
        self.stream_frame_times: Dict[str, deque] = {}  # Per-stream frame times
        self.active_streams = set()
        
        # Store latest annotated frames for each stream
        self.latest_annotated_frames: Dict[str, np.ndarray] = {}
        
        # Round robin CPU allocation
        self.cpu_allocation_index = 0
        self.cpu_cores_available = psutil.cpu_count(logical=False)
        self.stream_cpu_affinity: Dict[str, int] = {}  # Stream to CPU core mapping
        
        # Load model once
        logger.info(f"Loading YOLOv8 model from {model_path}...")
        self._load_model(device)
        logger.info("Model loaded successfully")
    
    def _load_model(self, device: Optional[str] = None):
        """Load YOLOv8 model with optimal settings"""
        try:
            self.model = YOLO(self.model_path)
            
            # Set device if specified
            if device:
                self.model.to(device)
            else:
                # Auto-detect best device
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.model.to('cuda')
                        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                    else:
                        logger.info("Using CPU")
                except ImportError:
                    logger.info("PyTorch not available, using default device")
            
            # Warm up model with dummy inference
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model.predict(
                dummy_image,
                verbose=False,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections
            )
            logger.info("Model warm-up completed")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def decode_image(self, image_data: str) -> np.ndarray:
        """
        Decode base64 encoded image to numpy array.
        
        Args:
            image_data: Base64 encoded image string
            
        Returns:
            Decoded image as numpy array (BGR format)
        """
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            return image
            
        except Exception as e:
            logger.error(f"Image decoding error: {e}")
            raise ValueError(f"Invalid image data: {e}")
    
    def predict(self, image: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Run inference on image and return detections.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (detections list, processing time in ms)
        """
        inference_start = time.time()
        
        try:
            # Run inference
            results = self.model.predict(
                image,
                verbose=False,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections
            )
            
            inference_time = (time.time() - inference_start) * 1000  # Convert to ms
            
            # Parse results
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        # Get box coordinates (x1, y1, x2, y2)
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())
                        label = self.model.names[cls]
                        
                        detections.append({
                            'label': label,
                            'conf': conf,
                            'bbox': box.tolist()
                        })
            
            # Update metrics
            self.inference_times.append(inference_time)
            self.total_frames_processed += 1
            
            return detections, inference_time
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate FPS
        if elapsed_time > 0:
            avg_fps = self.total_frames_processed / elapsed_time
        else:
            avg_fps = 0.0
        
        # Current FPS (last second)
        if len(self.frame_times) > 1:
            time_diff = self.frame_times[-1] - self.frame_times[0]
            if time_diff > 0:
                current_fps = (len(self.frame_times) - 1) / time_diff
            else:
                current_fps = 0.0
        else:
            current_fps = 0.0
        
        # Average latency
        avg_latency = np.mean(self.inference_times) if self.inference_times else 0.0
        
        # System resources
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.Process().memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        # GPU availability
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available()
        except ImportError:
            pass
        
        # Calculate stability (latency variance)
        stability_score = 100.0
        if len(self.inference_times) > 1:
            variance = np.var(list(self.inference_times))
            if avg_latency > 0:
                cv = (np.sqrt(variance) / avg_latency) * 100
                stability_score = max(0, 100 - cv)
        
        # Per-stream analytics summary
        stream_analytics_summary = {}
        for stream_name in self.active_streams:
            stream_analytics_summary[stream_name] = {
                'fps': self.get_stream_fps(stream_name),
                'latency_ms': np.mean(list(self.stream_latencies.get(stream_name, []))) if self.stream_latencies.get(stream_name) else 0.0,
                'stability': self._calculate_stability_score(stream_name),
                'cpu_core': self.stream_cpu_affinity.get(stream_name, -1)
            }
        
        return {
            'total_frames': self.total_frames_processed,
            'avg_latency_ms': round(avg_latency, 2),
            'avg_fps': round(avg_fps, 2),
            'current_fps': round(current_fps, 2),
            'stability_score': round(stability_score, 2),
            'cpu_usage_percent': round(cpu_usage, 2),
            'memory_usage_mb': round(memory_mb, 2),
            'gpu_available': gpu_available,
            'active_streams': len(self.active_streams),
            'stream_analytics': stream_analytics_summary
        }
    
    def update_stream_metrics(self, stream_name: str, latency: float):
        """Update metrics for specific stream with individual analytics"""
        current_time = time.time()
        
        # Initialize stream metrics if not exists
        if stream_name not in self.stream_metrics:
            self.stream_metrics[stream_name] = {
                'total_frames': 0,
                'start_time': current_time,
                'last_update': current_time
            }
            self.stream_latencies[stream_name] = deque(maxlen=100)
            self.stream_fps_history[stream_name] = deque(maxlen=100)
            self.stream_frame_times[stream_name] = deque(maxlen=100)
            
            # Round robin CPU allocation for new stream
            self._allocate_cpu_core(stream_name)
        
        # Update stream metrics
        self.stream_metrics[stream_name]['total_frames'] += 1
        self.stream_metrics[stream_name]['last_update'] = current_time
        self.stream_latencies[stream_name].append(latency)
        self.stream_frame_times[stream_name].append(current_time)
        self.active_streams.add(stream_name)
        
        # Calculate and store FPS
        fps = self._calculate_stream_fps(stream_name)
        if fps > 0:
            self.stream_fps_history[stream_name].append(fps)
    
    def _allocate_cpu_core(self, stream_name: str):
        """Allocate CPU core to stream using round robin algorithm"""
        if self.cpu_cores_available > 0:
            core = self.cpu_allocation_index % self.cpu_cores_available
            self.stream_cpu_affinity[stream_name] = core
            self.cpu_allocation_index += 1
            # Log CPU allocation (only metrics)
            if len(self.active_streams) % 10 == 0:  # Log every 10 streams
                logger.info(f"CPU Allocation - Stream: {stream_name}, Core: {core}, Active Streams: {len(self.active_streams)}")
    
    def _calculate_stream_fps(self, stream_name: str) -> float:
        """Calculate FPS for specific stream"""
        if stream_name not in self.stream_frame_times or len(self.stream_frame_times[stream_name]) < 2:
            return 0.0
        
        times = list(self.stream_frame_times[stream_name])
        time_diff = times[-1] - times[0] if len(times) > 1 else 0
        if time_diff > 0:
            return (len(times) - 1) / time_diff
        return 0.0
    
    def get_stream_fps(self, stream_name: str) -> float:
        """Get current FPS for specific stream"""
        if stream_name not in self.stream_fps_history or len(self.stream_fps_history[stream_name]) == 0:
            return 0.0
        return self.stream_fps_history[stream_name][-1]
    
    def get_stream_analytics(self, stream_name: str) -> Dict:
        """Get comprehensive analytics for a specific stream"""
        if stream_name not in self.stream_metrics:
            return {}
        
        metrics = self.stream_metrics[stream_name]
        latencies = list(self.stream_latencies.get(stream_name, []))
        fps_history = list(self.stream_fps_history.get(stream_name, []))
        
        # Calculate statistics
        avg_latency = np.mean(latencies) if latencies else 0.0
        min_latency = np.min(latencies) if latencies else 0.0
        max_latency = np.max(latencies) if latencies else 0.0
        
        current_fps = self.get_stream_fps(stream_name)
        avg_fps = np.mean(fps_history) if fps_history else 0.0
        
        # Calculate total time and throughput
        total_time = metrics['last_update'] - metrics['start_time']
        throughput = metrics['total_frames'] / total_time if total_time > 0 else 0.0
        
        return {
            'stream_name': stream_name,
            'total_frames': metrics['total_frames'],
            'current_fps': round(current_fps, 2),
            'average_fps': round(avg_fps, 2),
            'throughput_fps': round(throughput, 2),
            'average_latency_ms': round(avg_latency, 2),
            'min_latency_ms': round(min_latency, 2),
            'max_latency_ms': round(max_latency, 2),
            'cpu_core': self.stream_cpu_affinity.get(stream_name, -1),
            'stability_score': self._calculate_stability_score(stream_name)
        }
    
    def _calculate_stability_score(self, stream_name: str) -> float:
        """Calculate stability score based on latency variance"""
        if stream_name not in self.stream_latencies or len(self.stream_latencies[stream_name]) < 2:
            return 100.0
        
        latencies = list(self.stream_latencies[stream_name])
        if len(latencies) < 2:
            return 100.0
        
        # Lower variance = higher stability
        variance = np.var(latencies)
        mean_latency = np.mean(latencies)
        
        if mean_latency == 0:
            return 100.0
        
        # Stability score: 100 - (coefficient of variation * 100)
        cv = (np.sqrt(variance) / mean_latency) * 100
        stability = max(0, 100 - cv)
        
        return round(stability, 2)
    
    def _annotate_frame(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Annotate frame with bounding boxes and labels"""
        annotated = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            label = det['label']
            conf = det['conf']
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            label_text = f"{label} {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                annotated,
                label_text,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return annotated


@app.post("/inference", response_model=InferenceResponse)
async def process_inference(request: InferenceRequest):
    """
    Process inference request for a single frame.
    
    Args:
        request: Inference request with base64 encoded image
        
    Returns:
        Inference results with detections and performance metrics
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # Record start time for latency calculation
        request_start = time.time()
        client_timestamp = request.timestamp or request_start
        
        # Decode image
        image = inference_engine.decode_image(request.image_data)
        
        # Run inference
        detections, processing_time = inference_engine.predict(image)
        
        # Calculate total latency
        total_latency = (time.time() - request_start) * 1000  # Convert to ms
        
        # Update stream metrics
        inference_engine.update_stream_metrics(request.stream_name, total_latency)
        inference_engine.frame_times.append(time.time())
        
        # Get stream FPS
        stream_fps = inference_engine.get_stream_fps(request.stream_name)
        
        # Format detections
        detection_models = [
            Detection(**det) for det in detections
        ]
        
        # Create annotated frame for frontend display
        annotated_frame = inference_engine._annotate_frame(image.copy(), detections)
        inference_engine.latest_annotated_frames[request.stream_name] = annotated_frame
        
        # Create response (exact format as specified)
        response = InferenceResponse(
            timestamp=client_timestamp,
            frame_id=request.frame_id,
            stream_name=request.stream_name,
            latency_ms=total_latency,
            detections=detection_models
        )
        
        # Log metrics only (latency, FPS, stability) - not every request
        if request.frame_id % 30 == 0:  # Log every 30 frames
            stream_fps = inference_engine.get_stream_fps(request.stream_name)
            stream_analytics = inference_engine.get_stream_analytics(request.stream_name)
            logger.info(
                f"Metrics - Stream: {request.stream_name}, "
                f"Frame: {request.frame_id}, "
                f"Latency: {total_latency:.2f}ms, "
                f"FPS: {stream_fps:.2f}, "
                f"Stability: {stream_analytics.get('stability_score', 0):.2f}"
            )
        
        return response
        
    except ValueError as e:
        logger.error(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.post("/inference/batch")
async def process_batch_inference(requests: List[InferenceRequest]):
    """
    Process multiple inference requests in batch.
    Optimized for throughput with parallel processing.
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        results = []
        batch_start = time.time()
        
        # Process requests (can be parallelized further if needed)
        for request in requests:
            try:
                image = inference_engine.decode_image(request.image_data)
                detections, processing_time = inference_engine.predict(image)
                
                total_latency = (time.time() - batch_start) * 1000
                stream_fps = inference_engine.get_stream_fps(request.stream_name)
                
                detection_models = [Detection(**det) for det in detections]
                
                results.append(InferenceResponse(
                    timestamp=request.timestamp or time.time(),
                    frame_id=request.frame_id,
                    stream_name=request.stream_name,
                    latency_ms=total_latency,
                    detections=detection_models
                ))
            except Exception as e:
                logger.error(f"Error processing request {request.frame_id}: {e}")
                continue
        
        return results
        
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=PerformanceMetrics)
async def get_metrics():
    """Get current performance metrics"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    metrics = inference_engine.get_performance_metrics()
    return PerformanceMetrics(**metrics)


@app.get("/streams/{stream_name}/analytics")
async def get_stream_analytics(stream_name: str):
    """Get analytics for a specific stream"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    if stream_name not in inference_engine.stream_metrics:
        raise HTTPException(status_code=404, detail=f"Stream {stream_name} not found")
    
    analytics = inference_engine.get_stream_analytics(stream_name)
    return analytics


@app.get("/streams/analytics")
async def get_all_streams_analytics():
    """Get analytics for all active streams"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    # Clean up dead processes first
    dead_streams = []
    for stream_name, process in list(stream_processes.items()):
        if process.poll() is not None:
            dead_streams.append(stream_name)
    
    # Remove dead processes
    for stream_name in dead_streams:
        if stream_name in stream_processes:
            del stream_processes[stream_name]
        if stream_name in download_progress:
            del download_progress[stream_name]
    
    all_analytics = {}
    for stream_name in inference_engine.active_streams:
        all_analytics[stream_name] = inference_engine.get_stream_analytics(stream_name)
    
    return {
        "active_streams": len(inference_engine.active_streams),
        "streams": all_analytics
    }


@app.get("/streams/{stream_name}/frame")
async def get_annotated_frame(stream_name: str):
    """Get latest annotated frame for a stream as image"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        # Get latest annotated frame for this stream
        if stream_name in inference_engine.latest_annotated_frames:
            annotated_frame = inference_engine.latest_annotated_frames[stream_name]
            
            # Encode frame as JPEG with optimized quality for speed
            # Lower quality (75) for faster encoding/decoding while maintaining visual quality
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_bytes = buffer.tobytes()
            
            return Response(content=frame_bytes, media_type="image/jpeg")
        else:
            # Return placeholder if no frame available yet (even if stream not processing)
            # This allows frontend to show waiting state
            if PIL_AVAILABLE:
                img = Image.new('RGB', (640, 480), color='black')
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
                text = f"Stream: {stream_name}\nWaiting for frames...\n\nStart processing using:\npython client.py --streams <source> --names {stream_name}"
                draw.text((10, 10), text, fill='white', font=font)
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG')
                img_bytes.seek(0)
                return Response(content=img_bytes.read(), media_type="image/jpeg")
            else:
                # Fallback: create simple black image with OpenCV
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"Stream: {stream_name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(placeholder, "Waiting for frames...", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(placeholder, "Start processing to see video", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                _, buffer = cv2.imencode('.jpg', placeholder)
                return Response(content=buffer.tobytes(), media_type="image/jpeg")
    except Exception as e:
        logger.error(f"Error generating frame: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate frame: {e}")


@app.post("/streams/start")
async def start_stream(request: Dict):
    """
    Start processing a video stream from the frontend.
    Accepts stream configuration and starts client processing.
    Supports YouTube URLs by downloading them first.
    """
    try:
        stream_name = request.get('stream_name')
        source = request.get('source')
        source_type = request.get('source_type', 'auto')
        fps_limit = request.get('fps_limit', 60)  # Increased default FPS limit
        
        if not stream_name or not source:
            raise HTTPException(status_code=400, detail="stream_name and source are required")
        
        # Check if stream is already running
        if stream_name in stream_processes:
            process = stream_processes[stream_name]
            if process.poll() is None:  # Process is still running
                return {"status": "already_running", "stream_name": stream_name}
            else:
                # Process ended, remove it
                del stream_processes[stream_name]
        
        script_dir = Path(__file__).parent
        actual_source = source
        actual_source_type = source_type
        
        # Handle YouTube URLs
        if source_type == 'youtube' or ('youtube.com' in source or 'youtu.be' in source):
            try:
                logger.info(f"Downloading YouTube video: {source}")
                
                # Create downloads directory if it doesn't exist
                downloads_dir = script_dir / "downloads"
                downloads_dir.mkdir(exist_ok=True)
                
                # Generate output filename based on stream name
                output_filename = f"{stream_name}_{int(time.time())}.mp4"
                output_path = downloads_dir / output_filename
                
                # Initialize download progress tracking
                download_progress[stream_name] = {
                    "status": "downloading",
                    "progress": 0.0,
                    "message": "Starting download...",
                    "start_time": time.time()
                }
                
                # Use yt-dlp to download video with progress
                python_exe = sys.executable
                yt_dlp_cmd = [
                    python_exe, "-m", "yt_dlp",
                    "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                    "-o", str(output_path),
                    "--no-playlist",
                    "--newline",  # Print progress on new lines
                    source
                ]
                
                # Run yt-dlp to download video with progress tracking
                download_progress[stream_name]["message"] = "Connecting to YouTube..."
                download_progress[stream_name]["progress"] = 5.0
                
                download_process = subprocess.Popen(
                    yt_dlp_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=str(script_dir)
                )
                
                # Monitor progress in background thread
                def monitor_progress():
                    try:
                        for line in iter(download_process.stdout.readline, ''):
                            if not line:
                                break
                            line = line.strip()
                            if not line:
                                continue
                            
                            # Parse progress from yt-dlp output
                            if "%" in line and "download" in line.lower():
                                try:
                                    # Extract percentage (e.g., "[download] 45.2% of 10.5MiB")
                                    percent_match = re.search(r'(\d+\.?\d*)%', line)
                                    if percent_match:
                                        progress = float(percent_match.group(1))
                                        download_progress[stream_name]["progress"] = min(progress, 95.0)
                                        download_progress[stream_name]["message"] = f"Downloading... {progress:.1f}%"
                                except:
                                    pass
                            elif "Downloading" in line or "Extracting" in line or "Merging" in line:
                                download_progress[stream_name]["message"] = line[:60]  # Truncate long messages
                        
                        download_process.wait()
                        
                        if download_process.returncode == 0:
                            download_progress[stream_name]["progress"] = 100.0
                            download_progress[stream_name]["message"] = "Download complete!"
                            download_progress[stream_name]["status"] = "completed"
                        else:
                            download_progress[stream_name]["status"] = "failed"
                            download_progress[stream_name]["message"] = "Download failed"
                    except Exception as e:
                        download_progress[stream_name]["status"] = "failed"
                        download_progress[stream_name]["message"] = f"Error: {str(e)[:50]}"
                
                # Start progress monitoring in background
                progress_thread = threading.Thread(target=monitor_progress, daemon=True)
                progress_thread.start()
                
                # Wait for download to complete (with timeout)
                try:
                    download_process.wait(timeout=300)  # 5 minute timeout
                except subprocess.TimeoutExpired:
                    download_process.kill()
                    download_process.wait()
                    download_progress[stream_name]["status"] = "failed"
                    download_progress[stream_name]["message"] = "Download timed out"
                    raise subprocess.TimeoutExpired(yt_dlp_cmd, 300)
                
                if download_process.returncode != 0:
                    error_msg = download_process.stderr or download_process.stdout or "Unknown error"
                    logger.error(f"YouTube download failed: {error_msg}")
                    download_progress[stream_name]["status"] = "failed"
                    download_progress[stream_name]["message"] = "Download failed"
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download YouTube video: {error_msg[:200]}"
                    )
                
                # Check if file was downloaded
                if not output_path.exists():
                    # yt-dlp might have added extension, try to find the file
                    possible_files = list(downloads_dir.glob(f"{stream_name}_*.mp4"))
                    if possible_files:
                        output_path = max(possible_files, key=lambda p: p.stat().st_mtime)
                    else:
                        download_progress[stream_name]["status"] = "failed"
                        download_progress[stream_name]["message"] = "File not found"
                        raise HTTPException(
                            status_code=400,
                            detail="Video downloaded but file not found"
                        )
                
                actual_source = str(output_path)
                actual_source_type = 'file'
                download_progress[stream_name]["progress"] = 100.0
                download_progress[stream_name]["status"] = "completed"
                download_progress[stream_name]["message"] = "Download complete!"
                logger.info(f"YouTube video downloaded to: {actual_source}")
                
            except subprocess.TimeoutExpired as e:
                logger.error(f"YouTube download timed out: {e}")
                if stream_name in download_progress:
                    download_progress[stream_name]["status"] = "failed"
                    download_progress[stream_name]["message"] = "Download timed out"
                raise HTTPException(
                    status_code=408,
                    detail="YouTube download timed out. Please try a shorter video or check your connection."
                )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error downloading YouTube video: {e}", exc_info=True)
                error_msg = str(e)
                # Truncate long error messages
                if len(error_msg) > 300:
                    error_msg = error_msg[:300] + "..."
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to download YouTube video: {error_msg}"
                )
        
        # Build client.py command
        client_script = script_dir / "client.py"
        
        # Use the same Python interpreter that's running the server
        python_exe = sys.executable
        
        cmd = [
            python_exe,
            str(client_script),
            "--streams", actual_source,
            "--names", stream_name,
            "--fps-limit", str(fps_limit),
            "--types", actual_source_type
        ]
        
        # Start process in background
        try:
            # Create log directory for stream logs
            logs_dir = script_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
            
            # Redirect stdout/stderr to log files to prevent blocking
            stdout_file = logs_dir / f"{stream_name}_stdout.log"
            stderr_file = logs_dir / f"{stream_name}_stderr.log"
            
            # Open files in append mode - subprocess will handle file closing
            # Use 'a' mode so we can read immediately if needed, but subprocess writes won't block
            stdout_f = open(stdout_file, 'ab', buffering=0)  # Unbuffered binary
            stderr_f = open(stderr_file, 'ab', buffering=0)  # Unbuffered binary
            
            process = subprocess.Popen(
                cmd,
                stdout=stdout_f,
                stderr=stderr_f,
                cwd=str(script_dir),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                env=os.environ.copy()  # Pass environment variables
            )
            
            # Close file handles - subprocess has its own references
            # The OS will keep files open until process terminates
            stdout_f.close()
            stderr_f.close()
            
            # Give process a moment to start
            time.sleep(0.5)
            
            # Check if process started successfully
            if process.poll() is not None:
                # Process already exited - read error from log file
                try:
                    with open(stderr_file, 'rb') as f:
                        error_output = f.read().decode('utf-8', errors='ignore')
                    if not error_output:
                        with open(stdout_file, 'rb') as f:
                            error_output = f.read().decode('utf-8', errors='ignore')
                    error_msg = error_output[:500] if error_output else "Process exited immediately with no error output"
                except Exception:
                    error_msg = "Process exited immediately"
                logger.error(f"Process exited immediately: {error_msg}")
                raise Exception(f"Process exited immediately: {error_msg}")
            
            stream_processes[stream_name] = process
            
            logger.info(f"Started stream processing: {stream_name} from {source} (PID: {process.pid})")
            
            return {
                "status": "started",
                "stream_name": stream_name,
                "pid": process.pid
            }
        except subprocess.SubprocessError as e:
            logger.error(f"Subprocess error starting stream: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to start stream process: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting stream: {e}", exc_info=True)
        error_msg = str(e)
        # Truncate long error messages
        if len(error_msg) > 300:
            error_msg = error_msg[:300] + "..."
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {error_msg}")


@app.post("/streams/stop")
async def stop_stream(request: Dict = Body(...)):
    """Stop processing a video stream"""
    try:
        stream_name = request.get('stream_name')
        
        if not stream_name:
            raise HTTPException(status_code=400, detail="stream_name is required")
        
        if stream_name not in stream_processes:
            return {"status": "not_running", "stream_name": stream_name}
        
        process = stream_processes[stream_name]
        
        # Terminate process
        try:
            process.terminate()
            # Wait a bit for graceful shutdown
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                process.kill()
                process.wait()
            
            del stream_processes[stream_name]
            
            logger.info(f"Stopped stream processing: {stream_name}")
            
            return {"status": "stopped", "stream_name": stream_name}
            
        except Exception as e:
            logger.error(f"Error stopping stream {stream_name}: {e}")
            # Remove from dict even if termination failed
            if stream_name in stream_processes:
                del stream_processes[stream_name]
            raise HTTPException(status_code=500, detail=f"Failed to stop stream: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in stop_stream: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop stream: {str(e)}")


@app.get("/streams/status")
async def get_stream_status():
    """Get status of all running streams"""
    status = {}
    # Clean up dead processes
    dead_streams = []
    for stream_name, process in stream_processes.items():
        if process.poll() is not None:
            # Process has ended
            dead_streams.append(stream_name)
            status[stream_name] = {
                "running": False,
                "pid": process.pid,
                "return_code": process.returncode
            }
        else:
            status[stream_name] = {
                "running": True,
                "pid": process.pid,
                "return_code": None
            }
    
    # Remove dead processes
    for stream_name in dead_streams:
        del stream_processes[stream_name]
        # Also clean up download progress
        if stream_name in download_progress:
            del download_progress[stream_name]
    
    return {"streams": status}


@app.post("/streams/stop-all")
async def stop_all_streams():
    """Stop all running streams and clear state"""
    stopped_streams = []
    failed_streams = []
    
    # Stop all running processes
    for stream_name, process in list(stream_processes.items()):
        try:
            if process.poll() is None:  # Process is still running
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                stopped_streams.append(stream_name)
            else:
                # Process already dead, just remove it
                stopped_streams.append(stream_name)
        except Exception as e:
            logger.error(f"Error stopping stream {stream_name}: {e}")
            failed_streams.append(stream_name)
        finally:
            # Remove from dict regardless
            if stream_name in stream_processes:
                del stream_processes[stream_name]
            # Clean up download progress
            if stream_name in download_progress:
                del download_progress[stream_name]
    
    # Clear all state
    stream_processes.clear()
    download_progress.clear()
    
    logger.info(f"Stopped all streams. Stopped: {len(stopped_streams)}, Failed: {len(failed_streams)}")
    
    return {
        "status": "stopped_all",
        "stopped_streams": stopped_streams,
        "failed_streams": failed_streams,
        "total_stopped": len(stopped_streams)
    }


@app.post("/streams/clear")
async def clear_all_state():
    """Clear all stream state (processes, download progress, etc.)"""
    # Stop all running processes first
    for stream_name, process in list(stream_processes.items()):
        try:
            if process.poll() is None:  # Process is still running
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        except Exception as e:
            logger.warning(f"Error stopping stream {stream_name} during clear: {e}")
    
    # Clear all dictionaries
    stream_processes.clear()
    download_progress.clear()
    
    logger.info("Cleared all stream state")
    
    return {
        "status": "cleared",
        "message": "All stream state has been cleared"
    }


@app.get("/streams/{stream_name}/download-progress")
async def get_download_progress(stream_name: str):
    """Get YouTube download progress for a stream"""
    if stream_name not in download_progress:
        return {"status": "not_found", "progress": 0.0, "message": "No download in progress"}
    
    progress_data = download_progress[stream_name]
    return {
        "status": progress_data.get("status", "unknown"),
        "progress": progress_data.get("progress", 0.0),
        "message": progress_data.get("message", ""),
        "elapsed_time": time.time() - progress_data.get("start_time", time.time())
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if inference_engine is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Inference engine not initialized"}
        )
    
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "SmartEye Inference Server",
        "version": "1.0.0",
        "endpoints": {
            "inference": "/inference",
            "batch_inference": "/inference/batch",
            "metrics": "/metrics",
            "health": "/health",
            "streams_start": "/streams/start",
            "streams_stop": "/streams/stop",
            "streams_stop_all": "/streams/stop-all",
            "streams_clear": "/streams/clear",
            "streams_status": "/streams/status",
            "streams_analytics": "/streams/analytics",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    try:
        logger.info("Starting SmartEye Inference Server...")
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=False,  # Disable access logs - only metrics will be logged
            timeout_keep_alive=30,  # Keep connections alive
            limit_concurrency=100,  # Limit concurrent connections
            limit_max_requests=1000000  # Restart worker after N requests (prevents memory leaks) - increased for long-running video processing
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server crashed: {e}", exc_info=True)
        raise

