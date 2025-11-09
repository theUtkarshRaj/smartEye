"""
SmartEye - Ultra-Optimized Real-Time Vision Streaming System (YOLOv8)
Client Module: Ingests video streams (RTSP/webcam/video files), sends frames
for inference, retrieves results in real-time, and saves them to JSON files.
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import httpx
import numpy as np
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('client.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StreamConfig:
    """Configuration for a video stream"""
    
    def __init__(
        self,
        stream_name: str,
        source: str,
        source_type: str = "auto",  # "rtsp", "webcam", "file", "auto"
        fps_limit: Optional[int] = None,
        frame_skip: int = 0,  # Skip every N frames
        output_dir: str = "results"
    ):
        self.stream_name = stream_name
        self.source = source
        self.source_type = source_type
        self.fps_limit = fps_limit
        self.frame_skip = frame_skip
        self.output_dir = output_dir
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Output file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = os.path.join(
            output_dir,
            f"{stream_name}_{timestamp}.jsonl"
        )
        # Output video file path (annotated video with green boxes)
        # Using .avi extension with XVID codec for maximum compatibility
        self.output_video_file = os.path.join(
            output_dir,
            f"{stream_name}_{timestamp}_annotated.avi"
        )


def draw_detections(frame, detections, confidence_threshold=0.25):
    """
    Draw green bounding boxes and labels on frame.
    
    Args:
        frame: OpenCV image frame (BGR format)
        detections: List of detection dictionaries with 'bbox', 'label', 'conf'
        confidence_threshold: Minimum confidence to display
        
    Returns:
        Annotated frame with green bounding boxes
    """
    annotated_frame = frame.copy()
    
    for detection in detections:
        if detection.get('conf', 0) < confidence_threshold:
            continue
        
        # Get bounding box coordinates
        bbox = detection.get('bbox', [])
        if len(bbox) != 4:
            continue
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get label and confidence
        label = detection.get('label', 'unknown')
        conf = detection.get('conf', 0.0)
        
        # Draw green bounding box
        color = (0, 255, 0)  # Green (BGR format)
        thickness = 2
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label with confidence
        label_text = f"{label}: {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, font_thickness
        )
        
        # Draw green background rectangle for text
        cv2.rectangle(
            annotated_frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw black text
        cv2.putText(
            annotated_frame,
            label_text,
            (x1, y1 - 5),
            font,
            font_scale,
            (0, 0, 0),  # Black text
            font_thickness
        )
    
    return annotated_frame


class StreamProcessor:
    """
    Processes video streams and sends frames to inference server.
    Handles multiple concurrent streams with error recovery.
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.server_url = server_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Performance tracking
        self.stream_stats: Dict[str, Dict] = {}
        
        # HTTP client with connection pooling and better timeout settings
        timeout_config = httpx.Timeout(
            timeout=timeout,
            connect=10.0,  # Connection timeout
            read=timeout,  # Read timeout
            write=10.0,    # Write timeout
            pool=5.0       # Pool timeout
        )
        self.client = httpx.AsyncClient(
            timeout=timeout_config,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
            http2=False  # Disable HTTP/2 to avoid connection issues
        )
    
    async def encode_image(self, image: np.ndarray) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image: Image as numpy array (BGR format)
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Encode image to JPEG (more efficient than PNG)
            success, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                raise ValueError("Failed to encode image")
            
            # Convert to base64
            image_bytes = buffer.tobytes()
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            return image_b64
            
        except Exception as e:
            logger.error(f"Image encoding error: {e}")
            raise
    
    async def send_inference_request(
        self,
        stream_name: str,
        frame_id: int,
        image: np.ndarray,
        timestamp: float
    ) -> Optional[Dict]:
        """
        Send inference request to server with retry logic.
        
        Args:
            stream_name: Name of the stream
            frame_id: Frame identifier
            image: Image as numpy array
            timestamp: Client-side timestamp
            
        Returns:
            Inference result dictionary or None if failed
        """
        # Encode image
        image_data = await self.encode_image(image)
        
        # Prepare request
        request_data = {
            "stream_name": stream_name,
            "frame_id": frame_id,
            "image_data": image_data,
            "timestamp": timestamp
        }
        
        # Retry logic
        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    f"{self.server_url}/inference",
                    json=request_data
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Update statistics
                self._update_stats(stream_name, result)
                
                return result
                
            except httpx.TimeoutException:
                logger.warning(
                    f"Request timeout for {stream_name} frame {frame_id} "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to get inference after {self.max_retries} attempts")
                    
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as e:
                # Connection errors - server might be down or unreachable
                error_msg = str(e)
                if "All connection attempts failed" in error_msg or "Connection refused" in error_msg:
                    logger.warning(
                        f"Connection error for {stream_name} frame {frame_id}: Server may be down or restarting "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                else:
                    logger.warning(
                        f"Connection error for {stream_name} frame {frame_id}: {e} "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                if attempt < self.max_retries - 1:
                    # Longer delay for connection errors - exponential backoff
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    await asyncio.sleep(min(delay, 10.0))  # Cap at 10 seconds
                else:
                    logger.error(f"All connection attempts failed for {stream_name} frame {frame_id}")
                    return None
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error for {stream_name} frame {frame_id}: {e}")
                if e.response.status_code >= 500:  # Server error, retry
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                    else:
                        return None
                else:  # Client error, don't retry
                    return None
                    
            except Exception as e:
                logger.error(f"Unexpected error for {stream_name} frame {frame_id}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    return None
        
        return None
    
    def _update_stats(self, stream_name: str, result: Dict):
        """Update statistics for stream"""
        if stream_name not in self.stream_stats:
            self.stream_stats[stream_name] = {
                'total_frames': 0,
                'successful_frames': 0,
                'failed_frames': 0,
                'latencies': deque(maxlen=100),
                'fps_history': deque(maxlen=100),
                'last_update': time.time()
            }
        
        stats = self.stream_stats[stream_name]
        stats['total_frames'] += 1
        
        if result:
            stats['successful_frames'] += 1
            stats['latencies'].append(result.get('latency_ms', 0))
            if result.get('fps'):
                stats['fps_history'].append(result['fps'])
        else:
            stats['failed_frames'] += 1
    
    def get_stats(self, stream_name: str) -> Dict:
        """Get current statistics for stream"""
        if stream_name not in self.stream_stats:
            return {}
        
        stats = self.stream_stats[stream_name]
        latencies = list(stats['latencies'])
        
        return {
            'total_frames': stats['total_frames'],
            'successful_frames': stats['successful_frames'],
            'failed_frames': stats['failed_frames'],
            'success_rate': stats['successful_frames'] / stats['total_frames'] if stats['total_frames'] > 0 else 0,
            'avg_latency_ms': np.mean(latencies) if latencies else 0,
            'min_latency_ms': np.min(latencies) if latencies else 0,
            'max_latency_ms': np.max(latencies) if latencies else 0,
            'current_fps': np.mean(list(stats['fps_history'])) if stats['fps_history'] else 0
        }
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


class VideoStream:
    """
    Handles video stream capture from various sources.
    Supports RTSP, webcam, and video files.
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.last_frame_time = 0
        # FPS limiting disabled for maximum throughput
        # Removed frame_interval calculation - no FPS limiting
    
    def open(self) -> bool:
        """Open video stream"""
        try:
            source = self.config.source
            
            # Auto-detect source type
            if self.config.source_type == "auto":
                if source.isdigit():
                    source_type = "webcam"
                    source = int(source)
                elif source.startswith("rtsp://") or source.startswith("http://"):
                    source_type = "rtsp"
                else:
                    source_type = "file"
            else:
                source_type = self.config.source_type
                if source_type == "webcam" and source.isdigit():
                    source = int(source)
            
            # Open capture
            if source_type == "webcam":
                self.cap = cv2.VideoCapture(source)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            elif source_type == "rtsp":
                self.cap = cv2.VideoCapture(
                    source,
                    cv2.CAP_FFMPEG
                )
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            else:  # file
                self.cap = cv2.VideoCapture(source)
                # For video files, optimize for maximum throughput
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for responsiveness
            
            if not self.cap.isOpened():
                raise ValueError(f"Failed to open video source: {self.config.source}")
            
            logger.info(f"Opened {source_type} stream: {self.config.source}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening stream {self.config.stream_name}: {e}")
            return False
    
    def read_frame(self) -> Optional[tuple]:
        """
        Read next frame from stream.
        
        Returns:
            Tuple of (frame, timestamp) or None if failed
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        # Frame skipping (disabled for maximum FPS)
        # if self.config.frame_skip > 0:
        #     for _ in range(self.config.frame_skip):
        #         self.cap.read()
        
        # FPS limiting disabled for maximum throughput
        # Read frame immediately without delay
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            return None
        
        self.frame_count += 1
        self.last_frame_time = time.time()  # Update last frame time for tracking
        
        timestamp = time.time()
        return frame, timestamp
    
    def get_properties(self) -> Dict:
        """Get stream properties"""
        if self.cap is None:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': self.frame_count
        }
    
    def close(self):
        """Close video stream"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


async def process_stream(
    config: StreamConfig,
    processor: StreamProcessor
):
    """
    Process a single video stream.
    Captures frames, sends for inference, and saves results.
    """
    stream = VideoStream(config)
    
    if not stream.open():
        logger.error(f"Failed to open stream: {config.stream_name}")
        return
    
    logger.info(f"Processing stream: {config.stream_name}")
    
    # Wait for server to be ready (check health endpoint)
    logger.info("Checking server connection...")
    server_ready = False
    for check_attempt in range(10):  # Try 10 times
        try:
            import httpx
            check_client = httpx.Client(timeout=5.0)
            health_response = check_client.get(f"{processor.server_url}/health", timeout=5.0)
            if health_response.status_code == 200:
                server_ready = True
                logger.info("Server is ready")
                break
        except Exception:
            if check_attempt < 9:
                logger.warning(f"Server not ready, waiting... (attempt {check_attempt + 1}/10)")
                await asyncio.sleep(2)
            else:
                logger.error("Server is not responding. Please make sure the server is running.")
                return
    
    # Open output file
    output_file = open(config.output_file, 'w')
    
    # Initialize video writer for annotated video with green boxes
    video_writer = None
    video_fps = 30.0  # Default FPS
    video_width = None
    video_height = None
    
    try:
        frame_id = 0
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while True:
            # Read frame
            frame_data = stream.read_frame()
            
            if frame_data is None:
                # Only sleep if we're at end of video or stream closed
                # For live streams, continue immediately to check for new frames
                if hasattr(stream, 'cap') and stream.cap is not None:
                    # Check if stream is still open
                    if not stream.cap.isOpened():
                        logger.warning(f"{config.stream_name}: Stream closed")
                        break
                # Minimal delay only for async yield - removed to maximize FPS
                await asyncio.sleep(0.0001)  # Ultra-minimal delay for async yield
                continue
            
            frame, timestamp = frame_data
            frame_id += 1
            
            # Send for inference
            result = await processor.send_inference_request(
                config.stream_name,
                frame_id,
                frame,
                timestamp
            )
            
            if result:
                # Format result to match exact specification
                formatted_result = {
                    "timestamp": result.get('timestamp', timestamp),
                    "frame_id": result.get('frame_id', frame_id),
                    "stream_name": result.get('stream_name', config.stream_name),
                    "latency_ms": result.get('latency_ms', 0),
                    "detections": [
                        {
                            "label": det.get('label', ''),
                            "conf": det.get('conf', 0),
                            "bbox": det.get('bbox', [])
                        }
                        for det in result.get('detections', [])
                    ]
                }
                
                # Save result to JSONL file
                output_file.write(json.dumps(formatted_result) + '\n')
                output_file.flush()  # Ensure data is written
                
                # Initialize video writer on first frame
                if video_writer is None:
                    height, width = frame.shape[:2]
                    video_width = width
                    video_height = height
                    # Get FPS from stream if available
                    if hasattr(stream, 'cap') and stream.cap is not None:
                        video_fps = stream.cap.get(cv2.CAP_PROP_FPS) or 30.0
                    
                    # Create video writer with compatible codec
                    # Use XVID with AVI for maximum compatibility (works on all systems)
                    # Change extension to .avi for XVID
                    if config.output_video_file.endswith('.mp4'):
                        config.output_video_file = config.output_video_file.replace('.mp4', '.avi')
                    
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(
                        config.output_video_file,
                        fourcc,
                        video_fps,
                        (video_width, video_height)
                    )
                    
                    if video_writer.isOpened():
                        logger.info(f"Started saving annotated video: {config.output_video_file}")
                    else:
                        # Fallback to MJPG if XVID fails
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        video_writer = cv2.VideoWriter(
                            config.output_video_file,
                            fourcc,
                            video_fps,
                            (video_width, video_height)
                        )
                        if video_writer.isOpened():
                            logger.info(f"Started saving annotated video (MJPG): {config.output_video_file}")
                        else:
                            logger.error(f"Failed to create video writer: {config.output_video_file}")
                            video_writer = None
                
                # Draw green bounding boxes on frame
                detections = result.get('detections', [])
                annotated_frame = draw_detections(frame, detections, confidence_threshold=0.25)
                
                # Write annotated frame to video (always save, even if no detections)
                if video_writer is not None and video_writer.isOpened():
                    video_writer.write(annotated_frame)
            else:
                # No result from inference, but still write frame to video (without annotations)
                # Initialize video writer if not already done
                if video_writer is None:
                    height, width = frame.shape[:2]
                    video_width = width
                    video_height = height
                    if hasattr(stream, 'cap') and stream.cap is not None:
                        video_fps = stream.cap.get(cv2.CAP_PROP_FPS) or 30.0
                    # Create video writer with compatible codec
                    # Use XVID with AVI for maximum compatibility (works on all systems)
                    # Change extension to .avi for XVID
                    if config.output_video_file.endswith('.mp4'):
                        config.output_video_file = config.output_video_file.replace('.mp4', '.avi')
                    
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(
                        config.output_video_file,
                        fourcc,
                        video_fps,
                        (video_width, video_height)
                    )
                    
                    if video_writer.isOpened():
                        logger.info(f"Started saving annotated video: {config.output_video_file}")
                    else:
                        # Fallback to MJPG if XVID fails
                        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                        video_writer = cv2.VideoWriter(
                            config.output_video_file,
                            fourcc,
                            video_fps,
                            (video_width, video_height)
                        )
                        if video_writer.isOpened():
                            logger.info(f"Started saving annotated video (MJPG): {config.output_video_file}")
                
                # Write original frame if no detections
                if video_writer is not None and video_writer.isOpened():
                    video_writer.write(frame)
                
                consecutive_failures += 1
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        f"{config.stream_name}: Too many consecutive failures, "
                        "attempting to reconnect..."
                    )
                    stream.close()
                    await asyncio.sleep(2)
                    if not stream.open():
                        logger.error(f"{config.stream_name}: Failed to reconnect")
                        break
                    consecutive_failures = 0
            
            # Log metrics only (latency, FPS, stability) - every 30 frames
            if frame_id % 30 == 0:
                stats = processor.get_stats(config.stream_name)
                logger.info(
                    f"Metrics - Stream: {config.stream_name}, "
                    f"Frame: {frame_id}, "
                    f"Latency: {stats.get('avg_latency_ms', 0):.2f}ms, "
                    f"FPS: {stats.get('current_fps', 0):.2f}, "
                    f"Success: {stats.get('success_rate', 0)*100:.1f}%"
                )
            
            # No delay - maximize throughput
            # Removed all delays to achieve maximum FPS
            # Only yield control minimally for async event loop
            await asyncio.sleep(0)  # Yield control without delay
    
    except KeyboardInterrupt:
        logger.info(f"Interrupted: {config.stream_name}")
    except Exception as e:
        logger.error(f"Error processing {config.stream_name}: {e}")
    finally:
        stream.close()
        output_file.close()
        
        # Release video writer properly to ensure file is finalized
        if video_writer is not None:
            if video_writer.isOpened():
                video_writer.release()
            # Give a moment for file system to finalize the video file
            time.sleep(0.2)
            logger.info(f"Annotated video saved: {config.output_video_file}")
        
        logger.info(f"Finished processing: {config.stream_name}")
        logger.info(f"Results saved to: {config.output_file}")


async def process_multiple_streams(configs: List[StreamConfig], processor: StreamProcessor):
    """Process multiple streams concurrently"""
    tasks = [
        process_stream(config, processor)
        for config in configs
    ]
    
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Interrupted: Stopping all streams")
    finally:
        await processor.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="SmartEye Client - Real-time video inference client"
    )
    parser.add_argument(
        '--server',
        type=str,
        default='http://localhost:8000',
        help='Inference server URL'
    )
    parser.add_argument(
        '--streams',
        nargs='+',
        required=True,
        help='Video stream sources (RTSP URLs, webcam indices, or file paths)'
    )
    parser.add_argument(
        '--names',
        nargs='+',
        help='Stream names (default: stream_0, stream_1, ...)'
    )
    parser.add_argument(
        '--types',
        nargs='+',
        choices=['rtsp', 'webcam', 'file', 'auto'],
        default=['auto'],
        help='Source types for each stream'
    )
    parser.add_argument(
        '--fps-limit',
        type=int,
        default=None,  # No limit by default for maximum FPS
        help='Maximum FPS to process (applies to all streams). Set to None or 0 for unlimited FPS (default: unlimited)'
    )
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=0,
        help='Skip every N frames (for performance)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for JSON results'
    )
    
    args = parser.parse_args()
    
    # Create stream configurations
    stream_configs = []
    stream_names = args.names if args.names else [
        f"stream_{i}" for i in range(len(args.streams))
    ]
    
    # Extend types list if needed
    types = args.types
    if len(types) == 1:
        types = types * len(args.streams)
    
    for i, (source, name) in enumerate(zip(args.streams, stream_names)):
        config = StreamConfig(
            stream_name=name,
            source=source,
            source_type=types[i] if i < len(types) else 'auto',
            fps_limit=args.fps_limit,
            frame_skip=args.frame_skip,
            output_dir=args.output_dir
        )
        stream_configs.append(config)
    
    # Create processor
    processor = StreamProcessor(server_url=args.server)
    
    # Process streams
    logger.info(f"Starting client with {len(stream_configs)} stream(s)")
    logger.info(f"Server URL: {args.server}")
    
    try:
        asyncio.run(process_multiple_streams(stream_configs, processor))
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    finally:
        logger.info("Client shutdown complete")


if __name__ == "__main__":
    main()

 