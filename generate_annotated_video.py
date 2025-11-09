"""
Generate annotated video with bounding boxes from JSONL results.
Overlays detection results on the original video.
"""

import argparse
import json
import cv2
import os
from pathlib import Path
from typing import Dict, List, Optional

def draw_detections(frame, detections, confidence_threshold=0.25):
    """
    Draw bounding boxes and labels on frame.
    
    Args:
        frame: OpenCV image frame
        detections: List of detection dictionaries
        confidence_threshold: Minimum confidence to display
        
    Returns:
        Annotated frame
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
        
        # Draw bounding box
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
        
        # Draw background rectangle for text
        cv2.rectangle(
            annotated_frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw text
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


def create_annotated_video(
    video_path: str,
    jsonl_path: str,
    output_path: str,
    confidence_threshold: float = 0.25,
    fps: Optional[float] = None
):
    """
    Create annotated video from original video and JSONL results.
    
    Args:
        video_path: Path to original video file
        jsonl_path: Path to JSONL results file
        output_path: Path to output annotated video
        confidence_threshold: Minimum confidence to display detections
        fps: Output video FPS (default: same as input)
    """
    # Load results
    results = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                frame_id = result.get('frame_id', 0)
                results[frame_id] = result
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use provided FPS or original FPS
    output_fps = fps if fps else original_fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Could not create output video: {output_path}")
    
    print(f"Processing video: {video_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {output_fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Results loaded: {len(results)} frames")
    
    frame_id = 0
    processed_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_id += 1
        
        # Get detections for this frame
        if frame_id in results:
            detections = results[frame_id].get('detections', [])
            annotated_frame = draw_detections(frame, detections, confidence_threshold)
            processed_frames += 1
        else:
            # No detections for this frame, use original
            annotated_frame = frame
        
        # Write frame
        out.write(annotated_frame)
        
        # Progress indicator
        if frame_id % 30 == 0:
            progress = (frame_id / total_frames * 100) if total_frames > 0 else 0
            print(f"  Progress: {progress:.1f}% ({frame_id}/{total_frames} frames)")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"\nAnnotated video created: {output_path}")
    print(f"  Processed frames: {processed_frames}")
    print(f"  Total frames: {frame_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotated video with bounding boxes from JSONL results"
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Input video file path'
    )
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='Input JSONL results file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output video file path (default: input_video_annotated.mp4)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='Minimum confidence threshold for detections (default: 0.25)'
    )
    parser.add_argument(
        '--fps',
        type=float,
        help='Output video FPS (default: same as input)'
    )
    
    args = parser.parse_args()
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        video_path = Path(args.video)
        output_file = str(video_path.parent / f"{video_path.stem}_annotated.mp4")
    
    # Create annotated video
    try:
        create_annotated_video(
            args.video,
            args.jsonl,
            output_file,
            args.confidence,
            args.fps
        )
    except Exception as e:
        print(f"Error: {e}")
        import sys
        sys.exit(1)
    
    import sys
    sys.exit(0)


if __name__ == "__main__":
    exit(main())

