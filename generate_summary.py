"""
Generate summary JSON file from JSONL results.
Calculates total FPS, throughput, and other statistics.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

def calculate_summary(jsonl_file: str) -> Dict:
    """
    Calculate summary statistics from JSONL results file.
    
    Args:
        jsonl_file: Path to JSONL results file
        
    Returns:
        Dictionary with summary statistics
    """
    results = []
    detections_by_label = defaultdict(int)
    total_detections = 0
    latencies = []
    processing_times = []
    timestamps = []
    
    # Read JSONL file
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                results.append(result)
                
                # Collect timestamps
                if 'timestamp' in result:
                    timestamps.append(result['timestamp'])
                
                # Collect latencies
                if 'latency_ms' in result:
                    latencies.append(result['latency_ms'])
                
                # Collect processing times
                if 'processing_time_ms' in result:
                    processing_times.append(result['processing_time_ms'])
                
                # Count detections
                if 'detections' in result and result['detections']:
                    for detection in result['detections']:
                        total_detections += 1
                        if 'label' in detection:
                            detections_by_label[detection['label']] += 1
    
    if not results:
        return {
            "error": "No results found in file",
            "file": jsonl_file
        }
    
    # Calculate statistics
    total_frames = len(results)
    
    # Calculate FPS
    if len(timestamps) > 1:
        time_span = timestamps[-1] - timestamps[0]
        if time_span > 0:
            total_fps = total_frames / time_span
        else:
            total_fps = 0.0
    else:
        total_fps = 0.0
    
    # Calculate throughput (frames per second)
    throughput_fps = total_fps
    
    # Calculate average latency
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    
    # Calculate average processing time
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
    min_processing_time = min(processing_times) if processing_times else 0.0
    max_processing_time = max(processing_times) if processing_times else 0.0
    
    # Calculate detection statistics
    avg_detections_per_frame = total_detections / total_frames if total_frames > 0 else 0.0
    frames_with_detections = sum(1 for r in results if r.get('detections') and len(r['detections']) > 0)
    detection_rate = (frames_with_detections / total_frames * 100) if total_frames > 0 else 0.0
    
    # Get stream name
    stream_name = results[0].get('stream_name', 'unknown') if results else 'unknown'
    
    # Build summary
    summary = {
        "file_info": {
            "input_file": jsonl_file,
            "stream_name": stream_name,
            "total_frames": total_frames,
            "file_size_bytes": os.path.getsize(jsonl_file)
        },
        "performance": {
            "total_fps": round(total_fps, 2),
            "throughput_fps": round(throughput_fps, 2),
            "average_fps": round(throughput_fps, 2),  # Same as throughput
            "frames_processed": total_frames,
            "processing_duration_seconds": round(timestamps[-1] - timestamps[0], 2) if len(timestamps) > 1 else 0.0
        },
        "latency": {
            "average_ms": round(avg_latency, 2),
            "min_ms": round(min_latency, 2),
            "max_ms": round(max_latency, 2),
            "unit": "milliseconds"
        },
        "processing_time": {
            "average_ms": round(avg_processing_time, 2),
            "min_ms": round(min_processing_time, 2),
            "max_ms": round(max_processing_time, 2),
            "unit": "milliseconds"
        },
        "detections": {
            "total_detections": total_detections,
            "average_per_frame": round(avg_detections_per_frame, 2),
            "frames_with_detections": frames_with_detections,
            "detection_rate_percent": round(detection_rate, 2),
            "detections_by_label": dict(sorted(detections_by_label.items(), key=lambda x: x[1], reverse=True))
        },
        "throughput": {
            "frames_per_second": round(throughput_fps, 2),
            "total_frames": total_frames,
            "processing_rate": f"{round(throughput_fps, 2)} FPS"
        }
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary JSON from JSONL results"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path (default: input_file_summary.json)'
    )
    
    args = parser.parse_args()
    
    # Generate summary
    summary = calculate_summary(args.input)
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        input_path = Path(args.input)
        output_file = str(input_path.parent / f"{input_path.stem}_summary.json")
    
    # Write summary
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary generated: {output_file}")
    print(f"\nKey Statistics:")
    print(f"  Total FPS: {summary['performance']['total_fps']}")
    print(f"  Throughput: {summary['throughput']['frames_per_second']} FPS")
    print(f"  Total Frames: {summary['file_info']['total_frames']}")
    print(f"  Average Latency: {summary['latency']['average_ms']} ms")
    print(f"  Total Detections: {summary['detections']['total_detections']}")


if __name__ == "__main__":
    main()

