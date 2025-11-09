"""
Process JSONL results to generate summary and annotated video.
Combines summary generation and video annotation.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Process JSONL results to generate summary and annotated video"
    )
    parser.add_argument(
        '--jsonl',
        type=str,
        required=True,
        help='Input JSONL results file path'
    )
    parser.add_argument(
        '--video',
        type=str,
        help='Original video file path (for annotated video generation)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='Minimum confidence threshold for detections (default: 0.25)'
    )
    
    args = parser.parse_args()
    
    jsonl_path = Path(args.jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Processing Results")
    print("=" * 60)
    
    # Step 1: Generate summary
    print("\n1. Generating summary JSON...")
    summary_file = output_dir / f"{jsonl_path.stem}_summary.json"
    
    try:
        subprocess.run([
            'python', 'generate_summary.py',
            '--input', str(jsonl_path),
            '--output', str(summary_file)
        ], check=True)
        print(f"   ✓ Summary generated: {summary_file}")
    except subprocess.CalledProcessError as e:
        print(f"   ✗ Error generating summary: {e}")
        return 1
    
    # Step 2: Generate annotated video (if video provided)
    if args.video:
        print("\n2. Generating annotated video...")
        video_path = Path(args.video)
        annotated_video = output_dir / f"{video_path.stem}_annotated.mp4"
        
        try:
            subprocess.run([
                'python', 'generate_annotated_video.py',
                '--video', str(video_path),
                '--jsonl', str(jsonl_path),
                '--output', str(annotated_video),
                '--confidence', str(args.confidence)
            ], check=True)
            print(f"   ✓ Annotated video generated: {annotated_video}")
        except subprocess.CalledProcessError as e:
            print(f"   ✗ Error generating annotated video: {e}")
            return 1
    else:
        print("\n2. Skipping annotated video (no video file provided)")
        print("   Use --video to generate annotated video")
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    
    # Display summary
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print("\nSummary Statistics:")
        print(f"  Total FPS: {summary['performance']['total_fps']}")
        print(f"  Throughput: {summary['throughput']['frames_per_second']} FPS")
        print(f"  Total Frames: {summary['file_info']['total_frames']}")
        print(f"  Average Latency: {summary['latency']['average_ms']} ms")
        print(f"  Total Detections: {summary['detections']['total_detections']}")
    
    return 0


if __name__ == "__main__":
    exit(main())

