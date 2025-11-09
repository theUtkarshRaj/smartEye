"""
Simple test script to verify server connection and basic functionality.
Run this after starting the server to test the connection.
"""

import asyncio
import base64
import json
import time

import cv2
import httpx
import numpy as np


async def test_server_connection(server_url: str = "http://localhost:8000"):
    """Test server connection and basic inference"""
    
    print("Testing SmartEye Server Connection...")
    print(f"Server URL: {server_url}")
    print()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 1: Health Check
        print("Test 1: Health Check...")
        try:
            response = await client.get(f"{server_url}/health")
            response.raise_for_status()
            result = response.json()
            print(f"✓ Health check passed: {result}")
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return False
        print()
        
        # Test 2: Root Endpoint
        print("Test 2: Root Endpoint...")
        try:
            response = await client.get(f"{server_url}/")
            response.raise_for_status()
            result = response.json()
            print(f"✓ Root endpoint: {result.get('service', 'Unknown')}")
        except Exception as e:
            print(f"✗ Root endpoint failed: {e}")
        print()
        
        # Test 3: Metrics Endpoint
        print("Test 3: Metrics Endpoint...")
        try:
            response = await client.get(f"{server_url}/metrics")
            response.raise_for_status()
            result = response.json()
            print(f"✓ Metrics retrieved:")
            print(f"  - Total frames: {result.get('total_frames', 0)}")
            print(f"  - GPU available: {result.get('gpu_available', False)}")
            print(f"  - Active streams: {result.get('active_streams', 0)}")
        except Exception as e:
            print(f"✗ Metrics endpoint failed: {e}")
        print()
        
        # Test 4: Inference with Test Image
        print("Test 4: Inference Test...")
        try:
            # Create a test image (640x640 RGB)
            test_image = np.zeros((640, 640, 3), dtype=np.uint8)
            test_image[:320, :320] = [255, 0, 0]  # Red square
            test_image[320:, 320:] = [0, 255, 0]  # Green square
            
            # Encode image
            success, buffer = cv2.imencode('.jpg', test_image)
            if not success:
                raise ValueError("Failed to encode test image")
            
            image_bytes = buffer.tobytes()
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Prepare request
            request_data = {
                "stream_name": "test_stream",
                "frame_id": 1,
                "image_data": image_b64,
                "timestamp": time.time()
            }
            
            # Send inference request
            start_time = time.time()
            response = await client.post(
                f"{server_url}/inference",
                json=request_data
            )
            response.raise_for_status()
            elapsed_time = (time.time() - start_time) * 1000
            
            result = response.json()
            print(f"✓ Inference completed:")
            print(f"  - Latency: {result.get('latency_ms', 0):.2f}ms")
            print(f"  - Processing time: {result.get('processing_time_ms', 0):.2f}ms")
            print(f"  - Detections: {len(result.get('detections', []))}")
            print(f"  - Request time: {elapsed_time:.2f}ms")
            
            if result.get('detections'):
                print(f"  - Detection details:")
                for det in result['detections'][:3]:  # Show first 3
                    print(f"    * {det['label']}: {det['conf']:.2f}")
            
        except Exception as e:
            print(f"✗ Inference test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        print()
        
        # Test 5: Performance Check
        print("Test 5: Performance Check...")
        try:
            # Run multiple inference requests
            num_tests = 5
            latencies = []
            
            for i in range(num_tests):
                start_time = time.time()
                response = await client.post(
                    f"{server_url}/inference",
                    json=request_data
                )
                response.raise_for_status()
                elapsed = (time.time() - start_time) * 1000
                latencies.append(elapsed)
            
            avg_latency = np.mean(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
            
            print(f"✓ Performance test completed:")
            print(f"  - Average latency: {avg_latency:.2f}ms")
            print(f"  - Min latency: {min_latency:.2f}ms")
            print(f"  - Max latency: {max_latency:.2f}ms")
            print(f"  - Estimated FPS: {1000/avg_latency:.2f}")
            
        except Exception as e:
            print(f"✗ Performance test failed: {e}")
        print()
        
        print("=" * 50)
        print("All tests completed!")
        print("=" * 50)
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SmartEye server connection")
    parser.add_argument(
        '--server',
        type=str,
        default='http://localhost:8000',
        help='Server URL'
    )
    
    args = parser.parse_args()
    
    try:
        success = asyncio.run(test_server_connection(args.server))
        if success:
            print("\n✓ Server is ready for use!")
        else:
            print("\n✗ Some tests failed. Check server logs.")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

