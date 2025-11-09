#!/bin/bash
# Example script to run SmartEye with webcam

echo "Starting SmartEye Example..."
echo ""

echo "Step 1: Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Step 2: Starting server in background..."
python server.py &
SERVER_PID=$!

echo "Waiting for server to start..."
sleep 5

echo ""
echo "Step 3: Starting client with webcam..."
python client.py --streams 0 --names webcam_0 --fps-limit 30

echo ""
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null

echo ""
echo "Example completed!"

