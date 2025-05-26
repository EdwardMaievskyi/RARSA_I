#!/bin/bash

# Build and run the containerized application
echo "Building AI Research Agent container..."
docker-compose build

echo "Starting AI Research Agent..."
docker-compose up -d

echo "Waiting for container to start..."
sleep 5

echo "Checking container status..."
docker-compose ps

echo ""
echo "Application is running at http://localhost:7860"
echo "To stop: docker-compose down"
echo "To view logs: docker-compose logs -f"

# Show recent logs
echo ""
echo "=== Recent Logs ==="
docker-compose logs --tail=10 