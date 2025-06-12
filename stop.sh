echo "Stopping the server..."
docker-compose down
echo "Server stopped."
docker system prune -a -f --volumes
