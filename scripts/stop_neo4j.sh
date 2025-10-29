#!/bin/bash

# ============================================
# Stop Neo4j Docker Container
# ============================================

echo "🛑 Stopping Neo4j Docker container..."

if docker ps | grep -q neo4j; then
    docker stop neo4j
    echo "✅ Neo4j stopped successfully!"
else
    echo "ℹ️  Neo4j container is not running."
fi
