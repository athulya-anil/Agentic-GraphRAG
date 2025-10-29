#!/bin/bash

# ============================================
# Start Neo4j Docker Container
# ============================================

echo "🚀 Starting Neo4j Docker container..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Check if Neo4j container already exists
if docker ps -a | grep -q neo4j; then
    echo "📦 Neo4j container already exists."

    # Check if it's running
    if docker ps | grep -q neo4j; then
        echo "✅ Neo4j is already running!"
    else
        echo "▶️  Starting existing Neo4j container..."
        docker start neo4j
        echo "✅ Neo4j started successfully!"
    fi
else
    echo "📥 Creating new Neo4j container..."
    docker run \
        --name neo4j \
        -p 7474:7474 \
        -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/password123 \
        -d neo4j:5.15-community

    echo "⏳ Waiting for Neo4j to be ready..."
    sleep 10
    echo "✅ Neo4j started successfully!"
fi

echo ""
echo "📊 Neo4j Browser: http://localhost:7474"
echo "🔌 Bolt Connection: bolt://localhost:7687"
echo "👤 Username: neo4j"
echo "🔑 Password: password123"
echo ""
echo "To stop Neo4j: docker stop neo4j"
echo "To remove Neo4j: docker rm neo4j"
