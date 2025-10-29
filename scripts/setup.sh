#!/bin/bash

# ============================================
# Agentic GraphRAG - Quick Setup Script
# ============================================

echo "🚀 Setting up Agentic GraphRAG..."
echo ""

# Check Python version
echo "1️⃣  Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "2️⃣  Creating .env file..."
    cp .env.example .env
    echo "   ✅ .env file created"
    echo ""
    echo "   ⚠️  IMPORTANT: Edit .env and add your Groq API key!"
    echo "   Get your key from: https://console.groq.com/keys"
    echo ""
    read -p "   Press Enter to open .env in default editor..."
    ${EDITOR:-nano} .env
else
    echo "2️⃣  .env file already exists ✅"
fi

# Install dependencies
echo ""
echo "3️⃣  Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "4️⃣  Downloading spaCy model..."
python -m spacy download en_core_web_sm

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Make sure Docker Desktop is running"
echo "  2. Start Neo4j: ./scripts/start_neo4j.sh"
echo "  3. Test LLM client: python -m src.utils.llm_client"
echo ""
echo "See README.md for detailed usage instructions."
