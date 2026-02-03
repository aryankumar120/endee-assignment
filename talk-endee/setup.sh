#!/bin/bash

echo "Checking Python..."
echo "✓ Python $python_version found"
echo ""
echo "Checking Endee connection..."
echo "Installing Python dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""
echo "Setting up environment..."
echo "Checking configuration..."
set -e

echo "🚀 Talk Endee Setup"
echo "=================="
echo ""

echo "Checking Python..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version found"
echo ""

echo "Checking Endee connection..."
if curl -s http://localhost:8080/api/v1/index/list > /dev/null 2>&1; then
    echo "✓ Endee is running on http://localhost:8080"
else
    echo "⚠ Endee is not running or not reachable"
    echo "  Make sure to start Endee in another terminal:"
    echo "  cd /path/to/endee"
    echo "  export NDD_DATA_DIR=\$(pwd)/data"
    echo "  ./build/ndd-neon-darwin"
fi
echo ""

echo "Installing Python dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

echo "Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠ Created .env from .env.example"
    echo "  ⚠ IMPORTANT: Edit .env and add your GROQ_API_KEY"
    echo "  Get it from: https://docs.groq.ai/ (or your Groq Cloud dashboard)"
else
    echo "✓ .env already exists"
fi
echo ""

echo "Checking configuration..."
if grep -q "your_groq_api_key_here" .env; then
    echo "⚠ GROQ_API_KEY not set in .env"
    echo "  Please edit .env and add your GROQ API key"
    exit 1
else
    echo "✓ Configuration looks good"
fi
echo ""

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Ingest documents:"
echo "   python main.py ingest --directory data/sample_docs"
echo ""
echo "2. Query the system:"
echo "   python main.py query \"What is semantic search?\""
echo ""
echo "3. View system info:"
echo "   python main.py info"
echo ""
echo "Happy searching! 🎉"
