#!/bin/bash

# Create main project directory (if not already in it)
mkdir -p train_s
cd train_s

mkdir -p data
mkdir -p stock_qa_model
mkdir -p src
echo "Sub directories created: Proceeding to next steps"

echo "Creating python files"

touch src/train.py
touch src/inference.py
touch src/utils.py

echo "Python files created"

echo "Installing requirements.txt"
touch requirements.txt
echo "Requirements installed"


# Create README.md
touch README.md

