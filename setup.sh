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

echo "Project structure created successfully!"
echo "Next steps:"
echo "1. Paste the code into each file"
echo "2. Install requirements: pip install -r requirements.txt"
echo "3. Set up Kaggle API credentials"
echo "4. Run the training script: python src/train.py"