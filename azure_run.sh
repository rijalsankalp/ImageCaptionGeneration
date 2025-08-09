#!/usr/bin/env bash

# Azure ML Image Captioning Pipeline Script
# This script automates the entire workflow for image captioning ablation study


# Environment setup: Create or use Python 3.10 venv
PYTHON_VERSION="3.10"
VENV_DIR=".venv_imgcap"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python $PYTHON_VERSION virtual environment at $VENV_DIR..."
    python$PYTHON_VERSION -m venv $VENV_DIR
fi

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Using Python: $(which python)"
python --version

echo "Installing requirements..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Set up Git authentication if needed
if [ -f "setup_git_auth.sh" ]; then
    echo "Setting up Git authentication..."
    source setup_git_auth.sh
fi

# Define experiment configurations
EXPERIMENT_1="config/no_freeze_config.yaml"   # No freezing
EXPERIMENT_2="config/frozen_encoder_config.yaml"  # Frozen encoder

# Run Experiment 1: No Freezing
echo "Starting Experiment 1: Without Encoder Freezing"
python train.py --config $EXPERIMENT_1
python evaluate.py --config $EXPERIMENT_1 > eval_no_freeze.txt

# Run Experiment 2: Frozen Encoder  
echo "Starting Experiment 2: With Frozen Encoder"
python train.py --config $EXPERIMENT_2
python evaluate.py --config $EXPERIMENT_2 > eval_frozen.txt

# Generate Ablation Report
echo "Generating ablation report..."
python compile_report.py --config1 $EXPERIMENT_1 --config2 $EXPERIMENT_2 \
                        --eval1 eval_no_freeze.txt --eval2 eval_frozen.txt \
                        --output ablation_report.md

echo "Pipeline completed successfully!"
echo "Results available in ablation_report.md"

# Push results to GitHub
echo "Pushing results to GitHub..."

# Create results directory if it doesn't exist
mkdir -p results

# Copy model files and results to a results directory
cp -r final_model_no_freeze final_model_frozen best_model_no_freeze best_model_frozen results/
cp eval_no_freeze.txt eval_frozen.txt ablation_report.md results/
cp caption_sample_*.png results/ 2>/dev/null || true

# Git credentials should already be set up by setup_git_auth.sh
# If you're using this script directly, make sure Git is configured

# Add results to git, commit and push
git add results/
git add ablation_report.md
git commit -m "Add training results from Azure ML run $(date)"
git push origin main || echo "Failed to push to GitHub."

echo "GitHub push completed"
