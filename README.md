# Image Captioning with Vision-Text Models

This project implements an image captioning system using Vision-Encoder-Decoder architecture, combining ViT (Vision Transformer) for image encoding and GPT-2 for text generation. Originally built as a notebook, it has been refactored into a modular Python codebase with YAML-based configuration management, suitable for deployment on Azure.

## Project Structure

```
.
├── config/
│   ├── default_config.yaml       # Default configuration
│   ├── frozen_encoder_config.yaml # Configuration for frozen encoder experiment
│   └── no_freeze_config.yaml     # Configuration for non-frozen encoder experiment
├── data/
│   └── preprocess.py             # Data loading and preprocessing
├── models/
│   └── vit_gpt2.py               # Dataset classes and model utilities
├── utils/
│   └── config_utils.py           # Configuration loading and management utilities
├── train.py                      # Main training script
├── evaluate.py                   # Evaluation script with metrics and visualization
├── compile_report.py             # Generate reports for ablation studies
├── azure_run.sh                  # Shell script for running the pipeline on Azure
├── setup_git_auth.sh            # Script to set up GitHub authentication
├── generate_captions.py         # Script for generating captions for multiple images
├── quick_demo.py                # Simple demo script for testing the model
├── score.py                     # Scoring script for Azure ML deployment
├── make_executable.sh           # Script to make Python files executable
├── AZURE_DEPLOYMENT.md          # Azure ML deployment instructions
├── NOTEBOOK_README.md           # Information about the original notebook
├── requirements.txt             # Required Python packages
└── vitgpt2.ipynb                # Original notebook implementation (reference only)
```

## Setup and Requirements

- Python 3.10+
- PyTorch 1.12.0+, Transformers 4.21.0+
- datasets, evaluate
- pandas, PIL, matplotlib, tqdm
- PyYAML for configuration management

## Dataset

- Flickr8k dataset consisting of images and their corresponding captions
- Dataset split into training (80%) and evaluation (20%) sets

## Getting Started

### Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the Flickr8k dataset and place it in:
   - `flickr8k/Images/` - for images
   - `flickr8k/captions.txt` - for captions

## Model Architecture

- **Vision Encoder**: google/vit-base-patch16-224-in21k
- **Text Decoder**: GPT-2
- Two training variants (ablation study):
  1. Fine-tuning decoder only (encoder frozen)
  2. Fine-tuning both encoder and decoder

## Training Process

- Custom Dataset class for handling image-caption pairs
- Batch size: 4
- Learning rate: 5e-5
- Optimizer: AdamW
- Training for 5 epochs
- Evaluation metrics:
  - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
  - Training and validation losses
- Model checkpointing based on best validation loss

## Running the Code

### Configuration

The project uses YAML configuration files to manage training and evaluation parameters. This approach makes it easy to experiment with different settings without changing the code.

Main configuration files:

- `config/default_config.yaml`: Default configuration for all parameters
- `config/frozen_encoder_config.yaml`: Configuration specific to the frozen encoder experiment
- `config/no_freeze_config.yaml`: Configuration specific to the non-frozen encoder experiment

The configuration system includes:

- Data parameters (paths, preprocessing settings)
- Model architecture settings
- Training hyperparameters
- Generation parameters
- Evaluation options

### Local Training

```bash
# Train with default configuration
python train.py

# Train with specific configuration
python train.py --config config/frozen_encoder_config.yaml

# Override specific parameters from command line
python train.py --config config/frozen_encoder_config.yaml --learning_rate 1e-4 --batch_size 8
```

### Evaluation

```bash
# Evaluate with default configuration
python evaluate.py

# Evaluate with specific model path and configuration
python evaluate.py --config config/no_freeze_config.yaml --save_images --num_samples 10

# Evaluate a specific model with custom settings
python evaluate.py --model_path final_model_frozen --save_images

# Generate captions for new images
python generate_captions.py --image_path /path/to/images --model_path final_model

# Run a quick demo on a single image
python quick_demo.py --image_path test_image.jpg --model_path best_model
```

### Full Pipeline

To run the full training and evaluation pipeline for the ablation study:

```bash
# Run the complete pipeline (train, evaluate, and generate report)
bash azure_run.sh
```

This script will:

1. Set up the environment
2. Train models with both configurations (frozen and non-frozen encoder)
3. Evaluate both models
4. Generate an ablation study report comparing the results
5. Push the results to GitHub (if credentials are configured)

For Azure deployment and GitHub integration, see the `AZURE_DEPLOYMENT.md` file.

## Results

The evaluation includes:

- Visualization of training and evaluation losses
- ROUGE scores across epochs
- Sample predictions on test images
- Comparison between model variations

## Ablation Study

Comparison between:

- Decoder-only fine-tuning (encoder frozen)
- Full model fine-tuning (both encoder and decoder)

The compiled report demonstrates the performance differences between these approaches.

## Utility Scripts and Tools

### Configuration Management

The `utils/config_utils.py` provides tools for:

- Loading configuration from YAML files
- Overriding configuration with command-line arguments
- Type conversion for proper parameter handling

### Report Generation

Use the `compile_report.py` script to create comprehensive reports comparing different model configurations:

```bash
python compile_report.py --config1 config/no_freeze_config.yaml \
                         --config2 config/frozen_encoder_config.yaml \
                         --eval1 eval_no_freeze.txt \
                         --eval2 eval_frozen.txt \
                         --output ablation_report.md
```
