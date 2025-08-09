# Original Notebook

This notebook (`vitgpt2.ipynb`) contains the original implementation of the image captioning model using ViT and GPT-2. The code has been refactored into a modular Python codebase suitable for deployment.

## Usage

The notebook is kept for reference purposes. For production use, please use the modular Python scripts:

- `train.py` - For model training
- `evaluate.py` - For model evaluation
- `compile_report.py` - For generating ablation study reports

## Configuration

Training parameters are now managed through YAML configuration files in the `config/` directory.
See the README.md for detailed instructions on how to use the modular codebase.
