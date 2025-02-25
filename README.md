# Image Captioning with Vision-Text Models

This notebook implements an image captioning system using Vision-Encoder-Decoder architecture, combining ViT (Vision Transformer) for image encoding and GPT-2 for text generation. This project was completed as part of the Machine Learning course in December 2024.

## Setup and Requirements

- PyTorch
- Transformers
- datasets
- evaluate
- pandas
- PIL
- tqdm

## Dataset

- Flickr8k dataset consisting of images and their corresponding captions
- Dataset split into training (80%) and evaluation (20%) sets

## Model Architecture

-**Vision Encoder**: google/vit-base-patch16-224-in21k

-**Text Decoder**: GPT-2

- Two training variants:

1. Fine-tuning decoder only
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

## Results

The notebook includes visualization of:

- Training and evaluation losses
- ROUGE scores across epochs
- Sample predictions on test images

## Model Variations

Comparison between:

- Decoder-only fine-tuning
- Full model fine-tuning (both encoder and decoder)

The results section demonstrates the performance of both approaches on sample images.
