#!/usr/bin/env python
import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
from models.vit_gpt2 import ImageCaptioningDataset
from data.preprocess import load_and_split_data
from utils.config_utils import get_config

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Image Captioning Model")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--model_path", type=str,
                        help="Path to the trained model directory")
    parser.add_argument("--data_path", type=str,
                        help="Path to the captions dataset")
    parser.add_argument("--image_dir", type=str,
                        help="Path to the images directory")
    parser.add_argument("--num_samples", type=int,
                        help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int,
                        help="Batch size for evaluation")
    parser.add_argument("--save_images", action="store_true",
                        help="Save images with captions")
    return parser.parse_args()

def generate_caption(image_path, model, tokenizer, feature_extractor, device, max_length=128, num_beams=4, early_stopping=True):
    """Generate a caption for an image using the trained model."""
    img = Image.open(image_path)
    processed = feature_extractor(images=img, return_tensors="pt")
    pixel_values = processed["pixel_values"].to(device)
    
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values, 
            max_length=max_length, 
            num_beams=num_beams, 
            early_stopping=early_stopping
        )
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption, img

def main():
    args = parse_args()
    config = get_config(args.config, args=args)
    
    # Extract config values
    model_path = args.model_path or config['model']['output_dir']
    data_path = args.data_path or config['data']['data_path']
    image_dir = args.image_dir or config['data']['image_dir']
    num_samples = args.num_samples or int(config['evaluation']['num_samples'])
    save_images = args.save_images or bool(config['evaluation']['save_images'])
    max_target_length = int(config['data']['max_target_length'])
    image_encoder_model = config['model']['image_encoder_model']
    generation_config = config['generation']
    
    # Ensure generation config values are properly typed
    generation_config['max_length'] = int(generation_config['max_length'])
    generation_config['num_beams'] = int(generation_config['num_beams'])
    generation_config['early_stopping'] = bool(generation_config['early_stopping'])
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}")
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    feature_extractor = ViTImageProcessor.from_pretrained(image_encoder_model)
    
    # Move model to device
    model.to(device)
    
    # Load evaluation data
    _, eval_ds = load_and_split_data(data_path)
    
    # Create evaluation dataset and dataloader for batch evaluation if needed
    eval_dataset = ImageCaptioningDataset(eval_ds, tokenizer, feature_extractor, max_target_length=max_target_length, image_dir=image_dir)
    
    # Evaluate on individual samples with visualization
    total_samples = min(num_samples, len(eval_ds))
    print(f"Evaluating on {total_samples} samples")
    
    eval_results = []
    
    for i in range(total_samples):
        sample = eval_ds.iloc[i]
        image_path = os.path.join(image_dir, sample['image'])
        true_caption = sample['caption']
        
        # Generate caption using config parameters
        generated_caption, img = generate_caption(
            image_path, 
            model, 
            tokenizer, 
            feature_extractor, 
            device,
            max_length=generation_config['max_length']
        )
        
        print(f"\nSample {i+1}:")
        print(f"Image: {sample['image']}")
        print(f"True caption: {true_caption}")
        print(f"Generated caption: {generated_caption}")
        
        eval_results.append({
            'image': sample['image'],
            'true_caption': true_caption,
            'generated_caption': generated_caption
        })
        
        # Save image with caption if requested
        if save_images:
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.title(f"Generated: {generated_caption}\nTrue: {true_caption}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"caption_sample_{i+1}.png")
    
    # Calculate metrics
    avg_true_len = sum(len(r['true_caption'].split()) for r in eval_results) / total_samples
    avg_gen_len = sum(len(r['generated_caption'].split()) for r in eval_results) / total_samples
    
    print("\nEvaluation Summary:")
    print(f"Number of samples: {total_samples}")
    print(f"Average true caption length: {avg_true_len:.2f} words")
    print(f"Average generated caption length: {avg_gen_len:.2f} words")
    
    if args.save_images:
        print(f"Caption samples saved as PNG files")

if __name__ == "__main__":
    main()
