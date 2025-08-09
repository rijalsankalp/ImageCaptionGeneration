#!/usr/bin/env python
import os
import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
from utils.config_utils import get_config


def parse_args():
    parser = argparse.ArgumentParser(description="Generate captions for input images")
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--model_path", type=str,
                        help="Path to the trained model directory")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to a single image or directory of images")
    parser.add_argument("--output_dir", type=str, default="generated_captions",
                        help="Directory to save images with captions")
    parser.add_argument("--show_images", action="store_true",
                        help="Display images with captions")
    parser.add_argument("--save_images", action="store_true", default=True,
                        help="Save images with captions")
    return parser.parse_args()


def generate_caption(image_path, model, tokenizer, feature_extractor, device, generation_config):
    """Generate a caption for an image using the trained model."""
    img = Image.open(image_path).convert("RGB")
    processed = feature_extractor(images=img, return_tensors="pt")
    pixel_values = processed["pixel_values"].to(device)
    
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values, 
            max_length=int(generation_config['max_length']), 
            num_beams=int(generation_config['num_beams']), 
            early_stopping=bool(generation_config['early_stopping'])
        )
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption, img


def main():
    args = parse_args()
    config = get_config(args.config)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract config values
    model_path = args.model_path or config['model']['output_dir']
    image_encoder_model = config['model']['image_encoder_model']
    generation_config = config['generation']
    
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
    
    # Handle single image or directory
    if os.path.isfile(args.image_path):
        image_paths = [args.image_path]
    else:
        image_paths = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    print(f"Found {len(image_paths)} images")
    
    for i, img_path in enumerate(image_paths):
        try:
            # Generate caption
            caption, img = generate_caption(
                img_path, 
                model, 
                tokenizer, 
                feature_extractor, 
                device,
                generation_config
            )
            
            # Display filename and caption
            img_name = os.path.basename(img_path)
            print(f"\nImage {i+1}/{len(image_paths)}: {img_name}")
            print(f"Caption: {caption}")
            
            # Display image with caption
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.title(f"Caption: {caption}", fontsize=12)
            plt.axis('off')
            plt.tight_layout()
            
            # Save image with caption
            if args.save_images:
                output_file = os.path.join(args.output_dir, f"captioned_{os.path.basename(img_path)}")
                plt.savefig(output_file, bbox_inches='tight')
                print(f"Saved captioned image to {output_file}")
            
            # Show image if requested
            if args.show_images:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"\nCaption generation completed for {len(image_paths)} images")


if __name__ == "__main__":
    main()
