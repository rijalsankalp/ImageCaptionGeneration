
#!/usr/bin/env python
import os
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
from torch.optim import AdamW
from data.preprocess import load_and_split_data
from models.vit_gpt2 import ImageCaptioningDataset
from tqdm import tqdm
from utils.config_utils import get_config

def parse_args():
    parser = argparse.ArgumentParser(description="Train Image Captioning Model")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", 
                      help="Path to configuration file")
    parser.add_argument("--data_path", type=str, help="Path to the captions dataset")
    parser.add_argument("--image_dir", type=str, help="Path to the images directory")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder weights")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, help="Directory to save the model")
    return parser.parse_args()

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def main():
    args = parse_args()
    config = get_config(args.config, args=args)
    
    # Print configuration
    print(f"Using device: {DEVICE}")
    print(f"Freeze encoder: {config['model']['freeze_encoder']}")
    
    # Extract config values
    data_path = config['data']['data_path']
    image_dir = config['data']['image_dir']
    max_target_length = int(config['data']['max_target_length'])
    image_encoder_model = config['model']['image_encoder_model']
    text_decoder_model = config['model']['text_decoder_model']
    output_dir = config['model']['output_dir']
    batch_size = int(config['training']['batch_size'])
    learning_rate = float(config['training']['learning_rate'])
    num_epochs = int(config['training']['num_epochs'])
    
    # Data
    train_ds, eval_ds = load_and_split_data(data_path)

    # Tokenizer & Feature Extractor
    feature_extractor = ViTImageProcessor.from_pretrained(image_encoder_model)
    tokenizer = AutoTokenizer.from_pretrained(text_decoder_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Datasets
    train_dataset = ImageCaptioningDataset(train_ds, tokenizer, feature_extractor, max_target_length=max_target_length, image_dir=image_dir)
    eval_dataset = ImageCaptioningDataset(eval_ds, tokenizer, feature_extractor, max_target_length=max_target_length, image_dir=image_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    # Model
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(image_encoder_model, text_decoder_model, trust_remote_code=True)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model = torch.nn.DataParallel(model)
    model.to(DEVICE)


    # Freeze encoder if specified
    if config['model']['freeze_encoder']:
        for param in model.module.encoder.parameters():
            param.requires_grad = False
        print("Encoder weights frozen.")

    # Always freeze all GPT-2 decoder layers except the last block and lm_head
    decoder = model.module.decoder if hasattr(model.module, 'decoder') else model.module.transformer
    # Freeze all transformer blocks except the last
    for block in decoder.transformer.h[:-1]:
        for param in block.parameters():
            param.requires_grad = False
    print("All GPT-2 blocks except the last are frozen.")
    # Optionally freeze other decoder parts except lm_head and last block
    for name, param in decoder.named_parameters():
        if not (name.startswith('transformer.h.11') or name.startswith('lm_head')):
            if not name.startswith('transformer.h.'):  # freeze everything except blocks and lm_head
                param.requires_grad = False
    print("All GPT-2 decoder parameters except last block and lm_head are frozen.")

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training loop with progress bar and loss printout
    best_eval_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress:
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress.set_postfix({'loss': loss.item()})
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} done. Average loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        eval_loss = 0.0
        eval_progress = tqdm(eval_loader, desc=f"Evaluating epoch {epoch+1}")
        for batch in eval_progress:
            pixel_values = batch['pixel_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
                eval_loss += loss.item()
        avg_eval_loss = eval_loss / len(eval_loader)
        print(f"Eval Loss: {avg_eval_loss:.4f}")
        
        # Save best model
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            model.module.save_pretrained(config['model']['best_model_dir'])
            tokenizer.save_pretrained(config['model']['best_model_dir'])
            print(f"New best model saved to {config['model']['best_model_dir']}")

    # Save final model
    model.module.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Final model saved to {output_dir}")

if __name__ == "__main__":
    main()
