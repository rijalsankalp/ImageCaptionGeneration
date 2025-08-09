import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataframe, tokenizer, feature_extractor, max_target_length=128, image_dir="flickr8k/Images"):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_target_length = max_target_length
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        data = self.dataframe.iloc[idx]
        image_path = data['image']
        caption = data['caption']
        img = Image.open(os.path.join(self.image_dir, image_path))
        processed = self.feature_extractor(images=img, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)
        labels = self.tokenizer(caption, padding="max_length", max_length=self.max_target_length).input_ids
        if len(labels) < self.max_target_length:
            labels.extend([self.tokenizer.pad_token_id] * (self.max_target_length - len(labels)))
        return {'pixel_values': pixel_values, 'labels': torch.tensor(labels)}
