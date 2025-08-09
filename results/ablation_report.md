# Ablation Study Report

## Experiment Configurations

### Experiment 1: Without Encoder Freeze
```yaml
data:
  data_path: flickr8k/captions.txt
  image_dir: flickr8k/Images
  max_target_length: 128
  random_seed: 42
  train_test_split: 0.2
evaluation:
  metrics:
  - rouge
  num_samples: 5
  save_images: true
generation:
  early_stopping: true
  max_length: 128
  num_beams: 4
model:
  best_model_dir: best_model_no_freeze
  freeze_encoder: false
  image_encoder_model: google/vit-base-patch16-224-in21k
  output_dir: final_model_no_freeze
  text_decoder_model: gpt2
training:
  adam_epsilon: 1e-8
  batch_size: 4
  eval_steps: 500
  gradient_accumulation_steps: 1
  learning_rate: 5e-5
  num_epochs: 5
  save_steps: 500
  warmup_steps: 0
  weight_decay: 0.01
```

### Experiment 2: With Encoder Frozen
```yaml
data:
  data_path: flickr8k/captions.txt
  image_dir: flickr8k/Images
  max_target_length: 128
  random_seed: 42
  train_test_split: 0.2
evaluation:
  metrics:
  - rouge
  num_samples: 5
  save_images: true
generation:
  early_stopping: true
  max_length: 128
  num_beams: 4
model:
  best_model_dir: best_model_frozen
  freeze_encoder: true
  image_encoder_model: google/vit-base-patch16-224-in21k
  output_dir: final_model_frozen
  text_decoder_model: gpt2
training:
  adam_epsilon: 1e-8
  batch_size: 4
  eval_steps: 500
  gradient_accumulation_steps: 1
  learning_rate: 5e-5
  num_epochs: 5
  save_steps: 500
  warmup_steps: 0
  weight_decay: 0.01
```

## Results

### Experiment 1: Without Encoder Freeze
---


### Experiment 2: With Encoder Frozen
---


## Analysis

This section compares the performance of the two models with different configurations.
The key difference is whether the encoder weights were frozen during training.

Key metrics to compare:
- Caption generation quality
- Average caption length
- Training time and compute efficiency
