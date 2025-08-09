#!/usr/bin/env python
import argparse
import yaml
from utils.config_utils import get_config

def main():
    parser = argparse.ArgumentParser(description="Compile ablation experiment reports.")
    parser.add_argument('--config1', type=str, default="config/no_freeze_config.yaml", 
                      help="Configuration file for experiment 1")
    parser.add_argument('--config2', type=str, default="config/frozen_encoder_config.yaml",
                      help="Configuration file for experiment 2") 
    parser.add_argument('--eval1', type=str, help="Evaluation output file for Experiment 1")
    parser.add_argument('--eval2', type=str, help="Evaluation output file for Experiment 2")
    parser.add_argument('--output', type=str, default="ablation_report.md", help="Output report filename")
    args = parser.parse_args()
    
    # Load configs to include configuration details in the report
    config1 = get_config(args.config1)
    config2 = get_config(args.config2)
    
    # Determine experiment types based on configs
    exp1_type = "Without Encoder Freeze" if not config1['model']['freeze_encoder'] else "With Encoder Frozen"
    exp2_type = "Without Encoder Freeze" if not config2['model']['freeze_encoder'] else "With Encoder Frozen"
    
    # Get model paths from configs
    model1_path = config1['model']['output_dir']
    model2_path = config2['model']['output_dir']
    
    eval1_file = args.eval1 or f"eval_{model1_path.split('_')[-1]}.txt"
    eval2_file = args.eval2 or f"eval_{model2_path.split('_')[-1]}.txt"
    
    try:
        with open(eval1_file, 'r') as f:
            eval1 = f.read()
    except FileNotFoundError:
        eval1 = f"[Evaluation results not found for {model1_path}. Run evaluation first.]"
    
    try:
        with open(eval2_file, 'r') as f:
            eval2 = f.read()
    except FileNotFoundError:
        eval2 = f"[Evaluation results not found for {model2_path}. Run evaluation first.]"
    
    report = "# Ablation Study Report\n\n"
    report += "## Experiment Configurations\n\n"
    report += "### Experiment 1: " + exp1_type + "\n"
    report += "```yaml\n"
    report += yaml.dump(config1, default_flow_style=False)
    report += "```\n\n"
    report += "### Experiment 2: " + exp2_type + "\n"
    report += "```yaml\n"
    report += yaml.dump(config2, default_flow_style=False)
    report += "```\n\n"
    
    report += "## Results\n\n"
    report += "### Experiment 1: " + exp1_type + "\n"
    report += "---\n"
    report += eval1 + "\n\n"
    report += "### Experiment 2: " + exp2_type + "\n"
    report += "---\n"
    report += eval2 + "\n\n"
    
    report += "## Analysis\n\n"
    report += "This section compares the performance of the two models with different configurations.\n"
    report += "The key difference is whether the encoder weights were frozen during training.\n\n"
    report += "Key metrics to compare:\n"
    report += "- Caption generation quality\n"
    report += "- Average caption length\n"
    report += "- Training time and compute efficiency\n"
    
    with open(args.output, 'w') as f:
        f.write(report)
    print("Report generated and saved to", args.output)

if __name__ == '__main__':
    main()
