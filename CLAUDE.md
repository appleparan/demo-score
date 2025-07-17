# Demo-SCORE Repository Context

## Overview
This repository implements Demo-SCORE (Automatic Online Robot Dataset Curation), a system for automatically curating robot demonstration datasets based on online robot experience. Demo-SCORE uses classifiers trained on policy rollouts to identify and filter suboptimal demonstrations without manual curation.

**Paper**: [Demo-SCORE: Automatic Online Robot Dataset Curation](https://arxiv.org/abs/2503.03707)  
**Project Page**: https://anniesch.github.io/demo-score/  
**Authors**: Alec Lessing, Annie Chen

## Repository Structure

### Core Components

1. **`demo_score/`** - Main Demo-SCORE implementation
   - `demo_score/dataset.py` - Dataset handling for different formats (LeRobot, RoboDiff, ALOHA)
   - `demo_score/models.py` - Classifier models (MLPPoolMLP, StepwiseMLPClassifier)
   - `demo_score/train_stepwise_classifier.py` - Training stepwise classifiers
   - `demo_score/train_trajwise_classifier.py` - Training trajectory-wise classifiers
   - `demo_score/eval_stepwise_classifier.py` - Evaluation utilities
   - `demo_score/filters/` - Filtering implementations for different frameworks
   - `demo_score/run_sweep.py` - Hyperparameter sweep utilities

2. **`lerobot/`** - Fork of HuggingFace LeRobot for robot learning
   - Includes ACT, Diffusion Policy, and VQBET implementations
   - `lerobot/experiments/example/` - Example experiment scripts
   - Supports various robot environments and datasets

3. **`diffusion_policy/`** - Fork of Diffusion Policy for behavior cloning
   - Implements diffusion-based policies for robot learning
   - `diffusion_policy/experiments/example/` - Example experiment workflow
   - Supports various environments (PushT, Kitchen, etc.)

4. **`aloha_act/`** - Fork of ALOHA ACT for bimanual manipulation
   - Implements Action Chunking Transformer (ACT)
   - Includes simulation environments and real robot support

### Key Features

- **Multi-framework support**: Works with LeRobot, Diffusion Policy, and ALOHA ACT
- **Classifier training**: Stepwise and trajectory-wise classifiers for success prediction
- **Dataset filtering**: Automatic filtering of suboptimal demonstrations
- **Hyperparameter sweeps**: Systematic exploration of model configurations
- **Cross-validation**: Robust evaluation using different policy checkpoints

## Development Environment

### Primary Environment (Demo-SCORE + LeRobot)
```bash
conda create -y -n demoscore python=3.10
conda activate demoscore
cd lerobot && pip install -e ".[aloha]"
cd ../demo_score && pip install -e .
pip install tensorboard
```

### Additional Environments
- `robodiff`: For Diffusion Policy experiments
- `aloha`: For ALOHA ACT experiments

## Common Workflows

### 1. Full Demo-SCORE Pipeline
1. **Train base policy** with all demonstrations
2. **Collect rollouts** from multiple policy checkpoints
3. **Train classifiers** using sweep configurations
4. **Filter datasets** using trained classifiers
5. **Retrain policies** on filtered datasets

### 2. Adding New Environments
To make Demo-SCORE compatible with new codebases:
1. Modify `ClassifierDataset` in `demo_score/dataset.py` to handle new format
2. Create new filtering script in `demo_score/filters/`
3. Update `filter_sweep.py` to import the new filtering function

### 3. Classifier Training
- Use `train_stepwise_classifier.py` for step-level success prediction
- Use `train_trajwise_classifier.py` for trajectory-level success prediction
- Configure sweeps in experiment directories (e.g., `lerobot/experiments/example/step3_classifier_sweep.py`)

## Data Formats

The system supports multiple data formats:
- **LeRobot**: HuggingFace datasets format
- **RoboDiff**: Robomimic-style datasets
- **ALOHA**: HDF5 format for bimanual manipulation

## Model Architecture

### Classifiers
- **MLPPoolMLP**: Multi-layer perceptron with pooling for sequence data
- **StepwiseMLPClassifier**: Step-level classification
- **Transformer-based**: Optional transformer architectures for sequence modeling

### Features
- Positional encoding support
- Dropout regularization
- Configurable hidden dimensions
- Support for different pooling strategies

## Experiment Configuration

### Example Sweep Configuration
```python
model_dicts = {
    'small_stepwise': {'hidden_sizes': [8, 8]},
    'med_stepwise': {'hidden_sizes': [16, 16]},
    'large_stepwise': {'hidden_sizes': [32, 32]}
}

eps_dict = {
    'experiment_name': {
        "train": ["070000", "150000", "220000"],  # Training checkpoints
        "cross_val": ["300000"]  # Validation checkpoints
    }
}
```

## File Organization

### Key Files to Understand
- `README.md` - Main documentation and setup instructions
- `demo_score/dataset.py` - Core dataset handling logic
- `demo_score/models.py` - Classifier implementations
- `*/experiments/example/` - Example experiment workflows
- `demo_score/filters/` - Framework-specific filtering implementations

### Testing and Validation
- Use cross-validation across different policy checkpoints
- Evaluate on held-out test sets
- Monitor success rates and filtering effectiveness

## Performance Expectations

Demo-SCORE achieves:
- **15-35% higher absolute success rate** compared to policies trained on unfiltered data
- Effective identification of suboptimal demonstrations
- Robust performance across different robot learning frameworks

## Development Notes

- The repository contains forks of multiple established robotics repositories
- Each framework may require different conda environments
- Korean documentation available in `lerobot_한글_가이드.md`
- PI0 model integration guides available in `docs/`

## Common Commands

### Training
```bash
# Train base policy
python lerobot/scripts/train.py policy=act env=aloha [...]

# Train classifier
python -m demo_score.train_stepwise_classifier [...]

# Run sweep
python step3_classifier_sweep.py
```

### Evaluation
```bash
# Evaluate policy
python lerobot/scripts/eval.py [...]

# Apply filtering
python step4_apply_filter.py
```

This codebase represents a comprehensive system for improving robot learning through intelligent dataset curation, combining multiple state-of-the-art approaches in a unified framework.