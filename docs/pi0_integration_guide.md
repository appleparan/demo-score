# LeRobot PI0 Model Integration with Demo-SCORE

## Overview

This guide explains how to integrate the LeRobot PI0 (Policy Influence 0) model with Demo-SCORE's dataset curation system. The main challenge is adapting the PI0 model's `embed_prefix` method output (`prefix_embs`) to work with Demo-SCORE's input format.

## Current Demo-SCORE Architecture

Demo-SCORE uses a classifier-based approach to filter demonstration datasets:

1. **Dataset Format**: Uses `observation.state` tensors of shape `(sequence_length, state_dim)`
2. **Classifier Models**: MLPPoolMLP, TransformerClassifier, or StepwiseMLPClassifier
3. **Filtering Process**: Classifies demonstrations as successful/unsuccessful and filters accordingly

### Supported Dataset Formats

- **lerobot**: Uses `observation.state` from LeRobotDataset
- **robodiff**: Uses action sequences from HDF5 files
- **aloha**: Uses joint positions and velocities

## PI0 Model Integration Strategy

### 1. Understanding PI0 Prefix Embeddings

The PI0 model's `embed_prefix` method generates `prefix_embs` that represent:
- Context information from demonstration prefixes
- Encoded behavioral patterns
- Policy-relevant state representations

### 2. Adapting PI0 for Demo-SCORE

Since PI0 is not included in this repository's lerobot implementation, you'll need to:

#### Step 1: Add PI0 Model to Repository

```bash
# Add PI0 implementation to lerobot policies
mkdir -p lerobot/lerobot/common/policies/pi0
```

Create the following files:
- `lerobot/lerobot/common/policies/pi0/configuration_pi0.py`
- `lerobot/lerobot/common/policies/pi0/modeling_pi0.py`

#### Step 2: Modify Dataset Class

Extend the `ClassifierDataset` class to handle PI0 embeddings:

```python
# Add to demo_score/demo_score/dataset.py

class PI0ClassifierDataset(ClassifierDataset):
    def __init__(self, data_dir, pi0_model, data_root='./data', format='lerobot', **kwargs):
        super().__init__(data_dir, data_root, format=format, **kwargs)
        self.pi0_model = pi0_model
        self.pi0_model.eval()
        
    def _get_pi0_embeddings(self, observations):
        """
        Extract prefix embeddings from PI0 model
        
        Args:
            observations: Input observations tensor
            
        Returns:
            prefix_embs: Embedded representations from PI0
        """
        with torch.no_grad():
            # Assuming PI0FlowMatching has embed_prefix method
            prefix_embs = self.pi0_model.embed_prefix(observations)
        return prefix_embs
    
    def __getitem__(self, index):
        if self.format == 'lerobot':
            # Get standard observations
            obs, label = super().__getitem__(index)
            
            # Get PI0 embeddings
            prefix_embs = self._get_pi0_embeddings(obs)
            
            # Combine or replace with PI0 embeddings
            return prefix_embs, label
        else:
            return super().__getitem__(index)
```

#### Step 3: Create PI0-Compatible Classifier

```python
# Add to demo_score/demo_score/models.py

class PI0Classifier(nn.Module):
    def __init__(self, pi0_embed_dim, hidden_sizes=[128, 64], dropout_prob=0.3):
        super(PI0Classifier, self).__init__()
        
        layers = []
        input_size = pi0_embed_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, 1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, prefix_embs):
        """
        Args:
            prefix_embs: Output from PI0 embed_prefix method
        """
        # Handle different prefix_embs shapes
        if len(prefix_embs.shape) == 3:  # (batch, seq, embed_dim)
            prefix_embs = prefix_embs.mean(dim=1)  # Pool over sequence
        
        return torch.sigmoid(self.model(prefix_embs))
```

#### Step 4: Create PI0 Filter

```python
# Create demo_score/demo_score/filters/classifier_filter_pi0.py

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.pi0.modeling_pi0 import PI0FlowMatching
from pathlib import Path
from safetensors.torch import load_file
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def classifier_filter_pi0(orig_datasets, new_datasets_file, pi0_model, classifier_model, 
                         rew_thresh=3.99, classifier_thresh=0.5):
    """
    Filter datasets using PI0 embeddings and classifier
    
    Args:
        orig_datasets: List of original dataset paths
        new_datasets_file: Path to save filtered dataset
        pi0_model: Trained PI0FlowMatching model
        classifier_model: Trained classifier for PI0 embeddings
        rew_thresh: Reward threshold for success
        classifier_thresh: Classifier threshold for filtering
    """
    
    pi0_model = pi0_model.to(device)
    classifier_model = classifier_model.to(device)
    
    pi0_model.eval()
    classifier_model.eval()
    
    orig_datasets_lines = []
    
    # Parse original datasets
    for orig_datasets_file in orig_datasets:
        if '.txt' in orig_datasets_file:
            with open(orig_datasets_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    orig_datasets_lines.append(line)
        else:
            path = Path('/') / orig_datasets_file / "meta_data" / "episode_data_index.safetensors"
            ep_data_index = load_file(path)
            num_episodes = int(ep_data_index['to'].shape[0])
            line = orig_datasets_file + " " + ",".join([str(el) for el in list(range(num_episodes))])
            orig_datasets_lines.append(line)
    
    new_lines = []
    for line in orig_datasets_lines:
        fname = line.split(' ')[0]
        path = Path('/') / fname / "meta_data" / "episode_data_index.safetensors"
        ep_data_index = load_file(path)
        take_eps = []
        
        for episode_idx in line.split(' ')[1].split(','):
            episode_idx = int(episode_idx)
            dataset_repo_root = '/'
            dataset_repo_id = fname
            split = f"train[{int(ep_data_index['from'][episode_idx])}:{int(ep_data_index['to'][episode_idx])}]"
            
            old_ep_dataset = LeRobotDataset(dataset_repo_id, dataset_repo_root, split=split)
            
            # Check success
            success = False
            if 'next.reward' in old_ep_dataset.hf_dataset.features:
                final_rew = old_ep_dataset.hf_dataset['next.reward'][-1]
                if final_rew > rew_thresh:
                    success = True
            else:
                success = True
            
            if success:
                # Get observations
                inputs = old_ep_dataset.hf_dataset['observation.state'][:]
                inputs = torch.stack(inputs).to(device)
                
                with torch.inference_mode():
                    # Get PI0 embeddings
                    prefix_embs = pi0_model.embed_prefix(inputs.unsqueeze(0))
                    
                    # Classify using PI0 embeddings
                    pred = classifier_model(prefix_embs).squeeze(0)
                    
                    if pred.shape[0] > 1:
                        pred = pred.mean()
                    
                    if pred.item() > classifier_thresh:
                        take_eps.append(episode_idx)
        
        if len(take_eps) > 0:
            new_line = fname + " " + ",".join([str(el) for el in take_eps])
            new_lines.append(new_line)
    
    # Save filtered dataset
    os.makedirs(os.path.dirname(new_datasets_file), exist_ok=True)
    with open(new_datasets_file, 'w') as f:
        for nline in new_lines:
            f.write(nline + "\n")
```

## Usage Example

### 1. Training PI0-based Classifier

```python
from demo_score.dataset import PI0ClassifierDataset
from demo_score.models import PI0Classifier
from lerobot.common.policies.pi0.modeling_pi0 import PI0FlowMatching

# Load pre-trained PI0 model
pi0_model = PI0FlowMatching.from_pretrained("path/to/pi0/model")

# Create dataset with PI0 embeddings
dataset = PI0ClassifierDataset(
    data_dir="path/to/dataset",
    pi0_model=pi0_model,
    format='lerobot'
)

# Create classifier for PI0 embeddings
classifier = PI0Classifier(
    pi0_embed_dim=pi0_model.embed_dim,  # Get from PI0 model
    hidden_sizes=[128, 64]
)

# Train classifier
# ... training code here ...
```

### 2. Filtering Datasets

```python
from demo_score.filters.classifier_filter_pi0 import classifier_filter_pi0

# Filter datasets using PI0 embeddings
classifier_filter_pi0(
    orig_datasets=["path/to/original/dataset"],
    new_datasets_file="path/to/filtered/dataset.txt",
    pi0_model=pi0_model,
    classifier_model=trained_classifier,
    classifier_thresh=0.7
)
```

## Integration Steps Summary

1. **Add PI0 Model**: Include PI0FlowMatching implementation in the lerobot policies directory
2. **Extend Dataset Class**: Create PI0ClassifierDataset to handle PI0 embeddings
3. **Create PI0 Classifier**: Build a classifier that works with PI0 embeddings
4. **Implement Filter**: Create a PI0-specific filter for dataset curation
5. **Training Pipeline**: Train the classifier on PI0 embeddings
6. **Apply Filtering**: Use the trained classifier to filter datasets

## Key Considerations

### Embedding Dimensionality
- PI0 `prefix_embs` dimensions must match classifier input
- May need pooling or reshaping depending on PI0 output format

### Training Data
- Need successful/unsuccessful demonstrations for classifier training
- Consider using reward thresholds or manual labeling

### Performance Optimization
- Cache PI0 embeddings to avoid recomputation
- Use batch processing for large datasets
- Consider GPU memory usage for large embeddings

## Troubleshooting

### Common Issues

1. **PI0 Model Not Found**: Ensure PI0 implementation is properly added to lerobot policies
2. **Dimension Mismatch**: Check PI0 embedding dimensions and adjust classifier accordingly
3. **Memory Issues**: Use smaller batch sizes or implement embedding caching
4. **Performance**: Consider using lighter classifier architectures for real-time filtering

### Debugging Tips

- Print embedding shapes to verify compatibility
- Test on small datasets first
- Use visualization tools to understand embedding quality
- Monitor classifier training convergence

## Future Enhancements

1. **Multi-modal Integration**: Combine PI0 embeddings with other modalities
2. **Adaptive Thresholding**: Dynamic threshold adjustment based on dataset characteristics
3. **Ensemble Methods**: Combine multiple PI0 models for better filtering
4. **Online Learning**: Update classifier based on new demonstrations