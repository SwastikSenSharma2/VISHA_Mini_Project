import torch
import torch.nn as nn
import os

# Ensure the artifacts directory exists
artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'artifacts'))
os.makedirs(artifacts_dir, exist_ok=True)

# Simple placeholder model
temp_model = nn.Linear(1, 1)

# Save state dict
torch.save(temp_model.state_dict(), os.path.join(artifacts_dir, 'binary_classifier_latest.pt'))
print('Dummy binary checkpoint created at', os.path.join(artifacts_dir, 'binary_classifier_latest.pt'))
