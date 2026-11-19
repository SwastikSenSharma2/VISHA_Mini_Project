import torch
import torch.nn as nn
import os

# Define the dummy model that matches the one used in app/main.py
class DummyBinaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

model = DummyBinaryModel()
# Path to the checkpoint (relative to this script location)
ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ml', 'artifacts', 'binary_classifier_latest.pt'))
os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
torch.save(model.state_dict(), ckpt_path)
print('✓ Dummy binary checkpoint written to:', ckpt_path)
