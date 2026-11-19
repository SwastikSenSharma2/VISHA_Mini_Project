import argparse
import os
from pathlib import Path
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------------
# Simple dataset that reads the CSV manifest and loads images
class TaxonomyDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row['image_path'])
        label = row['label']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(label)

# Simple 2‑layer CNN (just for demo – replace with your real model)
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # After two stride‑2 convs, 224 -> 56 (approx)
        self.fc = nn.Linear(32 * 56 * 56, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='CSV with image_path,label')
    parser.add_argument('--binary-checkpoint', required=True, help='Path to binary model')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    # Load manifest
    ds = TaxonomyDataset(args.manifest,
                         transform=transforms.Compose([
                             transforms.Resize((224, 224)),
                             transforms.ToTensor(),
                         ]))
    # Simple train/val split (80/20) – ensure at least one val sample
    n = len(ds)
    val_n = max(1, int(0.2 * n))
    train_n = n - val_n
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_n, val_n])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)

    num_classes = ds.df['label'].nunique()
    model = TinyCNN(num_classes).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0
    checkpoint_dir = Path('ml') / 'runs' / f'taxonomy_{args.epochs}_{args.batch_size}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(args.device), lbls.to(args.device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(args.device), lbls.to(args.device)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total   += lbls.size(0)
        val_acc = correct / total if total else 0.0
        print(f'Epoch {epoch}/{args.epochs} – val_acc: {val_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = checkpoint_dir / 'best_model.pt'
            torch.save(model.state_dict(), ckpt_path)
            print(f'Best checkpoint saved to {ckpt_path}')

    print('Training finished.')

if __name__ == '__main__':
    main()
