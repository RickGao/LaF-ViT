import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (e.g., 'data/UTKFace').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []

        for filename in os.listdir(root_dir):
            if not (filename.endswith('.jpg') or filename.endswith('.png')):
                continue
            parts = filename.split('_')
            if len(parts) >= 4:
                self.images.append(filename)

        print(f"Found {len(self.images)} valid images in {root_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root_dir, filename)

        parts = filename.split('_')

        try:
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])
        except ValueError:

            print(f"Warning: Corrupt filename {filename}")
            age, gender, race = 0, 0, 0

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'age': torch.tensor(age, dtype=torch.float32),
            'gender': torch.tensor(gender, dtype=torch.long),
            'race': torch.tensor(race, dtype=torch.long),
            'filename': filename
        }


train_transforms = transforms.Compose([
    # scale=(0.95, 1.0):
    transforms.RandomResizedCrop(224, scale=(0.95, 1.0), ratio=(0.95, 1.05)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(degrees=10),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_dataloaders(data_dir, batch_size=32, train_split=0.8):
    from torch.utils.data import random_split

    full_dataset = UTKFaceDataset(data_dir, transform=None)

    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_dataset.dataset.transform = train_transforms
    train_ds = UTKFaceDataset(data_dir, transform=train_transforms)
    val_ds = UTKFaceDataset(data_dir, transform=val_transforms)

    indices = torch.randperm(len(train_ds)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = torch.utils.data.Subset(train_ds, train_indices)
    val_subset = torch.utils.data.Subset(val_ds, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
