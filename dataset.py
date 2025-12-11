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

        # 1. 遍历文件夹，建立文件列表
        # 我们只加载 valid 的文件 (包含3个下划线的文件名)
        for filename in os.listdir(root_dir):
            if not (filename.endswith('.jpg') or filename.endswith('.png')):
                continue

            # 文件名解析检查: ensure parsing won't crash later
            parts = filename.split('_')
            if len(parts) >= 4:  # 确保至少有 Age, Gender, Race, Date
                self.images.append(filename)

        print(f"Found {len(self.images)} valid images in {root_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root_dir, filename)

        # 2. 解析文件名获取 Labels
        # 格式: [age]_[gender]_[race]_[date].jpg
        parts = filename.split('_')

        try:
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])
        except ValueError:
            # 万一遇到文件名坏掉的情况，打印错误并返回一个 dummy 或者报错
            # 这里为了简单，我们假设 init 里的检查已经过滤了大部分错误
            print(f"Warning: Corrupt filename {filename}")
            age, gender, race = 0, 0, 0

        # 3. 加载图片 (使用 PIL)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
            # 返回一张全黑图防止 Crash，或者在这里 raise error
            image = Image.new('RGB', (224, 224))

        # 4. 应用变换 (Resize -> Tensor -> Norm)
        if self.transform:
            image = self.transform(image)

        # 5. 处理 Label 数据类型
        # Age: 用于回归 (Regression)，通常用 Float32
        # Gender/Race: 用于分类 (Classification)，必须用 Long (int64)
        return {
            'image': image,
            'age': torch.tensor(age, dtype=torch.float32),
            'gender': torch.tensor(gender, dtype=torch.long),
            'race': torch.tensor(race, dtype=torch.long),
            'filename': filename  # 方便后续 debug 或画图时知道是哪张图
        }


# ==========================================
# 定义 Transforms (预处理)
# ==========================================

# 训练集 Transform: 加入数据增强
train_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),  # 强制 224x224
    transforms.RandomHorizontalFlip(),  # 随机翻转，增加鲁棒性
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 标准归一化
])

# 测试/验证集 Transform: 只做 Resize 和 Normalize
val_transforms = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ==========================================
# 方便调用的 Helper Function
# ==========================================
def get_dataloaders(data_dir, batch_size=32, train_split=0.8):
    """
    自动切分训练集和验证集，并返回两个 DataLoader
    """
    from torch.utils.data import random_split

    # 加载完整数据集
    full_dataset = UTKFaceDataset(data_dir, transform=None)  # 先不加 transform

    # 计算切分数量
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 手动给切分后的 dataset 赋值 transform
    # 注意: subset.dataset 访问的是原始 dataset，这里有点 trick
    # 为了简单，通常我们在 Dataset 内部不区分 train/val transform，
    # 或者在这里这重写 dataset 的 transform 属性比较麻烦。
    # 
    # **简单做法**: 所有的都用 val_transforms (如果不做 heavy augmentation)
    # **标准做法**: 定义两个 Dataset 实例，一个给 train，一个给 val，但那样要读两遍文件列表。

    # 这里我们采用 "运行时覆盖" 的方式，或者简单点，直接在 Dataset 类里把 transform 写死
    # 但为了严谨，我们让 train_dataset 使用 train_transforms
    train_dataset.dataset.transform = train_transforms
    # 注意：这就意味着 val_dataset 也是 train_transforms (因为它们指向同一个父对象)
    # **修正方案**: 最好实例化两次

    train_ds = UTKFaceDataset(data_dir, transform=train_transforms)
    val_ds = UTKFaceDataset(data_dir, transform=val_transforms)

    # 重新做 split (需要固定 seed 保证两次 split 一样，或者直接简单粗暴如下)
    # 为了防止数据泄露，最稳妥的是：
    indices = torch.randperm(len(train_ds)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = torch.utils.data.Subset(train_ds, train_indices)
    val_subset = torch.utils.data.Subset(val_ds, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader