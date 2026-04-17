import io
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


def decode_image(cell) -> Image.Image:

    if isinstance(cell, bytes):
        return Image.open(io.BytesIO(cell)).convert("RGB")

    if isinstance(cell, dict):
        if "bytes" in cell and cell["bytes"] is not None:
            return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        if "path" in cell and cell["path"] is not None:
            return Image.open(cell["path"]).convert("RGB")

    raise ValueError(
        f"Cannot decode image from cell of type {type(cell)}.\n"
        f"Value preview: {str(cell)[:120]}\n"
        f"Please check your parquet file structure."
    )


def detect_columns(df: pd.DataFrame):

    image_candidates = ["image", "img", "pixel_values", "image_bytes", "file"]
    label_candidates = ["label", "target", "fake", "is_fake", "class", "y"]

    cols_lower = {c.lower(): c for c in df.columns}

    image_col = next((cols_lower[c] for c in image_candidates if c in cols_lower), None)
    label_col = next((cols_lower[c] for c in label_candidates if c in cols_lower), None)

    if image_col is None or label_col is None:
        raise ValueError(
            f"Could not auto-detect image/label columns.\n"
            f"Columns found: {list(df.columns)}\n"
            f"Manually set IMAGE_COL / LABEL_COL at the top of dataset.py."
        )

    print(f"[Dataset] Detected → image col: '{image_col}', label col: '{label_col}'")
    return image_col, label_col

class OpenFakeDataset(Dataset):

    def __init__(self, df: pd.DataFrame, image_col: str, label_col: str, transform=None):
        self.df        = df.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # image = decode_image(row[self.image_col])
        try:
            image = decode_image(row[self.image_col])
        except Exception as e:
            print(f"[Warning] Skipping bad image at index {idx}")
            return self.__getitem__((idx + 1) % len(self.df))

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)

        return image, label


def get_vit_transforms(train: bool = True):
    
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            norm,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            norm,
        ])


def get_efficientnet_transforms(train: bool = True):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.RandomCrop(240),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            norm,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.CenterCrop(240),
            transforms.ToTensor(),
            norm,
        ])


def build_dataloaders(
    parquet_path: str,
    model_name: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
):
    


    print(f"[Dataset] Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"[Dataset] {len(df)} rows | columns: {list(df.columns)}")

    image_col, label_col = detect_columns(df)
    
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()

    df[label_col] = df[label_col].map({
        "real": 0,
        "fake": 1
    })


    class_counts = df[label_col].value_counts().to_dict()
    print(f"[Dataset] Real (0): {class_counts.get(0,0)} | Fake (1): {class_counts.get(1,0)}")


    n       = len(df)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_val - n_test


    generator = torch.Generator().manual_seed(seed)
    train_sub, val_sub, test_sub = random_split(
        range(n), [n_train, n_val, n_test], generator=generator
    )


    train_df = df.iloc[list(train_sub)]
    val_df   = df.iloc[list(val_sub)]
    test_df  = df.iloc[list(test_sub)]
    print(f"[Dataset] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


    if model_name.lower() == "vit":
        train_tf = get_vit_transforms(train=True)
        eval_tf  = get_vit_transforms(train=False)
    elif model_name.lower() == "efficientnet":
        train_tf = get_efficientnet_transforms(train=True)
        eval_tf  = get_efficientnet_transforms(train=False)
    else:
        raise ValueError(f"model_name must be 'vit' or 'efficientnet', got: '{model_name}'")

    train_ds = OpenFakeDataset(train_df, image_col, label_col, transform=train_tf)
    val_ds   = OpenFakeDataset(val_df,   image_col, label_col, transform=eval_tf)
    test_ds  = OpenFakeDataset(test_df,  image_col, label_col, transform=eval_tf)


    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader, class_counts