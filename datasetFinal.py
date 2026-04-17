import io
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import pandas as pd
from datasets import load_dataset


def decode_image(cell) -> Image.Image:
    if isinstance(cell, bytes):
        return Image.open(io.BytesIO(cell)).convert("RGB")

    if isinstance(cell, dict):
        if "bytes" in cell and cell["bytes"] is not None:
            return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        if "path" in cell and cell["path"] is not None:
            return Image.open(cell["path"]).convert("RGB")

    raise ValueError(f"Cannot decode image from type {type(cell)}")



def detect_columns(df: pd.DataFrame):
    image_candidates = ["image", "img", "pixel_values", "image_bytes", "file"]
    label_candidates = ["label", "target", "fake", "is_fake", "class", "y"]

    cols_lower = {c.lower(): c for c in df.columns}

    image_col = next((cols_lower[c] for c in image_candidates if c in cols_lower), None)
    label_col = next((cols_lower[c] for c in label_candidates if c in cols_lower), None)

    if image_col is None or label_col is None:
        raise ValueError(f"Could not detect columns. Found: {list(df.columns)}")

    print(f"[Dataset] Detected → image: '{image_col}', label: '{label_col}'")
    return image_col, label_col



class OpenFakeDataset(Dataset):
    def __init__(self, df, image_col, label_col, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            image = decode_image(row[self.image_col])
        except Exception as e:
            print(f"[Warning] Skipping bad image at index {idx}")
            return self.__getitem__((idx + 1) % len(self.df))

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)

        return image, label



def get_vit_transforms(train=True):
    norm = transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])

    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            norm,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            norm,
        ])


def get_efficientnet_transforms(train=True):
    norm = transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])

    if train:
        return transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.RandomCrop(240),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
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
    dataset_name: str,
    model_name: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    data_files : list = None
):

    print(f"[Dataset] Loading from Hugging Face: {dataset_name}")

    # dataset = load_dataset(
    #     "parquet",
    #     data_files="hf://datasets/ComplexDataLab/OpenFake/data/test-*.parquet",
    #     split="train"
    # )
    
    dataset = load_dataset(
        "parquet",
        # data_files=[
        #         "hf://datasets/ComplexDataLab/OpenFake/data/test-00000-of-00007.parquet",
        #         "hf://datasets/ComplexDataLab/OpenFake/data/test-00001-of-00007.parquet",
        #     ],
        data_files=data_files,
        split="train"
    )
    df = dataset.to_pandas()

    print(f"[Dataset] Loaded {len(df)} rows")


    image_col, label_col = detect_columns(df)


    df[label_col] = df[label_col].astype(str).str.lower().str.strip()
    df[label_col] = df[label_col].map({"real": 0, "fake": 1})

    if df[label_col].isnull().any():
        raise ValueError("Unmapped labels found!")

    class_counts = df[label_col].value_counts().to_dict()
    print(f"[Dataset] Real: {class_counts.get(0,0)} | Fake: {class_counts.get(1,0)}")


    n = len(df)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx, test_idx = random_split(
        range(n), [n_train, n_val, n_test], generator=generator
    )

    train_df = df.iloc[list(train_idx)]
    val_df   = df.iloc[list(val_idx)]
    test_df  = df.iloc[list(test_idx)]

    print(f"[Dataset] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    if model_name.lower() == "vit":
        train_tf = get_vit_transforms(True)
        eval_tf  = get_vit_transforms(False)
    else:
        train_tf = get_efficientnet_transforms(True)
        eval_tf  = get_efficientnet_transforms(False)


    train_ds = OpenFakeDataset(train_df, image_col, label_col, train_tf)
    val_ds   = OpenFakeDataset(val_df,   image_col, label_col, eval_tf)
    test_ds  = OpenFakeDataset(test_df,  image_col, label_col, eval_tf)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader, class_counts