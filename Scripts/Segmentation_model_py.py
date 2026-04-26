import segmentation_models_pytorch as smp
import torchvision.transforms.v2 as T
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path


class XylemDataset(Dataset):
    #Detect any file format
    EXTS = {".tif", ".tiff", ".TIF", ".TIFF", ".png", ".PNG"}

    def __init__(self, img_dir, mask_dir, transform=None):
        self.imgs  = sorted(
            p for p in Path(img_dir).iterdir() if p.suffix in self.EXTS
        )
        self.masks = sorted(
            p for p in Path(mask_dir).iterdir() if p.suffix in self.EXTS
        )
        self.transform = transform

        assert len(self.imgs) > 0, f"No images found in {img_dir}"
        print(f"Images ({len(self.imgs)}) and masks ({len(self.masks)})")
        assert len(self.imgs) == len(self.masks), f"Quantity between images and maks does not match"
    
    @classmethod
    def from_lists(cls, img_paths, mask_paths, transform=None):
        obj = cls.__new__(cls)
        obj.imgs      = img_paths
        obj.masks     = mask_paths
        obj.transform = transform
        return obj

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.imgs[idx]), cv2.IMREAD_UNCHANGED)

        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        if mask.dtype == np.uint16:
            mask = (mask / 256).astype(np.uint8)
        mask = (mask > 127).astype(np.float32)

        out  = self.transform(image=img, mask=mask)
        img  = out["image"]
        mask = out["mask"].unsqueeze(0)
        return img, mask

# Albumentations
TARGET_SIZE = (1024, 1024)
train_transformations = A.Compose([
    A.PadIfNeeded(min_height=512, min_width=512,
                  border_mode=cv2.BORDER_REFLECT),
    A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1]),  # ← A.Resize, no T.Resize
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),tas
    A.ElasticTransform(p=0.3),
    A.CLAHE(clip_limit=3.0, p=0.5),
    A.GaussNoise(p=0.2),
    A.Normalize(),
    ToTensorV2(),
])

validate_transformations = A.Compose([
    A.PadIfNeeded(min_height=512, min_width=512,
                  border_mode=cv2.BORDER_REFLECT), 
    A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1]),  
    A.Normalize(),
    ToTensorV2(),
])

# Model

def build_model():
    return smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    )

# Loss 

dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss  = nn.BCEWithLogitsLoss()

def criterion(pred, target):
    return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)

# Metrics

def dice_score(pred_logits, target, threshold=0.5):
    """Dice coefficient sobre un batch. Devuelve valor entre 0 y 1."""
    pred = (torch.sigmoid(pred_logits) > threshold).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union        = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + 1) / (union + 1)).mean().item()

#Training

def train(
    img_dir, mask_dir,
    epochs=50, batch_size=4, lr=1e-4,
    val_split=0.2,              # 20% para validación
    save_path="xylem_unet.pth"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando: {device.upper()}")

    # Dataset completo — split en train/val
    full_ds  = XylemDataset(img_dir, mask_dir)   
    print("Found sizes:")
    for p in full_ds.imgs:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        print(f"  {p.name}: {img.shape}")
        n_val    = max(1, int(len(full_ds) * val_split))
        n_train  = len(full_ds) - n_val
    all_imgs  = sorted(p for p in Path(img_dir).iterdir()  if p.suffix in XylemDataset.EXTS)
    all_masks = sorted(p for p in Path(mask_dir).iterdir() if p.suffix in XylemDataset.EXTS)
    
    n_total = len(all_imgs)
    n_val   = max(1, int(n_total * val_split))

    n_train = n_total - n_val
    # split reproducible de rutas
    import random
    random.seed(42)
    indices   = list(range(n_total))
    random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    train_imgs  = [all_imgs[i]  for i in train_idx]
    train_masks = [all_masks[i] for i in train_idx]
    val_imgs    = [all_imgs[i]  for i in val_idx]
    val_masks   = [all_masks[i] for i in val_idx]

    train_ds = XylemDataset.from_lists(train_imgs, train_masks, transform=train_transformations)
    val_ds   = XylemDataset.from_lists(val_imgs,   val_masks,   transform=validate_transformations)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size,
                          shuffle=False, num_workers=0)

    model     = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=7, factor=0.5
    )

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks in train_dl:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_dl:
                imgs, masks = imgs.to(device), masks.to(device)
                preds     = model(imgs)
                val_loss += criterion(preds, masks).item()
                val_dice += dice_score(preds, masks)
        val_loss /= len(val_dl)
        val_dice /= len(val_dl)

        scheduler.step(val_loss)
        print(f"Epoch {epoch:3d}/{epochs} | train: {train_loss:.4f} | val: {val_loss:.4f} | dice: {val_dice:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  -> guardado (val_loss={best_val_loss:.4f})")

    print(f"\nListo. Modelo en: {save_path}")
    return model

# Inference

def predict_roi(roi_bgr: np.ndarray, model_path="xylem_unet.pth") -> np.ndarray:
    """Recibe ROI en BGR (OpenCV), devuelve máscara binaria uint8 del mismo tamaño."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    h, w = roi_bgr.shape[:2]

    if roi_bgr.dtype == np.uint16:
        roi_bgr = (roi_bgr / 256).astype(np.uint8)

    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    tf  = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])
    inp = tf(image=rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(inp))[0, 0].cpu().numpy()

    mask = (prob > 0.5).astype(np.uint8) * 255  # ✅ 255 para visualizar en OpenCV
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

# Entry point 

if __name__ == "__main__":
    IMG_DIR  = r"C:\Users\jandr\OneDrive - Universidad del rosario\Gui_xylem\ROIs_all"
    MASK_DIR = r"C:\Users\jandr\OneDrive - Universidad del rosario\Gui_xylem\Masks_all"

    train(
        img_dir   = IMG_DIR,
        mask_dir  = MASK_DIR,
        epochs    = 150,
        batch_size= 4,
        lr        = 1e-4,
        val_split = 0.2,
        save_path = "all_unet.pth"
    )
