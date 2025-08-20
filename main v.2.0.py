# ─────────────────────  ENV & FIRST IMPORTS  ─────────────────────
import os, warnings

os.environ["TORCH_DISABLE_NNPACK"] = "1"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore", message="Error fetching version info")

from typing import List, Dict
import random, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch

torch.backends.nnpack.enabled = False

# ─────────────────────  OTHER IMPORTS  ─────────────────────
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt

# ─────────────────────  CONFIG  ─────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEM = DEVICE.type == "cuda"
N_WORKERS = 4 if DEVICE.type == "cuda" else 0

# --- Пути к данным ---
ROOT = Path(".")  # текущая директория
DATA_DIR = ROOT / "data"
PNG_DIR = DATA_DIR / "Images"  # папка с изображениями
CSV_FILE = DATA_DIR / "point.csv"  # файл с оценками

IMG_SIZE = 256
MODEL_NAME = "efficientnet_b6"
EPOCHS = 60
MIN_EPOCHS = 20
EARLY_STOP = 10
BATCH_TRAIN = 16
BATCH_VAL = 4
LR = 2e-4
CLIP_NORM = 1.0

# ─────────────────────  AUGMENTATIONS  ─────────────────────
train_tf = A.Compose([
    A.ToFloat(max_value=255.0),
    A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    A.RandomResizedCrop((IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ToTensorV2(transpose_mask=True)
])

valid_tf = A.Compose([
    A.ToFloat(max_value=255.0),
    ToTensorV2(transpose_mask=True)
])


# ─────────────────────  DATASET  ─────────────────────
class PNGDataset(Dataset):
    def __init__(self, root: Path, df_split: pd.DataFrame, col: str, transform):
        self.root = root
        self.names = df_split["dicom_name"].tolist()
        self.labels = df_split[col].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        if not name.lower().endswith(".png"):
            name += ".png"
        img_path = self.root / name
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {img_path}") from e
        augmented = self.transform(image=img)
        image = augmented["image"]
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return image, label


# ─────────────────────  UTILS  ─────────────────────
def safe_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except ValueError:
        return float("nan")


def full_metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pred = (y_prob >= thr).astype(int)
    TP = ((pred == 1) & (y_true == 1)).sum()
    TN = ((pred == 0) & (y_true == 0)).sum()
    FP = ((pred == 1) & (y_true == 0)).sum()
    FN = ((pred == 0) & (y_true == 1)).sum()
    acc = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
    se = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    sp = TN / (TN + FP) if (TN + FP) > 0 else float("nan")
    return acc, se, sp


def load_data(csv_path: Path) -> pd.DataFrame:
    """Загружает point.csv, очищает и возвращает DataFrame"""
    df = pd.read_csv(csv_path, sep=";", engine="python")  # ; — разделитель
    print(f"Загружено {len(df)} строк из {csv_path}")

    # Проверка нужных колонок
    required = ["dicom_name", "Эксперт 1", "Эксперт 2", "Эксперт 3"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Нет колонки: {col} в файле {csv_path}")

    # Приводим оценки экспертов к числу
    for col in ["Эксперт 1", "Эксперт 2", "Эксперт 3"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def make_splits(df: pd.DataFrame, col: str):
    """Фильтруем только 0 и 1, исключаем -1"""
    clean = df[df[col].isin([0, 1])][["dicom_name", col]].astype({col: int})
    if len(clean) < 2 or clean[col].nunique() < 2:
        return None  # недостаточно данных или один класс
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_val_idx, test_idx = next(sss.split(clean, clean[col]))
    train_val, test = clean.iloc[train_val_idx], clean.iloc[test_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, val_idx = next(sss2.split(train_val, train_val[col]))
    train, val = train_val.iloc[train_idx], train_val.iloc[val_idx]

    return train, val, test


# ─────────────────────  PLOT  ─────────────────────
def plot_experiment_histories(experiment_histories, experiments_to_plot, metrics_to_plot=['loss', 'auc']):
    plt.figure(figsize=(12, 5 * len(metrics_to_plot)))
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(len(metrics_to_plot), 1, i)
        if metric == 'loss':
            tr, vl, ylabel, title = 'train_loss', 'val_loss', 'Loss', 'Train / Val Loss'
        else:
            tr, vl, ylabel, title = 'train_auc', 'val_auc', 'ROC-AUC', 'Train / Val ROC-AUC'
        for name in experiments_to_plot:
            if name in experiment_histories:
                h = experiment_histories[name]
                epochs = range(1, len(h[tr]) + 1)
                plt.plot(epochs, h[tr], '--', label=f'{name} (Train)')
                plt.plot(epochs, h[vl], '-', label=f'{name} (Val)')
                if metric == 'auc':
                    best = h.get('best_val_auc', float('nan'))
                    plt.annotate(f'{best:.4f}', xy=(1, 0), xycoords='axes fraction',
                                 xytext=(-5, 5), textcoords='offset points',
                                 ha='right', va='bottom', fontsize=10)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────  TRAIN ONE EXPERT  ─────────────────────
def train_one(df: pd.DataFrame, col: str) -> Dict:
    print(f"\nОбработка эксперта: {col}")
    split = make_splits(df, col)
    if split is None:
        print(f"[{col}] Пропущен — недостаточно валидных меток (0/1)")
        return {}

    tr_df, val_df, ts_df = split
    print(f"[{col}] Размеры: train={len(tr_df)}, val={len(val_df)}, test={len(ts_df)}")

    # Балансировка классов
    counts = np.bincount(tr_df[col])
    weights = 1.0 / counts
    sample_weights = [weights[label] for label in tr_df[col]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    tr_dl = DataLoader(PNGDataset(PNG_DIR, tr_df, col, train_tf),
                       batch_size=BATCH_TRAIN, sampler=sampler,
                       num_workers=N_WORKERS, pin_memory=PIN_MEM)
    val_dl = DataLoader(PNGDataset(PNG_DIR, val_df, col, valid_tf),
                        batch_size=BATCH_VAL, shuffle=False,
                        num_workers=N_WORKERS, pin_memory=PIN_MEM)
    ts_dl = DataLoader(PNGDataset(PNG_DIR, ts_df, col, valid_tf),
                       batch_size=BATCH_VAL, shuffle=False,
                       num_workers=N_WORKERS, pin_memory=PIN_MEM)

    # Модель
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
    best_auc = 0.0
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        y_true_train, y_pred_train = [], []
        for x, y in tqdm(tr_dl, desc=f"[{col}] Ep {epoch}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x).clamp(-15, 15)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            train_loss += loss.item()
            y_true_train.extend(y.cpu().numpy().ravel())
            y_pred_train.extend(torch.sigmoid(logits).detach().cpu().numpy().ravel())
        train_loss /= len(tr_dl)
        train_auc = safe_auc(y_true_train, y_pred_train)

        # --- VALIDATION ---
        model.eval()
        val_loss = 0.0
        y_true_val, y_pred_val = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x).clamp(-15, 15)
                val_loss += criterion(logits, y).item()
                y_true_val.extend(y.cpu().numpy().ravel())
                y_pred_val.extend(torch.sigmoid(logits).cpu().numpy().ravel())
        val_loss /= len(val_dl)
        val_auc = safe_auc(y_true_val, y_pred_val)
        scheduler.step()

        # Лог
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        print(f"Ep {epoch:02d}: trL={train_loss:.3f} AUC={train_auc:.3f} | vlL={val_loss:.3f} AUC={val_auc:.3f}")

        # Сохранение лучшей модели
        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            torch.save(model.state_dict(), ROOT / f"{col.replace(' ', '_')}_best.pth")
        elif epoch >= MIN_EPOCHS:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                print(f"Ранняя остановка на эпохе {epoch}")
                break

    # --- TEST ---
    model.load_state_dict(torch.load(ROOT / f"{col.replace(' ', '_')}_best.pth", map_location=DEVICE))
    model.eval()
    y_true_test, y_pred_test = [], []
    with torch.no_grad():
        for x, y in ts_dl:
            y_true_test.extend(y.numpy().ravel())
            y_pred_test.extend(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy().ravel())
    test_auc = safe_auc(y_true_test, y_pred_test)
    acc, se, sp = full_metrics(y_true_test, y_pred_test)
    print(f"[{col}] TEST AUC={test_auc:.3f} Acc={acc:.3f} Se={se:.3f} Sp={sp:.3f}")

    history['best_val_auc'] = best_auc
    history['num_epochs'] = len(history['train_loss'])
    return history


# ─────────────────────  MAIN  ─────────────────────
def main():
    # Загрузка данных
    df = load_data(CSV_FILE)
    expert_cols = ["Эксперт 1", "Эксперт 2", "Эксперт 3"]

    experiment_histories = {}
    for col in expert_cols:
        hist = train_one(df, col)
        if hist:
            experiment_histories[col] = hist

    # Сохранение истории
    with open("training_histories.pkl", "wb") as f:
        pickle.dump(experiment_histories, f)
    print("История обучения сохранена в training_histories.pkl")

    # Графики
    plot_experiment_histories(
        experiment_histories,
        experiments_to_plot=list(experiment_histories.keys()),
        metrics_to_plot=['loss', 'auc']
    )


if __name__ == "__main__":
    main()