import os
import random
import time
import copy
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import openpyxl  # нужен для чтения .xlsx (pandas использует как движок)

# =========================
# Пути и базовые проверки
# =========================
XLS_PATH = "data/point.xlsx"  # Файл с метками (.xlsx)
IMG_DIR = "data/Images"  # Папка с изображениями

if not os.path.exists(XLS_PATH):
    raise FileNotFoundError(f"❌ Файл с метками не найден: {XLS_PATH}")
if not os.path.exists(IMG_DIR):
    raise FileNotFoundError(f"❌ Папка с изображениями не найдена: {IMG_DIR}")

print(f"✅ Найден файл меток: {XLS_PATH}")
print(f"✅ Найдена папка изображений: {IMG_DIR}")


# =========================
# Фиксация случайности
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# =========================
# Трансформации
# =========================
# Попробуем взять нормализацию из весов модели; если не получится — используем ImageNet-стандарт
try:
    from torchvision.models import ResNet18_Weights

    weights = ResNet18_Weights.DEFAULT
    imagenet_mean = weights.meta["mean"]
    imagenet_std = weights.meta["std"]
except Exception:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.Resize((512, 512)),
    # Лёгкая аугментация (можно убрать, если это медицинские изображения и нужна строгая консистентность)
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

transform_val = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])


# =========================
# Кастомный Dataset
# =========================
class ClassificationDataset(Dataset):
    def __init__(self, img_dir, xls_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        try:
            df = pd.read_excel(xls_path, engine="openpyxl")
        except Exception as e:
            raise RuntimeError(f"❌ Не удалось прочитать Excel-файл: {e}")

        required_cols = ['dicom_name', 'Эксперт 1', 'Эксперт 2', 'Эксперт 3']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"❌ В Excel-файле нет столбца: '{col}'. Есть: {list(df.columns)}")

        # Убираем строки с пропусками
        df = df.dropna(subset=required_cols).copy()

        # Приводим оценки к int
        df[['Эксперт 1', 'Эксперт 2', 'Эксперт 3']] = df[['Эксперт 1', 'Эксперт 2', 'Эксперт 3']].astype(int)

        self.img_names = []
        self.labels = []

        for _, row in df.iterrows():
            votes = [row['Эксперт 1'], row['Эксперт 2'], row['Эксперт 3']]

            # Если хотя бы один эксперт поставил -1 — исключаем
            if -1 in votes:
                continue

            # Мажоритарное голосование
            majority_label = Counter(votes).most_common(1)[0][0]

            self.img_names.append(str(row['dicom_name']))
            self.labels.append(int(majority_label))

        if len(self.img_names) == 0:
            raise RuntimeError("❌ После фильтрации не осталось изображений. Проверьте метки и значения -1.")

        print(f"✅ Датасет создан: включено {len(self.img_names)} изображений (без меток -1)")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        label = self.labels[idx]

        # Ищем файл по возможным расширениям
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(self.img_dir, img_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        # Если не нашли — пробуем имя как есть (вдруг уже с расширением)
        if img_path is None:
            img_path = os.path.join(self.img_dir, img_name)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"❌ Изображение не найдено: {img_path}")

        # Загрузка изображения
        try:
            image = Image.open(img_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise IOError(f"❌ Ошибка при загрузке изображения {img_path}: {e}")

        # Трансформации
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# =========================
# Создание датасета и быстрая проверка визуализаций
# =========================
full_dataset = ClassificationDataset(IMG_DIR, XLS_PATH, transform=transform_val)  # для превью без аугментации

# Баланс классов
counts = Counter(full_dataset.labels)
print(f"📊 Распределение классов: {dict(counts)} (0 = 'Нет грыжи', 1 = 'Есть грыжа')")

# Тест: показать первый батч из 4 изображений
print("\n🔍 ПРОВЕРКА ДАТАСЕТА: показываем первый батч из 4 изображений")
_preview_loader = DataLoader(full_dataset, batch_size=4, shuffle=True, num_workers=0)
images, labels = next(iter(_preview_loader))
print(f"📌 Форма тензора изображений: {images.shape}")  # [4, 3, 512, 512]
print(f"📌 Метки: {labels.tolist()}")


# Де-нормализация для показа
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return tensor * std + mean


fig, axes = plt.subplots(1, 4, figsize=(16, 4))
class_names = {0: "Нет грыжи", 1: "Есть грыжа"}
imgs_to_show = denormalize(images.clone(), imagenet_mean, imagenet_std)

for i in range(min(4, images.size(0))):
    img = imgs_to_show[i].permute(1, 2, 0).numpy().clip(0, 1)
    axes[i].imshow(img)
    axes[i].set_title(f"{class_names.get(labels[i].item(), labels[i].item())}", fontsize=10)
    axes[i].axis("off")

plt.tight_layout()
plt.show()

# =========================
# Разделение на train/val и DataLoader'ы
# =========================
# 80/20 сплит
train_ratio = 0.8
train_size = int(train_ratio * len(full_dataset))
val_size = len(full_dataset) - train_size

# Важно: для обучения используем разные трансформации (train/val)
# Переопределим transform через Subset с доступом к исходному датасету
# Проще — создать два новых датасета с теми же именами/метками, но разными трансформами
indices = list(range(len(full_dataset)))
random.shuffle(indices)
train_indices = indices[:train_size]
val_indices = indices[train_size:]


# Вспомогательная "обёртка" над базовым датасетом с другим transform
class WrappedDataset(Dataset):
    def __init__(self, base_dataset: ClassificationDataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        # Берём исходный путь и метку косвенно: переиспользуем логику base, но меняем transform на лету
        img_name = self.base.img_names[idx]
        label = self.base.labels[idx]

        # Повтор той же логики загрузки файла, что и в base (чтобы не дублировать, можно было бы вынести метод)
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(self.base.img_dir, img_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            img_path = os.path.join(self.base.img_dir, img_name)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"❌ Изображение не найдено: {img_path}")

        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


train_dataset = WrappedDataset(full_dataset, train_indices, transform_train)
val_dataset = WrappedDataset(full_dataset, val_indices, transform_val)

# DataLoader'ы
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = dict(num_workers=0, pin_memory=True) if use_cuda else dict(num_workers=0)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, **kwargs)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, **kwargs)

print(f"\n🧩 Размеры: train={len(train_dataset)}, val={len(val_dataset)}")
print(f"🖥️ Устройство: {device}")


# =========================
# Модель
# =========================
def build_model(num_classes=2):
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    # Замена головой под нашу задачу
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


model = build_model(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# =========================
# Цикл обучения
# =========================
def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10):
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_start = time.time()

        for phase, loader in [('train', train_loader), ('val', val_loader)]:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = inputs.size(0)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels).item()
                total += batch_size

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print(f"{phase:>5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # Сохраняем лучшую по валидации
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())

        print(f"⏱️ Время эпохи: {time.time() - epoch_start:.1f} с")

    print(f"\n🏆 Лучшая val Acc: {best_acc:.4f}")
    model.load_state_dict(best_wts)
    return model


model = train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10)

# =========================
# Сохранение модели
# =========================
os.makedirs("outputs", exist_ok=True)
save_path = os.path.join("outputs", "cxr_binary_resnet18.pth")
torch.save(model.state_dict(), save_path)
print(f"✅ Модель сохранена: {save_path}")
