import os
import pydicom
from PIL import Image
import numpy as np

# Пути
DICOM_DIR = "data/dicomImages"
OUTPUT_DIR = "data/Images"

# Параметры конвертации
TARGET_SIZE = (512, 512)  # Можно изменить на (256, 256) при нехватке памяти
OUTPUT_FORMAT = "PNG"  # или "JPEG" (для JPEG лучше использовать grayscale и качество 95)

def convert_dicom_to_image():
    # Создаём папку для результатов
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Получаем все .dcm файлы
    dicom_files = [f for f in os.listdir(DICOM_DIR) if f.lower().endswith('.dcm')]
    print(f"Найдено {len(dicom_files)} DICOM-файлов.")

    for i, filename in enumerate(dicom_files):
        dicom_path = os.path.join(DICOM_DIR, filename)
        try:
            # Чтение DICOM
            ds = pydicom.dcmread(dicom_path)
            pixel_array = ds.pixel_array

            # Нормализация: масштабируем до 0-255
            if pixel_array.max() != pixel_array.min():
                img_normalized = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            else:
                img_normalized = np.zeros_like(pixel_array, dtype=np.uint8)

            # Преобразуем в PIL Image
            img = Image.fromarray(img_normalized)

            # Изменение размера
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

            # Формируем имя выходного файла
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}.{OUTPUT_FORMAT.lower()}")

            # Сохраняем
            img.save(output_path, format=OUTPUT_FORMAT, optimize=True, quality=95 if OUTPUT_FORMAT == "JPEG" else None)

            print(f"[{i+1}/{len(dicom_files)}] Сохранено: {output_path}")

        except Exception as e:
            print(f"[Ошибка] Не удалось обработать {filename}: {e}")

if __name__ == "__main__":
    convert_dicom_to_image()
    print("Конвертация завершена.")