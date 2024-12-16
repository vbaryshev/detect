import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T
from shapely.geometry import Polygon
import json
from torch.utils.data import Dataset
import os

class CustomDataset(Dataset):
    def __init__(self, images_dir, annotations, transforms=None):
        """
        Инициализация датасета.

        :param images_dir: Путь к каталогу с изображениями.
        :param annotations: Словарь с аннотациями.
        :param transforms: Трансформации для изображений.
        """
        self.images_dir = images_dir
        self.annotations = annotations
        self.transforms = transforms
        self.image_files = list(annotations.keys())
        
        # Создание сопоставления между классами и индексами
        self.class_to_idx = self._create_class_mapping()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    
    def _create_class_mapping(self):
        """
        Создание словаря для преобразования классов в индексы.
        """
        classes = set()
        for ann in self.annotations.values():
            classes.update(ann['labels'])
        class_to_idx = {cls: idx for idx, cls in enumerate(sorted(classes), start=1)}
        class_to_idx["__background__"] = 0  # Фоновый класс
        return class_to_idx

    def __getitem__(self, idx):
        """
        Получение элемента по индексу.

        :param idx: Индекс элемента.
        :return: Кортеж (изображение, цель).
        """
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        img = Image.open(img_path).convert("RGB")
        ann = self.annotations[img_filename]

        boxes = torch.as_tensor(ann['boxes'], dtype=torch.float32)
        labels = torch.as_tensor([self.class_to_idx[label] for label in ann['labels']], dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        """
        Возвращает длину датасета.
        """
        return len(self.image_files)


def load_annotations(csv_file):
    """
    Загрузка аннотаций из CSV-файла.

    :param csv_file: Путь к CSV-файлу с аннотациями.
    :return: Словарь с аннотациями.
    """
    df = pd.read_csv(csv_file)
    
    annotations = {}
    for _, row in df.iterrows():
        filename = row['#filename'] if '#filename' in row else row['filename']
        
        # Парсинг JSON-подобных строк
        shape_attrs = json.loads(row['region_shape_attributes'].replace("'", '"'))
        region_attrs = json.loads(row['region_attributes'].replace("'", '"'))
        
        object_type = region_attrs.get('type', 'unknown')
        
        if shape_attrs['name'] != 'polygon':
            continue  # Пропуск, если форма не полигон

        all_x = shape_attrs['all_points_x']
        all_y = shape_attrs['all_points_y']
        
        # Создание списка точек полигона
        polygon_points = list(zip(all_x, all_y))
        
        # Создание объекта Polygon для вычисления bounding box
        polygon = Polygon(polygon_points)
        xmin, ymin, xmax, ymax = polygon.bounds
        
        if filename not in annotations:
            annotations[filename] = {
                'boxes': [],
                'labels': []
            }
        
        annotations[filename]['boxes'].append([xmin, ymin, xmax, ymax])
        annotations[filename]['labels'].append(object_type)
        
    return annotations


def get_transform(train):
    """
    Получение трансформаций для данных.

    :param train: Флаг, является ли режим обучения.
    :return: Композиция трансформаций.
    """
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
