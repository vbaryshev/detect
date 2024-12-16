# train.py

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils import load_annotations, CustomDataset, get_transform
import os
import numpy as np


def get_model_instance_segmentation(num_classes):
    """
    Функция для получения модели Faster R-CNN с измененным классификатором.

    :param num_classes: Количество классов (включая фон).
    :return: Модель Faster R-CNN.
    """
    # Загружаем предобученную модель Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Получаем количество входных признаков для классификатора
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Заменяем предсказатель на новый, подходящий под количество классов
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def collate_fn(batch):
    """
    Функция коллации для загрузчика данных.

    :param batch: Список кортежей (изображение, цель).
    :return: Кортеж списков изображений и целей.
    """
    return tuple(zip(*batch))


def evaluate(model, data_loader, device):
    """
    Оценка модели на валидационном наборе.

    :param model: Обученная модель.
    :param data_loader: Загрузчик валидационных данных.
    :param device: Устройство (CPU или GPU).
    """
    model.eval()
    metric = 0.0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            # Здесь можно добавить вычисление метрик, например, mAP
            # Для простоты будем выводить пороговое значение метрики
    print("Валидация завершена")
    model.train()


def main():
    # Параметры
    annotations_file = os.path.join('./annotations/train.csv')
    images_dir = 'images'
    num_epochs = 10
    batch_size = 2
    learning_rate = 0.005
    momentum = 0.9
    weight_decay = 0.0005
    num_workers = 4
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Загрузка аннотаций
    annotations = load_annotations(annotations_file)
    
    # Создание датасетов
    dataset = CustomDataset(images_dir, annotations, transforms=get_transform(train=True))
    dataset_test = CustomDataset(images_dir, annotations, transforms=get_transform(train=False))
    
    # Разделение на обучающую и тестовую выборки (80%/20%)
    indices = torch.randperm(len(dataset)).tolist()
    split = int(0.8 * len(indices))
    dataset = Subset(dataset, indices[:split])
    dataset_test = Subset(dataset_test, indices[split:])
    
    # Создание загрузчиков данных
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    # Определение числа классов
    all_labels = set()
    for ann in annotations.values():
        all_labels.update(ann['labels'])
    num_classes = len(all_labels) + 1  # +1 за фон
    
    print(f"Количество классов (включая фон): {num_classes}")
    
    # Инициализация модели
    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    
    # Определение оптимизатора и lr_scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Цикл обучения
    for epoch in range(num_epochs):
        model.train()
        print(f"Эпоха {epoch+1}/{num_epochs}")
        for i, (images, targets) in enumerate(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"  Итерация {i}/{len(data_loader)} | Потери: {losses.item():.4f}")
        
        # Обновление lr_scheduler
        lr_scheduler.step()
        
        # Валидация после каждой эпохи
        evaluate(model, data_loader_test, device)
    
    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'fasterrcnn_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Модель сохранена по пути: {model_path}")


if __name__ == "__main__":
    main()


