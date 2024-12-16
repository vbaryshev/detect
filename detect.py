# detect.py

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from utils import CustomDataset, get_transform, load_annotations
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import os

def load_model(model_path, num_classes, device):
    """
    Загрузка обученной модели.

    :param model_path: Путь к файлу модели.
    :param num_classes: Количество классов.
    :param device: Устройство (CPU или GPU).
    :return: Загруженная модель.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, image_path, transforms, device, threshold=0.5):
    """
    Выполнение предсказания на одном изображении.

    :param model: Обученная модель.
    :param image_path: Путь к изображению.
    :param transforms: Трансформации для изображения.
    :param device: Устройство (CPU или GPU).
    :param threshold: Порог уверенности для отображения предсказаний.
    :return: Предсказания модели.
    """
    img = Image.open(image_path).convert("RGB")
    img_transformed = transforms(img).to(device)
    with torch.no_grad():
        prediction = model([img_transformed])
    return prediction


def visualize_prediction(image_path, prediction, idx_to_class, threshold=0.5):
    """
    Визуализация предсказаний на изображении.

    :param image_path: Путь к изображению.
    :param prediction: Предсказания модели.
    :param idx_to_class: Словарь сопоставления индексов классов с названиями.
    :param threshold: Порог уверенности для отображения предсказаний.
    """
    img = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    ax = plt.gca()

    for box, score, label in zip(prediction[0]['boxes'], prediction[0]['scores'], prediction[0]['labels']):
        if score >= threshold:
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin, f"{idx_to_class[label.item()]}: {score:.2f}", 
                    bbox=dict(facecolor='yellow', alpha=0.5), fontsize=12, color='black')
    
    plt.axis('off')
    plt.show()


def main():
    # Параметры
    model_path = os.path.join('models', 'fasterrcnn_model.pth')  # Путь к сохраненной модели
    test_image = os.path.join('images', 'A3.png')  # Замените на ваше тестовое изображение
    annotations_file = os.path.join('annotations', 'train.csv')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    threshold = 0.5  # Порог уверенности для отображения предсказаний

    # Загрузка аннотаций для определения числа классов
    annotations = load_annotations(annotations_file)
    all_labels = set()
    for ann in annotations.values():
        all_labels.update(ann['labels'])
    num_classes = len(all_labels) + 1  # +1 за фон

    # Создание сопоставления индексов классов с названиями
    class_to_idx = {}
    for img_ann in annotations.values():
        for label in img_ann['labels']:
            class_to_idx[label] = 1  # Начинаем с 1 для классов
    class_to_idx["__background__"] = 0
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Загрузка модели
    model = load_model(model_path, num_classes, device)
    print("Модель загружена и готова к использованию.")

    # Определение трансформаций
    transforms = get_transform(train=False)

    # Предсказание
    prediction = predict(model, test_image, transforms, device, threshold=threshold)

    # Визуализация
    visualize_prediction(test_image, prediction, idx_to_class, threshold=threshold)


if __name__ == "__main__":
    main()

