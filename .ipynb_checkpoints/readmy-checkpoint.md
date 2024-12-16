project/
├── annotations/
│   └── train.csv
├── images/
│   ├── A3.png
│   └── ... # другие изображения
├── train.py
├── detect.py
└── utils.py

1. utils.py — для загрузки и предобработки данных.

2. train.py — для обучения модели.

3. detect.py — для обнаружения объектов на новых изображениях.

▎1. utils.py

Этот модуль отвечает за загрузку аннотаций, создание пользовательского датасета и определение необходимых трансформаций.