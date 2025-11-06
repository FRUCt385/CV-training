from ultralytics import YOLO
from pathlib import Path




if __name__ == '__main__':
    model_dir = Path.cwd() / 'models'
    model = YOLO(model_dir / "yolo11n.pt")

    results = model.train(
        data=model_dir / 'face mask.yaml',  # путь к вашему data.yaml
        epochs=100,                      # число эпох
        imgsz=640,                      # размер изображения
        batch=16,                       # размер батча (уменьшите, если не хватает памяти)
        device='cuda',                  # 'cpu' или 'cuda' или 0,1 [список GPU]
        project=model_dir,
        patience=10,
        workers=4,                      # число workers для загрузки данных
        lr0=0.01,                       # начальный learning rate
        pretrained=True,                # использовать предобученные веса
        optimizer='auto'                 # или 'AdamW'
    )